"""
db_stream.py — Streamlit-safe live acquisition on top of the export pipeline.

The live monitor pulls and shapes data with the SAME code the historical backtest
uses (db_export). Each acquisition:

    export_table_to_csv(use_decode=True, decode_function="CPDecode", file_format="csv")
        -> decoded LONG csv (DateTime, CPScanIdx, X, Y, ...)
    format_channel_df(...)  -> WIDE frame: DateTime + ~80-100 numeric channel columns

That wide frame is the live buffer, and pages_live runs process_all_channels on it
exactly like the simulator runs it on an uploaded CSV.

WHY A "POLL ONCE" MODEL:
    The original streaming code was a blocking `while True: ... sleep()` generator,
    which cannot run inside Streamlit — the script must return control every cycle so
    the UI renders and the loop can be stopped. Here the cadence is owned by the page
    (st.rerun on a timer); this module only exposes the initial history load plus a
    single non-blocking incremental pull.

WATERMARK:
    build_sql_for_export filters with `v.DateTime >= :dt_filter`, so a poll re-fetches
    the watermark row itself; we drop rows <= watermark after formatting so only
    strictly-new scans are appended.
"""
from __future__ import annotations

import os
import pandas as pd

import db_export as dbx


# Decode settings shared by every live pull (mirrors the historical export script).
DECODE_FUNCTION = "CPDecode"
BLOB_COLUMN = "Delta"
SORT_COLUMN = "X"


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _decoded_base(cfg, suffix: str) -> str:
    """Base path (no extension) for a decoded long-csv scratch file for this DB."""
    d = os.path.dirname(cfg["cache_file"]) or "."
    return os.path.join(d, f"{cfg['key']}_decoded_{suffix}")


def _pull_decoded_long(cfg, output_base, datetime_filter=None, chunksize=None) -> str:
    """
    Run the CPDecode export to a LONG csv via the shared export pipeline and return
    the written path (output_base + '.csv').
    """
    _ensure_parent_dir(output_base)
    return dbx.export_table_to_csv(
        server=cfg["server"],
        database=cfg["database"],
        output_csv_path=output_base,
        username=cfg["username"],
        password=cfg["password"],
        backend="sqlalchemy",
        file_format="csv",
        use_decode=True,
        decode_function=DECODE_FUNCTION,
        blob_column=BLOB_COLUMN,
        table=cfg["table"],
        datetime_filter=datetime_filter,
        chunksize=chunksize,
        sort_column=SORT_COLUMN,
    )


def _long_csv_to_wide(long_csv_path: str) -> pd.DataFrame:
    """
    Read a decoded long csv and format it to the wide channel frame the RUL engine
    expects: a DateTime column plus one string-named column per chamber (X), values=Y.
    """
    long_df = pd.read_csv(long_csv_path)
    if long_df.empty:
        return pd.DataFrame(columns=["DateTime"])

    wide = dbx.format_channel_df(long_df)          # DateTime-indexed, columns = X
    if wide is None or wide.empty:
        return pd.DataFrame(columns=["DateTime"])

    wide = wide.reset_index()                      # DateTime back to a plain column
    wide.columns = ["DateTime"] + [str(c) for c in wide.columns[1:]]
    wide["DateTime"] = pd.to_datetime(wide["DateTime"])
    return wide.sort_values("DateTime").reset_index(drop=True)


def load_or_fetch_history(cfg, chunksize=200_000) -> pd.DataFrame:
    """
    Warm-start a DB's live buffer: read the wide CSV cache if present, else pull the
    full decoded history once (CPDecode export -> format_channel_df), cache the wide
    result, and return it. Always DateTime-sorted with string channel columns.
    """
    cache_file = cfg["cache_file"]
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, parse_dates=["DateTime"])
        df.columns = [str(c) for c in df.columns]
        if "DateTime" in df.columns:
            df = df.sort_values("DateTime").reset_index(drop=True)
        return df

    _ensure_parent_dir(cache_file)
    long_path = _pull_decoded_long(cfg, _decoded_base(cfg, "history"), datetime_filter=None, chunksize=chunksize)
    wide = _long_csv_to_wide(long_path)
    wide.to_csv(cache_file, index=False)
    return wide


def poll_new_data(cfg, watermark, max_rows=None):
    """
    Non-blocking incremental pull for one DB. Returns (wide_df, new_watermark, error).

    Exports only decoded scans with DateTime >= watermark, formats them to wide channel
    rows, then keeps strictly-new scans (DateTime > watermark). new_watermark is the max
    DateTime of the returned rows (unchanged if nothing new). max_rows caps the number of
    new wide rows appended in one cycle (0/None = no cap).
    """
    try:
        long_path = _pull_decoded_long(
            cfg, _decoded_base(cfg, "delta"), datetime_filter=watermark, chunksize=None
        )
        wide = _long_csv_to_wide(long_path)
    except Exception as e:
        return pd.DataFrame(), watermark, str(e)

    if wide.empty:
        return pd.DataFrame(), watermark, None

    if watermark:
        wm_ts = pd.to_datetime(watermark)
        wide = wide[wide["DateTime"] > wm_ts]
    if wide.empty:
        return pd.DataFrame(), watermark, None

    wide = wide.sort_values("DateTime").reset_index(drop=True)
    if max_rows:
        wide = wide.iloc[: int(max_rows)]

    new_wm = wide["DateTime"].max().isoformat()
    return wide, new_wm, None


def initial_watermark(buffer_df: pd.DataFrame) -> str:
    """ISO high-water mark from an existing buffer, or the epoch if empty."""
    if buffer_df is not None and not buffer_df.empty and "DateTime" in buffer_df.columns:
        return pd.to_datetime(buffer_df["DateTime"]).max().isoformat()
    return "1970-01-01T00:00:00"


def _append_to_cache(new_batch: pd.DataFrame, cache_filename: str):
    """
    Append rows to the wide CSV cache without risking column misalignment: reindex the
    batch to the existing header first, so a new/missing channel becomes NaN instead of
    shifting every subsequent value into the wrong column.
    """
    if not os.path.exists(cache_filename):
        _ensure_parent_dir(cache_filename)
        new_batch.to_csv(cache_filename, index=False)
        return
    try:
        header = pd.read_csv(cache_filename, nrows=0).columns.tolist()
        aligned = new_batch.reindex(columns=header)
        aligned.to_csv(cache_filename, mode="a", header=False, index=False)
    except Exception:
        # Never let cache-writing take down the live loop.
        pass


def append_batch(buffer_df: pd.DataFrame, new_batch: pd.DataFrame, cache_filename: str,
                 max_buffer_rows: int | None = None) -> pd.DataFrame:
    """
    Append a new batch to the in-memory buffer AND the wide CSV cache, keeping the buffer
    DateTime-sorted, de-duplicated on DateTime, and optionally bounded to the most recent
    `max_buffer_rows` so a long-running session can't grow without limit.
    """
    _append_to_cache(new_batch, cache_filename)
    combined = pd.concat([buffer_df, new_batch], ignore_index=True)
    if "DateTime" in combined.columns:
        combined = (
            combined.drop_duplicates(subset="DateTime", keep="last")
            .sort_values("DateTime")
            .reset_index(drop=True)
        )
    if max_buffer_rows and len(combined) > max_buffer_rows:
        combined = combined.iloc[-max_buffer_rows:].reset_index(drop=True)
    return combined
