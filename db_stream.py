"""
db_stream.py — Streamlit-safe SQL Server streaming helpers.

This is a reworked version of the original `diagnostic_utils.py` streaming code.

WHY THE REWORK:
    The original `stream_new_data()` is a blocking infinite generator
    (`while True: ... time.sleep(...)`). That model cannot run inside Streamlit:
    the script must hand control back to the runtime on every cycle so the UI can
    render, toggles work, and the loop can be stopped. Here the *cadence* is owned
    by the Streamlit page (via st.rerun on a timer); this module only exposes a
    single non-blocking "poll once" call plus the initial full-pull/cache logic.

BUG FIXES vs. the original snippet:
    1. Blocking loop removed  -> poll_new_data() returns immediately.
    2. Engines are cached per-DB (@st.cache_resource) instead of rebuilt per poll.
    3. CSV append reindexes to the cache header first, so column-order drift or a
       missing column can never silently misalign the cache file.
    4. Buffer is sorted by DateTime before any diff/rolling math downstream.
    5. Watermark handling documented: WHERE DateTime > wm is *strictly* greater, so
       rows sharing the exact max timestamp are intentionally not re-fetched.
    6. Errors are returned as structured status, not just printed, so each DB tab
       can surface its own connection state.
    7. Credentials are passed in (from config / st.secrets), never hardcoded.
"""
from __future__ import annotations

import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import URL


# ----------------------------
# Connection builder (unchanged API, kept for parity with diagnostic_utils)
# ----------------------------
def build_connection_url(
    server,
    database,
    username=None,
    password=None,
    driver="ODBC Driver 18 for SQL Server",
    trusted=False,
    encrypt="yes",
    trust_cert="yes",
    port=None,
):
    """Build a SQLAlchemy URL for SQL Server via the mssql+pyodbc dialect."""
    host = server if port is None else f"{server},{port}"
    query = {"driver": driver, "Encrypt": encrypt, "TrustServerCertificate": trust_cert}
    if trusted:
        query["Trusted_Connection"] = "yes"
        return URL.create("mssql+pyodbc", host=host, database=database, query=query)
    return URL.create(
        "mssql+pyodbc",
        username=username,
        password=password,
        host=host,
        database=database,
        query=query,
    )


@st.cache_resource(show_spinner=False)
def get_engine(server, database, username, password, driver="ODBC Driver 18 for SQL Server",
               trusted=False, encrypt="yes", trust_cert="yes", port=None):
    """
    Cached engine, one per unique connection tuple. @st.cache_resource keeps a single
    engine alive across reruns so we don't reopen a pool on every poll. The cache key is
    the full argument set, so different DBs get different engines automatically.
    """
    url = build_connection_url(server, database, username, password, driver,
                               trusted, encrypt, trust_cert, port)
    # fast_executemany only helps bulk INSERTs; harmless for our read-only use.
    return create_engine(url, fast_executemany=True)


def get_dynamic_columns(engine, table_name, search_string, include_datetime=True):
    """Return all column names in `table_name` whose name contains `search_string`."""
    query = text(
        """
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = :table
          AND COLUMN_NAME LIKE :search
        """
    )
    with engine.connect() as conn:
        df_cols = pd.read_sql_query(query, conn, params={"table": table_name, "search": f"%{search_string}%"})
    found = df_cols["COLUMN_NAME"].tolist()
    if include_datetime and "DateTime" not in found:
        found.insert(0, "DateTime")
    return found


def load_or_fetch_entire_database(engine, table, cache_filename="full_database_cache.csv"):
    """
    Load a local CSV cache if present, else fetch the whole table once and cache it.
    Kept close to the original, but always returns a DateTime-sorted frame so downstream
    diff/rolling math is well-defined.
    """
    if os.path.exists(cache_filename):
        df = pd.read_csv(cache_filename, parse_dates=["DateTime"])
    else:
        query = text(f"SELECT * FROM {table} ORDER BY DateTime ASC")
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn)
        df.to_csv(cache_filename, index=False)
    if "DateTime" in df.columns:
        df = df.sort_values("DateTime").reset_index(drop=True)
    return df


def _append_to_cache(new_batch: pd.DataFrame, cache_filename: str):
    """
    Append rows to the CSV cache WITHOUT risking column misalignment.

    The original `to_csv(mode='a', header=False)` trusts that batch column order
    matches the file's header forever. Here we read the existing header (cheap: one
    row) and reindex the batch to it before appending. Missing columns become NaN
    instead of shifting every subsequent value into the wrong column.
    """
    if not os.path.exists(cache_filename):
        new_batch.to_csv(cache_filename, index=False)
        return
    try:
        header = pd.read_csv(cache_filename, nrows=0).columns.tolist()
        aligned = new_batch.reindex(columns=header)
        aligned.to_csv(cache_filename, mode="a", header=False, index=False)
    except Exception:
        # Never let cache-writing take down the live loop.
        pass


def poll_new_data(engine, table, watermark, columns=None, max_rows=None):
    """
    Non-blocking single poll: fetch every row with DateTime strictly greater than
    `watermark`, ordered ascending. Returns (dataframe, new_watermark, error_str).

    - dataframe: new rows (possibly empty).
    - new_watermark: updated ISO high-water mark (unchanged if nothing new).
    - error_str: None on success, else a short message for the UI.

    `columns` may be an explicit list to SELECT; None means SELECT *.
    """
    if columns:
        # De-dupe while preserving order; guarantee DateTime is present and first.
        seen, ordered = set(), []
        for c in ["DateTime"] + list(columns):
            if c not in seen:
                seen.add(c)
                ordered.append(c)
        col_sql = ", ".join(ordered)
    else:
        col_sql = "*"

    top_sql = f"TOP {int(max_rows)} " if max_rows else ""
    query = text(
        f"SELECT {top_sql}{col_sql} FROM {table} WHERE DateTime > :last_dt ORDER BY DateTime ASC"
    )
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn, params={"last_dt": watermark})
        if not df.empty:
            new_wm = pd.to_datetime(df["DateTime"]).max().isoformat()
            return df, new_wm, None
        return df, watermark, None
    except Exception as e:
        return pd.DataFrame(), watermark, str(e)


def initial_watermark(buffer_df: pd.DataFrame) -> str:
    """ISO high-water mark from an existing buffer, or the epoch if empty."""
    if buffer_df is not None and not buffer_df.empty and "DateTime" in buffer_df.columns:
        return pd.to_datetime(buffer_df["DateTime"]).max().isoformat()
    return "1970-01-01T00:00:00"


def append_batch(buffer_df: pd.DataFrame, new_batch: pd.DataFrame, cache_filename: str,
                 max_buffer_rows: int | None = None) -> pd.DataFrame:
    """
    Append a new batch to the in-memory buffer AND the on-disk cache, keeping the
    buffer sorted by DateTime and (optionally) bounded to the most recent
    `max_buffer_rows` so a long-running live session can't grow without limit.
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
