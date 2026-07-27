"""
live_features.py — Optional Shutter diff/cumsum feature charts for the live page.

This is the feature-engineering from your original streaming snippet, cleaned up:
    * sorts by time before .diff() (concat/stream order is not guaranteed monotonic)
    * guards against missing Shutter columns instead of indexing [0] blindly
    * renames the mislabeled "_ema" — the original used .rolling().mean() (a SIMPLE
      moving average), not an EMA. Here it is offered as either, explicitly.
    * computes only on the tail window so a growing live buffer stays responsive
      instead of recomputing a 10k-row rolling mean over the entire history each poll.

Events on the live page are driven by the RUL engine, not by these charts; this is
purely a secondary diagnostic view the user can toggle on.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def find_shutter_columns(df):
    """Return (close_time_cols, open_count_cols) present in df."""
    close_cols = [c for c in df.columns if "Shutter_Close_Time_1" in c]
    count_cols = [c for c in df.columns if "Shutter_Open_Count" in c]
    return close_cols, count_cols


def build_shutter_figures(buffer_df, diff_periods=10, roll_window=10_000,
                          use_ema=False, tail_rows=50_000):
    """
    Build one Plotly figure per Shutter close-time column.

    Returns a list of (column_name, figure). Empty list if no shutter columns exist.

    diff_periods : lag for .diff() (original used 10)
    roll_window  : smoothing window over the squared diff (original used 10_000)
    use_ema      : True -> ewm() (a real EMA); False -> rolling().mean() (original behavior)
    tail_rows    : only compute over the most recent N rows for responsiveness
    """
    close_cols, count_cols = find_shutter_columns(buffer_df)
    if not close_cols:
        return []

    work = buffer_df.copy()
    if "DateTime" in work.columns:
        work = work.sort_values("DateTime").set_index("DateTime")   # FIX: sort before diff
    if tail_rows and len(work) > tail_rows:
        work = work.iloc[-tail_rows:]

    figures = []
    for col in close_cols:
        d = work[col].diff(diff_periods)
        d2 = d.pow(2)
        if use_ema:
            smooth = d2.ewm(span=roll_window, adjust=False).mean()
            smooth_name = f"{col}_diff_sq_ema"
        else:
            smooth = d2.rolling(window=roll_window, min_periods=1).mean()
            smooth_name = f"{col}_diff_sq_movavg"
        cum = smooth.cumsum()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=work.index, y=work[col], mode="lines", name=col))
        fig.add_trace(go.Scatter(x=work.index, y=d, mode="lines", name=f"{col}_diff"))
        fig.add_trace(go.Scatter(x=work.index, y=d2, mode="lines", name=f"{col}_diff_sq"))
        fig.add_trace(go.Scatter(x=work.index, y=smooth, mode="lines", name=smooth_name))
        fig.add_trace(go.Scatter(x=work.index, y=cum, mode="lines", name=f"{smooth_name}_cumsum"))
        if count_cols:                                              # FIX: guard [0]
            fig.add_trace(go.Scatter(x=work.index, y=work[count_cols[0]],
                                     mode="lines", name=count_cols[0]))
        fig.update_layout(title=f"Live Stream — {col} & Derived Metrics",
                          xaxis_title="DateTime", yaxis_title="Value",
                          template="plotly_white", height=380,
                          margin=dict(t=40, b=40, l=40, r=10))
        figures.append((col, fig))
    return figures
