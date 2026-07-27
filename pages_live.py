"""
pages_live.py — Live multi-database monitoring.

WHAT THIS DOES
    Streams new rows from up to 3 SQL Server databases, appends them to a per-DB
    in-memory buffer (+ CSV cache), runs the SHARED RUL engine (lib_core) on the
    live buffer exactly as the simulator runs it on an uploaded CSV, and emits a
    status-change event (reusing send_custom_telegram) whenever a channel crosses
    a health threshold.

USABILITY MODEL (the point of the request)
    * Each configured DB has an independent ON/OFF toggle. Any subset can run.
    * Each ACTIVE DB gets its OWN tab (st.tabs), so when all three run their charts,
      audit logs, and connection state stay individually readable and never interleave.
    * A single global Start/Stop drives the polling cadence for whatever is active.

STREAMLIT-SAFE LOOP
    No while-True. When "live" is on, the page polls every active DB once, renders,
    then schedules the next cycle with st.rerun() after `poll_interval` seconds. The
    user can stop, toggle DBs, or leave the page at any time — control always returns
    to the runtime between polls.
"""
import time
import numpy as np
import pandas as pd
import streamlit as st

from lib_core import (
    get_available_channels, load_my_sensor_data, AVAILABLE_MODELS,
    evaluate_all_models, build_dynamic_std_fn, solve_rul_root,
    send_custom_telegram, format_rul_str, SENTINEL_ALREADY_REACHED,
)
import db_stream as dbs
from live_features import build_shutter_figures, find_shutter_columns


# =========================================================
# DATABASE REGISTRY
# =========================================================
# The three live sources. `password` falls back to st.secrets so nothing is hardcoded
# in the file; edit `key`/`label`/connection fields to match your environment.
def get_db_registry():
    def _secret(name, default=None):
        try:
            return st.secrets[name]
        except Exception:
            return default

    return [
        {
            "key": "alunorf_1",
            "label": "Alunorf 1",
            "server": "dev-mars",
            "database": "136816",
            "table": "ValuesHotMillProfileGaugeService",
            "username": _secret("DB1_USER", "sa"),
            "password": _secret("DB1_PASSWORD", None),
            "cache_file": "alunorf_1_cache.csv",
        },
        {
            "key": "alunorf_2",
            "label": "Alunorf 2",
            "server": "dev-mars",
            "database": "136817",
            "table": "ValuesHotMillProfileGaugeService",
            "username": _secret("DB2_USER", "sa"),
            "password": _secret("DB2_PASSWORD", None),
            "cache_file": "alunorf_2_cache.csv",
        },
        {
            "key": "alunorf_3",
            "label": "Alunorf 3",
            "server": "dev-mars",
            "database": "136818",
            "table": "ValuesHotMillProfileGaugeService",
            "username": _secret("DB3_USER", "sa"),
            "password": _secret("DB3_PASSWORD", None),
            "cache_file": "alunorf_3_cache.csv",
        },
    ]


# =========================================================
# PER-DB STATE (namespaced so tabs never bleed into each other)
# =========================================================
def _db_state(key):
    """Return the mutable state dict for one DB, creating it on first use."""
    root = st.session_state.setdefault("live_db_state", {})
    if key not in root:
        root[key] = {
            "buffer": None,          # in-memory DataFrame
            "watermark": None,       # ISO high-water mark
            "channel_states": {},    # {channel: 'ok'|'warning'|'critical'}
            "channel_ruls": {},      # {channel: last nominal RUL}
            "audit_log": [],         # newest-first list of event dicts
            "conn_error": None,      # last connection error string
            "last_poll": None,       # timestamp of last successful poll
            "rows_total": 0,
            "initialized": False,    # has the full pull / cache load happened
        }
    return root[key]


def _reset_db_state(key):
    st.session_state.get("live_db_state", {}).pop(key, None)


# =========================================================
# CORE: run the shared RUL engine on a live buffer for one DB
# =========================================================
def evaluate_buffer_for_events(cfg, state, params_ui):
    """
    Feed the DB's live buffer through the SAME pipeline the simulator uses
    (load_my_sensor_data -> evaluate_all_models -> solve_rul_root), and append a
    status-change event to the DB's audit log whenever a channel changes health.

    params_ui carries the thresholds and options chosen in the sidebar.
    """
    buffer_df = state["buffer"]
    if buffer_df is None or buffer_df.empty or "DateTime" not in buffer_df.columns:
        return

    # load_my_sensor_data expects a DateTime index (same shape as parse_raw_csv output).
    df_indexed = buffer_df.copy()
    df_indexed["DateTime"] = pd.to_datetime(df_indexed["DateTime"])
    df_indexed = df_indexed.set_index("DateTime")
    df_indexed = df_indexed[~df_indexed.index.duplicated(keep="first")].sort_index()

    target_threshold = params_ui["target_threshold"]
    rul_warn_days = params_ui["rul_warn_days"]
    rul_crit_days = params_ui["rul_crit_days"]
    outlier_factor = params_ui["iqr_factor"]
    outlier_window = params_ui["iqr_window"]
    lookback_days = params_ui["lookback_days"]
    dry_run = params_ui["dry_run"]

    priority = {m: i for i, m in enumerate(AVAILABLE_MODELS)}
    channels = get_available_channels(df_indexed)

    for ch in channels:
        smooth, raw, elapsed = load_my_sensor_data(
            df_indexed, col=ch, outlier_factor=outlier_factor, outlier_window=outlier_window
        )
        if elapsed is None or elapsed.empty:
            continue

        # Use the most recent `lookback_days` daily steps, like the simulator engine.
        n = len(elapsed)
        start = max(0, n - lookback_days)
        time_data = elapsed.iloc[start:].values
        sensor_smooth = smooth.iloc[start:].values
        if np.isnan(sensor_smooth).sum() > len(sensor_smooth) * 0.8:
            continue

        top_models, _ = evaluate_all_models(time_data, sensor_smooth, priority)
        new_status, nom_rul, upper_rul, lower_rul = "ok", None, None, None
        best_model = None

        if top_models:
            best_model = list(top_models.keys())[0]
            p = top_models[best_model]["params"]
            func = top_models[best_model]["func"]
            t_max = np.max(time_data) if np.max(time_data) > 0 else 1.0

            sensor_raw = raw.iloc[start:].values
            time_norm = time_data / t_max
            preds = func(time_norm, *p)
            residuals = pd.Series(sensor_raw - preds)
            rolling_std = residuals.rolling(window=20, min_periods=1).std().bfill().fillna(0)
            valid_std = ~np.isnan(rolling_std)
            std_slope, std_intercept = (
                np.polyfit(time_norm[valid_std], rolling_std[valid_std], 1)
                if valid_std.sum() > 1 else (0.0, rolling_std.iloc[-1])
            )
            dyn_std_fn = build_dynamic_std_fn(std_slope, std_intercept)

            nom_time = solve_rul_root(func, p, target_threshold, t_max, dyn_std_fn, 1.645, "nominal")
            upper_time = solve_rul_root(func, p, target_threshold, t_max, dyn_std_fn, 1.645, "upper")
            lower_time = solve_rul_root(func, p, target_threshold, t_max, dyn_std_fn, 1.645, "lower")

            def calc_rul(t_val):
                return t_val if t_val in ("Safe", SENTINEL_ALREADY_REACHED) else t_val - np.max(time_data)

            nom_rul, upper_rul, lower_rul = calc_rul(nom_time), calc_rul(upper_time), calc_rul(lower_time)

            if isinstance(nom_rul, (float, int)):
                if nom_rul < rul_crit_days:
                    new_status = "critical"
                elif nom_rul < rul_warn_days:
                    new_status = "warning"

        old_status = state["channel_states"].get(ch, "ok")
        old_rul = state["channel_ruls"].get(ch, None)

        if new_status != old_status:
            api_result = send_custom_telegram(
                channel=ch,
                status=new_status,
                nominal_rul_days=nom_rul if isinstance(nom_rul, (int, float)) else None,
                early_rul_days=upper_rul if isinstance(upper_rul, (int, float)) else None,
                late_rul_days=lower_rul if isinstance(lower_rul, (int, float)) else None,
                limit_value=target_threshold,
                dry_run=dry_run,
                best_model=best_model or "Unknown",
            )
            err_msg = "" if api_result["sent"] or dry_run else f" ❌ SERVER ERROR: {api_result.get('error', 'Unknown')}"
            state["audit_log"].insert(0, {
                "ts": pd.Timestamp.now().strftime("%H:%M:%S"),
                "channel": ch,
                "status": new_status,
                "old_status": old_status,
                "new_rul_str": format_rul_str(nom_rul, short=True),
                "old_rul_str": format_rul_str(old_rul, short=True),
                "payload": api_result["body"],
                "err_msg": err_msg,
            })
            state["channel_states"][ch] = new_status
        state["channel_ruls"][ch] = nom_rul


# =========================================================
# POLL: one cycle for one DB
# =========================================================
def poll_one_db(cfg, params_ui):
    state = _db_state(cfg["key"])

    if cfg["password"] is None:
        state["conn_error"] = "No password configured (set DBx_PASSWORD in secrets)."
        return

    try:
        engine = dbs.get_engine(cfg["server"], cfg["database"], cfg["username"], cfg["password"])
    except Exception as e:
        state["conn_error"] = f"Engine build failed: {e}"
        return

    # First cycle for this DB: full pull / cache load + set the watermark.
    if not state["initialized"]:
        try:
            buffer_df = dbs.load_or_fetch_entire_database(engine, cfg["table"], cfg["cache_file"])
            state["buffer"] = buffer_df
            state["watermark"] = dbs.initial_watermark(buffer_df)
            state["rows_total"] = len(buffer_df)
            state["initialized"] = True
            state["conn_error"] = None
        except Exception as e:
            state["conn_error"] = f"Initial load failed: {e}"
            return

    # Incremental poll.
    new_batch, new_wm, err = dbs.poll_new_data(
        engine, cfg["table"], state["watermark"], columns=None, max_rows=params_ui["max_rows_per_poll"]
    )
    if err:
        state["conn_error"] = err
        return
    state["conn_error"] = None
    state["last_poll"] = pd.Timestamp.now().strftime("%H:%M:%S")

    if not new_batch.empty:
        state["buffer"] = dbs.append_batch(
            state["buffer"], new_batch, cfg["cache_file"],
            max_buffer_rows=params_ui["max_buffer_rows"]
        )
        state["watermark"] = new_wm
        state["rows_total"] = len(state["buffer"])
        # Only re-evaluate when new data actually arrived (saves compute).
        evaluate_buffer_for_events(cfg, state, params_ui)


# =========================================================
# RENDER: one DB's tab
# =========================================================
def render_db_tab(cfg, params_ui):
    state = _db_state(cfg["key"])
    key = cfg["key"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows buffered", f"{state['rows_total']:,}")
    active_alerts = sum(1 for s in state["channel_states"].values() if s in ("warning", "critical"))
    c2.metric("Active alerts", active_alerts)
    c3.metric("Last poll", state["last_poll"] or "—")
    c4.metric("Events logged", len(state["audit_log"]))

    if state["conn_error"]:
        st.error(f"Connection issue: {state['conn_error']}")
    elif not state["initialized"]:
        st.info("Waiting for first poll. Press **Start Live Monitoring** above.")

    # --- Optional secondary chart: Shutter feature engineering ---
    if params_ui["show_shutter"] and state["buffer"] is not None and not state["buffer"].empty:
        close_cols, _ = find_shutter_columns(state["buffer"])
        if close_cols:
            with st.expander("🔧 Shutter diff / cumsum diagnostics", expanded=False):
                figs = build_shutter_figures(
                    state["buffer"], diff_periods=params_ui["diff_periods"],
                    roll_window=params_ui["roll_window"], use_ema=params_ui["use_ema"],
                )
                for _col, fig in figs:
                    st.plotly_chart(fig, use_container_width=True, key=f"shutter_{key}_{_col}")

    # --- Active alerts summary ---
    if active_alerts:
        st.markdown("#### ⚠️ Active Alerts")
        alert_map = {ch: s for ch, s in state["channel_states"].items() if s in ("warning", "critical")}
        cols = st.columns(min(len(alert_map), 6))
        for i, (ch, s) in enumerate(alert_map.items()):
            icon = "🔴" if s == "critical" else "🟡"
            cols[i % 6].markdown(f"{icon} **Ch {ch}** — {format_rul_str(state['channel_ruls'].get(ch), short=True)}")

    # --- Per-DB audit log (isolated to this tab) ---
    st.markdown("#### 📡 Event Log")
    st.caption(f"Newest first · {cfg['label']} · Mode: **{'Live' if not params_ui['dry_run'] else 'Dry Run'}**")
    with st.container(height=360):
        if not state["audit_log"]:
            st.write("No events yet.")
        for entry in state["audit_log"][:50]:
            icon = "🔴" if entry["status"] == "critical" else "🟡" if entry["status"] == "warning" else "🟢"
            title = f"{icon} {entry['ts']} · Ch {entry['channel']} ({entry['old_status'].upper()} ➔ {entry['status'].upper()})"
            with st.expander(title):
                st.write(f"**RUL:** {entry['old_rul_str']} ➔ **{entry['new_rul_str']}**")
                if entry.get("err_msg"):
                    st.error(entry["err_msg"])
                st.json(entry["payload"])

    if st.button(f"🗑️ Reset {cfg['label']} buffer & log", key=f"reset_{key}"):
        _reset_db_state(key)
        st.rerun()


# =========================================================
# PAGE ENTRY
# =========================================================
def render():
    st.title("📶 Live Multi-Database Monitoring")
    st.caption("Stream new records from one or more databases and evaluate them for RUL events in real time.")

    registry = get_db_registry()

    # ---------------- Sidebar controls ----------------
    with st.sidebar:
        st.header("Live Sources")
        st.caption("Activate any combination. Each active DB gets its own tab.")
        active_keys = []
        for cfg in registry:
            on = st.toggle(cfg["label"], value=st.session_state.get(f"active_{cfg['key']}", False),
                           key=f"active_{cfg['key']}")
            if on:
                active_keys.append(cfg["key"])

        st.divider()
        st.header("Loop")
        live = st.toggle("▶️ Start Live Monitoring", key="live_running",
                         help="When on, every active DB is polled on the interval below.")
        poll_interval = st.number_input("Poll interval (s)", 5, 3600, 60,
                                        help="Seconds between polls. The DB-side function default is 60s.")
        max_rows_per_poll = st.number_input("Max rows per poll", 0, 1_000_000, 0,
                                            help="0 = no cap. Caps a single SELECT so a huge backlog can't stall the UI.")
        max_buffer_rows = st.number_input("Max buffer rows (in memory)", 0, 10_000_000, 500_000,
                                          help="0 = unbounded. Otherwise keep only the most recent N rows per DB.")

        st.divider()
        st.header("RUL Parameters")
        target_threshold = st.number_input("Failure Threshold Limit", value=0.2, format="%.3f")
        rul_warn_days = st.number_input("🟡 Warning (< days)", value=30)
        rul_crit_days = st.number_input("🔴 Critical (< days)", value=10)
        lookback_days = st.slider("Lookback Window (Steps)", 100, 1000, 300)
        iqr_window = st.slider("IQR Window Size", 5, 50, 20)
        iqr_factor = st.slider("IQR Factor", 0.5, 3.0, 1.5)
        dry_run = not st.toggle("📡 Live API Sending", value=False,
                                help="OFF = Dry Run (events computed & logged, nothing POSTed).")

        st.divider()
        st.header("Shutter Diagnostics (optional)")
        show_shutter = st.toggle("Show shutter diff/cumsum charts", value=False)
        diff_periods = st.number_input("diff() periods", 1, 500, 10, disabled=not show_shutter)
        roll_window = st.number_input("smoothing window", 10, 100_000, 10_000, disabled=not show_shutter)
        use_ema = st.toggle("Use EMA instead of moving average", value=False, disabled=not show_shutter)

    params_ui = {
        "target_threshold": target_threshold, "rul_warn_days": rul_warn_days,
        "rul_crit_days": rul_crit_days, "lookback_days": lookback_days,
        "iqr_window": iqr_window, "iqr_factor": iqr_factor, "dry_run": dry_run,
        "max_rows_per_poll": (max_rows_per_poll or None),
        "max_buffer_rows": (max_buffer_rows or None),
        "show_shutter": show_shutter, "diff_periods": diff_periods,
        "roll_window": roll_window, "use_ema": use_ema,
    }

    if not active_keys:
        st.info("👈 Activate at least one database in the sidebar to begin.")
        return

    # ---------------- Poll active DBs (one cycle per rerun) ----------------
    if live:
        with st.spinner("Polling active databases..."):
            for cfg in registry:
                if cfg["key"] in active_keys:
                    poll_one_db(cfg, params_ui)

    # ---------------- One tab per ACTIVE db (stays individually readable) ----------------
    active_cfgs = [c for c in registry if c["key"] in active_keys]
    status_dot = lambda c: ("🔴" if any(s == "critical" for s in _db_state(c["key"])["channel_states"].values())
                            else "🟡" if any(s == "warning" for s in _db_state(c["key"])["channel_states"].values())
                            else "🟢")
    tabs = st.tabs([f"{status_dot(c)} {c['label']}" for c in active_cfgs])
    for tab, cfg in zip(tabs, active_cfgs):
        with tab:
            render_db_tab(cfg, params_ui)

    # ---------------- Schedule next cycle (Streamlit-safe; no while-True) ----------------
    if live:
        time.sleep(poll_interval)
        st.rerun()
