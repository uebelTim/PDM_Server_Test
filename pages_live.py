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
import numpy as np
import pandas as pd
import streamlit as st

from lib_core import (
    process_all_channels, AVAILABLE_MODELS, fit_and_plotly_model,
    evaluate_all_models, build_dynamic_std_fn, solve_rul_root,
    send_custom_telegram, delete_telegrams, format_rul_str, SENTINEL_ALREADY_REACHED,
)
import db_stream as dbs
from live_features import build_shutter_figures, find_shutter_columns


# =========================================================
# DATABASE REGISTRY
# =========================================================
# The three live sources. All databases share ONE username/password, read from
# st.secrets (DB_USER / DB_PASSWORD) so nothing is hardcoded in the file.
# Only the database id, label and cache file differ per source.
def get_db_registry():
    def _secret(name, default=None):
        try:
            return st.secrets[name]
        except Exception:
            return default

    shared_user = _secret("DB_USER", "sa")
    shared_password = _secret("DB_PASSWORD", None)

    sources = [
        {"key": "alunorf_1", "label": "Alunorf 1", "database": "136816"},
        {"key": "alunorf_2", "label": "Alunorf 2", "database": "140988"},
        {"key": "Elval", "label": "Alunorf 3", "database": "134520"},
    ]

    return [
        {
            "key": s["key"],
            "label": s["label"],
            "server": "dev-mars",
            "database": s["database"],
            # BLOB table: db_stream joins it to its Values… partner and decodes it
            # via CPDecode, then pivots to one column per chamber.
            "table": "BlobsHotMillProfileGaugeServiceChannel",
            "username": shared_user,
            "password": shared_password,
            "cache_file": f"data/{s['key']}_cache.csv",
        }
        for s in sources
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
            "selected_channel": None,  # channel whose prediction chart is shown
        }
    return root[key]


def _reset_db_state(key):
    st.session_state.get("live_db_state", {}).pop(key, None)


def _select_channel(key, ch):
    """on_click callback: runs before the rerun's widgets are drawn, so the button
    reflects the new selection (red) on the SAME click instead of one click late."""
    _db_state(key)["selected_channel"] = ch


def _clear_channel(key):
    _db_state(key)["selected_channel"] = None


# =========================================================
# When did the breach actually happen?
# =========================================================
def _breach_time(series, threshold, sustained_days):
    """
    First DateTime where `series` reaches/exceeds `threshold` and then stays >= threshold
    for at least `sustained_days` days — the point we treat the limit as *truly* crossed.
    Returns the FIRST such sustained crossing (short spikes that fall back below are
    skipped), or None if no run is long enough.

    Detection runs on the channel's least-lagged available envelope (the rolling-max
    `raw` series), NOT the heavily-smoothed EMA, so the reported time tracks the actual
    crossing instead of trailing it by months. `series` is daily-indexed.
    """
    s = series.dropna()
    if s.empty:
        return None
    above = (s >= threshold).values
    idx = s.index
    need = max(1, int(sustained_days))
    n = len(s)
    i = 0
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:
            j += 1
        # Contiguous above-run spans idx[i]..idx[j-1] inclusive.
        if (idx[j - 1] - idx[i]).days + 1 >= need:
            return idx[i]
        i = j
    return None


def _indexed_buffer(buffer_df):
    """
    Buffer -> DateTime-indexed, de-duplicated, sorted frame — the shape the RUL engine
    (load_my_sensor_data / process_all_channels) expects, same as parse_raw_csv output.
    Returns None if the buffer is empty or has no DateTime column.
    """
    if buffer_df is None or buffer_df.empty or "DateTime" not in buffer_df.columns:
        return None
    df = buffer_df.copy()
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.set_index("DateTime")
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


# =========================================================
# CORE: run the shared RUL engine on a live buffer for one DB
# =========================================================
def evaluate_buffer_for_events(cfg, state, params_ui, force_send=False):
    """
    Feed the DB's live buffer through the SAME pipeline the simulator uses
    (load_my_sensor_data -> evaluate_all_models -> solve_rul_root), and append a
    status-change event to the DB's audit log whenever a channel changes health.

    params_ui carries the thresholds and options chosen in the sidebar.

    force_send=True additionally (re)sends a telegram for every channel currently in
    warning/critical even if its status did not change — used when Live API Sending is
    switched on so alerts already present get pushed to the server.
    """
    # load_my_sensor_data expects a DateTime index (same shape as parse_raw_csv output).
    df_indexed = _indexed_buffer(state["buffer"])
    if df_indexed is None:
        return

    target_threshold = params_ui["target_threshold"]
    rul_warn_days = params_ui["rul_warn_days"]
    rul_crit_days = params_ui["rul_crit_days"]
    outlier_factor = params_ui["iqr_factor"]
    outlier_window = params_ui["iqr_window"]
    lookback_days = params_ui["lookback_days"]
    dry_run = params_ui["dry_run"]
    sustained_days = params_ui.get("sustained_days", 7)

    priority = {m: i for i, m in enumerate(AVAILABLE_MODELS)}

    # Same processing as the historical backtest: build per-channel smooth/raw/elapsed
    # series for every channel up front, then evaluate each channel's recent window.
    all_channel_data = process_all_channels(df_indexed, outlier_factor, outlier_window)

    for ch, data in all_channel_data.items():
        smooth, raw, elapsed = data["smooth"], data["raw"], data["elapsed"]
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

        status_changed = new_status != old_status
        # Send on a real status change, OR when force_send is set and the channel is
        # currently in an alert state (used to push already-active alerts on turn-on).
        resend = force_send and not status_changed and new_status in ("warning", "critical")

        if status_changed or resend:
            # Timestamp the event with the DATA time, not wall-clock. For a truly-crossed
            # channel that's the first moment the raw envelope stayed above the limit for
            # `sustained_days`; for a predicted/recovery change (no sustained crossing yet)
            # it's the latest step the assessment was made on.
            breach_dt = _breach_time(raw, target_threshold, sustained_days)
            if breach_dt is None:
                raw_valid = raw.dropna()
                breach_dt = raw_valid.index.max() if not raw_valid.empty else pd.Timestamp.now()
            event_dt = pd.Timestamp(breach_dt)
            # Match _now_iso()'s shape: timezone-AWARE, seconds precision. A naive
            # "2022-09-10T00:00:00" (no offset) is not what the server accepts.
            event_iso = event_dt.to_pydatetime().astimezone().isoformat(timespec="seconds")

            api_result = send_custom_telegram(
                channel=ch,
                status=new_status,
                nominal_rul_days=nom_rul if isinstance(nom_rul, (int, float)) else None,
                early_rul_days=upper_rul if isinstance(upper_rul, (int, float)) else None,
                late_rul_days=lower_rul if isinstance(lower_rul, (int, float)) else None,
                limit_value=target_threshold,
                dry_run=dry_run,
                best_model=best_model or "Unknown",
                event_timestamp=event_iso,
                gauge_number=cfg["database"],
            )
            err_msg = "" if api_result["sent"] or dry_run else f" ❌ SERVER ERROR: {api_result.get('error', 'Unknown')}"
            state["audit_log"].insert(0, {
                "ts": event_dt.strftime("%Y-%m-%d %H:%M"),
                "channel": ch,
                "status": new_status,
                "old_status": old_status,
                "new_rul_str": format_rul_str(nom_rul, short=True),
                "old_rul_str": format_rul_str(old_rul, short=True),
                "payload": api_result["body"],
                "err_msg": err_msg,
                "resent": resend,   # True => pushed to server without a status change
            })
            state["channel_states"][ch] = new_status
        state["channel_ruls"][ch] = nom_rul


# =========================================================
# POLL: one cycle for one DB
# =========================================================
def poll_one_db(cfg, params_ui):
    state = _db_state(cfg["key"])

    if cfg["password"] is None:
        state["conn_error"] = "No password configured (set DB_PASSWORD in secrets)."
        return

    # First cycle for this DB: full decode history load (CPDecode export -> wide
    # channels) + set the watermark. The export pipeline builds its own connection.
    if not state["initialized"]:
        with st.spinner(f"Loading {cfg['label']} history via CPDecode… (first load can take a while)"):
            try:
                buffer_df = dbs.load_or_fetch_history(cfg)
                state["buffer"] = buffer_df
                state["watermark"] = dbs.initial_watermark(buffer_df)
                state["rows_total"] = len(buffer_df)
                state["initialized"] = True
                state["conn_error"] = None
            except Exception as e:
                state["conn_error"] = f"Initial load failed: {e}"
                return
        # Evaluate the ENTIRE loaded history once so channels that are already past
        # threshold surface immediately, instead of only reacting to brand-new scans.
        with st.spinner(f"Evaluating {cfg['label']} history…"):
            evaluate_buffer_for_events(cfg, state, params_ui)

    # Incremental poll.
    new_batch, new_wm, err = dbs.poll_new_data(
        cfg, state["watermark"], max_rows=params_ui["max_rows_per_poll"]
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
# RENDER: one channel's RUL prediction chart (same viz as the simulator)
# =========================================================
def _render_channel_prediction(cfg, state, params_ui, ch):
    """
    Recompute the selected channel through the shared pipeline and draw the same
    fit_and_plotly_model chart the simulator shows for its foreground channel.
    process_all_channels is @st.cache_data, so this reuses the work already done during
    evaluation for the current buffer instead of refitting from scratch.
    """
    df_indexed = _indexed_buffer(state["buffer"])
    if df_indexed is None:
        st.info("No buffered data to plot yet.")
        return

    all_channel_data = process_all_channels(df_indexed, params_ui["iqr_factor"], params_ui["iqr_window"])
    if ch not in all_channel_data:
        st.info(f"Channel {ch} has no valid processed data to plot.")
        return

    data = all_channel_data[ch]
    elapsed_full, smooth_full, raw_full = data["elapsed"], data["smooth"], data["raw"]
    n = len(elapsed_full)
    lookback = int(params_ui["lookback_days"])
    threshold = params_ui["target_threshold"]

    # Choose which slice of the timeseries to show. For a channel that has a sustained
    # crossing, center a lookback-sized window ON that crossing so the plot shows the
    # transition (not the flat "still above" tail). Otherwise show the most recent window.
    breach_dt = _breach_time(raw_full, threshold, params_ui.get("sustained_days", 7))
    if breach_dt is not None and n > 0:
        pos = int(elapsed_full.index.get_indexer([breach_dt])[0])
        if pos < 0:
            pos = int(elapsed_full.index.searchsorted(breach_dt))
        start = max(0, pos - lookback // 2)
        end = min(n, start + lookback)
        start = max(0, end - lookback)   # re-clamp if we hit the right edge
        centered = True
    else:
        start, end = max(0, n - lookback), n
        centered = False

    t_data = elapsed_full.iloc[start:end].values
    s_smooth = smooth_full.iloc[start:end]
    s_raw = raw_full.iloc[start:end].values

    priority = {m: i for i, m in enumerate(AVAILABLE_MODELS)}
    top_models, _ = evaluate_all_models(t_data, s_smooth.values, priority)
    if not top_models:
        st.info(f"Not enough valid data to fit a model for Channel {ch}.")
        return

    best_model = list(top_models.keys())[0]
    # elapsed_days are measured from the full series origin; pass it so the chart's
    # x-axis shows real DateTimes instead of elapsed days.
    x_origin = pd.Timestamp(elapsed_full.index.min())
    fig, _, _ = fit_and_plotly_model(
        time_raw=t_data, sensor_smooth=s_smooth, sensor_raw=s_raw,
        model_choice=best_model, thresholds=[threshold],
        precomputed_params=top_models[best_model]["params"],
        channel_name=str(ch), smoothed_rul=state["channel_ruls"].get(ch),
        x_origin=x_origin,
    )

    if centered:
        # The fit's smooth curve is drawn from the full-series origin out to 1.5·t_max, so
        # clamp the visible x-range to the shown window (with a little padding) instead of
        # letting that long tail zoom the view out to the whole history.
        win_start = x_origin + pd.to_timedelta(float(t_data.min()), unit="D")
        win_end = x_origin + pd.to_timedelta(float(t_data.max()), unit="D")
        pad = (win_end - win_start) * 0.05
        fig.update_xaxes(range=[win_start - pad, win_end + pad])
        st.caption(f"Window of {lookback} steps centered on the crossing "
                   f"({pd.Timestamp(breach_dt).date()}).")

    st.plotly_chart(fig, use_container_width=True, key=f"pred_{cfg['key']}_{ch}")


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

    # --- Active alerts (click a channel to open its prediction chart below) ---
    if active_alerts:
        st.markdown("#### ⚠️ Active Alerts")
        st.caption("Click a channel to view its RUL prediction chart below.")
        alert_map = {ch: s for ch, s in state["channel_states"].items() if s in ("warning", "critical")}
        cols = st.columns(min(len(alert_map), 6))
        for i, (ch, s) in enumerate(alert_map.items()):
            icon = "🔴" if s == "critical" else "🟡"
            label = f"{icon} Ch {ch} · {format_rul_str(state['channel_ruls'].get(ch), short=True)}"
            is_sel = state.get("selected_channel") == ch
            # on_click sets the selection BEFORE this rerun redraws the buttons, so the
            # clicked one turns primary (red) on the same click, not the next one.
            cols[i % 6].button(label, key=f"alert_{key}_{ch}", use_container_width=True,
                               type="primary" if is_sel else "secondary",
                               on_click=_select_channel, args=(key, ch))

    # --- Per-DB audit log (isolated to this tab) ---
    st.markdown("#### 📡 Event Log")
    st.caption(f"Newest first · {cfg['label']} · Mode: **{'Live' if not params_ui['dry_run'] else 'Dry Run'}**")
    with st.container(height=360):
        if not state["audit_log"]:
            st.write("No events yet.")
        for entry in state["audit_log"][:50]:
            icon = "🔴" if entry["status"] == "critical" else "🟡" if entry["status"] == "warning" else "🟢"
            resent_tag = " · ↻ re-sent" if entry.get("resent") else ""
            title = f"{icon} {entry['ts']} · Ch {entry['channel']} ({entry['old_status'].upper()} ➔ {entry['status'].upper()}){resent_tag}"
            with st.expander(title):
                st.write(f"**RUL:** {entry['old_rul_str']} ➔ **{entry['new_rul_str']}**")
                if entry.get("err_msg"):
                    st.error(entry["err_msg"])
                st.json(entry["payload"])

    # --- Channel prediction viewer (populated by clicking an alert above) ---
    sel = state.get("selected_channel")
    if sel is not None:
        st.divider()
        c_head, c_clear = st.columns([5, 1])
        c_head.markdown(f"#### 📈 Channel {sel} — RUL Prediction")
        c_clear.button("✖ Clear", key=f"clear_sel_{key}", use_container_width=True,
                       on_click=_clear_channel, args=(key,))
        _render_channel_prediction(cfg, state, params_ui, sel)

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
        sustained_days = st.number_input("Sustained days above limit", 1, 365, 7,
                                         help="A channel is treated as having truly crossed the limit only once it stays above it for this many consecutive days. Also drives the reported breach date.")
        rul_warn_days = st.number_input("🟡 Warning (< days)", value=30)
        rul_crit_days = st.number_input("🔴 Critical (< days)", value=10)
        lookback_days = st.slider("Lookback Window (Steps)", 100, 3000, 300)
        iqr_window = st.slider("IQR Window Size", 5, 50, 20)
        iqr_factor = st.slider("IQR Factor", 0.5, 3.0, 1.5)
        live_sending = st.toggle("📡 Live API Sending", value=False,
                                 help="OFF = Dry Run (events computed & logged, nothing POSTed).")
        dry_run = not live_sending
        # Detect the OFF->ON transition so already-active alerts get pushed on turn-on.
        if live_sending and not st.session_state.get("_live_sending_prev", False):
            st.session_state["_resync_pending"] = True
        st.session_state["_live_sending_prev"] = live_sending

        st.divider()
        st.header("Server")
        if st.button("🗑️ Delete All Server Entries", type="primary", use_container_width=True,
                     help="Sends a global DELETE to the visualization server, clearing its dashboard."):
            ok, msg = delete_telegrams()
            if ok:
                st.success(f"Server cleared ({msg}).")
                # Reset local alert state so the current alerts re-fire and re-send,
                # and clear each DB's audit/event log to match the cleared server.
                for _c in registry:
                    _s = _db_state(_c["key"])
                    _s["channel_states"] = {}
                    _s["channel_ruls"] = {}
                    _s["audit_log"] = []
                st.session_state["_resync_pending"] = True
            else:
                st.error(f"Delete failed: {msg}")

        st.divider()
        st.header("Shutter Diagnostics (optional)")
        show_shutter = st.toggle("Show shutter diff/cumsum charts", value=False)
        diff_periods = st.number_input("diff() periods", 1, 500, 10, disabled=not show_shutter)
        roll_window = st.number_input("smoothing window", 10, 100_000, 10_000, disabled=not show_shutter)
        use_ema = st.toggle("Use EMA instead of moving average", value=False, disabled=not show_shutter)

    params_ui = {
        "target_threshold": target_threshold, "rul_warn_days": rul_warn_days,
        "rul_crit_days": rul_crit_days, "lookback_days": lookback_days,
        "sustained_days": sustained_days,
        "iqr_window": iqr_window, "iqr_factor": iqr_factor, "dry_run": dry_run,
        "max_rows_per_poll": (max_rows_per_poll or None),
        "max_buffer_rows": (max_buffer_rows or None),
        "show_shutter": show_shutter, "diff_periods": diff_periods,
        "roll_window": roll_window, "use_ema": use_ema,
    }

    if not active_keys:
        st.info("👈 Activate at least one database in the sidebar to begin.")
        return

    # ---------------- Poll + render inside an auto-refreshing fragment ----------------
    # A top-level time.sleep()+st.rerun() re-instantiated st.tabs on every cycle, which
    # left a greyed-out, non-interactive "ghost" duplicate tab bar behind (the previous
    # run's tab component wasn't reconciled before the next rerun began). st.fragment
    # with run_every isolates the refresh to this block, so the tabs re-render cleanly
    # and the sidebar no longer blocks while waiting for the next poll.
    @st.fragment(run_every=(poll_interval if live else None))
    def _live_view():
        if live:
            with st.spinner("Polling active databases..."):
                for cfg in registry:
                    if cfg["key"] in active_keys:
                        poll_one_db(cfg, params_ui)

        # Live API Sending was just turned on (or the server was cleared): push every
        # currently-active alert to the server once. Guarded by a session flag so the
        # fragment's periodic auto-reruns don't repeat it.
        if not dry_run and st.session_state.get("_resync_pending"):
            with st.spinner("Pushing active alerts to server..."):
                for cfg in registry:
                    if cfg["key"] in active_keys:
                        evaluate_buffer_for_events(cfg, _db_state(cfg["key"]), params_ui, force_send=True)
            st.session_state["_resync_pending"] = False

        active_cfgs = [c for c in registry if c["key"] in active_keys]
        status_dot = lambda c: ("🔴" if any(s == "critical" for s in _db_state(c["key"])["channel_states"].values())
                                else "🟡" if any(s == "warning" for s in _db_state(c["key"])["channel_states"].values())
                                else "🟢")
        tabs = st.tabs([f"{status_dot(c)} {c['label']}" for c in active_cfgs])
        for tab, cfg in zip(tabs, active_cfgs):
            with tab:
                render_db_tab(cfg, params_ui)

    _live_view()
