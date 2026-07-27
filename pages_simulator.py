"""
pages_simulator.py — The original CSV time-travel simulator, unchanged in behavior,
wrapped in render() so the multipage router can call it. All algorithm + Telegram
logic now lives in lib_core (imported below) so this page and the live page share
identical math.
"""
import time
import numpy as np
import pandas as pd
import streamlit as st

from lib_core import (
    parse_raw_csv, get_available_channels, load_my_sensor_data, process_all_channels,
    AVAILABLE_MODELS, evaluate_all_models, build_dynamic_std_fn, solve_rul_root,
    fit_and_plotly_model, send_custom_telegram, delete_telegrams,
    format_rul_str, SENTINEL_ALREADY_REACHED,
)


def render():
    # =========================================================
    # 3. STREAMLIT UI - LEFT SIDEBAR
    # =========================================================
    st.title("🏭 Machine Health Time-Travel Simulator")

    with st.sidebar:
        st.header("1. Data Ingestion")
        uploaded_file = st.file_uploader(
            "Upload Timeseries (CSV)", 
            type=['csv'],
            help="Upload your raw historical timeseries data. The script will parse 'DateTime' and filter out errors automatically."
        )

        if uploaded_file is not None and st.session_state.df is None:
            st.session_state.df = parse_raw_csv(uploaded_file)

        st.divider()
        st.header("2. Algorithm Parameters")

        available_channels = get_available_channels(st.session_state.df) if st.session_state.df is not None else []

        if st.session_state.selected_channel not in available_channels and available_channels:
            st.session_state.selected_channel = available_channels[0]

        sel_idx = available_channels.index(st.session_state.selected_channel) if st.session_state.selected_channel in available_channels else 0

        chosen_ch = st.selectbox(
            "Target Channel (Foreground UI)", 
            options=available_channels,
            index=sel_idx,
            help="Select the specific sensor channel you want to visualize in the main Plotly chart. (Note: The engine evaluates ALL channels in the background)."
        )
        if chosen_ch != st.session_state.selected_channel:
            st.session_state.selected_channel = chosen_ch
            st.rerun()

        target_threshold = st.number_input(
            "Failure Threshold Limit", 
            value=0.2, 
            format="%.3f",
            help="The physical limit. When the prediction curve crosses this value, the Remaining Useful Life (RUL) hits 0 days."
        )

        st.subheader("Health Thresholds")
        rul_ok_days = st.number_input("🟢 OK (> days)", value=30, help="If the Nominal RUL is greater than this number, the status is deemed 'OK'.")
        rul_warn_days = st.number_input("🟡 Warning (< days)", value=30, help="If the Nominal RUL drops below this number (but is above Critical), the status changes to 'Warning'.")
        rul_crit_days = st.number_input("🔴 Critical (< days)", value=10, help="If the Nominal RUL drops below this number, the status changes to 'Critical'.")

        st.subheader("Data Slice & Filtering")
        lookback_days = st.slider("Lookback Window (Steps)", min_value=100, max_value=1000, value=300, help="How many previous daily steps to feed into the prediction engine at any given moment.")
        iqr_window = st.slider("IQR Window Size", 5, 50, 20, help="The rolling window size used to detect and filter out erratic spikes in the data.")
        iqr_factor = st.slider("IQR Factor", 0.5, 3.0, 1.5, help="The strictness of the outlier filter. A lower factor filters out more data. 1.5 is standard.")

        rul_ema_window = st.slider(
            "RUL EMA Window (Days)", 
            min_value=1, max_value=20, value=1,
            help="Applies a Pandas EMA to the history of predicted RULs to smooth out fluctuations. 1 = No smoothing."
        )


    # =========================================================
    # 4. STREAMLIT UI - MAIN LAYOUT
    # =========================================================
    if st.session_state.df is not None:
        all_channel_data = process_all_channels(st.session_state.df, iqr_factor, iqr_window)
        if not all_channel_data:
            st.error("No valid data could be extracted from the dataset.")
            st.stop()

        first_ch = list(all_channel_data.keys())[0]
        max_steps = len(all_channel_data[first_ch]['elapsed']) - 1

        # --- INDEPENDENT STATE INITIALIZATION ---
        if 'channel_states' not in st.session_state:
            st.session_state.channel_states = {ch: 'ok' for ch in all_channel_data.keys()}

        if 'channel_ruls' not in st.session_state:
            st.session_state.channel_ruls = {ch: None for ch in all_channel_data.keys()}

        if 'channel_rul_history' not in st.session_state:
            st.session_state.channel_rul_history = {ch: [] for ch in all_channel_data.keys()}

        if 'last_evaluated_step' not in st.session_state:
            st.session_state.last_evaluated_step = -1

        if len(st.session_state.audit_log) == 0:
            st.session_state.audit_log.insert(0, {
                "step": 0, "channel": "SYSTEM", "status": "info", 
                "payload": {"message": "System Startup. All channels initialized to OK."}
            })

        col_main, col_log = st.columns([3, 1])

        with col_main:
            st.subheader("Simulation Controls")

            st.markdown("**Timeline Navigation**")
            step_slider = st.slider(
                "Jump to Timestep", 
                0, max_steps, st.session_state.current_step, 
                key="slider_step", label_visibility="collapsed",
                help="Drag this slider to time-travel through the historic dataset."
            )
            if step_slider != st.session_state.current_step: st.session_state.current_step = step_slider

            st.write("") 
            c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1.5])

            with c1:
                step_size = st.number_input("Step Size", min_value=1, max_value=50, value=1, help="How many timesteps to advance per tick.")
            with c2:
                st.write("") 
                if st.button("⏭️ Next Step", use_container_width=True, help=f"Advance the simulation by {step_size} timestep(s)."):
                    st.session_state.current_step = min(st.session_state.current_step + step_size, max_steps)
            with c3:
                st.write("") 
                if st.button("▶️ Auto-Play" if not st.session_state.is_playing else "⏸️ Pause", use_container_width=True):
                    st.session_state.is_playing = not st.session_state.is_playing
            with c4:
                delay = st.number_input("Auto-Play Delay (s)", min_value=0.1, max_value=5.0, value=0.5, help="Pause between steps.")
            with c5:
                st.write("") 
                st.toggle("📡 Live API Sending", key="live_sending", help="When ON (Live): Real JSON payloads are POSTed to your visualization server.")
                dry_run_mode = not st.session_state.live_sending

            # --- BACKGROUND ENGINE (EVALUATES ALL CHANNELS) ---
            if st.session_state.current_step != st.session_state.last_evaluated_step:
                with st.spinner("Calculating Backend Edge States..."):
                    start_idx = max(0, st.session_state.current_step - lookback_days)
                    end_idx = st.session_state.current_step + 1
                    priority = {m: i for i, m in enumerate(AVAILABLE_MODELS)}

                    for ch, data in all_channel_data.items():
                        time_data = data['elapsed'].iloc[start_idx:end_idx].values
                        sensor_data_smooth = data['smooth'].iloc[start_idx:end_idx].values

                        if np.isnan(sensor_data_smooth).sum() > len(sensor_data_smooth) * 0.8: continue

                        top_models, _ = evaluate_all_models(time_data, sensor_data_smooth, priority)
                        new_status, raw_nom_rul, upper_rul, lower_rul = "ok", None, None, None
                        context_df_data = []

                        if top_models:
                            best_model = list(top_models.keys())[0]
                            params = top_models[best_model]['params']
                            t_max = np.max(time_data) if np.max(time_data) > 0 else 1.0

                            sensor_raw = data['raw'].iloc[start_idx:end_idx].values
                            time_norm = time_data / t_max
                            preds = top_models[best_model]['func'](time_norm, *params)
                            residuals = pd.Series(sensor_raw - preds)
                            rolling_std = residuals.rolling(window=20, min_periods=1).std().bfill().fillna(0)

                            valid_std = ~np.isnan(rolling_std)
                            std_slope, std_intercept = np.polyfit(time_norm[valid_std], rolling_std[valid_std], 1) if valid_std.sum() > 1 else (0.0, rolling_std.iloc[-1])
                            dyn_std_fn = build_dynamic_std_fn(std_slope, std_intercept)

                            func = top_models[best_model]['func']
                            nom_time = solve_rul_root(func, params, target_threshold, t_max, dyn_std_fn, 1.645, 'nominal')
                            upper_time = solve_rul_root(func, params, target_threshold, t_max, dyn_std_fn, 1.645, 'upper')
                            lower_time = solve_rul_root(func, params, target_threshold, t_max, dyn_std_fn, 1.645, 'lower')

                            def calc_rul(t_val): return t_val if t_val in ('Safe', SENTINEL_ALREADY_REACHED) else t_val - np.max(time_data)

                            raw_nom_rul, upper_rul, lower_rul = calc_rul(nom_time), calc_rul(upper_time), calc_rul(lower_time)

                            # --- TABULAR PANDAS EMA WITH LAST-5 CONTEXT ---
                            # nom_rul is the EMA-SMOOTHED value that drives the status
                            # thresholds. raw_nom_rul is the un-smoothed prediction shown
                            # on the chart. We expose both so they can be compared.
                            nom_rul = raw_nom_rul
                            ema_nom_rul = raw_nom_rul
                            if isinstance(raw_nom_rul, (float, int)):
                                history = st.session_state.channel_rul_history[ch]
                                history = [row for row in history if row['step'] != st.session_state.current_step]
                                history.append({'step': st.session_state.current_step, 'rul': raw_nom_rul})
                                history = sorted(history, key=lambda x: x['step'])
                                st.session_state.channel_rul_history[ch] = history

                                valid_history = [row for row in history if row['step'] <= st.session_state.current_step]

                                # LAG FIX 1: Only smooth over a bounded, recent window instead of
                                # the entire descending tail. A long run of large early RUL values
                                # (e.g. 200 -> 80 -> 40) would otherwise anchor the average high and
                                # delay the warning by several steps. We keep the most recent
                                # (2*span + 1) steps, enough for the EMA to be well-defined without
                                # dragging in stale pre-jump values.
                                ema_lookback = max(5, 2 * int(rul_ema_window) + 1)
                                valid_history = valid_history[-ema_lookback:]

                                valid_ruls = [row['rul'] for row in valid_history]
                                valid_steps = [row['step'] for row in valid_history]

                                if valid_ruls:
                                    # LAG FIX 2: adjust=True computes a normalized weighted average
                                    # rather than a recursive seed from the first value, so a series
                                    # that has settled near 21 reads ~21 immediately instead of
                                    # decaying toward it over several steps.
                                    ema_series = pd.Series(valid_ruls).ewm(span=rul_ema_window, adjust=True).mean()
                                    ema_nom_rul = ema_series.iloc[-1]
                                    nom_rul = ema_nom_rul

                                    # Extract context for the Audit Log
                                    last_5_steps = valid_steps[-5:]
                                    last_5_raw = valid_ruls[-5:]
                                    last_5_ema = ema_series.iloc[-5:].tolist()

                                    context_df_data = [
                                        {
                                            "Step": last_5_steps[i], 
                                            "Raw RUL": format_rul_str(last_5_raw[i], short=True), 
                                            "EMA RUL": format_rul_str(last_5_ema[i], short=True)
                                        } 
                                        for i in range(len(last_5_steps))
                                    ]

                            if isinstance(nom_rul, (float, int)):
                                if nom_rul < rul_crit_days: new_status = "critical"
                                elif nom_rul < rul_warn_days: new_status = "warning"

                        old_status = st.session_state.channel_states.get(ch, 'ok')
                        old_rul = st.session_state.channel_ruls.get(ch, None)

                        if new_status != old_status:
                            api_result = send_custom_telegram(
                                channel=ch, 
                                status=new_status,
                                nominal_rul_days=nom_rul if isinstance(nom_rul, (int, float)) else None,
                                early_rul_days=upper_rul if isinstance(upper_rul, (int, float)) else None,
                                late_rul_days=lower_rul if isinstance(lower_rul, (int, float)) else None,
                                limit_value=target_threshold, 
                                dry_run=dry_run_mode,
                                best_model=best_model if top_models else "Unknown"
                            )

                            err_msg = "" if api_result["sent"] or dry_run_mode else f" ❌ SERVER ERROR: {api_result.get('error', 'Unknown')}"

                            st.session_state.audit_log.insert(0, {
                                "step": st.session_state.current_step, 
                                "channel": ch, 
                                "status": new_status,
                                "old_status": old_status,
                                "new_rul_str": format_rul_str(nom_rul, short=True),
                                "old_rul_str": format_rul_str(old_rul, short=True),
                                "payload": api_result["body"], 
                                "err_msg": err_msg,
                                "context_df": context_df_data  # <--- Added context data here
                            })
                            st.session_state.channel_states[ch] = new_status

                        st.session_state.channel_ruls[ch] = nom_rul

                st.session_state.last_evaluated_step = st.session_state.current_step


            # --- VISUALIZATION (Foreground) ---
            target_ch = st.session_state.selected_channel
            if target_ch in all_channel_data:
                start_idx = max(0, st.session_state.current_step - lookback_days)
                end_idx = st.session_state.current_step + 1
                vis_data = all_channel_data[target_ch]

                t_data = vis_data['elapsed'].iloc[start_idx:end_idx].values
                s_smooth, s_raw = vis_data['smooth'].iloc[start_idx:end_idx], vis_data['raw'].iloc[start_idx:end_idx].values

                top_models, _ = evaluate_all_models(t_data, s_smooth.values, {m: i for i, m in enumerate(AVAILABLE_MODELS)})
                if top_models:
                    b_model = list(top_models.keys())[0]
                    # channel_ruls stores the EMA-smoothed RUL (the value that drives the
                    # status/log). Pass it so the chart shows Raw and EMA side by side.
                    smoothed_rul = st.session_state.channel_ruls.get(target_ch, None)
                    fig, _, _ = fit_and_plotly_model(
                        time_raw=t_data, sensor_smooth=s_smooth, sensor_raw=s_raw, 
                        model_choice=b_model, thresholds=[target_threshold], 
                        precomputed_params=top_models[b_model]['params'],
                        channel_name=target_ch, smoothed_rul=smoothed_rul
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else: st.info(f"Not enough valid data to plot prediction for Channel {target_ch}.")

            # --- ACTIVE ALERTS WIDGET ---
            active_alerts = {ch: status for ch, status in st.session_state.channel_states.items() if status in ['warning', 'critical']}
            if active_alerts:
                st.divider()
                st.markdown("### ⚠️ Active Alerts")
                st.caption("Click an alert below to instantly load that channel into the prediction viewer above.")

                btn_cols = st.columns(min(len(active_alerts), 8))

                for idx, (alert_ch, alert_status) in enumerate(active_alerts.items()):
                    col = btn_cols[idx % 8]
                    icon = "🔴" if alert_status == "critical" else "🟡"

                    is_current = (alert_ch == st.session_state.selected_channel)
                    btn_type = "primary" if is_current else "secondary"

                    with col:
                        if st.button(f"{icon} Ch {alert_ch}", key=f"btn_alert_{alert_ch}", type=btn_type, use_container_width=True):
                            st.session_state.selected_channel = alert_ch
                            st.rerun()

        # --- RIGHT DRAWER: Telegram Audit Log ---
        with col_log:
            st.subheader("Server Management")
            if st.button("🗑️ Delete All Server Entries", type="primary", use_container_width=True, help="Sends a global DELETE command to the visualization server, resetting the dashboard."):
                if st.session_state.live_sending:
                    success, msg = delete_telegrams()
                    if success: 
                        st.success("Server cleared!")
                        st.session_state.channel_states = {ch: 'ok' for ch in st.session_state.channel_states.keys()}
                        st.session_state.channel_ruls = {ch: None for ch in st.session_state.channel_ruls.keys()}
                    else: 
                        st.error(f"Failed: {msg}")
                else:
                    st.warning("DRY RUN: Delete simulated. Local states reset.")
                    st.session_state.channel_states = {ch: 'ok' for ch in st.session_state.channel_states.keys()}
                    st.session_state.channel_ruls = {ch: None for ch in st.session_state.channel_ruls.keys()}
                st.session_state.audit_log = []

            st.divider()
            st.subheader("📡 Server Audit Log")
            mode_text = "Live" if st.session_state.live_sending else "Dry Run"
            st.caption(f"Showing last 50 events | Mode: **{mode_text}**")

            with st.container(height=800):
                for entry in st.session_state.audit_log[:50]:
                    if entry["channel"] == "SYSTEM":
                        st.info("🟢 SYSTEM START: All channels initialized OK.")
                        continue

                    icon = "🔴" if entry["status"] == "critical" else "🟡" if entry["status"] == "warning" else "🟢"
                    title = f"{icon} Step {entry['step']} | Ch {entry['channel']} ({entry.get('old_status', 'ok').upper()} ➔ {entry['status'].upper()})"

                    with st.expander(title):
                        st.write(f"**Calculated RUL:** {entry.get('old_rul_str', 'Unknown')} ➔ **{entry['new_rul_str']}**")

                        # --- NEW: RENDER THE TREND CONTEXT TABLE ---
                        if entry.get("context_df"):
                            st.markdown("**Last 5 Steps (Trend Context):**")
                            # Format as a clean Pandas dataframe without the index column
                            df_context = pd.DataFrame(entry["context_df"]).set_index("Step")
                            st.dataframe(df_context, use_container_width=True)

                        if entry.get("err_msg"): st.error(entry["err_msg"])
                        st.json(entry["payload"])

        # --- AUTO-PLAY EXECUTION ---
        if st.session_state.is_playing:
            if st.session_state.current_step < max_steps:
                time.sleep(delay)
                st.session_state.current_step = min(st.session_state.current_step + step_size, max_steps)
                st.rerun()
            else:
                st.session_state.is_playing = False
                st.success("Simulation Complete!")

    else:
        st.info("👈 Please upload a historic dataset in the sidebar to begin the simulation.")
