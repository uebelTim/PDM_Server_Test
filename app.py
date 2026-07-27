"""
app.py — Entry point for the Machine Health dashboard (multipage).

Run with:  streamlit run app.py

Pages:
    • Time-Travel Simulator  — the original CSV-driven simulator (pages_simulator.py)
    • Live Monitoring        — new multi-database live streaming + RUL events (pages_live.py)

Both pages import their shared math/Telegram logic from lib_core.py, so there is a
single source of truth for the algorithm. set_page_config lives here (it must be the
first Streamlit call and may only run once per session).
"""
import streamlit as st

st.set_page_config(page_title="Machine Health Suite", layout="wide")

# --- Shared session-state defaults used by the simulator page ---
_defaults = {
    "current_step": 20,
    "is_playing": False,
    "df": None,
    "audit_log": [],
    "live_sending": False,
    "selected_channel": None,
}
for k, v in _defaults.items():
    st.session_state.setdefault(k, v)

import pages_simulator
import pages_live

simulator_page = st.Page(pages_simulator.render, title="Historical Backtesting",
                         icon="🏭", url_path="simulator", default=True)
live_page = st.Page(pages_live.render, title="Live Monitoring",
                    icon="📶", url_path="live")

nav = st.navigation([simulator_page, live_page])
nav.run()
