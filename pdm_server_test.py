import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit, brentq
from sklearn.metrics import mean_squared_error
import math
import time
import requests
import json
from datetime import datetime, timezone

# =========================================================
# 0. CONFIG & CONSTANTS
# =========================================================
st.set_page_config(page_title="Machine Health Simulator", layout="wide")

ENDPOINT = "https://mars01.dcims.ims/api/events"
DELETE_ENDPOINT = "https://mars01.dcims.ims/api/events"

API_TOKEN = st.secrets["API_TOKEN"]

GAUGE_NUMBER = "140988"  # Hardcoded System ID

HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json",
}

RUL_HORIZON = 180  # days; "Safe" (never reached) and anything beyond this saturate here.
MIN_MSE_FLOOR = 5e-5
SENTINEL_ALREADY_REACHED = -999.0

# --- INDEPENDENT STATE INITIALIZATION ---
if 'current_step' not in st.session_state: st.session_state.current_step = 20
if 'is_playing' not in st.session_state: st.session_state.is_playing = False
if 'df' not in st.session_state: st.session_state.df = None
if 'audit_log' not in st.session_state: st.session_state.audit_log = []
if 'live_sending' not in st.session_state: st.session_state.live_sending = False
if 'selected_channel' not in st.session_state: st.session_state.selected_channel = None

# --- HELPER FORMATTER FOR RUL DISPLAY ---
def format_rul_str(rul_val, short=False):
    """Formats RUL values into readable strings for UI and Logs."""
    suffix = "D" if short else " Days"
    if rul_val == 'Safe' or rul_val is None: return "Safe"
    if rul_val == SENTINEL_ALREADY_REACHED or (isinstance(rul_val, (float, int)) and rul_val < 0): return "Breached"
    if isinstance(rul_val, (float, int)): return f"{rul_val:.1f}{suffix}"
    return "Unknown"


# =========================================================
# 1. MATHEMATICAL MODELS & DATA PROCESSING
# =========================================================
@st.cache_data
def parse_raw_csv(file_obj):
    file_obj.seek(0)
    df = pd.read_csv(file_obj, parse_dates=['DateTime'])
    df.set_index('DateTime', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    return df

def get_available_channels(df):
    cols = [col for col in df.columns if 'Error' not in col]
    if 'Thermo_Valve_Temperature_DeviationPct' in cols:
        cols.remove('Thermo_Valve_Temperature_DeviationPct')
    return [str(c) for c in cols]

def rolling_iqr_filter(data, window=20, factor=1.5, center=True, keep_nans=True):
    def _filter_series(series):
        s = pd.to_numeric(series, errors='coerce')
        original_nans = s.isna()
        Q1 = s.rolling(window=window, center=center, min_periods=1).quantile(0.25)
        Q3 = s.rolling(window=window, center=center, min_periods=1).quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - (factor * IQR), Q3 + (factor * IQR)
        mask = (s >= lower_bound) & (s <= upper_bound)
        s_filtered = s.where(mask, np.nan).interpolate(method='linear', limit_direction='both')
        if keep_nans:
            s_filtered.loc[original_nans] = np.nan
        return s_filtered
    return _filter_series(data) if isinstance(data, pd.Series) else data.apply(_filter_series) if isinstance(data, pd.DataFrame) else None

@st.cache_data
def load_my_sensor_data(df, col='32', outlier_factor=1.5, outlier_window=20, target_freq='1D'):
    freq = '4h'
    df_resampled = df.resample(freq).mean(numeric_only=True)
    if col not in df_resampled.columns:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    df_select = df_resampled[[col]].copy().interpolate(method='time', limit=1)
    df_select = rolling_iqr_filter(df_select, factor=outlier_factor, window=outlier_window)

    window = int(24 / 4)  # 1 Day
    df_select[f'{col}_max'] = df_select[col].rolling(window=window * 5, min_periods=1).max()
    df_select[f'{col}_max_ema'] = df_select[f'{col}_max'].ewm(span=window * 5, adjust=False, ignore_na=True).mean()

    # Resample to the requested target frequency
    df_target = df_select.resample(target_freq).mean(numeric_only=True)

    # Fractional elapsed days via total_seconds (avoids integer-day truncation)
    origin = df_target.index.min()
    df_target['elapsed_days'] = (df_target.index - origin).total_seconds() / 86400.0

    return df_target[f'{col}_max_ema'], df_target[f'{col}_max'], df_target['elapsed_days']

@st.cache_data
def process_all_channels(df, outlier_factor, outlier_window):
    channels = get_available_channels(df)
    all_data = {}
    for ch in channels:
        smooth, raw, elapsed = load_my_sensor_data(df, col=ch, outlier_factor=outlier_factor, outlier_window=outlier_window)
        if not elapsed.empty: all_data[ch] = {'smooth': smooth, 'raw': raw, 'elapsed': elapsed}
    return all_data

# ---------------------------------------------------------
# 1. The Standardized Mathematical Models
# ---------------------------------------------------------
def linear_model(t, m, c): return m * t + c
def logarithmic_model(t, a, b, d): return a * np.log1p(np.clip(b * t, 0, np.inf)) + d
def shifted_exponential_model(t, a, b, t0, d): return a * np.exp(np.clip(b * (t - t0), -50, 50)) + d
def softplus_model(t, a, b, t0, d): return a * np.logaddexp(0, b * (t - t0)) + d
def gompertz_model(t, a, b, c, d): return a * np.exp(-b * np.exp(-c * t)) + d
def arctan_model(t, L, k, t0, d): return L * (np.arctan(k * (t - t0)) / np.pi + 0.5) + d
def linear_sine_model(t, a, b, c, m, d):
    # a: amplitude, b: frequency, c: phase shift, m: linear slope, d: vertical offset
    return a * np.sin(b * t + c) + (m * t + d)

# Ordered by requested default priority (Highest to Lowest)
AVAILABLE_MODELS = [
    'Linear',
    'Logarithmic',
    'Trending Sine',
    'Softplus',
    'Shifted Exponential',
    #'Gompertz',
]

# ---------------------------------------------------------
# 1a. Analytical Jacobians (Exact Partial Derivatives)
# ---------------------------------------------------------
def linear_jac(t, m, c):
    dm = t
    dc = np.ones_like(t)
    return np.column_stack((dm, dc))

def logarithmic_jac(t, a, b, d):
    u = np.clip(b * t, 0, np.inf)
    da = np.log1p(u)
    db = (a * t) / (1 + u)
    db[b * t < 0] = 0.0
    dd = np.ones_like(t)
    return np.column_stack((da, db, dd))

def shifted_exponential_jac(t, a, b, t0, d):
    u = np.clip(b * (t - t0), -50, 50)
    exp_u = np.exp(u)
    da = exp_u
    db = a * (t - t0) * exp_u
    dt0 = -a * b * exp_u
    clip_mask = (b * (t - t0) <= -50) | (b * (t - t0) >= 50)
    db[clip_mask] = 0.0
    dt0[clip_mask] = 0.0
    dd = np.ones_like(t)
    return np.column_stack((da, db, dt0, dd))

def softplus_jac(t, a, b, t0, d):
    x = b * (t - t0)
    da = np.logaddexp(0, x)
    sigmoid = np.where(x >= 0,
                       1.0 / (1.0 + np.exp(-x)),
                       np.exp(x) / (1.0 + np.exp(x)))
    db = a * (t - t0) * sigmoid
    dt0 = -a * b * sigmoid
    dd = np.ones_like(t)
    return np.column_stack((da, db, dt0, dd))

def gompertz_jac(t, a, b, c, d):
    u = np.exp(-c * t)
    E = np.exp(-b * u)
    da = E
    db = -a * u * E
    dc = a * b * t * u * E
    dd = np.ones_like(t)
    return np.column_stack((da, db, dc, dd))

def linear_sine_jac(t, a, b, c, m, d):
    phase = b * t + c
    cos_p = np.cos(phase)
    da = np.sin(phase)
    db = a * t * cos_p
    dc = a * cos_p
    dm = t
    dd = np.ones_like(t)
    return np.column_stack((da, db, dc, dm, dd))

# ---------------------------------------------------------
# 1b. CENTRALIZED MODEL CONFIG (single source of truth)
# ---------------------------------------------------------
def build_models_config(y_min, y_max, y_range):
    d_lo, d_hi = y_min - 0.2, y_max + 0.2
    return {
        'Linear': {'func': linear_model, 'jac': linear_jac, 'p0': [y_range, y_min], 'bounds': ([-np.inf, d_lo], [np.inf, d_hi])},
        'Logarithmic': {'func': logarithmic_model, 'jac': logarithmic_jac, 'p0': [y_range * 0.5, 10.0, y_min], 'bounds': ([1e-5, 1e-5, d_lo], [y_range * 10.0, 500.0, d_hi])},
        'Shifted Exponential': {'func': shifted_exponential_model, 'jac': shifted_exponential_jac, 'p0': [y_range * 0.1, 5.0, 0.5, y_min], 'bounds': ([1e-5, 0.01, 0.0, d_lo], [y_range * 5.0, 50.0, 1.0, d_hi])},
        'Softplus': {'func': softplus_model, 'jac': softplus_jac, 'p0': [y_range * 0.5, 10.0, 0.5, y_min], 'bounds': ([1e-3, 1e-3, 0.0, d_lo], [y_range * 10.0, 500.0, 1.0, d_hi])},
        'Gompertz': {'func': gompertz_model, 'jac': gompertz_jac, 'p0': [y_range * 1.1, 1.0, 0.1, y_min], 'bounds': ([y_range * 0.8, 0.01, 1e-4, d_lo], [max(2.0, y_range * 2.2), 100.0, 50.0, d_hi])},
        'Trending Sine': {'func': linear_sine_model, 'jac': linear_sine_jac, 'p0': [y_range * 0.1, 3 * 2 * np.pi, 0.0, y_range, y_min], 'bounds': ([1e-5, 2 * np.pi, -np.pi, -np.inf, d_lo], [y_range * 2.0, 10 * np.pi, np.pi, np.inf, d_hi])},
    }

# ---------------------------------------------------------
# 2. Master Fitting Function (AIC & Priority Router)
# ---------------------------------------------------------
def evaluate_all_models(time_data, sensor_data, priority_ranking, eval_window=None, warm_start=None, maxfev=10000, min_mse_floor=MIN_MSE_FLOOR):
    time_arr, sensor_arr = np.asarray(time_data), np.asarray(sensor_data)
    t_max = np.max(time_arr)
    time_norm = time_arr / t_max if t_max > 0 else time_arr
    valid_mask = ~np.isnan(sensor_arr)
    if valid_mask.sum() < 10: return {}, {}
    t_fit, y_fit = time_norm[valid_mask], sensor_arr[valid_mask]
    y_min, y_max = float(np.min(y_fit)), float(np.max(y_fit))

    models_config, results = build_models_config(y_min, y_max, y_max - y_min), {}
    for name in AVAILABLE_MODELS:
        if name not in models_config: continue
        config = models_config[name]
        try:
            p0 = config['p0']
            k = len(p0)
            if warm_start is not None and name in warm_start and warm_start[name] is not None:
                try: p0 = np.clip(warm_start[name], config['bounds'][0], config['bounds'][1])
                except Exception: p0 = config['p0']
            params, _ = curve_fit(config['func'], t_fit, y_fit, p0=p0, jac=config.get('jac'), bounds=config['bounds'], method='trf', maxfev=maxfev)
            preds = config['func'](t_fit, *params)
            if eval_window is not None and eval_window < len(y_fit):
                y_eval, preds_eval = y_fit[-eval_window:], preds[-eval_window:]
            else:
                y_eval, preds_eval = y_fit, preds
            n = len(y_eval)
            mse = mean_squared_error(y_eval, preds_eval)
            # Tail-weighted pseudo-AIC with MSE floor (prevents log-domain blow-ups on near-perfect fits)
            aic = n * np.log(max(mse, min_mse_floor)) + 2 * k
            results[name] = {'params': params, 'mse': mse, 'aic': aic, 'func': config['func']}
        except Exception as e:
            results[name] = {'params': None, 'mse': float('inf'), 'aic': float('inf'), 'error': str(e), 'func': config['func']}

    valid_results = {n: d for n, d in results.items() if d['aic'] != float('inf')}
    if not valid_results: return {}, results
    best_aic = min(d['aic'] for d in valid_results.values())
    competitive = {n: d for n, d in valid_results.items() if (d['aic'] - best_aic) <= 2.0}
    return dict(sorted(competitive.items(), key=lambda item: priority_ranking.get(item[0], 99))), dict(sorted(results.items(), key=lambda item: item[1].get('aic', float('inf'))))

def build_dynamic_std_fn(std_slope, std_intercept):
    return lambda t_norm_val: np.maximum(0.0, std_slope * t_norm_val + std_intercept)

def solve_rul_root(func, params, target_val, t_max, get_dynamic_std, sigma_factor, mode='nominal', t_hi=200.0):
    t_inf = 10000.0
    ceil_base = func(t_inf, *params)
    ceil_val = ceil_base + get_dynamic_std(t_inf) * sigma_factor if mode == 'upper' else ceil_base - get_dynamic_std(t_inf) * sigma_factor if mode == 'lower' else ceil_base
    if ceil_val < target_val: return 'Safe'

    def target_equation(t_guess):
        base_val = func(t_guess, *params)
        return (base_val + get_dynamic_std(t_guess) * sigma_factor) - target_val if mode == 'upper' else (base_val - get_dynamic_std(t_guess) * sigma_factor) - target_val if mode == 'lower' else base_val - target_val

    try:
        if target_equation(1e-9) >= 0: return SENTINEL_ALREADY_REACHED
        hi, f_hi, expand = t_hi, target_equation(t_hi), 0
        while f_hi < 0 and expand < 8: hi *= 2.0; f_hi = target_equation(hi); expand += 1
        if f_hi < 0: return None
        return brentq(target_equation, 1e-9, hi, xtol=1e-9, maxiter=200) * t_max
    except: return None

def fit_and_plotly_model(time_raw, sensor_smooth, sensor_raw, model_choice, thresholds=None, precomputed_params=None, sigma_factor=1.645, channel_name=""):
    time_arr, sensor_arr, sensor_raw_arr = np.asarray(time_raw, dtype=float), np.asarray(sensor_smooth, dtype=float), np.asarray(sensor_raw, dtype=float)
    t_max = np.max(time_arr) if np.max(time_arr) > 0 else 1.0
    time_norm = time_arr / t_max
    valid_mask = ~np.isnan(sensor_arr)
    t_fit, y_fit = time_norm[valid_mask], sensor_arr[valid_mask]
    
    if len(y_fit) == 0: return go.Figure(), pd.Series(dtype=float), pd.DataFrame()

    y_min, y_max = float(np.min(y_fit)), float(np.max(y_fit))
    config = build_models_config(y_min, y_max, y_max - y_min)[model_choice]
    params = precomputed_params
    func = config['func']

    residuals = pd.Series(sensor_raw_arr - func(time_norm, *params))
    rolling_std = residuals.rolling(window=20, min_periods=1).std().bfill().fillna(0)
    valid_std = ~np.isnan(rolling_std)
    std_slope, std_intercept = np.polyfit(time_norm[valid_std], rolling_std[valid_std], 1) if valid_std.sum() > 1 else (0.0, rolling_std.iloc[-1])
    get_dynamic_std = build_dynamic_std_fn(std_slope, std_intercept)

    rul_records = []
    nom_rul = None
    if thresholds:
        for thresh in thresholds:
            nom_time = solve_rul_root(func, params, thresh, t_max, get_dynamic_std, sigma_factor, 'nominal')
            upper_time = solve_rul_root(func, params, thresh, t_max, get_dynamic_std, sigma_factor, 'upper')
            lower_time = solve_rul_root(func, params, thresh, t_max, get_dynamic_std, sigma_factor, 'lower')

            def calc_rul(t_val):
                return t_val if t_val in ('Safe', SENTINEL_ALREADY_REACHED) else t_val - np.max(time_arr) if isinstance(t_val, float) else np.nan

            nom_rul, upper_rul, lower_rul = calc_rul(nom_time), calc_rul(upper_time), calc_rul(lower_time)
            status = 'Never Reached (Safe)' if nom_rul == 'Safe' else 'Already Reached' if isinstance(nom_rul, float) and nom_rul < 0 else 'Predicted Future'
            rul_records.append({'Threshold': thresh, 'Status': status, 'Nominal_RUL': nom_rul, 'Upper_Band_RUL': upper_rul, 'Lower_Band_RUL': lower_rul})

    time_smooth_norm = np.linspace(0, 1.5, 500)
    time_smooth_converted = time_smooth_norm * t_max
    smooth_preds = func(time_smooth_norm, *params)
    dynamic_std_smooth = get_dynamic_std(time_smooth_norm)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_arr, y=sensor_raw_arr, mode='markers', name='Raw Data', marker=dict(color='gray', size=5, opacity=0.7)))
    fig.add_trace(go.Scatter(x=time_smooth_converted, y=smooth_preds, mode='lines', name=f'{model_choice} Fit', line=dict(color='blue', width=2.5)))
    fig.add_trace(go.Scatter(x=time_smooth_converted, y=smooth_preds - (dynamic_std_smooth * sigma_factor), mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=time_smooth_converted, y=smooth_preds + (dynamic_std_smooth * sigma_factor), mode='lines', fill='tonexty', fillcolor='rgba(0, 0, 255, 0.15)', name='Confidence Band', line=dict(width=0)))
    
    if thresholds:
        for thresh in thresholds: fig.add_hline(y=thresh, line_dash="dash", line_color="red", annotation_text=f"Target: {thresh}")

    display_rul = format_rul_str(nom_rul, short=False)
    font_color = "red" if (isinstance(nom_rul, (float, int)) and nom_rul < 10) or display_rul == "Breached" else "black"
    
    fig.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper",
        text=f"<b>Predicted RUL: {display_rul}</b>",
        showarrow=False,
        font=dict(size=16, color=font_color),
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="black", borderwidth=1, borderpad=6
    )

    max_target = max(thresholds) if thresholds else 0
    absolute_y_max = max(y_max, max_target)
    absolute_y_min = min(y_min, min(thresholds) if thresholds else y_min)
    y_padding = max((absolute_y_max - absolute_y_min) * 0.15, 0.1)
    
    fig.update_layout(
        title=f"RUL Prediction — Channel {channel_name} | Best Fit: {model_choice}",
        xaxis_title="Elapsed Days", 
        yaxis_title="Sensor Value",
        yaxis=dict(range=[absolute_y_min - y_padding, absolute_y_max + y_padding]),
        template="plotly_white", 
        margin=dict(t=50, b=50, l=50, r=50)
    )
    return fig, pd.Series(dtype=float), pd.DataFrame(rul_records)


# =========================================================
# 2. TELEGRAM API HANDLING
# =========================================================
def _now_iso():
    return datetime.now().astimezone().isoformat(timespec='seconds')

def build_custom_telegram(channel, status, nominal_rul_days=None, early_rul_days=None, late_rul_days=None, sigma_factor=1.645, limit_value=0.2, best_model="Unknown"):
    cdf = 0.5 * (1.0 + math.erf(sigma_factor / (2 ** 0.5)))

    def _to_duration(rul): return f"P{max(0, int(round(float(rul))))}D" if not pd.isna(rul) and rul is not None else None
    def _num(x): return round(float(x), 2) if not pd.isna(x) and x is not None else None

    return {
        "type": "pdm_status_changed",
        "timestamp": _now_iso(),
        "gaugeNumber": GAUGE_NUMBER,
        "status": status,
        "component": {
            "type": "ionization_chamber_kg20/10",
            "name": f"Channel Nr. {channel}",
            "label": str(channel)
        },
        "metric": {
            "type": "prediction",
            "symbol": "RUL",
            "name": "Remaining Useful Life",
            "value": _to_duration(nominal_rul_days),
            "unit": "days",
            "confidence": 0.5,
            "limit": float(limit_value) if limit_value is not None else 0.2
        },
        "metadata": {
            "rul_early_days": _num(early_rul_days), 
            "rul_early_confidence": round(1.0 - cdf, 4),
            "rul_nominal_days": _num(nominal_rul_days), 
            "rul_nominal_confidence": 0.5,
            "rul_late_days": _num(late_rul_days), 
            "rul_late_confidence": round(cdf, 4),
            "confidence_definition": "chance_threshold_crossed",
            "sigma_factor": round(float(sigma_factor), 4),
            "model": best_model
        }
    }

def send_custom_telegram(channel, status, dry_run=True, **kwargs):
    body = build_custom_telegram(channel, status, **kwargs)
    if dry_run: return {"sent": False, "status_code": "DRY_RUN", "body": body}
    try:
        resp = requests.post(ENDPOINT, headers=HEADERS, json=[body], timeout=5.0, verify=False)
        resp.raise_for_status()
        return {"sent": True, "status_code": resp.status_code, "body": body}
    except Exception as e:
        return {"sent": False, "status_code": "ERROR", "body": body, "error": str(e)}

def delete_telegrams():
    try:
        resp = requests.delete(DELETE_ENDPOINT, headers=HEADERS, timeout=10.0, verify=False)
        resp.raise_for_status()
        return True, f"HTTP {resp.status_code} (all deleted)"
    except Exception as e: return False, str(e)


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
        min_value=1, max_value=20, value=4,
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
                        nom_rul = raw_nom_rul
                        if isinstance(raw_nom_rul, (float, int)):
                            history = st.session_state.channel_rul_history[ch]
                            history = [row for row in history if row['step'] != st.session_state.current_step]
                            history.append({'step': st.session_state.current_step, 'rul': raw_nom_rul})
                            history = sorted(history, key=lambda x: x['step'])
                            st.session_state.channel_rul_history[ch] = history
                            
                            valid_history = [row for row in history if row['step'] <= st.session_state.current_step]
                            valid_ruls = [row['rul'] for row in valid_history]
                            valid_steps = [row['step'] for row in valid_history]
                            
                            if valid_ruls:
                                ema_series = pd.Series(valid_ruls).ewm(span=rul_ema_window, adjust=False).mean()
                                nom_rul = ema_series.iloc[-1]
                                
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
                fig, _, _ = fit_and_plotly_model(
                    time_raw=t_data, sensor_smooth=s_smooth, sensor_raw=s_raw, 
                    model_choice=b_model, thresholds=[target_threshold], 
                    precomputed_params=top_models[b_model]['params'],
                    channel_name=target_ch
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