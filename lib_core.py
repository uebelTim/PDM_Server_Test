"""
lib_core.py — Shared algorithm + Telegram/API logic for the Machine Health dashboard.

Extracted verbatim from the original single-file app so BOTH the simulator page and
the live-monitoring page use identical RUL math and identical event payloads.
"""
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import datetime
from scipy.optimize import curve_fit, brentq
from sklearn.metrics import mean_squared_error
import streamlit as st

# =========================================================
# 0. CONSTANTS  (single source of truth for both pages)
# =========================================================
ENDPOINT = "https://mars01.dcims.ims/api/events"
DELETE_ENDPOINT = "https://mars01.dcims.ims/api/events"

# API token is read lazily so importing this module never crashes if secrets
# are missing (e.g. during unit tests). Pages that actually POST will provide it.
def get_api_token():
    try:
        return st.secrets["API_TOKEN"]
    except Exception:
        return None

def get_headers():
    return {
        "Authorization": f"Bearer {get_api_token()}",
        "Content-Type": "application/json",
    }

GAUGE_NUMBER = "140988"  # Hardcoded System ID

RUL_HORIZON = 180  # days; "Safe" (never reached) and anything beyond this saturate here.
MIN_MSE_FLOOR = 5e-5
SENTINEL_ALREADY_REACHED = -999.0


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

def fit_and_plotly_model(time_raw, sensor_smooth, sensor_raw, model_choice, thresholds=None, precomputed_params=None, sigma_factor=1.645, channel_name="", smoothed_rul=None, x_origin=None):
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

    # When an x_origin (a DateTime) is supplied, plot the x-axis as real calendar dates
    # (origin + elapsed days) instead of elapsed-day floats; the fit's future tail then
    # lands on future dates. Without it the behavior is unchanged (elapsed days).
    if x_origin is not None:
        x_origin = pd.Timestamp(x_origin)
        x_raw = x_origin + pd.to_timedelta(time_arr, unit='D')
        x_smooth = x_origin + pd.to_timedelta(time_smooth_converted, unit='D')
        x_axis_title = "DateTime"
    else:
        x_raw, x_smooth, x_axis_title = time_arr, time_smooth_converted, "Elapsed Days"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_raw, y=sensor_raw_arr, mode='markers', name='Raw Data', marker=dict(color='gray', size=5, opacity=0.7)))
    fig.add_trace(go.Scatter(x=x_smooth, y=smooth_preds, mode='lines', name=f'{model_choice} Fit', line=dict(color='blue', width=2.5)))
    fig.add_trace(go.Scatter(x=x_smooth, y=smooth_preds - (dynamic_std_smooth * sigma_factor), mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=x_smooth, y=smooth_preds + (dynamic_std_smooth * sigma_factor), mode='lines', fill='tonexty', fillcolor='rgba(0, 0, 255, 0.15)', name='Confidence Band', line=dict(width=0)))
    
    if thresholds:
        for thresh in thresholds: fig.add_hline(y=thresh, line_dash="dash", line_color="red", annotation_text=f"Target: {thresh}")

    # The chart's own nom_rul is the RAW (un-smoothed) prediction. The status engine
    # decides alerts from the EMA-smoothed value, so we surface both here to make the
    # comparison explicit — this is the number that actually drives the log.
    display_rul = format_rul_str(nom_rul, short=False)
    raw_is_danger = (isinstance(nom_rul, (float, int)) and nom_rul < 10) or display_rul == "Breached"

    if smoothed_rul is not None:
        display_ema = format_rul_str(smoothed_rul, short=False)
        ema_is_danger = (isinstance(smoothed_rul, (float, int)) and smoothed_rul < 10) or display_ema == "Breached"
        annotation_text = (
            f"<b>Raw RUL: {display_rul}</b><br>"
            f"<span style='font-size:13px'>EMA RUL (status): {display_ema}</span>"
        )
        font_color = "red" if ema_is_danger else "black"
    else:
        annotation_text = f"<b>Predicted RUL: {display_rul}</b>"
        font_color = "red" if raw_is_danger else "black"

    fig.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper",
        text=annotation_text,
        align="left",
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
        xaxis_title=x_axis_title,
        yaxis_title="Sensor Value",
        yaxis=dict(range=[absolute_y_min - y_padding, absolute_y_max + y_padding]),
        template="plotly_white",
        margin=dict(t=50, b=50, l=50, r=50)
    )
    # On a datetime axis, force day-level tick/hover labels so ticks read e.g.
    # "2026-01-14" instead of collapsing to just the month.
    if x_origin is not None:
        fig.update_xaxes(tickformat="%Y-%m-%d", hoverformat="%Y-%m-%d")
    return fig, pd.Series(dtype=float), pd.DataFrame(rul_records)


# =========================================================
# 2. TELEGRAM API HANDLING
# =========================================================
def _now_iso():
    return datetime.now().astimezone().isoformat(timespec='seconds')

def build_custom_telegram(channel, status, nominal_rul_days=None, early_rul_days=None, late_rul_days=None, sigma_factor=1.645, limit_value=0.2, best_model="Unknown", event_timestamp=None, gauge_number=None):
    cdf = 0.5 * (1.0 + math.erf(sigma_factor / (2 ** 0.5)))

    def _to_duration(rul): return f"P{max(0, int(round(float(rul))))}D" if not pd.isna(rul) and rul is not None else None
    def _num(x):
        # RUL day count for the wire payload. The server rejects both negative numbers
        # and null, so:
        #   * 'Safe' / no-crossing bands arrive here as None -> saturate at RUL_HORIZON
        #     (a long life), which also keeps the band ordering early <= nominal <= late
        #     valid instead of collapsing an optimistic band to 0.
        #   * breached channels carry SENTINEL_ALREADY_REACHED (-999) / negative RULs,
        #     meaning "0 days remaining" -> clamp to 0.
        if x is None or pd.isna(x):
            return float(RUL_HORIZON)
        try:
            return max(0.0, round(float(x), 2))
        except (TypeError, ValueError):
            return float(RUL_HORIZON)

    return {
        "type": "pdm_status_changed",
        # When the status change actually occurred (e.g. the historical moment a channel
        # breached the limit), not when this telegram was built. Defaults to now.
        "timestamp": event_timestamp or _now_iso(),
        # Which gauge/system this alert is for. The live page monitors several databases,
        # each its own gauge, so it passes the per-DB number; defaults to GAUGE_NUMBER.
        "gaugeNumber": str(gauge_number) if gauge_number is not None else GAUGE_NUMBER,
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
        resp = requests.post(ENDPOINT, headers=get_headers(), json=[body], timeout=5.0, verify=False)
        resp.raise_for_status()
        return {"sent": True, "status_code": resp.status_code, "body": body}
    except Exception as e:
        return {"sent": False, "status_code": "ERROR", "body": body, "error": str(e)}

def delete_telegrams():
    try:
        resp = requests.delete(DELETE_ENDPOINT, headers=get_headers(), timeout=10.0, verify=False)
        resp.raise_for_status()
        return True, f"HTTP {resp.status_code} (all deleted)"
    except Exception as e: return False, str(e)


