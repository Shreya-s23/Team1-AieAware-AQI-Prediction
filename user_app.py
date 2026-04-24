"""
user_app.py — AirAware User Dashboard (read-only)
Run with: streamlit run user_app.py --server.port 8501

Users can only VIEW the pre-trained model, metrics, and visualizations.
No training, no dataset uploads, no model changes.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import time

from utils import (
    authenticate, register_user, is_logged_in, current_user, logout,
    load_csv, find_target, auto_detect_target,
    model_exists, load_model, load_metrics, load_admin_data,
    classify_aqi, aqi_advice, AQI_LEVELS, predict_next,
)
from theme import get_theme, inject_base_css
from charts import (
    chart_trend, chart_rolling, chart_monthly_bar,
    chart_heatmap, chart_radar, chart_forecast,
    chart_importance, chart_correlation,
    chart_distribution, chart_alerts, chart_hourly
)
from sidebar import render_sidebar

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AirAware",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={}
)

for k, v in [("dark_mode", False), ("auth_page", "login")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
#  AUTH PAGES
# ─────────────────────────────────────────────────────────────────────────────
def show_auth():
    C = get_theme(st.session_state.dark_mode)
    inject_base_css(C, "user")

    _, col, _ = st.columns([1, 2, 1])
    with col:
        page = st.session_state.auth_page

        st.markdown(f"""
        <div style="background:{C['card']};border:1px solid {C['border']};
             border-radius:16px;padding:2.2rem 1.8rem;margin-top:3rem">
            <div style="text-align:center;margin-bottom:1.5rem">
                <div style="font-family:'Sora',sans-serif;font-size:1.9rem;
                     font-weight:800;letter-spacing:-0.05em;color:{C['text']}">
                    Air<span style="color:{C['accent']}">Aware</span>
                </div>
                <div style="font-size:0.58rem;text-transform:uppercase;
                     letter-spacing:0.14em;color:{C['text3']};margin-top:3px">
                    Air Quality Intelligence
                </div>
                <div style="margin-top:0.8rem">
                    <span style="background:{C['accent']}15;color:{C['accent']};
                         border:1px solid {C['accent']}30;border-radius:5px;
                         padding:3px 10px;font-size:0.62rem;font-weight:700;
                         text-transform:uppercase;letter-spacing:0.08em">
                        {'Sign In' if page == 'login' else 'Create Account'}
                    </span>
                </div>
            </div>
            <div style="height:1px;background:{C['border']};margin:1rem 0"></div>
        """, unsafe_allow_html=True)

        if page == "login":
            _show_login_form(C)
        else:
            _show_signup_form(C)

        # Toggle between login and signup
        if page == "login":
            st.markdown(f"""
            <div style="text-align:center;font-size:0.74rem;color:{C['text3']};margin-top:1rem">
                New to AirAware?
            </div>""", unsafe_allow_html=True)
            if st.button("Create an Account", use_container_width=True, key="go_signup"):
                st.session_state.auth_page = "signup"
                st.rerun()
        else:
            st.markdown(f"""
            <div style="text-align:center;font-size:0.74rem;color:{C['text3']};margin-top:1rem">
                Already have an account?
            </div>""", unsafe_allow_html=True)
            if st.button("Sign In", use_container_width=True, key="go_login"):
                st.session_state.auth_page = "login"
                st.rerun()

        # Dark mode toggle on auth page
        dark_t = st.toggle("Dark Mode", value=st.session_state.dark_mode, key="auth_dark")
        if dark_t != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_t
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def _show_login_form(C):
    username = st.text_input("Username", placeholder="your username", key="login_user")
    password = st.text_input("Password", type="password", placeholder="••••••••", key="login_pw")
    if st.button("Sign In", use_container_width=True, key="btn_login"):
        ok, info = authenticate(username, password)
        if ok:
            if info.get("role") == "admin":
                st.error("Admin accounts must use the Admin Portal.")
            else:
                st.session_state.logged_in = True
                st.session_state.user_info = info
                st.rerun()
        else:
            st.error("Incorrect username or password.")


def _show_signup_form(C):
    name     = st.text_input("Full Name", placeholder="Your name", key="signup_name")
    username = st.text_input("Username", placeholder="choose a username", key="signup_user")
    password = st.text_input("Password", type="password",
                              placeholder="At least 6 characters", key="signup_pw")
    confirm  = st.text_input("Confirm Password", type="password",
                              placeholder="Re-enter password", key="signup_confirm")

    if st.button("Create Account", use_container_width=True, key="btn_signup"):
        if not name or not username or not password:
            st.error("All fields are required.")
        elif password != confirm:
            st.error("Passwords do not match.")
        else:
            ok, msg = register_user(username, password, name)
            if ok:
                st.success(msg + " Please sign in.")
                st.session_state.auth_page = "login"
                st.rerun()
            else:
                st.error(msg)


# ─────────────────────────────────────────────────────────────────────────────
#  TOPBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_topbar(C, n_records=None, target=None):
    parts = []
    if n_records: parts.append(f"{n_records:,} records")
    if target: parts.append(f"target: {target}")
    meta = "  /  ".join(parts) if parts else "no dataset loaded"
    st.markdown(f"""
    <div class="topbar">
        <div class="topbar-brand">
            Air<span class="accent">Aware</span>
        </div>
        <div class="topbar-right">
            <span class="role-badge">Viewer</span>
            <span><span class="live-dot"></span>{meta}</span>
        </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL METRICS — READ ONLY
# ─────────────────────────────────────────────────────────────────────────────
def show_model_metrics(C):
    """Display trained model metrics — read only for users."""
    saved = load_metrics()
    if not saved:
        st.markdown(f"""
        <div class="banner" style="--bc:{C['text3']}">
            <div class="banner-title">No trained model available</div>
            <div class="banner-text">
                The administrator has not yet trained a model.
                Please check back later.
            </div>
        </div>""", unsafe_allow_html=True)
        return

    st.markdown(f"""
    <div class="banner" style="--bc:{C['accent3']}">
        <div class="banner-title">Trained Model — {saved.get('dataset_name','—')}</div>
        <div class="banner-text">
            Target column: {saved.get('target_col','—')} |
            Read-only access — contact admin to update the model.
        </div>
    </div>""", unsafe_allow_html=True)

    r2   = saved.get("r2", 0)
    r2_color = C["good"] if r2 > 0.9 else (C["moderate"] if r2 > 0.7 else C["bad"])

    for col_w, (lbl, val, desc, vc) in zip(st.columns(3), [
        ("R2 Score", str(saved.get("r2","—")), "Model accuracy (1.0 = perfect)", r2_color),
        ("MAE", str(saved.get("mae","—")), "Mean absolute error", C["accent"]),
        ("RMSE", str(saved.get("rmse","—")), "Root mean squared error", C["accent"]),
    ]):
        with col_w:
            st.markdown(f"""
            <div class="score-box">
                <div class="score-num" style="color:{vc}">{val}</div>
                <div class="score-key">{lbl}</div>
                <div class="score-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)


def show_model_visualizations(C):
    """Show pre-computed model forecast chart if model + data are available."""
    if not model_exists():
        st.info("No trained model found. Ask your administrator to train a model.")
        return

    # Try to load model for feature importance
    model, feats = load_model()
    if model is None:
        st.warning("Could not load model file.")
        return

    saved = load_metrics()
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-lbl">Feature Importance — Trained Model</div>', unsafe_allow_html=True)

    # Show feature importance (no predictions needed — just importance from loaded model)
    from charts import _setup
    import matplotlib.pyplot as plt
    import pandas as pd

    _setup(C)
    imp = pd.DataFrame({"Feature": feats, "Importance": model.feature_importances_})
    imp = imp.sort_values("Importance", ascending=True).tail(12)
    q33 = imp["Importance"].quantile(0.33)
    q66 = imp["Importance"].quantile(0.66)
    colors = [C["accent"] if v >= q66 else (C["accent3"] if v >= q33 else C["text3"]) for v in imp["Importance"]]
    fig, ax = plt.subplots(figsize=(9, len(imp)*0.36+0.8))
    fig.patch.set_facecolor(C["mpl_bg"])
    ax.barh(imp["Feature"], imp["Importance"], color=colors, alpha=0.85, height=0.55)
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance (Read Only)", pad=8, fontweight='semibold')
    ax.spines[["top","right","left"]].set_visible(False)
    ax.grid(True, axis='x', alpha=0.25)
    ax.tick_params(left=False)
    plt.tight_layout(pad=0.8)
    st.markdown('<div class="cw">', unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=True)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────
def render_kpis(df, TARGET, C):
    aqi_mean = df[TARGET].mean()
    aqi_max  = df[TARGET].max()
    good_pct = (df[TARGET] <= 50).sum() / max(len(df),1) * 100
    alert_ct = (df[TARGET] > 100).sum()
    alert_pct= alert_ct / max(len(df),1) * 100
    cat, color = classify_aqi(aqi_mean)
    for col_w, (label, val, sub, clr) in zip(st.columns(4), [
        ("Mean AQI",  f"{aqi_mean:.1f}", cat, color),
        ("Peak AQI",  f"{aqi_max:.0f}",  "Maximum", C["bad"]),
        ("Good Air",  f"{good_pct:.0f}%",f"{(df[TARGET]<=50).sum():,} records", C["good"]),
        ("Alerts",    f"{alert_ct:,}",   f"{alert_pct:.1f}%", C["moderate"] if alert_pct < 20 else C["bad"]),
    ]):
        with col_w:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-accent-bar" style="--sc:{clr}"></div>
                <div class="stat-label">{label}</div>
                <div class="stat-value" style="--sv:{clr}">{val}</div>
                <div class="stat-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB VIEWS
# ─────────────────────────────────────────────────────────────────────────────
def tab_overview(df, TARGET, selected_poll, C):
    aqi_val = df[TARGET].iloc[-1]
    label, color = classify_aqi(aqi_val)
    st.markdown(f"""
    <div class="banner" style="--bc:{color}">
        <div class="banner-title">Current — {label} (AQI {aqi_val:.0f})</div>
        <div class="banner-text">{aqi_advice(label)}</div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown('<div class="cw"><div class="cw-title">Pollutant Time Series</div>', unsafe_allow_html=True)
        if selected_poll in df.columns:
            fig = chart_trend(df, selected_poll, C)
            st.pyplot(fig, use_container_width=True); plt.close()
        else:
            st.info(f"'{selected_poll}' not in dataset.")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="cw"><div class="cw-title">Pollutant Profile</div>', unsafe_allow_html=True)
        fig_r = chart_radar(df, C)
        if fig_r:
            st.pyplot(fig_r, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="cw"><div class="cw-title">Rolling Averages</div>', unsafe_allow_html=True)
        st.pyplot(chart_rolling(df, TARGET, C), use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="cw"><div class="cw-title">Monthly Pattern</div>', unsafe_allow_html=True)
        st.pyplot(chart_monthly_bar(df, TARGET, C), use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    if len(df) >= 24:
        st.markdown('<div class="cw"><div class="cw-title">AQI Heatmap</div>', unsafe_allow_html=True)
        st.pyplot(chart_heatmap(df, TARGET, C), use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)


def tab_model(C):
    """Read-only model tab for users."""
    st.markdown(f"""
    <div style="margin-bottom:1rem">
        <span class="readonly-badge">Read Only — Model managed by Admin</span>
    </div>""", unsafe_allow_html=True)
    show_model_metrics(C)
    show_model_visualizations(C)


def tab_alerts_user(df, TARGET, alert_thr, C):
    aqi_vals = df[TARGET]
    is_norm = aqi_vals.abs().max() < 20
    threshold = ((alert_thr - float(aqi_vals.mean())) / (float(aqi_vals.std()) or 1.0)
                 if is_norm else float(alert_thr))
    alert_df  = df[df[TARGET] > threshold]
    alert_pct = len(alert_df) / max(len(df),1) * 100

    if len(alert_df) == 0:
        bc, bt, bb = C["good"], "All Clear", "All readings within threshold."
    elif alert_pct > 30:
        bc, bt, bb = C["bad"], f"High Alert — {len(alert_df):,} records exceed threshold", \
                     f"{alert_pct:.1f}% above AQI {alert_thr}."
    else:
        bc, bt, bb = C["moderate"], f"Caution — {len(alert_df):,} exceedances", \
                     f"{alert_pct:.1f}% above AQI {alert_thr}."

    st.markdown(f'<div class="banner" style="--bc:{bc}"><div class="banner-title">{bt}</div>'
                f'<div class="banner-text">{bb}</div></div>', unsafe_allow_html=True)

    for col_w, (lbl, val, clr) in zip(st.columns(4), [
        ("Total Records", f"{len(df):,}", C["text2"]),
        ("Alert Records", f"{len(alert_df):,}", C["bad"] if len(alert_df) > 0 else C["good"]),
        ("Alert Rate", f"{alert_pct:.1f}%", C["moderate"]),
        ("Peak AQI", f"{alert_df[TARGET].max():.0f}" if len(alert_df) > 0 else "—", C["bad"]),
    ]):
        with col_w:
            st.markdown(f"""
            <div class="stat-card" style="padding:1rem 1.2rem">
                <div class="stat-accent-bar" style="--sc:{clr}"></div>
                <div class="stat-label">{lbl}</div>
                <div class="stat-value" style="--sv:{clr};font-size:1.6rem">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="cw">', unsafe_allow_html=True)
    st.pyplot(chart_alerts(df, TARGET, threshold, C), use_container_width=True); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)


def tab_analysis_user(df, TARGET, C):
    corr_cols = [c for c in ["PM2.5","PM10","NO2(GT)","CO(GT)","Temperature","Humidity","AirQualityIndex"] if c in df.columns]
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown('<div class="cw"><div class="cw-title">Correlation Matrix</div>', unsafe_allow_html=True)
        if len(corr_cols) >= 2:
            fig_c = chart_correlation(df, corr_cols, C)
            if fig_c:
                st.pyplot(fig_c, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="cw"><div class="cw-title">AQI Distribution</div>', unsafe_allow_html=True)
        st.pyplot(chart_distribution(df, TARGET, C), use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    if corr_cols:
        st.markdown('<div class="sec-lbl">Statistical Summary</div>', unsafe_allow_html=True)
        st.dataframe(df[corr_cols].describe().round(3), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-lbl">AQI Category Breakdown</div>', unsafe_allow_html=True)
    for col_w, (lo, hi, label, color) in zip(st.columns(len(AQI_LEVELS)), AQI_LEVELS):
        count = ((df[TARGET] >= lo) & (df[TARGET] <= hi)).sum()
        pct = count / max(len(df),1) * 100
        with col_w:
            st.markdown(f"""
            <div class="cat-cell" style="--cbg:{color}12;--cborder:{color}30;--ccolor:{color}">
                <div class="cat-name">{label}</div>
                <div class="cat-pct">{pct:.1f}%</div>
                <div class="cat-count">{count:,}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="cw"><div class="cw-title">Hourly AQI Profile</div>', unsafe_allow_html=True)
    st.pyplot(chart_hourly(df, TARGET, C), use_container_width=True); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)


def predict_next_24h(df, TARGET, model, feats):
    """Predict next 24 hours using iterative predictions"""
    if len(df) == 0 or model is None:
        current_aqi = df[TARGET].iloc[-1] if len(df) > 0 else 0
        return [current_aqi] * 24
    
    predictions = []
    temp_df = df.copy()
    
    for _ in range(24):
        pred = predict_next(temp_df, TARGET, model, feats)
        if pred is None:
            pred = temp_df[TARGET].iloc[-1]
        predictions.append(pred)
        
        # Add the prediction to temp_df for next prediction
        new_row = temp_df.iloc[-1:].copy()
        new_row['Datetime'] = new_row['Datetime'] + pd.Timedelta(hours=1)
        new_row[TARGET] = pred
        temp_df = pd.concat([temp_df, new_row], ignore_index=True)
    
    return predictions
def render_dynamic_dashboard(df, TARGET, C):
    # Top bar with live time
    current_time = datetime.datetime.now().strftime("%H:%M")
    st.markdown(f"""
    <div style="display:flex;align-items:space-between;margin-bottom:16px;flex-wrap:wrap;gap:8px">
        <h1 style="font-family:'Syne',sans-serif;font-size:17px;font-weight:700;color:{C['text']};letter-spacing:-0.3px;margin:0">
            India Air Quality Dashboard
        </h1>
        <span style="font-size:11px;color:{C['text3']};background:{C['card']};padding:3px 8px;border-radius:20px;border:0.5px solid {C['border']};align-self:center">
            Live {current_time}
        </span>
    </div>""", unsafe_allow_html=True)

    # Region selector (simplified, since we have one dataset)
    region_name = "Current Dataset"
    current_aqi = df[TARGET].iloc[-1] if len(df) > 0 else 0
    label, color = classify_aqi(current_aqi)
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;flex-wrap:wrap">
        <label style="font-size:12px;color:{C['text3']};font-weight:500;white-space:nowrap">Region</label>
        <span style="flex:1;min-width:180px;font-family:'DM Sans',sans-serif;font-size:13px;padding:7px 12px;border:0.5px solid {C['border']};border-radius:8px;background:{C['card']};color:{C['text']}">
            {region_name}
        </span>
        <div style="padding:5px 14px;border-radius:20px;font-size:12px;font-weight:500;white-space:nowrap;background:{color}22;color:{color};border:0.5px solid {color}55">
            {label}
        </div>
    </div>""", unsafe_allow_html=True)

    # AQI Category Breakdown
    st.markdown(f'<div style="font-size:11px;font-weight:500;color:{C["text3"]};text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px">AQI Category Breakdown</div>', unsafe_allow_html=True)
    cols = st.columns(6)
    for i, (lo, hi, label, color) in enumerate(AQI_LEVELS):
        count = ((df[TARGET] >= lo) & (df[TARGET] <= hi)).sum()
        pct = count / max(len(df),1) * 100
        with cols[i]:
            st.markdown(f"""
            <div style="border-radius:8px;padding:8px 4px;text-align:center;border:0.5px solid {C['border']};background:{C['card']}">
                <div style="width:8px;height:8px;border-radius:50%;margin:0 auto 4px;background:{color}"></div>
                <div style="font-size:9px;color:{C['text3']};margin-bottom:2px">{lo}–{hi}</div>
                <div style="font-size:9px;font-weight:500;color:{C['text']};line-height:1.2">{label}</div>
                <div style="font-size:8px;color:{C['text3']};margin-top:2px">{pct:.1f}%</div>
            </div>""", unsafe_allow_html=True)

    # Real-time Metrics
    st.markdown(f'<div style="font-size:11px;font-weight:500;color:{C["text3"]};text-transform:uppercase;letter-spacing:0.8px;margin:16px 0 8px 0">Real-time Metrics — {region_name}</div>', unsafe_allow_html=True)
    metrics_cols = st.columns(5)
    pollutants = ["AQI", "PM2.5", "PM10", "NO2", "CO"]
    units = ["index", "μg/m³", "μg/m³", "μg/m³", "mg/m³"]
    cols_map = {"AQI": TARGET, "PM2.5": "PM2.5", "PM10": "PM10", "NO2": "NO2(GT)", "CO": "CO(GT)"}
    for i, (poll, unit) in enumerate(zip(pollutants, units)):
        col_name = cols_map[poll]
        if col_name in df.columns:
            val = df[col_name].iloc[-1] if len(df) > 0 else 0
            if poll == "AQI":
                _, color = classify_aqi(val)
            else:
                color = C['text']
        else:
            val = "—"
            color = C['text3']
        with metrics_cols[i]:
            st.markdown(f"""
            <div style="background:{C['card']};border-radius:8px;padding:12px 14px">
                <div style="font-size:11px;color:{C['text3']};margin-bottom:4px">{poll}</div>
                <div style="font-size:22px;font-weight:500;color:{color};line-height:1">{val if isinstance(val, str) else f"{val:.1f}" if poll == "CO" else f"{val:.0f}"}</div>
                <div style="font-size:11px;color:{C['text3']};margin-top:2px">{unit}</div>
            </div>""", unsafe_allow_html=True)

    # Next 24 Hour Prediction
    st.markdown(f'<div style="font-size:11px;font-weight:500;color:{C["text3"]};text-transform:uppercase;letter-spacing:0.8px;margin:16px 0 8px 0">Next 24 Hour AQI Prediction</div>', unsafe_allow_html=True)
    
    # Load model for prediction
    model, feats = load_model()
    predictions = []
    if model is not None and feats is not None and len(df) > 0:
        # Predict next 24 hours
        predictions = predict_next_24h(df, TARGET, model, feats)
    else:
        current_aqi = df[TARGET].iloc[-1] if len(df) > 0 else 0
        predictions = [current_aqi] * 24
    
    next_pred = predictions[0] if predictions else current_aqi
    label_pred, color_pred = classify_aqi(next_pred)
    
    st.markdown(f"""
    <div style="background:{C['card']};border:0.5px solid {C['border']};border-radius:12px;padding:12px 14px;margin-bottom:16px;display:flex;align-items:center;gap:12px;flex-wrap:wrap">
        <div><div style="font-size:11px;color:{C['text3']};margin-bottom:2px">Current</div><div style="font-size:20px;font-weight:500;color:{color};transition:color 0.4s">{current_aqi:.0f}</div></div>
        <div style="font-size:18px;color:{C['text3']}">→</div>
        <div style="flex:1;min-width:160px">
            <div style="font-size:11px;color:{C['text3']};margin-bottom:4px">Predicted in 1 hour</div>
            <div style="height:8px;border-radius:4px;background:{C['card']};overflow:hidden;border:1px solid {C['border']}">
                <div style="height:100%;border-radius:4px;background:{color_pred};width:{min(next_pred/500*100,100)}%;transition:width 0.8s ease,background 0.5s"></div>
            </div>
            <div style="display:flex;justify-content:space-between;margin-top:2px">
                <span style="font-size:9px;color:{C['text3']}">0</span>
                <span style="font-size:9px;color:{C['text3']}">500</span>
            </div>
        </div>
        <div><div style="font-size:11px;color:{C['text3']};margin-bottom:2px">Predicted</div><div style="font-size:20px;font-weight:500;color:{color_pred};transition:color 0.4s">{next_pred:.0f}</div></div>
        <div style="padding:5px 14px;border-radius:20px;font-size:10px;font-weight:500;background:{color_pred}22;color:{color_pred};border:0.5px solid {color_pred}55">
            {label_pred}
        </div>
    </div>""", unsafe_allow_html=True)

    # Awareness Flashcards
    st.markdown(f'<div style="font-size:11px;font-weight:500;color:{C["text3"]};text-transform:uppercase;letter-spacing:0.8px;margin:16px 0 8px 0">Awareness Flashcards</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    # Flashcard 1: Predictive Meter
    with col1:
        st.markdown(f"""
        <div style="background:{C['card']};border:0.5px solid {C['border']};border-radius:12px;overflow:hidden;min-height:320px;display:flex;flex-direction:column">
            <div style="padding:10px 12px 0;display:flex;align-items:center;gap:6px">
                <div style="width:20px;height:20px;border-radius:50%;background:{C['card']};border:0.5px solid {C['border']};display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:500;color:{C['text3']};flex-shrink:0">1</div>
                <div style="font-size:11px;font-weight:500;color:{C['text3']};font-family:'Syne',sans-serif">Current AQI & 24h Prediction</div>
            </div>
            <div style="flex:1;padding:10px 12px 12px;display:flex;flex-direction:column">
                <div style="margin-bottom:10px">
                    <div style="font-size:36px;font-weight:700;font-family:'Syne',sans-serif;line-height:1;color:{color};transition:color 0.4s">{current_aqi:.0f}</div>
                    <div style="font-size:11px;font-weight:500;margin-top:2px;color:{color};transition:color 0.4s">{label}</div>
                </div>
                <div style="height:10px;border-radius:5px;background:linear-gradient(to right,{C['good']} 0%,{C['moderate']} 20%,{C['bad']} 40%,{C['text3']} 60%,{C['text3']} 100%);margin-bottom:6px;position:relative">
                    <div style="position:absolute;top:-3px;width:16px;height:16px;background:{C['card']};border:2px solid {C['text']};border-radius:50%;left:{min(current_aqi/500*100,100)}%;transition:left 0.8s ease"></div>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:8px;color:{C['text3']};margin-bottom:8px">
                    <span>Good</span><span>Moderate</span><span>Unhealthy</span><span>Hazardous</span>
                </div>
                <div style="font-size:10px;color:{C['text3']};font-weight:500;margin-bottom:6px">24-hour forecast</div>
                <div style="flex:1;display:flex;flex-direction:column;gap:4px">
        """, unsafe_allow_html=True)
        
        # 24-hour bars with predicted values
        if predictions:
            hours_indices = [0, 3, 7, 11, 15, 17, 19, 23]  # 1h, 4h, 8h, 12h, 16h, 18h, 20h, 24h
            hour_labels = ['1h','4h','8h','12h','16h','18h','20h','24h']
            for i, (label, idx) in enumerate(zip(hour_labels, hours_indices)):
                pred = predictions[idx] if idx < len(predictions) else predictions[-1]
                pred_label, bar_color = classify_aqi(pred)
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:6px">
                    <span style="font-size:9px;color:{C['text3']};width:28px;flex-shrink:0">{label}</span>
                    <div style="flex:1;height:6px;background:{C['card']};border-radius:3px;overflow:hidden;border:1px solid {C['border']}">
                        <div style="height:100%;border-radius:3px;background:{bar_color};width:{min(pred/500*100,100)}%;transition:width 0.6s ease"></div>
                    </div>
                    <span style="font-size:9px;color:{C['text']};width:24px;text-align:right;flex-shrink:0">{pred:.0f}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="font-size:10px;color:{C["text3"]}">No predictions available</div>', unsafe_allow_html=True)
        
        st.markdown('</div></div>', unsafe_allow_html=True)
    
    # Flashcard 2: Pollution Effects
    with col2:
        # Initialize image rotation state
        if 'pollution_image_idx' not in st.session_state:
            st.session_state.pollution_image_idx = 0
        if 'last_rotation_time' not in st.session_state:
            st.session_state.last_rotation_time = datetime.datetime.now()
        
        # Rotate image every 3 seconds
        current_time = datetime.datetime.now()
        if (current_time - st.session_state.last_rotation_time).seconds >= 3:
            st.session_state.pollution_image_idx = (st.session_state.pollution_image_idx + 1) % 5
            st.session_state.last_rotation_time = current_time
        
        pollution_scenarios = [
            {
                "title": "Urban Smog",
                "description": "Heavy particulate matter from vehicle exhaust and industrial emissions creates visible smog, reducing visibility and causing respiratory irritation.",
                "color": C['bad']
            },
            {
                "title": "Industrial Pollution",
                "description": "Factory emissions release toxic gases and fine particles that can travel long distances, affecting air quality in surrounding communities.",
                "color": C['text3']
            },
            {
                "title": "Crop Burning",
                "description": "Agricultural waste burning releases massive amounts of PM2.5 and CO₂, contributing significantly to seasonal air pollution spikes.",
                "color": C['moderate']
            },
            {
                "title": "Construction Dust",
                "description": "Building activities generate airborne dust particles that contain silica and other harmful substances, affecting nearby residents.",
                "color": C['bad']
            },
            {
                "title": "Traffic Congestion",
                "description": "Idling vehicles in heavy traffic continuously emit pollutants, creating hotspots of poor air quality along major roadways.",
                "color": C['text3']
            }
        ]
        
        current_scenario = pollution_scenarios[st.session_state.pollution_image_idx]
        
        st.markdown(f"""
        <div style="background:{C['card']};border:0.5px solid {C['border']};border-radius:12px;overflow:hidden;min-height:320px;display:flex;flex-direction:column">
            <div style="padding:10px 12px 0;display:flex;align-items:center;gap:6px">
                <div style="width:20px;height:20px;border-radius:50%;background:{C['card']};border:0.5px solid {C['border']};display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:500;color:{C['text3']};flex-shrink:0">2</div>
                <div style="font-size:11px;font-weight:500;color:{C['text3']};font-family:'Syne',sans-serif">Pollution Effects</div>
            </div>
            <div style="flex:1;padding:10px 12px 12px;display:flex;flex-direction:column;gap:6px">
                <div style="flex:1;border-radius:8px;overflow:hidden;min-height:140px;position:relative;background:linear-gradient(135deg, {current_scenario['color']}22 0%, {C['card']} 100%);border:1px solid {C['border']};display:flex;flex-direction:column;align-items:center;justify-content:center;padding:20px">
                    <div style="width:60px;height:60px;border-radius:50%;background:{current_scenario['color']}44;border:2px solid {current_scenario['color']};display:flex;align-items:center;justify-content:center;margin-bottom:12px">
                        <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="{current_scenario['color']}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>
                        </svg>
                    </div>
                    <div style="font-size:14px;font-weight:600;color:{C['text']};text-align:center;margin-bottom:8px">{current_scenario['title']}</div>
                    <div style="font-size:12px;color:{C['text3']};text-align:center;line-height:1.4">{current_scenario['description']}</div>
                </div>
                <div style="font-size:10px;color:{C['text3']};line-height:1.4;text-align:center;padding:0 4px">
                    Air pollution causes respiratory diseases affecting millions daily.
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
    
    # Flashcard 3: Citizen Daily Actions
    with col3:
        st.markdown(f"""
        <div style="background:{C['card']};border:0.5px solid {C['border']};border-radius:12px;overflow:hidden;min-height:320px;display:flex;flex-direction:column">
            <div style="padding:10px 12px 0;display:flex;align-items:center;gap:6px">
                <div style="width:20px;height:20px;border-radius:50%;background:{C['card']};border:0.5px solid {C['border']};display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:500;color:{C['text3']};flex-shrink:0">3</div>
                <div style="font-size:11px;font-weight:500;color:{C['text3']};font-family:'Syne',sans-serif">Air (Prevention and Control of Pollution) Act, 1981</div>
            </div>
            <div style="flex:1;padding:10px 12px 12px;display:flex;flex-direction:column">
                <div style="display:flex;flex-direction:column;gap:7px">
        """, unsafe_allow_html=True)
        
        actions = [
            ("Use public transport or carpool", "Vehicle exhaust is a primary source of urban PM2.5 and NO₂. Choosing buses, metro, or shared rides directly reduces tailpipe emissions regulated under BS-VI norms."),
            ("Avoid burning waste, leaves, or crop residue", "Open burning is prohibited under the Air Act and actively penalised by State Pollution Control Boards. Composting is the government-endorsed alternative."),
            ("Monitor AQI daily via the SAMEER app", "MoEFCC's official app (System of Air Quality and Weather Forecasting And Research) gives real-time AQI for your area so you can plan outdoor activity accordingly."),
            ("Report industrial or vehicular violations", "Citizens can lodge complaints with their State Pollution Control Board or CPCB against industries violating emission norms — a right backed by Section 18 of the Air Act."),
            ("Switch to cleaner cooking and energy sources", "Replace biomass and kerosene with LPG or electric options. CPCB identifies indoor combustion as a major contributor to household and ambient PM levels."),
            ("Plant and protect trees in your locality", "Green belt requirements are mandated for all industrial setups under this Act. Citizens can mirror this by supporting urban tree drives and protecting existing green cover.")
        ]
        
        for title, desc in actions:
            st.markdown(f"""
            <div style="display:flex;gap:8px;align-items:flex-start">
                <div style="width:22px;height:22px;border-radius:6px;background:{C['card']};border:0.5px solid {C['border']};display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px">
                    <svg width="11" height="11" viewBox="0 0 12 12" fill="none" stroke="{C['text3']}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="6" cy="4" r="2"/><path d="M2 11c0-2.2 1.8-4 4-4s4 1.8 4 4"/><line x1="6" y1="8" x2="6" y2="9"/>
                    </svg>
                </div>
                <div>
                    <div style="font-size:24px;font-weight:500;color:{C['text']};margin-bottom:1px">{title}</div>
                    <div style="font-size:24px;color:{C['text3']};line-height:1.4">{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        
        st.markdown('</div></div></div>', unsafe_allow_html=True)
def main():
    if not is_logged_in():
        show_auth()
        return

    C = get_theme(st.session_state.dark_mode)
    inject_base_css(C, "user")

    uploaded, selected_poll, alert_thr = render_sidebar(C, "user")

    df_raw = None
    TARGET = None

    if uploaded:
        df_raw = load_csv(uploaded)
        TARGET = find_target(df_raw)
        if TARGET is None:
            candidates = auto_detect_target(df_raw)
            if candidates:
                TARGET = st.selectbox("Select target column:", candidates)
        render_topbar(C, len(df_raw), TARGET)
    else:
        # Try to load admin data
        df_raw, TARGET = load_admin_data()
        if df_raw is not None and TARGET is not None:
            render_topbar(C, len(df_raw), TARGET)
        else:
            render_topbar(C)

    if df_raw is not None and TARGET is not None:
        c1, c2, ci = st.columns([1,1,3])
        with c1:
            start_d = st.date_input("From", df_raw["Datetime"].min().date(),
                                    min_value=df_raw["Datetime"].min().date(),
                                    max_value=df_raw["Datetime"].max().date())
        with c2:
            end_d = st.date_input("To", df_raw["Datetime"].max().date(),
                                  min_value=df_raw["Datetime"].min().date(),
                                  max_value=df_raw["Datetime"].max().date())
        with ci:
            days = (pd.Timestamp(end_d) - pd.Timestamp(start_d)).days
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:20px;height:100%;
                 padding-top:8px;font-size:0.72rem;color:{C['text3']}">
                <span>{days} day range</span>
                <span>{len(df_raw):,} total records</span>
                <span style="color:{C['accent']};font-weight:600">{TARGET}</span>
            </div>""", unsafe_allow_html=True)

        df = df_raw[
            (df_raw["Datetime"] >= pd.Timestamp(start_d)) &
            (df_raw["Datetime"] <= pd.Timestamp(end_d))
        ].copy()

        if len(df) == 0:
            st.warning("No data in selected range.")
            return

        st.markdown("<br>", unsafe_allow_html=True)
        render_dynamic_dashboard(df, TARGET, C)
        st.markdown("<br>", unsafe_allow_html=True)

        # Keep the old tabs for additional views
        tabs = st.tabs(["Model Metrics", "Alerts", "Analysis"])
        with tabs[0]: tab_model(C)
        with tabs[1]: tab_alerts_user(df, TARGET, alert_thr, C)
        with tabs[2]: tab_analysis_user(df, TARGET, C)

    else:
        # No dataset — still show model metrics
        st.markdown(f"""
        <div class="banner" style="--bc:{C['accent']}">
            <div class="banner-title">Welcome, {current_user().get('name','')}</div>
            <div class="banner-text">
                Upload a CSV from the sidebar to view air quality charts.
                You can also view the pre-trained model metrics below.
            </div>
        </div>""", unsafe_allow_html=True)

        tab_model(C)

    st.markdown(f"""
    <div class="footer">
        <p>AirAware — Air Quality Intelligence Platform</p>
        <p>View-only access · Contact admin to update the model</p>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
