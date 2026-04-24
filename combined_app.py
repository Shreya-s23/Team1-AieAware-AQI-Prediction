"""
combined_app.py — AirAware Unified Dashboard
Run with: streamlit run combined_app.py

Combines user_app.py and admin_app.py into a single role-based app.
Admins can access user interface via toggle.

Folder structure:
  your_project/
  ├── combined_app.py          ← this file
  ├── utils.py
  ├── theme.py
  ├── charts.py
  ├── users.json            (auto-created)
  ├── aqi_xgboost_model.pkl (auto-created on first train)
  ├── aqi_feature_columns.pkl
  └── aqi_model_metrics.json
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from utils import (
    authenticate, is_logged_in, current_user, logout,
    load_csv, auto_detect_target, find_target,
    engineer_features, train_model, predict_next,
    model_exists, load_model, load_metrics, save_metrics,
    classify_aqi, aqi_advice, AQI_LEVELS,
    MODEL_FILE, FEATS_FILE
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
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={}
)

# Session state initialization
for k in ["dark_mode", "role", "admin_df", "admin_target", "retrain_done",
          "user_df", "user_target", "admin_view_user", "admin_nav"]:
    if k not in st.session_state:
        st.session_state[k] = "Overview" if k == "admin_nav" else (False if k in ["dark_mode", "retrain_done", "admin_view_user"] else None)

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# ROLE SELECTION
# ─────────────────────────────────────────────────────────────────────────────
def show_role_selection():
    C = get_theme(st.session_state.dark_mode)
    inject_base_css(C, "role")

    # Header
    st.markdown(f"""
    <div class="auth-wrap">
        <div class="auth-logo">
            <div class="auth-logo-name">Air<span class="accent">Aware</span></div>
            <div class="auth-logo-sub">Air Quality Intelligence</div>
        </div>
        <div style="text-align:center;margin-bottom:1.2rem">
            <span class="auth-role-tag">Select Your Role</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Center card
    _, col, _ = st.columns([1, 2, 1])

    with col:
        st.markdown(f"""
        <div style="
            background:{C['card']};
            border:1px solid {C['border']};
            border-radius:16px;
            padding:2rem 1.8rem;
            max-width:400px;
            margin:0 auto;
            text-align:center;
        ">
        """, unsafe_allow_html=True)

        # Buttons (FIXED INDENTATION)
        st.markdown(
            '<div style="display:flex;flex-direction:column;gap:1rem">',
            unsafe_allow_html=True
        )

        if st.button("👤 User Access", use_container_width=True, type="primary", key="user_btn"):
            st.session_state.role = "user"
            st.rerun()

        if st.button("🛡️ Admin Access", use_container_width=True, key="admin_btn"):
            st.session_state.role = "admin"
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        # Dark mode toggle
        dark_t = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode, key="role_dark")
        if dark_t != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_t
            st.rerun()

        # Close card
        st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# AUTH GATE
# ─────────────────────────────────────────────────────────────────────────────
def show_auth(role):
    C = get_theme(st.session_state.dark_mode)
    inject_base_css(C, role)

    role_title = "Admin Portal" if role == "admin" else "User Dashboard"
    role_color = "#7c3aed" if role == "admin" else "#059669"

    st.markdown(f"""
    <div class="auth-wrap">
        <div class="auth-logo">
            <div class="auth-logo-name">Air<span class="accent">Aware</span></div>
            <div class="auth-logo-sub">Air Quality Intelligence</div>
        </div>
        <div style="text-align:center;margin-bottom:1.2rem">
            <span class="auth-role-tag">{role_title}</span>
        </div>
        <div class="auth-divider"></div>
    </div>""", unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown(f"""
        <div style="background:{C['card']};border:1px solid {C['border']};
             border-radius:16px;padding:2rem 1.8rem;">
            <div style="text-align:center;margin-bottom:1.5rem">
                <div style="font-family:'Sora',sans-serif;font-size:1.8rem;
                     font-weight:800;letter-spacing:-0.05em;color:{C['text']}">
                    Air<span style="color:{role_color}">Aware</span>
                </div>
                <div style="font-size:0.58rem;text-transform:uppercase;
                     letter-spacing:0.14em;color:{C['text3']};margin-top:3px">
                    {role_title}
                </div>
                <div style="margin-top:0.8rem">
                    <span style="background:{role_color}18;color:{role_color};
                         border:1px solid {role_color}30;border-radius:5px;
                         padding:3px 10px;font-size:0.62rem;font-weight:700;
                         text-transform:uppercase;letter-spacing:0.08em">
                        {role.upper()}
                    </span>
                </div>
            </div>
            <div style="height:1px;background:{C['border']};margin:1rem 0"></div>
        """, unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="user" if role == "user" else "admin")
        password = st.text_input("Password", type="password", placeholder="••••••••")

        if st.button("Sign In", use_container_width=True):
            ok, info = authenticate(username, password)
            if ok and info.get("role") == role:
                st.session_state.logged_in = True
                st.session_state.user_info = info
                st.rerun()
            elif ok and info.get("role") != role:
                st.error(f"This account does not have {role} privileges.")
            else:
                st.error("Invalid username or password.")

        if role == "user":
            st.markdown('<div style="text-align:center;margin-top:1rem">', unsafe_allow_html=True)
            if st.button("Create Account", key="signup"):
                st.session_state.show_signup = True
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        dark_t = st.toggle("Dark Mode", value=st.session_state.dark_mode, key="auth_dark")
        if dark_t != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_t
            st.rerun()

        default_creds = "admin / admin123" if role == "admin" else "user / user123"
        st.markdown(f"""
            <div style="text-align:center;font-size:0.72rem;
                 color:{C['text3']};margin-top:1rem">
                Default credentials: {default_creds}
            </div>
        </div>""", unsafe_allow_html=True)

    if st.session_state.get("show_signup", False):
        st.markdown(f"""
        <div style="background:{C['card']};border:1px solid {C['border']};
             border-radius:16px;padding:2rem 1.8rem;margin-top:1rem;">
            <div style="text-align:center;margin-bottom:1.5rem">
                <div style="font-family:'Sora',sans-serif;font-size:1.5rem;
                     font-weight:800;color:{C['text']}">
                    Create New Account
                </div>
            </div>
        """, unsafe_allow_html=True)

        name = st.text_input("Full Name", placeholder="Your full name")
        new_username = st.text_input("Username", placeholder="Choose a username")
        new_password = st.text_input("Password", type="password", placeholder="••••••••")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="••••••••")

        if st.button("Sign Up", use_container_width=True):
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                from utils import register_user
                ok, msg = register_user(new_username, new_password, name)
                if ok:
                    st.success(msg)
                    st.session_state.show_signup = False
                    st.rerun()
                else:
                    st.error(msg)

        if st.button("Back to Login", use_container_width=True):
            st.session_state.show_signup = False
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

# SIDEBAR (Admin)
# ─────────────────────────────────────────────────────────────────────────────
def render_admin_sidebar(C):
    with st.sidebar:
        st.header("AirAware")
        st.subheader("Admin Dashboard")
        user = current_user()
        st.write(f"Signed in as {user.get('name','Administrator')}")
        st.write("Admin")
        st.markdown("---")
        st.subheader("Interface")
        view_user = st.toggle("User View", value=st.session_state.admin_view_user)
        if view_user != st.session_state.admin_view_user:
            st.session_state.admin_view_user = view_user
            st.rerun()
        st.markdown("---")
        st.subheader("Training Dataset")
        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded:
            st.write(f"Loaded {uploaded.name}")
        st.markdown("---")
        st.subheader("Alert Threshold")
        alert_thr = st.slider("AQI", 50, 300, 100, label_visibility="collapsed")
        from utils import classify_aqi
        t_label, t_color = classify_aqi(alert_thr)
        st.write(f"{alert_thr} - {t_label}")
        st.markdown("---")
        st.subheader("Pollutant View")
        poll_opts = ["AirQualityIndex","PM2.5","PM10","NO2(GT)","CO(GT)","Temperature","Humidity"]
        selected_poll = st.selectbox("Pollutant", poll_opts, label_visibility="collapsed")
        st.markdown("---")
        st.subheader("Appearance")
        dark_t = st.toggle("Dark Mode", value=st.session_state.dark_mode)
        if dark_t != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_t
            st.rerun()
        st.markdown("---")
        if st.button("Sign Out", use_container_width=True):
            logout()
            st.rerun()
    return uploaded, selected_poll, alert_thr

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR (User)
# ─────────────────────────────────────────────────────────────────────────────
def render_user_sidebar(C):
    with st.sidebar:
        st.header("AirAware")
        st.subheader("Air Quality Intelligence")
        user = current_user()
        st.write(f"Signed in as {user.get('name','User')}")
        st.write("Viewer")
        st.markdown("---")
        st.subheader("Load Dataset (View Only)")
        uploaded = st.file_uploader("Upload CSV to view", type=["csv"], label_visibility="collapsed")
        st.markdown("---")
        st.subheader("Pollutant View")
        poll_opts = ["AirQualityIndex","PM2.5","PM10","NO2(GT)","CO(GT)","Temperature","Humidity"]
        selected_poll = st.selectbox("Pollutant", poll_opts, label_visibility="collapsed")
        st.markdown("---")
        st.subheader("Alert Threshold")
        alert_thr = st.slider("AQI", 50, 300, 100, label_visibility="collapsed")
        from utils import classify_aqi
        t_label, t_color = classify_aqi(alert_thr)
        st.write(f"{alert_thr} - {t_label}")
        st.markdown("---")
        st.write("View Only Mode")
        st.markdown("---")
        st.subheader("Appearance")
        dark_t = st.toggle("Dark Mode", value=st.session_state.dark_mode)
        if dark_t != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_t
            st.rerun()
        st.markdown("---")
        if st.button("Sign Out", use_container_width=True):
            logout()
            st.rerun()
    return uploaded, selected_poll, alert_thr

# ─────────────────────────────────────────────────────────────────────────────
# TOPBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_topbar(C, role, n_records=None, target=None):
    role_badge = "Admin" if role == "admin" else "User"
    role_color = "#7c3aed" if role == "admin" else "#059669"
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
            <span class="role-badge" style="background:{role_color}18;color:{role_color}">{role_badge}</span>
            <span><span class="live-dot"></span>{meta}</span>
        </div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# NAVBAR (Admin)
# ─────────────────────────────────────────────────────────────────────────────
def render_admin_navbar(C):
    nav_options = ["Overview", "Forecast", "Retrain", "Alerts", "Analysis", "System"]
    cols = st.columns(len(nav_options))
    for i, opt in enumerate(nav_options):
        with cols[i]:
            if st.button(opt, key=f"nav_{opt}", use_container_width=True):
                st.session_state.admin_nav = opt
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────
def render_kpis(df, TARGET, C):
    aqi_mean = df[TARGET].mean()
    aqi_max  = df[TARGET].max()
    good_pct = (df[TARGET] <= 50).sum() / max(len(df),1) * 100
    alert_ct = (df[TARGET] > 100).sum()
    alert_pct= alert_ct / max(len(df),1) * 100
    cat, color = classify_aqi(aqi_mean)

    cards = [
        ("Mean AQI",    f"{aqi_mean:.1f}", cat, color),
        ("Peak AQI",    f"{aqi_max:.0f}",  "Maximum recorded", C["bad"]),
        ("Good Air",    f"{good_pct:.0f}%", f"{(df[TARGET]<=50).sum():,} clean records", C["good"]),
        ("Alerts",      f"{alert_ct:,}",   f"{alert_pct:.1f}% above 100", C["moderate"] if alert_pct < 20 else C["bad"]),
    ]
    for col_w, (label, val, sub, clr) in zip(st.columns(4), cards):
        with col_w:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-accent-bar" style="--sc:{clr}"></div>
                <div class="stat-label">{label}</div>
                <div class="stat-value" style="--sv:{clr}">{val}</div>
                <div class="stat-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# USER TABS (Simplified)
# ─────────────────────────────────────────────────────────────────────────────
def user_tab_overview(df, TARGET, selected_poll, C):
    if df is None or df.empty:
        st.error("No data available.")
        return

    import datetime
    live_time = datetime.datetime.now().strftime('%H:%M')

    aqi_val = df[TARGET].iloc[-1]
    label, color = classify_aqi(aqi_val)

    # Forecast next hour AQI
    forecast_aqi = aqi_val
    try:
        model, feats = load_model()
        if model and feats:
            last_row = df[feats].iloc[-1:].values
            forecast_aqi = model.predict(last_row)[0]
    except:
        pass
    forecast_label, forecast_color = classify_aqi(forecast_aqi)

    # Pollutant values
    pm25 = df.get('PM2.5', pd.Series([0])).iloc[-1] if 'PM2.5' in df.columns else 0
    pm10 = df.get('PM10', pd.Series([0])).iloc[-1] if 'PM10' in df.columns else 0
    no2 = df.get('NO2(GT)', pd.Series([0])).iloc[-1] if 'NO2(GT)' in df.columns else df.get('NO2', pd.Series([0])).iloc[-1] if 'NO2' in df.columns else 0
    co = df.get('CO(GT)', pd.Series([0])).iloc[-1] if 'CO(GT)' in df.columns else df.get('CO', pd.Series([0])).iloc[-1] if 'CO' in df.columns else 0

    # AQI levels
    levels = [
        (0, 50, 'Good', '#22c55e'),
        (51, 100, 'Satisfactory', '#86efac'),
        (101, 200, 'Moderate', '#facc15'),
        (201, 300, 'Poor', '#f97316'),
        (301, 400, 'Very Poor', '#ef4444'),
        (401, 500, 'Severe', '#7c3aed')
    ]

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;700&family=DM+Sans:wght@300;400;500&display=swap');
    .wrap {{padding:12px 0;max-width:680px;margin:0 auto}}
    .top-bar {{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;flex-wrap:wrap;gap:8px}}
    .top-bar h1 {{font-family:'Syne',sans-serif;font-size:24px;font-weight:700;color:{C['text']};letter-spacing:-0.3px}}
    .top-bar span {{font-size:11px;color:{C['text3']};background:{C['card']};padding:3px 8px;border-radius:20px;border:0.5px solid {C['border']}}}
    .region-bar {{display:flex;align-items:center;gap:10px;margin-bottom:16px;flex-wrap:wrap}}
    .region-bar label {{font-size:12px;color:{C['text3']};font-weight:500;white-space:nowrap}}
    .region-bar select {{flex:1;min-width:180px;font-family:'DM Sans',sans-serif;font-size:13px;padding:7px 12px;border:0.5px solid {C['border']};border-radius:8px;background:{C['card']};color:{C['text']};cursor:pointer}}
    .aqi-badge {{padding:5px 14px;border-radius:20px;font-size:12px;font-weight:500;white-space:nowrap;transition:all 0.3s}}
    .section-label {{font-size:14px;font-weight:500;color:{C['text3']};text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px}}
    .cat-grid {{display:grid;grid-template-columns:repeat(6,1fr);gap:6px;margin-bottom:16px}}
    .cat-item {{border-radius:8px;padding:8px 4px;text-align:center;border:0.5px solid {C['border']};background:{C['card']}}}
    .cat-item .cat-dot {{width:8px;height:8px;border-radius:50%;margin:0 auto 4px}}
    .cat-item .cat-range {{font-size:14px;color:{C['text3']};margin-bottom:2px}}
    .cat-item .cat-name {{font-size:14px;font-weight:500;color:{C['text']};line-height:1.2}}
    .metrics-grid {{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:16px}}
    .metric-card {{background:{C['card']};border-radius:8px;padding:12px 14px;border:0.5px solid {C['border']}}}
    .metric-card .m-label {{font-size:14px;color:{C['text3']};margin-bottom:4px}}
    .metric-card .m-val {{font-size:22px;font-weight:500;color:{C['text']};line-height:1}}
    .metric-card .m-unit {{font-size:14px;color:{C['text3']};margin-top:2px}}
    .next-hour {{background:{C['card']};border:0.5px solid {C['border']};border-radius:12px;padding:12px 14px;margin-bottom:16px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}}
    .next-hour .nh-label {{font-size:14px;color:{C['text3']};margin-bottom:2px}}
    .next-hour .nh-val {{font-size:20px;font-weight:500;color:{C['text']}}}
    .nh-bar {{flex:1;min-width:160px;height:8px;border-radius:4px;background:{C['card']};overflow:hidden}}
    .nh-fill {{height:100%;border-radius:4px;transition:width 0.8s ease}}
    .nh-arrow {{font-size:18px;color:{C['text3']}}}
    .stSelectbox label {{font-size:14px !important;}}
    .flashcard-row {{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:12px}}
    .flashcard {{background:{C['card']};border:0.5px solid {C['border']};border-radius:12px;overflow:hidden;min-height:320px;display:flex;flex-direction:column;transition:all 0.3s ease}}
    .flashcard:hover {{transform:translateY(-5px);box-shadow:0 10px 25px rgba(0,0,0,0.1);background:{C['bg']}}}
    .fc-head {{padding:10px 12px 0;display:flex;align-items:center;gap:6px}}
    .fc-num {{width:20px;height:20px;border-radius:50%;background:{C['card']};border:0.5px solid {C['border']};display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:500;color:{C['text3']};flex-shrink:0}}
    .fc-title {{font-size:11px;font-weight:500;color:{C['text3']};font-family:'Syne',sans-serif}}
    .fc-body {{flex:1;padding:10px 12px 12px;display:flex;flex-direction:column}}
    .pred-current {{margin-bottom:10px}}
    .pred-current .pc-num {{font-size:36px;font-weight:700;font-family:'Syne',sans-serif;line-height:1;transition:color 0.4s}}
    .pred-current .pc-cat {{font-size:14px;font-weight:500;margin-top:2px;transition:color 0.4s}}
    .meter-track {{height:10px;border-radius:5px;background:linear-gradient(to right,#22c55e 0%,#86efac 16%,#facc15 33%,#fb923c 50%,#ef4444 67%,#7c3aed 83%,#1e0a4a 100%);margin-bottom:6px;position:relative}}
    .meter-pointer {{position:absolute;top:-3px;width:16px;height:16px;background:{C['card']};border:2px solid {C['text']};border-radius:50%;transform:translateX(-50%);transition:left 0.8s ease}}
    .meter-labels {{display:flex;justify-content:space-between;font-size:14px;color:{C['text3']};margin-bottom:8px}}
    .pred-24h {{font-size:14px;color:{C['text3']};font-weight:500;margin-bottom:6px}}
    .pred-bars {{display:flex;flex-direction:column;gap:4px}}
    .pred-row {{display:flex;align-items:center;gap:6px}}
    .pred-row .pr-time {{font-size:14px;color:{C['text3']};width:28px;flex-shrink:0}}
    .pred-row .pr-bar-wrap {{flex:1;height:6px;background:{C['card']};border-radius:3px;overflow:hidden}}
    .pred-row .pr-bar {{height:100%;border-radius:3px;transition:width 0.6s ease}}
    .pred-row .pr-val {{font-size:14px;color:{C['text']};width:24px;text-align:right;flex-shrink:0}}
    .img-slider {{flex:1;display:flex;flex-direction:column;gap:6px}}
    .img-frame {{flex:1;border-radius:8px;overflow:hidden;min-height:140px;position:relative;background:{C['card']}}}
    .img-slide {{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;opacity:0;transition:opacity 0.6s ease}}
    .img-slide.active {{opacity:1}}
    .img-dots {{display:flex;gap:4px;justify-content:center}}
    .img-dot {{width:5px;height:5px;border-radius:50%;background:{C['border']};cursor:pointer;transition:background 0.3s}}
    .img-dot.active {{background:{C['text']}}}
    .img-caption {{font-size:10px;color:{C['text3']};line-height:1.4;text-align:center;padding:0 4px}}
    .carousel {{position:relative;overflow:hidden;flex:1;border-radius:8px;min-height:140px;background:{C['card']}}}
    .slide {{position:absolute;inset:0;opacity:0;transition:opacity 0.6s ease;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:10px}}
    .slide.active {{opacity:1}}
    .slide svg {{width:60px;height:60px;margin-bottom:8px}}
    .slide text {{font-size:8px;fill:{C['text']}}}
    .dots {{display:flex;gap:4px;justify-content:center;margin-top:6px}}
    .dot {{width:8px;height:8px;border-radius:50%;background:{C['border']};transition:background 0.3s}}
    .dot.active {{background:{C['text']}}}
    .caption {{font-size:14px;color:{C['text3']};line-height:1.4;text-align:center;padding:0 4px;margin-top:6px;position:relative;height:40px}}
    .slide-caption {{opacity:0;position:absolute;inset:0;transition:opacity 0.6s ease}}
    .slide-caption.active {{opacity:1}}
    .slide-caption:nth-child(1) {{animation:fade 15s infinite}}
    .slide-caption:nth-child(2) {{animation:fade 15s infinite 3s}}
    .slide-caption:nth-child(3) {{animation:fade 15s infinite 6s}}
    .slide-caption:nth-child(4) {{animation:fade 15s infinite 9s}}
    .slide-caption:nth-child(5) {{animation:fade 15s infinite 12s}}
    .slide:nth-child(1) {{animation:fade 15s infinite}}
    .slide:nth-child(2) {{animation:fade 15s infinite 3s}}
    .slide:nth-child(3) {{animation:fade 15s infinite 6s}}
    .slide:nth-child(4) {{animation:fade 15s infinite 9s}}
    .slide:nth-child(5) {{animation:fade 15s infinite 12s}}
    @keyframes fade {{0%,20% {{opacity:1}} 25%,100% {{opacity:0}}}}
    .gov-list {{display:flex;flex-direction:column;gap:8px}}
    .gov-title {{font-size:24px;font-weight:600;color:{C['text']};margin-bottom:8px;text-align:center}}
    .gov-item {{display:flex;gap:8px;align-items:flex-start}}
    .gov-icon {{width:22px;height:22px;border-radius:6px;background:{C['card']};border:0.5px solid {C['border']};display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px}}
    .gov-text .gt-title {{font-size:10px;font-weight:500;color:{C['text']};margin-bottom:1px}}
    .gov-text .gt-desc {{font-size:14px;color:{C['text3']};line-height:1.4}}
    </style>
    """, unsafe_allow_html=True)

    # Top bar
    st.markdown(f"""
    <div class="wrap">
      <div class="top-bar">
        <h1>Air Quality Dashboard</h1>
        <span id="live-time">Live {live_time}</span>
      </div>
    """, unsafe_allow_html=True)

    # Region selector (pollutant)
    selected_poll = st.selectbox("Select Pollutant View", ["AirQualityIndex", "PM2.5", "PM10", "NO2", "CO"], index=0, key="poll_select")
    st.markdown(f"""
      <div class="aqi-badge" style="background:{color}22;color:{color};border:0.5px solid {color}55">{label}</div>
    </div>
    """, unsafe_allow_html=True)

    # AQI Category Breakdown
    st.markdown('<div class="section-label">AQI Category Breakdown</div>', unsafe_allow_html=True)
    cat_html = '<div class="cat-grid">'
    for lo, hi, name, col in levels:
        cat_html += f'<div class="cat-item"><div class="cat-dot" style="background:{col}"></div><div class="cat-range">{lo}–{hi}</div><div class="cat-name">{name}</div></div>'
    cat_html += '</div>'
    st.markdown(cat_html, unsafe_allow_html=True)

    # Metrics
    st.markdown('<div class="section-label">Real-time Metrics</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metrics-grid">
      <div class="metric-card"><div class="m-label">AQI Index</div><div class="m-val" style="color:{color}">{aqi_val:.0f}</div><div class="m-unit">index</div></div>
      <div class="metric-card"><div class="m-label">PM2.5</div><div class="m-val">{pm25:.1f}</div><div class="m-unit">μg/m³</div></div>
      <div class="metric-card"><div class="m-label">PM10</div><div class="m-val">{pm10:.1f}</div><div class="m-unit">μg/m³</div></div>
      <div class="metric-card"><div class="m-label">NO₂</div><div class="m-val">{no2:.1f}</div><div class="m-unit">μg/m³</div></div>
      <div class="metric-card"><div class="m-label">CO</div><div class="m-val">{co:.2f}</div><div class="m-unit">mg/m³</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Next Hour Prediction
    st.markdown('<div class="section-label">Next Hour AQI Prediction</div>', unsafe_allow_html=True)
    fill_width = min(forecast_aqi / 500 * 100, 100)
    st.markdown(f"""
    <div class="next-hour">
      <div><div class="nh-label">Current</div><div class="nh-val" style="color:{color}">{aqi_val:.0f}</div></div>
      <div class="nh-arrow">→</div>
      <div style="flex:1;min-width:160px"><div class="nh-label" style="margin-bottom:4px">Predicted in 1 hour</div><div class="nh-bar"><div class="nh-fill" style="width:{fill_width}%;background:{forecast_color}"></div></div></div>
      <div><div class="nh-label">Predicted</div><div class="nh-val" style="color:{forecast_color}">{forecast_aqi:.0f}</div></div>
      <div class="aqi-badge" style="background:{forecast_color}22;color:{forecast_color};border:0.5px solid {forecast_color}55">{forecast_label}</div>
    </div>
    """, unsafe_allow_html=True)

    # Flashcards
    st.markdown('<div class="section-label">Awareness Flashcards</div>', unsafe_allow_html=True)
    st.markdown('<div class="flashcard-row">', unsafe_allow_html=True)

    # Flashcard 1: Predictive Meter
    pointer_left = min(aqi_val / 500 * 100, 100)
    pred_rows = []
    for i in range(8):
        time = ['1h','4h','8h','12h','16h','18h','20h','24h'][i]
        val = forecast_aqi + (i * 2)  # dummy progression
        width = min(val / 500 * 100, 100)
        pred_rows.append(f'<div class="pred-row"><span class="pr-time">{time}</span><div class="pr-bar-wrap"><div class="pr-bar" style="width:{width}%;background:{forecast_color}"></div></div><span class="pr-val">{val:.0f}</span></div>')
    pred_html = ''.join(pred_rows)
    st.markdown(f"""
    <div class="flashcard">
      <div class="fc-head"><div class="fc-num">1</div><div class="fc-title">Predictive Meter</div></div>
      <div class="fc-body">
        <div class="pred-current"><div class="pc-num" style="color:{color}">{aqi_val:.0f}</div><div class="pc-cat" style="color:{color}">{label}</div></div>
        <div class="meter-track"><div class="meter-pointer" style="left:{pointer_left}%"></div></div>
        <div class="meter-labels"><span>Good</span><span>Mod</span><span>Poor</span><span>Severe</span></div>
        <div class="pred-24h">24-hour forecast</div>
        <div class="pred-bars">{pred_html}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Flashcard 2: Pollution Effects
    st.markdown(f"""
    <div class="flashcard">
      <div class="fc-head"><div class="fc-num">2</div><div class="fc-title">Pollution Effects</div></div>
      <div class="fc-body">
        <div class="carousel">
          <div class="slide">
            <svg viewBox="0 0 100 100">
              <circle cx="50" cy="50" r="30" fill="#ccc"/>
              <text x="50" y="55" text-anchor="middle" font-size="8">Smog</text>
            </svg>
          </div>
          <div class="slide">
            <svg viewBox="0 0 100 100">
              <circle cx="50" cy="40" r="15" fill="#f0f"/>
              <text x="50" y="45" text-anchor="middle" font-size="6">Head</text>
              <ellipse cx="50" cy="70" rx="20" ry="15" fill="#f0f"/>
              <text x="50" y="75" text-anchor="middle" font-size="6">Lungs</text>
            </svg>
          </div>
          <div class="slide">
            <svg viewBox="0 0 100 100">
              <circle cx="50" cy="35" r="10" fill="#ff0"/>
              <text x="50" y="40" text-anchor="middle" font-size="6">Child</text>
              <rect x="40" y="45" width="20" height="30" fill="#ff0"/>
              <text x="50" y="60" text-anchor="middle" font-size="6">Body</text>
            </svg>
          </div>
          <div class="slide">
            <svg viewBox="0 0 100 100">
              <rect x="30" y="50" width="40" height="30" fill="#888"/>
              <text x="50" y="65" text-anchor="middle" font-size="6">Factory</text>
              <rect x="35" y="30" width="30" height="20" fill="#888"/>
              <text x="50" y="40" text-anchor="middle" font-size="6">Chimney</text>
              <circle cx="50" cy="25" r="5" fill="#000"/>
              <text x="50" y="28" text-anchor="middle" font-size="4">Smoke</text>
            </svg>
          </div>
          <div class="slide">
            <svg viewBox="0 0 100 100">
              <polygon points="50,80 40,60 60,60" fill="#f00"/>
              <text x="50" y="70" text-anchor="middle" font-size="6">Fire</text>
              <rect x="35" y="80" width="30" height="10" fill="#8B4513"/>
              <text x="50" y="87" text-anchor="middle" font-size="6">Stubble</text>
            </svg>
          </div>
        </div>
        <div class="dots">
          <span class="dot active"></span>
          <span class="dot"></span>
          <span class="dot"></span>
          <span class="dot"></span>
          <span class="dot"></span>
        </div>
        <div class="caption">
          <div class="slide-caption active">Smog: Heavy particulate matter from vehicle exhaust and industrial emissions creates visible smog, reducing visibility and causing respiratory irritation.</div>
          <div class="slide-caption">Health Impact: Air pollution causes respiratory diseases affecting millions daily, with long-term effects on cardiovascular health.</div>
          <div class="slide-caption">Children at Risk: Children exposed to toxic air face stunted lung growth and lifelong illness, impacting development and education.</div>
          <div class="slide-caption">Industrial Emissions: Factories release pollutants that contribute to PM2.5 and NO₂ levels, regulated but often exceeding safe limits.</div>
          <div class="slide-caption">Stubble Burning: Farmers burning crop residue releases harmful smoke, banned but still practiced, worsening air quality in rural areas.</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Flashcard 3: Govt Guidelines
    st.markdown("""
    <div class="flashcard">
      <div class="fc-head"><div class="fc-num">3</div><div class="fc-title">Govt. Guidelines</div></div>
      <div class="fc-body">
        <div class="gov-list">
          <div class="gov-title">Control of Air Pollution (Grant, Refusal or Cancellation of Consent) Guidelines, 2025</div>
          <div class="gov-item">
            <div class="gov-text"><div class="gt-desc">Use public transport or carpool — Vehicle exhaust is a primary source of urban PM2.5 and NO₂. Choosing buses, metro, or shared rides directly reduces tailpipe emissions regulated under BS-VI norms.</div></div>
          </div>
          <div class="gov-item">
            <div class="gov-text"><div class="gt-desc">Avoid burning waste, leaves, or crop residue — Open burning is prohibited under the Air Act and actively penalised by State Pollution Control Boards. Composting is the government-endorsed alternative.</div></div>
          </div>
          <div class="gov-item">
            <div class="gov-text"><div class="gt-desc">Monitor AQI daily via the SAMEER app — MoEFCC's official app (System of Air Quality and Weather Forecasting And Research) gives real-time AQI for your area so you can plan outdoor activity accordingly.</div></div>
          </div>
          <div class="gov-item">
            <div class="gov-text"><div class="gt-desc">Report industrial or vehicular violations — Citizens can lodge complaints with their State Pollution Control Board or CPCB against industries violating emission norms — a right backed by Section 18 of the Air Act.</div></div>
          </div>
          <div class="gov-item">
            <div class="gov-text"><div class="gt-desc">Switch to cleaner cooking and energy sources — Replace biomass and kerosene with LPG or electric options. CPCB identifies indoor combustion as a major contributor to household and ambient PM levels.</div></div>
          </div>
          <div class="gov-item">
            <div class="gov-text"><div class="gt-desc">Plant and protect trees in your locality — Green belt requirements are mandated for all industrial setups under this Act. Citizens can mirror this by supporting urban tree drives and protecting existing green cover, which absorbs particulate matter and CO₂.</div></div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

def user_tab_model_metrics(C):
    saved = load_metrics()
    if saved:
        st.markdown(f"""
        <div class="banner" style="--bc:{C['accent3']}">
            <div class="banner-title">Model Performance</div>
            <div class="banner-text">
                R2: {saved.get('r2','—')}  |  MAE: {saved.get('mae','—')}  |
                RMSE: {saved.get('rmse','—')}
            </div>
        </div>""", unsafe_allow_html=True)

        # Display AQI data visualization
        try:
            df = st.session_state.get("retrain_df")
            target = st.session_state.get("retrain_target")
            if df is not None and target in df.columns:
                st.markdown('<div class="cw"><div class="cw-title">AQI Data Visualization</div>', unsafe_allow_html=True)
                fig = chart_trend(df, target, C)
                st.pyplot(fig, use_container_width=True); plt.close()
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Do not display any message if no dataset is available
                pass
        except Exception as e:
            st.error(f"Unable to generate AQI visualization: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

def user_tab_alerts(df, TARGET, alert_thr, C):
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

def user_tab_analysis(df, TARGET, C):
    st.markdown('<div class="cw"><div class="cw-title">AQI Category Breakdown</div>', unsafe_allow_html=True)
    
    # Calculate actual distribution from data
    aqi_vals = df[TARGET]
    total_readings = len(aqi_vals)
    
    categories = [
        ("Good", 0, 50, "#22c55e"),
        ("Moderate", 51, 100, "#86efac"),
        ("Unhealthy for Sensitive", 101, 150, "#facc15"),
        ("Unhealthy", 151, 200, "#f97316"),
        ("Very Unhealthy", 201, 300, "#ef4444"),
        ("Hazardous", 301, 500, "#7c3aed")
    ]
    
    for label, min_val, max_val, color in categories:
        count = ((aqi_vals >= min_val) & (aqi_vals <= max_val)).sum()
        pct = (count / total_readings * 100) if total_readings > 0 else 0
        st.markdown(f"""
        <div class="cat-cell" style="--cbg:{color}12;--cborder:{color}30;--ccolor:{color}">
            <div class="cat-name">{label}</div>
            <div class="cat-pct">{pct:.1f}%</div>
            <div class="cat-count">{count} readings</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;margin-top:1rem;color:#666;font-size:0.9rem">Analysis of air quality distribution in the selected data range.</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ADMIN TABS (Full)
# ─────────────────────────────────────────────────────────────────────────────
def admin_tab_overview(df, TARGET, selected_poll, C):
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
            st.info(f"Column '{selected_poll}' not in dataset.")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="cw"><div class="cw-title">Pollutant Profile</div>', unsafe_allow_html=True)
        fig_r = chart_radar(df, C)
        if fig_r:
            st.pyplot(fig_r, use_container_width=True); plt.close()
        else:
            st.markdown('<p style="font-size:0.76rem;color:#94a3b8">Not enough pollutant columns.</p>',
                        unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="cw"><div class="cw-title">Rolling Averages</div>', unsafe_allow_html=True)
        fig_r2 = chart_rolling(df, TARGET, C)
        st.pyplot(fig_r2, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="cw"><div class="cw-title">Monthly Pattern</div>', unsafe_allow_html=True)
        fig_m = chart_monthly_bar(df, TARGET, C)
        st.pyplot(fig_m, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    if len(df) >= 24:
        st.markdown('<div class="cw"><div class="cw-title">AQI Heatmap — Hour vs Day</div>', unsafe_allow_html=True)
        fig_h = chart_heatmap(df, TARGET, C)
        st.pyplot(fig_h, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

def admin_tab_forecast(df, df_raw, TARGET, C):
    st.markdown('<div class="sec-lbl">XGBoost Forecast Engine</div>', unsafe_allow_html=True)

    # If no dataset uploaded, prompt to upload in this tab
    if df_raw is None:
        st.markdown('<div class="sec-lbl">Upload Dataset for Forecasting</div>', unsafe_allow_html=True)
        uploaded_forecast = st.file_uploader("Upload CSV for Forecasting", type=["csv"], key="forecast_upload")
        if uploaded_forecast:
            df_raw = load_csv(uploaded_forecast)
            TARGET = find_target(df_raw)
            st.session_state["admin_df"] = df_raw
            st.session_state["admin_target"] = TARGET
            st.success("Dataset loaded. Retrain the model if needed.")
            st.rerun()
        return

    saved = load_metrics()
    if saved:
        st.markdown(f"""
        <div class="banner" style="--bc:{C['accent3']}">
            <div class="banner-title">Existing Model — {saved.get('dataset_name','unknown dataset')}</div>
            <div class="banner-text">
                R2: {saved.get('r2','—')}  |  MAE: {saved.get('mae','—')}  |
                RMSE: {saved.get('rmse','—')}
            </div>
        </div>""", unsafe_allow_html=True)

    if st.button("Train Model on Current Dataset", key="btn_train"):
        with st.spinner("Engineering features and training..."):
            try:
                df_eng = engineer_features(df_raw.copy(), TARGET)
                model, X, X_test, y_test, preds, mae, rmse, r2 = train_model(df_eng, TARGET)
                st.session_state.update({
                    "model": model, "X": X,
                    "y_test": y_test, "preds": preds,
                    "metrics": (mae, rmse, r2),
                    "model_trained": True
                })
                joblib.dump(model, MODEL_FILE)
                joblib.dump(list(X.columns), FEATS_FILE)
                save_metrics(mae, rmse, r2, TARGET, "current dataset")
                st.success(f"Model trained — R2: {r2:.4f}  |  MAE: {mae:.3f}  |  RMSE: {rmse:.3f}")
            except Exception as e:
                st.error(f"Training failed: {e}")

    if not st.session_state.get("model_trained"):
        if saved and model_exists():
            st.info("A trained model exists from a previous session. Click Train to retrain, or proceed to view the saved metrics above.")
        else:
            st.markdown(f"""
            <div style="padding:1.2rem;background:{C['card']};border:1px solid {C['border']};
                 border-radius:10px;font-size:0.8rem;color:{C['text2']}">
                No model trained yet. Upload a dataset and click Train Model above.
            </div>""", unsafe_allow_html=True)
        return

    mae, rmse, r2 = st.session_state["metrics"]
    model  = st.session_state["model"]
    X      = st.session_state["X"]
    y_test = st.session_state["y_test"]
    preds  = st.session_state["preds"]

    r2_color = C["good"] if r2 > 0.9 else (C["moderate"] if r2 > 0.7 else C["bad"])
    for col_w, (lbl, val, desc, vc) in zip(st.columns(3), [
        ("R2 Score", f"{r2:.4f}", "1.0 = perfect fit", r2_color),
        ("MAE", f"{mae:.3f}", "Mean absolute error", C["accent"]),
        ("RMSE", f"{rmse:.3f}", "Root mean squared error", C["accent"]),
    ]):
        with col_w:
            st.markdown(f"""
            <div class="score-box">
                <div class="score-num" style="color:{vc}">{val}</div>
                <div class="score-key">{lbl}</div>
                <div class="score-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="cw"><div class="cw-title">Forecast Visualization</div>', unsafe_allow_html=True)
    fig_f = chart_forecast(y_test, preds, C)
    st.pyplot(fig_f, use_container_width=True); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    c_fi, c_imp = st.columns([2, 1])
    with c_fi:
        st.markdown('<div class="cw"><div class="cw-title">Feature Importance</div>', unsafe_allow_html=True)
        fig_imp = chart_importance(model, list(X.columns), C)
        st.pyplot(fig_imp, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)
    with c_imp:
        st.markdown(f'<div style="background:{C["card"]};border:1px solid {C["border"]};border-radius:12px;padding:1.2rem">', unsafe_allow_html=True)
        st.markdown('<div class="sec-lbl">Top Features</div>', unsafe_allow_html=True)
        imp_df = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
        imp_df = imp_df.sort_values("Importance", ascending=False)
        for _, row in imp_df.head(8).iterrows():
            pct = row["Importance"] / imp_df["Importance"].max() * 100
            st.markdown(f"""
            <div class="prog">
                <div class="prog-header">
                    <span>{row['Feature']}</span>
                    <span class="prog-val">{row['Importance']:.4f}</span>
                </div>
                <div class="prog-track">
                    <div class="prog-fill" style="width:{pct:.0f}%"></div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Next Prediction Section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="cw"><div class="cw-title">Next Hour AQI Prediction</div>', unsafe_allow_html=True)
    
    # Use session state for df_raw and TARGET
    df_raw = st.session_state.get("admin_df")
    TARGET = st.session_state.get("admin_target")
    if df_raw is not None and TARGET in df_raw.columns:
        try:
            next_pred = predict_next(df_raw, TARGET, model, list(X.columns))
            if next_pred is not None:
                pred_label, pred_color = classify_aqi(next_pred)
                st.markdown(f"""
                <div style="text-align:center;padding:1.5rem;background:{C['card']};border:1px solid {C['border']};border-radius:12px;">
                    <div style="font-size:2rem;font-weight:800;color:{pred_color}">{next_pred:.1f}</div>
                    <div style="font-size:1.1rem;font-weight:600;color:{pred_color}">{pred_label}</div>
                    <div style="font-size:0.9rem;color:{C['text2']};margin-top:0.5rem">Predicted AQI for next hour</div>
                </div>""", unsafe_allow_html=True)
                
                advice = aqi_advice(pred_label)
                st.markdown(f"""
                <div style="margin-top:1rem;padding:1rem;background:{pred_color}12;border:1px solid {pred_color}30;border-radius:8px;">
                    <div style="font-weight:600;color:{pred_color}">Health Advice:</div>
                    <div style="color:{C['text']};margin-top:0.3rem">{advice}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.error("Unable to generate prediction.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.info("Upload a dataset and train the model to see predictions.")
    
    st.markdown('</div>', unsafe_allow_html=True)

    if os.path.exists(MODEL_FILE):
        st.markdown("<br>", unsafe_allow_html=True)
        with open(MODEL_FILE, "rb") as f:
            st.download_button("Download Trained Model (.pkl)", f,
                               "aqi_xgboost_model.pkl", mime="application/octet-stream")

def admin_tab_retrain(C):
    st.markdown('<div class="sec-lbl">Retrain on New Dataset</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="banner" style="--bc:#7c3aed">
        <div class="banner-title">Admin Retraining Panel</div>
        <div class="banner-text">
            Upload any CSV dataset. The system auto-detects numeric columns.
            You can manually select the target column if needed.
        </div>
    </div>""", unsafe_allow_html=True)

    # Adjust file upload box styling based on theme
    box_style = f"background:{'#ffffff' if not st.session_state.dark_mode else '#000000'};color:{C['text']};border:1px solid {C['border']};padding:1rem;border-radius:8px"
    new_file = st.file_uploader("Upload New Training CSV", type=["csv"], key="retrain_upload", label_visibility="collapsed")

    if new_file:
        try:
            df_new = load_csv(new_file)
            new_file.seek(0)
            st.success(f"File loaded: {new_file.name}  ({len(df_new):,} rows, {len(df_new.columns)} columns)")

            with st.expander("Preview columns"):
                st.write(df_new.dtypes.rename("dtype").reset_index().rename(columns={"index":"Column"}))

            candidates = auto_detect_target(df_new)
            auto_t = find_target(df_new)
            default_idx = candidates.index(auto_t) if auto_t and auto_t in candidates else 0

            c_sel, c_info = st.columns([1, 2])
            with c_sel:
                chosen_target = st.selectbox(
                    "Select Target Column",
                    candidates, index=default_idx,
                    help="Choose the column to forecast (e.g. AQI, PM2.5, etc.)"
                )
            with c_info:
                if chosen_target in df_new.columns:
                    v = df_new[chosen_target].dropna()
                    st.markdown(f"""
                    <div class="sb-info" style="margin-top:1.5rem">
                        <div class="sb-info-title">Selected column preview</div>
                        <div class="sb-info-val">
                            Min: {v.min():.2f}  |  Max: {v.max():.2f}<br>
                            Mean: {v.mean():.2f}  |  Records: {len(v):,}
                        </div>
                    </div>""", unsafe_allow_html=True)

            dataset_label = new_file.name

            if st.button("Retrain Model on This Dataset", key="btn_retrain_new"):
                with st.spinner(f"Retraining on '{dataset_label}' using target '{chosen_target}'..."):
                    try:
                        df_eng = engineer_features(df_new.copy(), chosen_target)
                        model, X, X_test, y_test, preds, mae, rmse, r2 = train_model(df_eng, chosen_target)
                        joblib.dump(model, MODEL_FILE)
                        joblib.dump(list(X.columns), FEATS_FILE)
                        save_metrics(mae, rmse, r2, chosen_target, dataset_label)
                        st.session_state.update({
                            "model": model, "X": X,
                            "y_test": y_test, "preds": preds,
                            "metrics": (mae, rmse, r2),
                            "model_trained": True,
                            "retrain_done": True,
                            "retrain_df": df_new,
                            "retrain_target": chosen_target
                        })
                        st.success(f"Retrained on '{chosen_target}' — R2: {r2:.4f}  |  MAE: {mae:.3f}  |  RMSE: {rmse:.3f}")
                    except Exception as e:
                        st.error(f"Retraining failed: {e}")

        except Exception as e:
            st.error(f"Could not parse file: {e}")

    if st.session_state.get("retrain_done") and st.session_state.get("model_trained"):
        st.markdown("---")
        st.markdown('<div class="sec-lbl">Retrained Model Visualizations</div>', unsafe_allow_html=True)

        model  = st.session_state["model"]
        X      = st.session_state["X"]
        y_test = st.session_state["y_test"]
        preds  = st.session_state["preds"]
        mae, rmse, r2 = st.session_state["metrics"]
        C_curr = get_theme(st.session_state.dark_mode)

        r2_color = C_curr["good"] if r2 > 0.9 else (C_curr["moderate"] if r2 > 0.7 else C_curr["bad"])
        for col_w, (lbl, val, desc, vc) in zip(st.columns(3), [
            ("R2 Score", f"{r2:.4f}", "1.0 = perfect", r2_color),
            ("MAE", f"{mae:.3f}", "Mean absolute error", C_curr["accent"]),
            ("RMSE", f"{rmse:.3f}", "Root mean squared error", C_curr["accent"]),
        ]):
            with col_w:
                st.markdown(f"""
                <div class="score-box">
                    <div class="score-num" style="color:{vc}">{val}</div>
                    <div class="score-key">{lbl}</div>
                    <div class="score-desc">{desc}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="cw"><div class="cw-title">Retrained Forecast</div>', unsafe_allow_html=True)
        fig_f = chart_forecast(y_test, preds, C_curr)
        st.pyplot(fig_f, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="cw"><div class="cw-title">Feature Importance — New Model</div>', unsafe_allow_html=True)
        fig_imp = chart_importance(model, list(X.columns), C_curr)
        st.pyplot(fig_imp, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

        rt = st.session_state.get("retrain_target")
        rd = st.session_state.get("retrain_df")
        if rt and rd is not None and rt in rd.columns:
            st.markdown('<div class="cw"><div class="cw-title">AQI Distribution — New Dataset</div>', unsafe_allow_html=True)
            fig_dist = chart_distribution(rd, rt, C_curr)
            st.pyplot(fig_dist, use_container_width=True); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="cw"><div class="cw-title">Trend — New Dataset</div>', unsafe_allow_html=True)
            fig_tr = chart_trend(rd, rt, C_curr)
            st.pyplot(fig_tr, use_container_width=True); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

def admin_tab_alerts(df, TARGET, alert_thr, C):
    aqi_vals = df[TARGET]
    is_norm = aqi_vals.abs().max() < 20
    threshold = ((alert_thr - float(aqi_vals.mean())) / (float(aqi_vals.std()) or 1.0)
                 if is_norm else float(alert_thr))

    alert_df  = df[df[TARGET] > threshold]
    alert_pct = len(alert_df) / max(len(df),1) * 100

    if len(alert_df) == 0:
        bc, bt, bb = C["good"], "All Clear — No breaches", "All readings within threshold."
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
    fig_al = chart_alerts(df, TARGET, threshold, C)
    st.pyplot(fig_al, use_container_width=True); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    if len(alert_df) > 0:
        st.markdown('<div class="sec-lbl">Alert Log</div>', unsafe_allow_html=True)
        disp_cols = ["Datetime", TARGET] + [c for c in ["PM2.5","NO2(GT)","CO(GT)"] if c in alert_df.columns]
        disp = alert_df[disp_cols].copy().head(50)
        disp["Datetime"] = disp["Datetime"].dt.strftime("%Y-%m-%d %H:%M")
        disp[disp.columns[1:]] = disp[disp.columns[1:]].round(2)
        disp["Severity"] = disp[TARGET].apply(lambda v: classify_aqi(v)[0])
        st.dataframe(disp, use_container_width=True, hide_index=True)
        st.download_button("Export Alert Log", disp.to_csv(index=False).encode(),
                           "alert_log.csv", "text/csv")

def admin_tab_analysis(df, TARGET, C):
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
        fig_d = chart_distribution(df, TARGET, C)
        st.pyplot(fig_d, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-lbl">Statistical Summary</div>', unsafe_allow_html=True)
    if corr_cols:
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
    fig_h = chart_hourly(df, TARGET, C)
    st.pyplot(fig_h, use_container_width=True); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

def admin_tab_system(C):
    st.markdown('<div class="sec-lbl">System Status</div>', unsafe_allow_html=True)
    saved = load_metrics()

    info_items = [
        ("Model File", MODEL_FILE,
         f"{os.path.getsize(MODEL_FILE)/1024:.1f} KB" if os.path.exists(MODEL_FILE) else "Not found"),
        ("Feature File", FEATS_FILE,
         "Present" if os.path.exists(FEATS_FILE) else "Not found"),
        ("Model Status",
         "Trained" if st.session_state.get("model_trained") or model_exists() else "Not trained",
         f"R2: {saved.get('r2','—')}  |  MAE: {saved.get('mae','—')}  |  RMSE: {saved.get('rmse','—')}"
         if saved else "—"),
    ]
    for col_w, (lbl, name, val) in zip(st.columns(3), info_items):
        ok = "Not found" not in val and "Not trained" not in val
        clr = C["good"] if ok else C["text3"]
        with col_w:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-accent-bar" style="--sc:{clr}"></div>
                <div class="stat-label">{lbl}</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                     color:{C['text2']};margin:4px 0">{name}</div>
                <div style="font-size:0.7rem;color:{clr};font-weight:600">{val}</div>
            </div>""", unsafe_allow_html=True)

    if saved:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-lbl">Last Training Details</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sb-info" style="margin:0">
            <div class="sb-info-val">
                Dataset: {saved.get('dataset_name','—')}<br>
                Target column: {saved.get('target_col','—')}<br>
                R2: {saved.get('r2','—')}  |  MAE: {saved.get('mae','—')}  |  RMSE: {saved.get('rmse','—')}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-lbl">Registered Users</div>', unsafe_allow_html=True)
    from utils import load_users
    users = load_users()
    user_data = [{"Username": u, "Name": d.get("name",""), "Role": d.get("role","")}
                 for u, d in users.items()]
    st.dataframe(pd.DataFrame(user_data), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not st.session_state.get("role"):
        show_role_selection()
        return

    role = st.session_state.role

    if not is_logged_in():
        show_auth(role)
        return

    C = get_theme(st.session_state.dark_mode)
    inject_base_css(C, role)

    # Determine effective role for sidebar
    effective_role = "admin" if role == "admin" and not st.session_state.admin_view_user else "user"
    uploaded, selected_poll, alert_thr = render_sidebar(C, effective_role, is_combined=True)

    # Admin view toggle
    if role == "admin" and st.session_state.admin_view_user:
        # Show user interface for admin
        render_topbar(C, "user", None, None)

        df_raw = None
        TARGET = None
        if uploaded:
            df_raw = load_csv(uploaded)
            TARGET = find_target(df_raw)
            if TARGET is None:
                candidates = auto_detect_target(df_raw)
                if candidates:
                    TARGET = st.selectbox("Select target column:", candidates)
            render_topbar(C, "user", len(df_raw), TARGET)

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
            render_kpis(df, TARGET, C)
            st.markdown("<br>", unsafe_allow_html=True)

            tabs = st.tabs(["Overview", "Model Metrics", "Alerts", "Analysis"])
            with tabs[0]: user_tab_overview(df_raw, TARGET, selected_poll, C)
            with tabs[1]: user_tab_model_metrics(C)
            with tabs[2]: user_tab_alerts(df, TARGET, alert_thr, C)
            with tabs[3]: user_tab_analysis(df, TARGET, C)

        else:
            st.markdown(f"""
            <div class="banner" style="--bc:#059669">
                <div class="banner-title">No dataset loaded</div>
                <div class="banner-text">
                    Upload a CSV from the sidebar to view air quality data.
                </div>
            </div>""", unsafe_allow_html=True)

            saved = load_metrics()
            if saved:
                st.markdown('<div class="sec-lbl">Model Metrics</div>', unsafe_allow_html=True)
                for col_w, (lbl, val) in zip(st.columns(3), [
                    ("R2 Score", str(saved.get("r2","—"))),
                    ("MAE", str(saved.get("mae","—"))),
                    ("RMSE", str(saved.get("rmse","—"))),
                ]):
                    with col_w:
                        st.markdown(f"""
                        <div class="score-box">
                            <div class="score-num">{val}</div>
                            <div class="score-key">{lbl}</div>
                        </div>""", unsafe_allow_html=True)

            tabs = st.tabs(["Model Metrics"])
            with tabs[0]: user_tab_model_metrics(C)

    else:
        # Normal admin or user interface
        df_raw = None
        TARGET = None
        if uploaded:
            df_raw = load_csv(uploaded)
            TARGET = find_target(df_raw)
            if role == "admin":
                st.session_state["admin_df"] = df_raw
                st.session_state["admin_target"] = TARGET
            if TARGET is None:
                candidates = auto_detect_target(df_raw)
                if candidates:
                    TARGET = st.selectbox("Select target column:", candidates)
            render_topbar(C, role, len(df_raw), TARGET)
        else:
            render_topbar(C, role)

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
            render_kpis(df, TARGET, C)
            st.markdown("<br>", unsafe_allow_html=True)

            if role == "admin":
                render_admin_navbar(C)
                nav = st.session_state.get("admin_nav", "Overview")
                if nav == "Overview":
                    admin_tab_overview(df, TARGET, selected_poll, C)
                elif nav == "Forecast":
                    admin_tab_forecast(df, df_raw, TARGET, C)
                elif nav == "Retrain":
                    admin_tab_retrain(C)
                elif nav == "Alerts":
                    admin_tab_alerts(df, TARGET, alert_thr, C)
                elif nav == "Analysis":
                    admin_tab_analysis(df, TARGET, C)
                elif nav == "System":
                    admin_tab_system(C)
            else:
                tabs = st.tabs(["Overview", "Model Metrics", "Alerts"])
                with tabs[0]: user_tab_overview(df, TARGET, selected_poll, C)
                with tabs[1]: user_tab_model_metrics(C)
                with tabs[2]: user_tab_alerts(df, TARGET, alert_thr, C)

        else:
            # No dataset
            if role == "admin":
                st.markdown(f"""
                <div class="banner" style="--bc:#7c3aed">
                    <div class="banner-title">No dataset loaded</div>
                    <div class="banner-text">
                        Upload a CSV from the sidebar to view full dashboard.
                        The existing trained model is still accessible via the Retrain tab below.
                    </div>
                </div>""", unsafe_allow_html=True)

                saved = load_metrics()
                if saved:
                    st.markdown('<div class="sec-lbl">Existing Model Metrics</div>', unsafe_allow_html=True)
                    for col_w, (lbl, val) in zip(st.columns(3), [
                        ("R2 Score", str(saved.get("r2","—"))),
                        ("MAE", str(saved.get("mae","—"))),
                        ("RMSE", str(saved.get("rmse","—"))),
                    ]):
                        with col_w:
                            st.markdown(f"""
                            <div class="score-box">
                                <div class="score-num">{val}</div>
                                <div class="score-key">{lbl}</div>
                            </div>""", unsafe_allow_html=True)

                render_admin_navbar(C)
                nav = st.session_state.get("admin_nav", "Overview")
                if nav == "Overview":
                    st.info("No dataset loaded. Upload a CSV to view overview.")
                elif nav == "Forecast":
                    st.info("No dataset loaded. Upload a CSV to train and forecast.")
                elif nav == "Retrain":
                    admin_tab_retrain(C)
                elif nav == "Alerts":
                    st.info("No dataset loaded. Upload a CSV to view alerts.")
                elif nav == "Analysis":
                    st.info("No dataset loaded. Upload a CSV to view analysis.")
                elif nav == "System":
                    admin_tab_system(C)
            else:
                # Show full dashboard even without uploaded data - use demo data
                import joblib
                
                # Create demo data for display
                demo_data = {
                    'Datetime': pd.date_range('2024-01-01', periods=100, freq='h'),
                    'AirQualityIndex': np.random.normal(120, 30, 100).clip(0, 500),
                    'PM2.5': np.random.normal(60, 20, 100).clip(0, 200),
                    'PM10': np.random.normal(100, 30, 100).clip(0, 300),
                    'NO2': np.random.normal(40, 15, 100).clip(0, 100),
                    'CO': np.random.normal(1.5, 0.5, 100).clip(0, 5),
                }
                demo_df = pd.DataFrame(demo_data)
                demo_target = 'AirQualityIndex'
                
                st.markdown(f"""
                <div class="banner" style="--bc:#059669">
                    <div class="banner-title">Demo Dashboard</div>
                    <div class="banner-text">
                        Viewing sample air quality data. Upload a CSV from the sidebar to view your own data.
                    </div>
                </div>""", unsafe_allow_html=True)

                tabs = st.tabs(["Overview", "Model Metrics", "Alerts"])
                with tabs[0]: user_tab_overview(demo_df, demo_target, 'PM2.5', C)
                with tabs[1]: user_tab_model_metrics(C)
                with tabs[2]: 
                    # Demo alerts with sample data
                    st.info("Demo mode: Showing alerts for sample data.")
                    user_tab_alerts(demo_df, demo_target, 100, C)

    # Footer
    st.markdown(f"""
    <div class="footer" style="padding:10px 0;text-align:center;color:{C['text3']};font-size:0.85rem">
        <p>Forecasting Air Quality | Powered by AirAware</p>
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
