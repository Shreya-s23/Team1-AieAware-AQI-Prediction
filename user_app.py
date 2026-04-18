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

from utils import (
    authenticate, register_user, is_logged_in, current_user, logout,
    load_csv, find_target, auto_detect_target,
    model_exists, load_model, load_metrics,
    classify_aqi, aqi_advice, AQI_LEVELS,
)
from theme import get_theme, inject_base_css
from charts import (
    chart_trend, chart_rolling, chart_monthly_bar,
    chart_heatmap, chart_radar, chart_forecast,
    chart_importance, chart_correlation,
    chart_distribution, chart_alerts, chart_hourly
)

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
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar(C, df=None):
    with st.sidebar:
        user = current_user()
        st.markdown(f"""
        <div class="sb-brand">
            <div class="sb-brand-name">Air<span class="accent">Aware</span></div>
            <div class="sb-brand-sub">Air Quality Intelligence</div>
        </div>""", unsafe_allow_html=True)

        # User info
        st.markdown(f"""
        <div class="sb-info" style="margin-top:0.8rem">
            <div class="sb-info-title">Signed in as</div>
            <div class="sb-info-val">
                {user.get('name','User')}<br>
                <span style="color:{C['accent']};font-size:0.65rem">Viewer</span>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)

        # Dataset upload (VIEW ONLY — user can upload their own data to view)
        st.markdown('<div class="sb-sec"><span class="sb-sec-lbl">Load Dataset (View Only)</span></div>',
                    unsafe_allow_html=True)
        with st.container():
            st.markdown('<div style="padding:0 1.3rem">', unsafe_allow_html=True)
            uploaded = st.file_uploader("Upload CSV to view", type=["csv"],
                                        label_visibility="collapsed",
                                        help="You can upload data to view charts. You cannot retrain the model.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)

        # Pollutant selector
        st.markdown('<div class="sb-sec"><span class="sb-sec-lbl">Pollutant View</span></div>',
                    unsafe_allow_html=True)
        with st.container():
            st.markdown('<div style="padding:0 1.3rem">', unsafe_allow_html=True)
            poll_opts = ["AirQualityIndex","PM2.5","PM10","NO2(GT)","CO(GT)","Temperature","Humidity"]
            selected_poll = st.selectbox("Pollutant", poll_opts, label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)

        # Alert threshold (view-only, just for display)
        st.markdown('<div class="sb-sec"><span class="sb-sec-lbl">Alert Threshold</span></div>',
                    unsafe_allow_html=True)
        with st.container():
            st.markdown('<div style="padding:0 1.3rem">', unsafe_allow_html=True)
            alert_thr = st.slider("AQI", 50, 300, 100, 10, label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)

        t_label, t_color = classify_aqi(alert_thr)
        st.markdown(f"""
        <div style="margin:0.4rem 1.3rem;background:{t_color}12;border:1px solid {t_color}30;
             border-radius:8px;padding:0.7rem;text-align:center">
            <div style="font-family:'Sora',sans-serif;font-size:1.2rem;
                 font-weight:800;color:{t_color}">{alert_thr}</div>
            <div style="font-size:0.6rem;font-weight:700;color:{t_color};
                 text-transform:uppercase;letter-spacing:0.07em">{t_label}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)

        # Read-only notice
        st.markdown(f"""
        <div style="margin:0.5rem 1.3rem">
            <span class="readonly-badge">View Only Mode</span>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)

        # Dark mode
        st.markdown('<div class="sb-sec"><span class="sb-sec-lbl">Appearance</span></div>',
                    unsafe_allow_html=True)
        with st.container():
            st.markdown('<div style="padding:0 1.3rem">', unsafe_allow_html=True)
            dark_t = st.toggle("Dark Mode", value=st.session_state.dark_mode)
            if dark_t != st.session_state.dark_mode:
                st.session_state.dark_mode = dark_t
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div style="padding:0.6rem 1.3rem">', unsafe_allow_html=True)
            if st.button("Sign Out", use_container_width=True):
                logout()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    return uploaded, selected_poll, alert_thr


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


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not is_logged_in():
        show_auth()
        return

    C = get_theme(st.session_state.dark_mode)
    inject_base_css(C, "user")

    uploaded, selected_poll, alert_thr = render_sidebar(C)

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
        render_kpis(df, TARGET, C)
        st.markdown("<br>", unsafe_allow_html=True)

        tabs = st.tabs(["Overview", "Model Metrics", "Alerts", "Analysis"])
        with tabs[0]: tab_overview(df, TARGET, selected_poll, C)
        with tabs[1]: tab_model(C)
        with tabs[2]: tab_alerts_user(df, TARGET, alert_thr, C)
        with tabs[3]: tab_analysis_user(df, TARGET, C)

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