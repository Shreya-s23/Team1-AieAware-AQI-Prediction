import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AirAware — Air Quality Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={}
)

# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for key, default in [
    ("dark_mode", False),
    ("model_trained", False),
    ("active_section", "overview"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────────────────────
#  THEME
# ─────────────────────────────────────────────────────────────────────────────
def theme():
    if st.session_state.dark_mode:
        return dict(
            bg="#1a1a2e",
            surface="#16213e",
            card="#0f3460",
            card2="#1a2a4a",
            border="#2a3f6f",
            border2="#3a5080",
            accent="#e94560",
            accent2="#c73652",
            accent3="#0f9b8e",
            text="#e8eaf6",
            text2="#9fa8da",
            text3="#5c6bc0",
            good="#00c896",
            moderate="#ffa726",
            bad="#ef5350",
            hazard="#ab47bc",
            plot_bg="#16213e",
            mpl_bg="#16213e",
            is_dark=True,
        )
    else:
        return dict(
            bg="#f5f7fa",
            surface="#ffffff",
            card="#ffffff",
            card2="#f0f4ff",
            border="#e2e8f0",
            border2="#cbd5e1",
            accent="#2563eb",
            accent2="#1d4ed8",
            accent3="#059669",
            text="#1e293b",
            text2="#475569",
            text3="#94a3b8",
            good="#059669",
            moderate="#d97706",
            bad="#dc2626",
            hazard="#7c3aed",
            plot_bg="#ffffff",
            mpl_bg="#ffffff",
            is_dark=False,
        )

# ─────────────────────────────────────────────────────────────────────────────
#  CSS INJECTION
# ─────────────────────────────────────────────────────────────────────────────
def inject_css(C):
    dark = C['is_dark']
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif !important;
    background: {C['bg']} !important;
    color: {C['text']} !important;
}}

.stApp {{
    background: {C['bg']} !important;
}}

/* Hide default menu/deploy */
#MainMenu {{ visibility: hidden !important; display: none !important; }}
footer {{ visibility: hidden !important; }}
header {{ visibility: hidden !important; }}
[data-testid="stToolbar"] {{ display: none !important; }}
button[kind="header"] {{ display: none !important; }}
.stDeployButton {{ display: none !important; }}
[data-testid="stDecoration"] {{ display: none !important; }}

/* ── SIDEBAR ─────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background: {C['surface']} !important;
    border-right: 1px solid {C['border']} !important;
    width: 280px !important;
}}
section[data-testid="stSidebar"] > div {{
    padding: 0 !important;
    background: {C['surface']} !important;
}}
section[data-testid="stSidebar"] * {{
    color: {C['text']} !important;
}}
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {{
    margin-top: 4px;
}}
section[data-testid="stSidebar"] label {{
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: {C['text2']} !important;
}}
section[data-testid="stSidebar"] .stSelectbox > div > div {{
    background: {C['card2']} !important;
    border-color: {C['border']} !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
}}
section[data-testid="stSidebar"] .stFileUploader {{
    background: {C['card2']} !important;
    border: 1px dashed {C['border2']} !important;
    border-radius: 10px !important;
    padding: 8px !important;
}}

/* ── MAIN CONTENT ────────────────────────────── */
.block-container {{
    padding: 0 2rem 3rem 2rem !important;
    max-width: 100% !important;
}}

/* ── TOP BAR ─────────────────────────────────── */
.topbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.5rem 0 1.5rem 0;
    border-bottom: 1px solid {C['border']};
    margin-bottom: 2rem;
}}
.topbar-brand {{
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: {C['text']};
    letter-spacing: -0.02em;
}}
.topbar-brand span {{
    color: {C['accent']};
}}
.topbar-meta {{
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: {C['text3']};
    display: flex;
    align-items: center;
    gap: 20px;
}}
.live-indicator {{
    display: flex;
    align-items: center;
    gap: 6px;
}}
.live-dot {{
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: {C['good']};
    animation: blink 2.5s ease-in-out infinite;
}}
@keyframes blink {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.3; }}
}}

/* ── STAT CARDS ──────────────────────────────── */
.stat-card {{
    background: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    transition: box-shadow 0.2s;
}}
.stat-card:hover {{
    box-shadow: 0 4px 20px {'rgba(0,0,0,0.2)' if dark else 'rgba(0,0,0,0.08)'};
}}
.stat-card-accent {{
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 12px 12px 0 0;
    background: var(--accent-color, {C['accent']});
}}
.stat-label {{
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {C['text3']};
    margin-bottom: 0.5rem;
}}
.stat-value {{
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: var(--stat-color, {C['text']});
    letter-spacing: -0.03em;
    line-height: 1;
    margin-bottom: 0.4rem;
}}
.stat-sub {{
    font-size: 0.72rem;
    color: {C['text3']};
    font-weight: 400;
}}

/* ── SECTION LABEL ───────────────────────────── */
.sec-label {{
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: {C['text3']};
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid {C['border']};
}}

/* ── CHART CARD ──────────────────────────────── */
.chart-card {{
    background: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1.2rem;
}}
.chart-card-title {{
    font-size: 0.78rem;
    font-weight: 600;
    color: {C['text2']};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}}
.chart-card-sub {{
    font-size: 0.7rem;
    color: {C['text3']};
    margin-bottom: 1rem;
}}

/* ── STATUS PILL ─────────────────────────────── */
.status-pill {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--pill-bg);
    color: var(--pill-color);
    border: 1px solid var(--pill-border);
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    margin-bottom: 1.5rem;
}}
.status-dot {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--pill-color);
}}

/* ── AQI BANNER ──────────────────────────────── */
.aqi-banner {{
    background: {C['card']};
    border: 1px solid {C['border']};
    border-left: 4px solid var(--banner-color);
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.4rem;
    margin-bottom: 1.5rem;
}}
.aqi-banner-title {{
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--banner-color);
    margin-bottom: 0.25rem;
}}
.aqi-banner-text {{
    font-size: 0.78rem;
    color: {C['text2']};
}}

/* ── METRICS ROW ─────────────────────────────── */
.metric-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid {C['border']};
    font-size: 0.78rem;
}}
.metric-row:last-child {{ border-bottom: none; }}
.metric-key {{ color: {C['text2']}; }}
.metric-val {{
    font-family: 'DM Mono', monospace;
    font-weight: 500;
    color: {C['text']};
}}

/* ── SCORE BOX ───────────────────────────────── */
.score-box {{
    background: {C['card2']};
    border: 1px solid {C['border']};
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
}}
.score-num {{
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: {C['accent']};
    letter-spacing: -0.03em;
}}
.score-key {{
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {C['text3']};
    font-weight: 600;
    margin-top: 4px;
}}
.score-desc {{
    font-size: 0.68rem;
    color: {C['text3']};
    margin-top: 4px;
}}

/* ── SIDEBAR INNER ───────────────────────────── */
.sb-brand {{
    padding: 1.6rem 1.4rem 1.2rem;
    border-bottom: 1px solid {C['border']};
}}
.sb-brand-name {{
    font-family: 'DM Serif Display', serif;
    font-size: 1.2rem;
    color: {C['text']};
    letter-spacing: -0.02em;
}}
.sb-brand-name span {{ color: {C['accent']}; }}
.sb-brand-sub {{
    font-size: 0.62rem;
    color: {C['text3']};
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 600;
    margin-top: 2px;
}}
.sb-section {{
    padding: 1rem 1.4rem 0.5rem;
}}
.sb-section-label {{
    font-size: 0.6rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: {C['text3']};
    display: block;
    margin-bottom: 0.6rem;
}}
.sb-divider {{
    height: 1px;
    background: {C['border']};
    margin: 0.6rem 1.4rem;
}}
.sb-status {{
    margin: 0.8rem 1.4rem;
    background: {C['card2']};
    border: 1px solid {C['border']};
    border-radius: 8px;
    padding: 0.8rem;
    display: flex;
    align-items: center;
    gap: 10px;
}}
.sb-status-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: {C['good']};
    flex-shrink: 0;
    animation: blink 2.5s infinite;
}}
.sb-status-text {{
    font-size: 0.72rem;
    color: {C['text2']};
    line-height: 1.4;
}}
.sb-aqi-preview {{
    margin: 0.6rem 1.4rem;
    background: {C['card2']};
    border: 1px solid {C['border']};
    border-radius: 8px;
    padding: 0.8rem;
    text-align: center;
}}
.sb-aqi-val {{
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: var(--t, {C['accent']});
    letter-spacing: -0.03em;
}}
.sb-aqi-label {{
    font-size: 0.62rem;
    font-weight: 600;
    color: var(--t, {C['text3']});
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}

/* ── BUTTONS ─────────────────────────────────── */
.stButton > button {{
    background: {C['accent']} !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.15s ease !important;
    letter-spacing: 0.01em !important;
}}
.stButton > button:hover {{
    background: {C['accent2']} !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px {'rgba(233,69,96,0.3)' if dark else 'rgba(37,99,235,0.3)'} !important;
}}

/* ── STREAMLIT OVERRIDES ─────────────────────── */
.stSelectbox label, .stSlider label, .stDateInput label, .stFileUploader label,
.stToggle label, .stMultiSelect label, .stNumberInput label {{
    color: {C['text2']} !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}}
div[data-testid="stMarkdownContainer"] p {{ color: {C['text']} !important; }}
.stSpinner > div {{ border-top-color: {C['accent']} !important; }}
.stSuccess {{ border-radius: 8px !important; }}
.stError {{ border-radius: 8px !important; }}
.stInfo {{ border-radius: 8px !important; }}
.stWarning {{ border-radius: 8px !important; }}

[data-testid="stMetricValue"] {{
    font-family: 'DM Serif Display', serif !important;
    font-weight: 400 !important;
    color: {C['accent']} !important;
}}

/* ── TABS ─────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px !important;
    background: {C['card']} !important;
    border-radius: 10px !important;
    padding: 4px !important;
    border: 1px solid {C['border']} !important;
    margin-bottom: 1.5rem !important;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 7px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    color: {C['text2']} !important;
    padding: 8px 18px !important;
}}
.stTabs [aria-selected="true"] {{
    background: {C['accent']} !important;
    color: white !important;
    font-weight: 600 !important;
}}

/* ── DATAFRAME ───────────────────────────────── */
.stDataFrame {{
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid {C['border']} !important;
}}

/* ── WELCOME CARD ────────────────────────────── */
.welcome-card {{
    max-width: 560px;
    margin: 3rem auto;
    background: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 16px;
    padding: 2.5rem;
}}
.welcome-title {{
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: {C['text']};
    letter-spacing: -0.03em;
    margin-bottom: 0.5rem;
}}
.welcome-title span {{ color: {C['accent']}; }}
.welcome-sub {{
    font-size: 0.85rem;
    color: {C['text2']};
    line-height: 1.7;
    margin-bottom: 1.5rem;
}}

/* ── SCHEMA TABLE ────────────────────────────── */
.schema-tbl {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.76rem;
}}
.schema-tbl th {{
    background: {C['card2']};
    color: {C['text2']};
    padding: 7px 10px;
    text-align: left;
    font-weight: 600;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
.schema-tbl td {{
    padding: 7px 10px;
    border-bottom: 1px solid {C['border']};
    color: {C['text']};
}}
.schema-tbl tr:last-child td {{ border-bottom: none; }}
.req {{ color: {C['good']}; font-weight: 700; }}
.opt {{ color: {C['text3']}; }}

/* ── AQI CATEGORY GRID ───────────────────────── */
.cat-cell {{
    text-align: center;
    padding: 0.8rem 0.4rem;
    background: var(--cat-bg);
    border: 1px solid var(--cat-border);
    border-radius: 10px;
}}
.cat-name {{
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--cat-color);
    margin-bottom: 4px;
    line-height: 1.3;
}}
.cat-pct {{
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: var(--cat-color);
    letter-spacing: -0.03em;
}}
.cat-count {{
    font-size: 0.62rem;
    color: {C['text3']};
    margin-top: 3px;
    font-family: 'DM Mono', monospace;
}}

/* ── PROGRESS BAR ────────────────────────────── */
.prog-wrap {{
    margin-bottom: 0.8rem;
}}
.prog-label {{
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    margin-bottom: 3px;
    color: {C['text2']};
}}
.prog-val {{
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: {C['accent']};
}}
.prog-track {{
    background: {C['border']};
    border-radius: 3px;
    height: 5px;
    overflow: hidden;
}}
.prog-fill {{
    height: 100%;
    border-radius: 3px;
    background: {C['accent']};
    transition: width 0.4s ease;
}}

/* ── ADMIN CARD ──────────────────────────────── */
.admin-card {{
    background: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1rem;
}}
.admin-card-title {{
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {C['text3']};
    margin-bottom: 1rem;
}}

/* ── FOOTER ──────────────────────────────────── */
.footer {{
    border-top: 1px solid {C['border']};
    margin-top: 3rem;
    padding-top: 1.2rem;
    text-align: center;
}}
.footer p {{
    font-size: 0.68rem;
    color: {C['text3']};
    margin-bottom: 0.2rem;
    line-height: 1.6;
}}
</style>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  AQI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
AQI_LEVELS = [
    (0,   50,  "Good",                           "#059669"),
    (51,  100, "Moderate",                       "#d97706"),
    (101, 150, "Unhealthy — Sensitive",          "#ea580c"),
    (151, 200, "Unhealthy",                      "#dc2626"),
    (201, 300, "Very Unhealthy",                 "#7c3aed"),
    (301, 999, "Hazardous",                      "#9f1239"),
]

def classify_aqi(val):
    val = float(max(0, val))
    for lo, hi, label, color in AQI_LEVELS:
        if lo <= val <= hi:
            return label, color
    return "Hazardous", "#9f1239"

def aqi_advice(label):
    d = {
        "Good": "Air quality is satisfactory. Outdoor activities are safe for all.",
        "Moderate": "Acceptable quality. Unusually sensitive individuals may notice mild effects.",
        "Unhealthy — Sensitive": "Children, elderly and those with respiratory conditions should reduce prolonged outdoor exertion.",
        "Unhealthy": "Everyone may begin to experience adverse health effects. Limit outdoor activity.",
        "Very Unhealthy": "Health alert — everyone should avoid prolonged outdoor exertion.",
        "Hazardous": "Emergency conditions. The entire population is likely to be affected. Stay indoors.",
    }
    return d.get(label, "Monitor air quality conditions.")


# ─────────────────────────────────────────────────────────────────────────────
#  MATPLOTLIB STYLE
# ─────────────────────────────────────────────────────────────────────────────
def set_mpl(C):
    plt.rcParams.update({
        "figure.facecolor":  C["mpl_bg"],
        "axes.facecolor":    C["mpl_bg"],
        "axes.edgecolor":    C["border"],
        "axes.labelcolor":   C["text2"],
        "axes.titlecolor":   C["text"],
        "axes.titlesize":    9.5,
        "axes.labelsize":    8,
        "xtick.color":       C["text3"],
        "ytick.color":       C["text3"],
        "xtick.labelsize":   7.5,
        "ytick.labelsize":   7.5,
        "text.color":        C["text"],
        "grid.color":        C["border"],
        "grid.linestyle":    "--",
        "grid.alpha":        0.5,
        "legend.facecolor":  C["mpl_bg"],
        "legend.edgecolor":  C["border"],
        "legend.labelcolor": C["text"],
        "legend.fontsize":   7.5,
        "font.family":       "DejaVu Sans",
        "font.size":         8.5,
    })

# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_csv(file):
    df = pd.read_csv(file)
    if "Date" in df.columns and "Time" in df.columns:
        df["Datetime"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce")
    elif "date" in df.columns:
        df["Datetime"] = pd.to_datetime(df["date"], errors="coerce")
    elif "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    elif "datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        # Try first column
        df["Datetime"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    df = df.dropna(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)
    return df

def find_target(df):
    # Check for known column names
    for col in ["AirQualityIndex", "aqi_value", "AQI", "aqi", "air_quality_index"]:
        if col in df.columns:
            return col
    # Auto-detect: find numeric column that could be AQI (values 0-500 range)
    for col in df.select_dtypes(include=np.number).columns:
        vals = df[col].dropna()
        if len(vals) > 0 and 0 <= vals.median() <= 500 and col not in ["Date", "Time", "Hour", "DayOfWeek", "Month"]:
            return col
    return None

def auto_detect_target(df):
    """Returns a list of candidate target columns"""
    candidates = []
    known = ["AirQualityIndex", "aqi_value", "AQI", "aqi", "air_quality_index"]
    for col in known:
        if col in df.columns:
            candidates.append(col)
    # Also add numeric columns that look like AQI
    for col in df.select_dtypes(include=np.number).columns:
        if col not in candidates and col not in ["Date","Time","Hour","DayOfWeek","Month","year","month","day"]:
            vals = df[col].dropna()
            if len(vals) > 0 and vals.min() >= 0:
                candidates.append(col)
    return candidates

# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df, TARGET):
    df = df.copy()
    # Keep all numeric columns (flexible for different datasets)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    keep = [c for c in numeric_cols] + ["Datetime"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.ffill().bfill()
    for col in df.select_dtypes(include=np.number).columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:
            df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    df["month"]     = df["Datetime"].dt.month
    df["quarter"]   = df["Datetime"].dt.quarter
    df["day_of_week"] = df["Datetime"].dt.dayofweek
    df["hour"]      = df["Datetime"].dt.hour
    df["day_of_year"] = df["Datetime"].dt.dayofyear
    if TARGET in df.columns:
        df["target_lag_1"]       = df[TARGET].shift(1)
        df["target_lag_2"]       = df[TARGET].shift(2)
        df["target_lag_24"]      = df[TARGET].shift(24)
        df["target_roll_mean_3"] = df[TARGET].rolling(3).mean()
        df["target_roll_mean_24"]= df[TARGET].rolling(24).mean()
        df["target_roll_std_3"]  = df[TARGET].rolling(3).std()
    df = df.fillna(df.median(numeric_only=True))
    return df

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train_model(df, TARGET):
    drop_cols = [TARGET, "Datetime", "Date", "Time"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.select_dtypes(include=np.number)
    y = df[TARGET]
    # Remove rows with NaN in target
    mask = y.notna()
    X, y = X[mask], y[mask]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(
        n_estimators=400, learning_rate=0.06, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        early_stopping_rounds=20, random_state=42, verbosity=0
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    return model, X, X_test, y_test, preds, mae, rmse, r2


# ─────────────────────────────────────────────────────────────────────────────
#  CHART FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def chart_trend(df, col, C, title=""):
    set_mpl(C)
    fig, ax = plt.subplots(figsize=(10, 3))
    x, y = df["Datetime"], df[col]
    ax.fill_between(x, y, alpha=0.1, color=C["accent"])
    ax.plot(x, y, color=C["accent"], lw=1.5, alpha=0.9)
    if len(y) > 12:
        ma = y.rolling(12, center=True).mean()
        ax.plot(x, ma, color=C["accent3"], lw=2, alpha=0.7, linestyle="--", label="12-pt avg")
        ax.legend(framealpha=0.4, loc="upper right")
    ax.set_title(title or col, pad=8, fontweight='semibold')
    ax.set_ylabel(col, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(axis='x', rotation=20)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.8)
    return fig

def chart_aqi_heatmap(df, TARGET, C):
    set_mpl(C)
    df2 = df.copy()
    df2["hour"] = df2["Datetime"].dt.hour
    df2["day"]  = df2["Datetime"].dt.day_name()
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = df2.pivot_table(index="day", columns="hour", values=TARGET, aggfunc="mean")
    pivot = pivot.reindex([d for d in day_order if d in pivot.index])
    fig, ax = plt.subplots(figsize=(12, 3))
    cmap = LinearSegmentedColormap.from_list("aqi",
        ["#059669","#d97706","#dc2626","#7c3aed"])
    sns.heatmap(pivot, ax=ax, cmap=cmap, annot=False,
                linewidths=0.3, linecolor=C["bg"],
                cbar_kws={"shrink": 0.7, "label": "Avg AQI"})
    ax.set_title("AQI by Hour and Day of Week", pad=8, fontweight='semibold')
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("")
    plt.tight_layout(pad=0.5)
    return fig

def chart_monthly_bar(df, TARGET, C):
    set_mpl(C)
    df2 = df.copy()
    df2["month"] = df2["Datetime"].dt.month
    monthly = df2.groupby("month")[TARGET].mean()
    labels  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    mlabels = [labels[m-1] for m in monthly.index]
    q33 = np.percentile(monthly.values, 33)
    q66 = np.percentile(monthly.values, 66)
    colors = [C["good"] if v < q33 else (C["moderate"] if v < q66 else C["bad"]) for v in monthly.values]
    fig, ax = plt.subplots(figsize=(9, 3))
    bars = ax.bar(mlabels, monthly.values, color=colors, width=0.55, alpha=0.85, edgecolor='none')
    for bar, val in zip(bars, monthly.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.0f}", ha='center', va='bottom', fontsize=7.5, color=C["text2"])
    ax.set_title("Monthly Average AQI", pad=8, fontweight='semibold')
    ax.spines[["top","right","left"]].set_visible(False)
    ax.grid(True, axis='y', alpha=0.3)
    ax.tick_params(left=False)
    plt.tight_layout(pad=0.5)
    return fig

def chart_radar(df, C):
    cols = [c for c in ["PM2.5","PM10","NO2(GT)","CO(GT)","Temperature","Humidity"] if c in df.columns]
    if len(cols) < 3:
        return None
    set_mpl(C)
    means = df[cols].mean()
    maxes = df[cols].max().replace(0, 1)
    vals  = (means / maxes).values.tolist()
    N = len(cols)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    vals += vals[:1]
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.plot(angles, vals, color=C["accent"], lw=2)
    ax.fill(angles, vals, color=C["accent"], alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cols, fontsize=7.5, color=C["text2"])
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%","50%","75%","100%"], fontsize=6, color=C["text3"])
    ax.grid(color=C["border"], linestyle='--', alpha=0.4)
    ax.spines['polar'].set_color(C["border"])
    ax.set_title("Pollutant Profile", fontsize=8.5, color=C["text"], pad=12, fontweight='semibold')
    plt.tight_layout()
    return fig

def chart_rolling(df, TARGET, C):
    set_mpl(C)
    fig, ax = plt.subplots(figsize=(10, 3))
    roll7  = df[TARGET].rolling(7).mean()
    roll24 = df[TARGET].rolling(24).mean()
    ax.fill_between(df["Datetime"], df[TARGET], alpha=0.05, color=C["accent"])
    ax.plot(df["Datetime"], df[TARGET], color=C["text3"], lw=0.6, alpha=0.5, label="Raw")
    ax.plot(df["Datetime"], roll7,  color=C["accent"],  lw=1.8, alpha=0.9, label="7-pt avg")
    ax.plot(df["Datetime"], roll24, color=C["moderate"], lw=1.5, alpha=0.8, linestyle="--", label="24-pt avg")
    ax.set_title("Rolling Averages", pad=8, fontweight='semibold')
    ax.legend(framealpha=0.4)
    ax.spines[["top","right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(axis='x', rotation=20)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.8)
    return fig

def chart_forecast(y_test, preds, C):
    set_mpl(C)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), gridspec_kw={"width_ratios": [2.5, 1]})
    ax1 = axes[0]
    idx = np.arange(len(y_test))
    ax1.fill_between(idx, y_test.values, alpha=0.1, color=C["accent"])
    ax1.plot(idx, y_test.values, color=C["accent"], lw=1.5, label="Actual", alpha=0.9)
    ax1.plot(idx, preds, color=C["moderate"], lw=1.5, linestyle="--", label="Predicted", alpha=0.9)
    ax1.set_title("Actual vs Predicted", pad=8, fontweight='semibold')
    ax1.legend(framealpha=0.4)
    ax1.spines[["top","right"]].set_visible(False)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("AQI")
    ax2 = axes[1]
    ax2.scatter(y_test.values, preds, color=C["accent"], alpha=0.3, s=6)
    mn = min(y_test.min(), preds.min())
    mx = max(y_test.max(), preds.max())
    ax2.plot([mn, mx], [mn, mx], color=C["bad"], lw=1.5, linestyle="--", alpha=0.6)
    ax2.set_title("Scatter — Actual vs Predicted", pad=8, fontweight='semibold')
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    ax2.spines[["top","right"]].set_visible(False)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.8)
    return fig

def chart_importance(model, feature_names, C, top_n=12):
    set_mpl(C)
    imp = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
    imp = imp.sort_values("Importance", ascending=True).tail(top_n)
    colors = [C["accent"] if v >= imp["Importance"].quantile(0.66)
              else (C["accent3"] if v >= imp["Importance"].quantile(0.33) else C["text3"])
              for v in imp["Importance"]]
    fig, ax = plt.subplots(figsize=(7, top_n * 0.38 + 0.8))
    ax.barh(imp["Feature"], imp["Importance"], color=colors, alpha=0.85, height=0.55)
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance", pad=8, fontweight='semibold')
    ax.spines[["top","right","left"]].set_visible(False)
    ax.grid(True, axis='x', alpha=0.3)
    ax.tick_params(left=False)
    plt.tight_layout(pad=0.8)
    return fig

def chart_correlation(df, cols, C):
    set_mpl(C)
    avail = [c for c in cols if c in df.columns]
    if len(avail) < 2:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.5))
    cmap = "RdBu_r" if not C["is_dark"] else LinearSegmentedColormap.from_list(
        "corr", ["#e94560","#1a1a2e","#0f9b8e"])
    corr = df[avail].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax, cmap=cmap, center=0,
                square=True, linewidths=0.4, linecolor=C["bg"],
                annot_kws={"size": 7.5}, cbar_kws={"shrink": 0.7})
    ax.set_title("Correlation Matrix", pad=8, fontweight='semibold')
    plt.tight_layout(pad=0.5)
    return fig

def chart_distribution(df, TARGET, C):
    set_mpl(C)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    data = df[TARGET].dropna()
    n, bins, patches = ax.hist(data, bins=30, edgecolor='none', alpha=0.8)
    for patch, x in zip(patches, bins[:-1]):
        _, color = classify_aqi(x)
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axvline(data.mean(), color=C["text2"], lw=1.5, linestyle="--", label=f"Mean: {data.mean():.1f}")
    ax.axvline(data.median(), color=C["accent"], lw=1.5, linestyle=":", label=f"Median: {data.median():.1f}")
    ax.legend(framealpha=0.4)
    ax.set_title("AQI Distribution", pad=8, fontweight='semibold')
    ax.set_xlabel("AQI")
    ax.set_ylabel("Frequency")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(pad=0.5)
    return fig

def chart_alerts(df, TARGET, threshold, C):
    set_mpl(C)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(df["Datetime"], df[TARGET], color=C["text3"], lw=0.8, alpha=0.6)
    ax.fill_between(df["Datetime"], df[TARGET], threshold,
                    where=df[TARGET] > threshold, color=C["bad"], alpha=0.2, label="Alert Zone")
    above = df[df[TARGET] > threshold]
    if len(above):
        ax.scatter(above["Datetime"], above[TARGET], color=C["bad"], s=12, zorder=5, alpha=0.7)
    ax.axhline(threshold, color=C["bad"], lw=1.5, linestyle="--", alpha=0.8, label=f"Threshold ({threshold:.0f})")
    ax.set_title("Alert Monitor", pad=8, fontweight='semibold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(framealpha=0.4)
    ax.tick_params(axis='x', rotation=20)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.8)
    return fig

def chart_hourly_profile(df, TARGET, C):
    set_mpl(C)
    df2 = df.copy()
    df2["hour"] = df2["Datetime"].dt.hour
    hourly = df2.groupby("hour")[TARGET].agg(["mean","std"]).reset_index()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(hourly["hour"],
                    hourly["mean"] - hourly["std"],
                    hourly["mean"] + hourly["std"],
                    alpha=0.1, color=C["accent"])
    ax.plot(hourly["hour"], hourly["mean"], color=C["accent"], lw=2, marker='o', markersize=4)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average AQI")
    ax.set_title("Hourly AQI Profile (with std dev band)", pad=8, fontweight='semibold')
    ax.spines[["top","right"]].set_visible(False)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.8)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar(C):
    with st.sidebar:
        # Brand
        st.markdown(f"""
        <div class="sb-brand">
            <div class="sb-brand-name">Air<span>Aware</span></div>
            <div class="sb-brand-sub">Air Quality Intelligence</div>
        </div>""", unsafe_allow_html=True)

        # Theme
        st.markdown(f'<div class="sb-section"><span class="sb-section-label">Display</span></div>',
                    unsafe_allow_html=True)
        with st.container():
            st.markdown('<div style="padding: 0 1.4rem;">', unsafe_allow_html=True)
            dark_toggle = st.toggle("Dark Mode", value=st.session_state.dark_mode)
            st.markdown('</div>', unsafe_allow_html=True)
        if dark_toggle != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_toggle
            st.rerun()

        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

        # Data upload
        st.markdown(f'<div class="sb-section"><span class="sb-section-label">Dataset</span></div>',
                    unsafe_allow_html=True)
        with st.container():
            st.markdown('<div style="padding: 0 1.4rem;">', unsafe_allow_html=True)
            uploaded = st.file_uploader("Upload CSV", type=["csv"],
                                        label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

        # Pollutant selector
        st.markdown(f'<div class="sb-section"><span class="sb-section-label">Pollutant View</span></div>',
                    unsafe_allow_html=True)
        with st.container():
            st.markdown('<div style="padding: 0 1.4rem;">', unsafe_allow_html=True)
            pollutant_options = ["AirQualityIndex","PM2.5","PM10","NO2(GT)","CO(GT)","Temperature","Humidity"]
            selected_pollutant = st.selectbox("Pollutant", pollutant_options, label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

        # Alert threshold
        st.markdown(f'<div class="sb-section"><span class="sb-section-label">Alert Threshold</span></div>',
                    unsafe_allow_html=True)
        with st.container():
            st.markdown('<div style="padding: 0 1.4rem;">', unsafe_allow_html=True)
            alert_threshold = st.slider("AQI Threshold", 50, 300, 100, 10, label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)

        # AQI preview
        t_label, t_color = classify_aqi(alert_threshold)
        st.markdown(f"""
        <div class="sb-aqi-preview">
            <div class="sb-aqi-val" style="--t:{t_color}">{alert_threshold}</div>
            <div class="sb-aqi-label" style="--t:{t_color}">{t_label}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

        # Admin mode
        st.markdown(f'<div class="sb-section"><span class="sb-section-label">Access</span></div>',
                    unsafe_allow_html=True)
        with st.container():
            st.markdown('<div style="padding: 0 1.4rem;">', unsafe_allow_html=True)
            admin_mode = st.toggle("Admin Mode", key="admin_toggle")
            st.markdown('</div>', unsafe_allow_html=True)

        # Status
        if uploaded:
            st.markdown(f"""
            <div class="sb-status">
                <div class="sb-status-dot"></div>
                <div class="sb-status-text">Dataset loaded<br>
                    <span style="font-family:monospace;font-size:0.65rem;color:{C['text3']}">
                        {uploaded.name}
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="sb-status">
                <div style="width:8px;height:8px;border-radius:50%;
                    background:{C['text3']};flex-shrink:0;"></div>
                <div class="sb-status-text">No dataset loaded</div>
            </div>""", unsafe_allow_html=True)

        # Credits
        st.markdown(f"""
        <div style="padding:1rem 1.4rem 0.5rem;margin-top:auto">
            <div style="font-size:0.6rem;color:{C['text3']};line-height:1.9;text-transform:uppercase;letter-spacing:0.08em">
                XGBoost · Streamlit<br>
                CPCB · OpenAQ · Kaggle
            </div>
        </div>""", unsafe_allow_html=True)

    return uploaded, selected_pollutant, alert_threshold, admin_mode


# ─────────────────────────────────────────────────────────────────────────────
#  TOP BAR
# ─────────────────────────────────────────────────────────────────────────────
def render_topbar(C, n_records=None, target=None):
    meta_parts = []
    if n_records:
        meta_parts.append(f"{n_records:,} records")
    if target:
        meta_parts.append(f"target: {target}")
    meta_str = "  /  ".join(meta_parts) if meta_parts else "no data loaded"

    st.markdown(f"""
    <div class="topbar">
        <div class="topbar-brand">Air<span>Aware</span></div>
        <div class="topbar-meta">
            <div class="live-indicator">
                <div class="live-dot"></div>
                <span>live</span>
            </div>
            <span>{meta_str}</span>
        </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────
def render_kpis(df, TARGET, C):
    aqi_mean = df[TARGET].mean()
    aqi_max  = df[TARGET].max()
    good_pct = (df[TARGET] <= 50).sum() / max(len(df),1) * 100
    alert_ct = (df[TARGET] > 100).sum()
    alert_pct = alert_ct / max(len(df),1) * 100
    cat, color = classify_aqi(aqi_mean)

    cards = [
        ("Mean AQI", f"{aqi_mean:.1f}", cat, color),
        ("Peak AQI", f"{aqi_max:.0f}", "Maximum recorded", C["bad"]),
        ("Good Air", f"{good_pct:.0f}%", f"{(df[TARGET]<=50).sum():,} clean records", C["good"]),
        ("Alerts", f"{alert_ct:,}", f"{alert_pct:.1f}% above threshold", C["moderate"] if alert_pct < 20 else C["bad"]),
    ]
    cols = st.columns(4)
    for col_w, (label, value, sub, color) in zip(cols, cards):
        with col_w:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-card-accent" style="--accent-color:{color}"></div>
                <div class="stat-label">{label}</div>
                <div class="stat-value" style="--stat-color:{color}">{value}</div>
                <div class="stat-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  WELCOME SCREEN
# ─────────────────────────────────────────────────────────────────────────────
def render_welcome(C):
    st.markdown(f"""
    <div class="welcome-card">
        <div class="welcome-title">Air<span>Aware</span></div>
        <p class="welcome-sub">
            Upload a CSV dataset from the sidebar to begin exploring air quality
            trends, XGBoost forecasts, alert monitoring, and pollutant analysis.
        </p>
        <div style="margin-top:1.2rem">
            <p style="font-size:0.68rem;font-weight:700;text-transform:uppercase;
                      letter-spacing:0.1em;color:{C['text3']};margin-bottom:0.8rem">
                Expected Columns
            </p>
            <table class="schema-tbl">
                <tr><th>Column</th><th>Type</th><th>Required</th></tr>
                <tr><td>Date</td><td>Date string</td><td class="req">Required</td></tr>
                <tr><td>Time</td><td>Time string</td><td class="req">Required</td></tr>
                <tr><td>AirQualityIndex</td><td>Float</td><td class="req">Required</td></tr>
                <tr><td>PM2.5</td><td>Float (ug/m3)</td><td class="req">Required</td></tr>
                <tr><td>PM10</td><td>Float (ug/m3)</td><td class="opt">Optional</td></tr>
                <tr><td>NO2(GT)</td><td>Float</td><td class="opt">Optional</td></tr>
                <tr><td>CO(GT)</td><td>Float</td><td class="opt">Optional</td></tr>
                <tr><td>Temperature</td><td>Float (C)</td><td class="opt">Optional</td></tr>
                <tr><td>Humidity</td><td>Float (%)</td><td class="opt">Optional</td></tr>
            </table>
            <p style="font-size:0.68rem;color:{C['text3']};margin-top:0.8rem;line-height:1.6">
                For datasets with different column names, AirAware will auto-detect
                the target column or allow you to select it manually in Admin Mode.
            </p>
        </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
def tab_overview(df, TARGET, selected_pollutant, C):
    aqi_val = df[TARGET].iloc[-1] if len(df) > 0 else df[TARGET].mean()
    label, color = classify_aqi(aqi_val)
    advice = aqi_advice(label)
    st.markdown(f"""
    <div class="aqi-banner" style="--banner-color:{color}">
        <div class="aqi-banner-title">Current Status — {label} (AQI {aqi_val:.0f})</div>
        <div class="aqi-banner-text">{advice}</div>
    </div>""", unsafe_allow_html=True)

    col_chart, col_radar = st.columns([3, 1])
    with col_chart:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="chart-card-title">Pollutant Time Series</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chart-card-sub">Selected: {selected_pollutant}</div>', unsafe_allow_html=True)
        if selected_pollutant in df.columns:
            fig = chart_trend(df, selected_pollutant, C)
            st.pyplot(fig, use_container_width=True)
            plt.close()
        else:
            st.info(f"Column '{selected_pollutant}' not found in dataset.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_radar:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-card-title">Pollutant Profile</div>', unsafe_allow_html=True)
        fig_r = chart_radar(df, C)
        if fig_r:
            st.pyplot(fig_r, use_container_width=True)
            plt.close()
        else:
            st.markdown(f'<p style="font-size:0.78rem;color:{C["text3"]}">Not enough pollutant columns for radar chart.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    col_roll, col_month = st.columns(2)
    with col_roll:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-card-title">Rolling Averages</div>', unsafe_allow_html=True)
        fig_roll = chart_rolling(df, TARGET, C)
        st.pyplot(fig_roll, use_container_width=True)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_month:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-card-title">Monthly Pattern</div>', unsafe_allow_html=True)
        fig_m = chart_monthly_bar(df, TARGET, C)
        st.pyplot(fig_m, use_container_width=True)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    if len(df) >= 24:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-card-title">AQI Heatmap — Hour vs Day</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-card-sub">Average AQI by time of day and day of week</div>', unsafe_allow_html=True)
        fig_h = chart_aqi_heatmap(df, TARGET, C)
        st.pyplot(fig_h, use_container_width=True)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB: FORECAST
# ─────────────────────────────────────────────────────────────────────────────
def tab_forecast(df, df_raw, TARGET, C):
    st.markdown('<div class="sec-label">XGBoost Forecast Engine</div>', unsafe_allow_html=True)

    if st.button("Train Model and Generate Forecast", key="btn_train"):
        with st.spinner("Engineering features, training model, evaluating..."):
            try:
                df_eng = engineer_features(df_raw.copy(), TARGET)
                model, X, X_test, y_test, preds, mae, rmse, r2 = train_model(df_eng, TARGET)
                st.session_state.update({
                    "model": model, "X": X,
                    "y_test": y_test, "preds": preds,
                    "metrics": (mae, rmse, r2),
                    "model_trained": True
                })
                joblib.dump(model, "aqi_xgboost_model.pkl")
                joblib.dump(list(X.columns), "aqi_feature_columns.pkl")
                st.success(f"Model trained — R2: {r2:.4f}  |  MAE: {mae:.3f}  |  RMSE: {rmse:.3f}")
            except Exception as e:
                st.error(f"Training failed: {e}")

    if not st.session_state.get("model_trained"):
        st.markdown(f"""
        <div style="padding:1.5rem;background:{C['card']};border:1px solid {C['border']};
             border-radius:10px;font-size:0.82rem;color:{C['text2']}">
            Model has not been trained yet. Click the button above to begin training
            on the current dataset using XGBoost with lag features and rolling statistics.
        </div>""", unsafe_allow_html=True)
        return

    mae, rmse, r2 = st.session_state["metrics"]
    model  = st.session_state["model"]
    X      = st.session_state["X"]
    y_test = st.session_state["y_test"]
    preds  = st.session_state["preds"]

    # Score boxes
    c1, c2, c3 = st.columns(3)
    r2_color = C["good"] if r2 > 0.9 else (C["moderate"] if r2 > 0.7 else C["bad"])
    for col_w, label, val, desc, vcolor in [
        (c1, "R2 Score", f"{r2:.4f}", "1.0 = perfect fit", r2_color),
        (c2, "MAE", f"{mae:.3f}", "Mean absolute error (AQI units)", C["accent"]),
        (c3, "RMSE", f"{rmse:.3f}", "Root mean squared error", C["accent"]),
    ]:
        with col_w:
            st.markdown(f"""
            <div class="score-box">
                <div class="score-num" style="color:{vcolor}">{val}</div>
                <div class="score-key">{label}</div>
                <div class="score-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-card-title">Forecast Visualization</div>', unsafe_allow_html=True)
    fig_f = chart_forecast(y_test, preds, C)
    st.pyplot(fig_f, use_container_width=True)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    col_fi, col_stats = st.columns([2, 1])
    with col_fi:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-card-title">Feature Importance</div>', unsafe_allow_html=True)
        fig_imp = chart_importance(model, list(X.columns), C)
        st.pyplot(fig_imp, use_container_width=True)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_stats:
        st.markdown('<div class="admin-card">', unsafe_allow_html=True)
        st.markdown('<div class="admin-card-title">Top Features</div>', unsafe_allow_html=True)
        imp_df = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
        imp_df = imp_df.sort_values("Importance", ascending=False)
        for _, row in imp_df.head(8).iterrows():
            pct_fill = row["Importance"] / imp_df["Importance"].max() * 100
            st.markdown(f"""
            <div class="prog-wrap">
                <div class="prog-label">
                    <span>{row['Feature']}</span>
                    <span class="prog-val">{row['Importance']:.4f}</span>
                </div>
                <div class="prog-track">
                    <div class="prog-fill" style="width:{pct_fill:.0f}%"></div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if os.path.exists("aqi_xgboost_model.pkl"):
        st.markdown("<br>", unsafe_allow_html=True)
        with open("aqi_xgboost_model.pkl", "rb") as f:
            st.download_button("Download Trained Model (.pkl)", f,
                               "aqi_xgboost_model.pkl", mime="application/octet-stream")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB: ALERTS
# ─────────────────────────────────────────────────────────────────────────────
def tab_alerts(df, TARGET, alert_threshold, C):
    st.markdown('<div class="sec-label">Alert Monitor</div>', unsafe_allow_html=True)

    aqi_vals = df[TARGET]
    is_norm  = aqi_vals.abs().max() < 20
    if is_norm:
        t_mean = float(aqi_vals.mean())
        t_std  = float(aqi_vals.std()) or 1.0
        threshold = (alert_threshold - t_mean) / t_std
    else:
        threshold = float(alert_threshold)

    alert_df  = df[df[TARGET] > threshold]
    alert_pct = len(alert_df) / max(len(df), 1) * 100

    if len(alert_df) == 0:
        banner_color = C["good"]
        banner_title = "All Clear — No threshold breaches detected"
        banner_text  = "All readings are within the configured safe AQI threshold."
    elif alert_pct > 30:
        banner_color = C["bad"]
        banner_title = f"High Alert — {len(alert_df):,} records exceed threshold"
        banner_text  = f"{alert_pct:.1f}% of readings above AQI {alert_threshold:.0f}. Immediate attention recommended."
    else:
        banner_color = C["moderate"]
        banner_title = f"Caution — {len(alert_df):,} threshold exceedances"
        banner_text  = f"{alert_pct:.1f}% of readings above AQI {alert_threshold:.0f}. Monitor sensitive groups."

    st.markdown(f"""
    <div class="aqi-banner" style="--banner-color:{banner_color}">
        <div class="aqi-banner-title">{banner_title}</div>
        <div class="aqi-banner-text">{banner_text}</div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    stats_data = [
        ("Total Records", f"{len(df):,}", C["text2"]),
        ("Alert Records", f"{len(alert_df):,}", C["bad"] if len(alert_df) > 0 else C["good"]),
        ("Alert Rate", f"{alert_pct:.1f}%", C["moderate"]),
        ("Peak AQI (alerts)", f"{alert_df[TARGET].max():.0f}" if len(alert_df) > 0 else "—", C["bad"]),
    ]
    for col_w, (label, val, color) in zip([c1,c2,c3,c4], stats_data):
        with col_w:
            st.markdown(f"""
            <div class="stat-card" style="padding:1rem 1.2rem">
                <div class="stat-card-accent" style="--accent-color:{color}"></div>
                <div class="stat-label">{label}</div>
                <div class="stat-value" style="--stat-color:{color};font-size:1.6rem">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fig_al = chart_alerts(df, TARGET, threshold, C)
    st.pyplot(fig_al, use_container_width=True)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    if len(alert_df) > 0:
        st.markdown('<div class="sec-label">Alert Log</div>', unsafe_allow_html=True)
        disp_cols = ["Datetime", TARGET] + [c for c in ["PM2.5","NO2(GT)","CO(GT)"] if c in alert_df.columns]
        disp = alert_df[disp_cols].copy().head(50)
        disp["Datetime"] = disp["Datetime"].dt.strftime("%Y-%m-%d %H:%M")
        for col in disp.columns[1:]:
            disp[col] = disp[col].round(2)
        disp["Severity"] = disp[TARGET].apply(lambda v: classify_aqi(v)[0])
        st.dataframe(disp, use_container_width=True, hide_index=True)
        csv_data = disp.to_csv(index=False).encode()
        st.download_button("Export Alert Log as CSV", csv_data, "alert_log.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB: ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def tab_analysis(df, TARGET, C):
    corr_cols = [c for c in ["PM2.5","PM10","NO2(GT)","CO(GT)","Temperature","Humidity","AirQualityIndex","aqi_value"] if c in df.columns]

    col_corr, col_dist = st.columns([3, 2])
    with col_corr:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-card-title">Correlation Matrix</div>', unsafe_allow_html=True)
        if len(corr_cols) >= 2:
            fig_c = chart_correlation(df, corr_cols, C)
            if fig_c:
                st.pyplot(fig_c, use_container_width=True)
                plt.close()
        else:
            st.markdown(f'<p style="font-size:0.78rem;color:{C["text3"]}">Not enough columns for correlation matrix.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_dist:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-card-title">AQI Distribution</div>', unsafe_allow_html=True)
        fig_d = chart_distribution(df, TARGET, C)
        st.pyplot(fig_d, use_container_width=True)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Statistical Summary</div>', unsafe_allow_html=True)
    stat_cols = [c for c in corr_cols if c in df.columns]
    if stat_cols:
        stats_df = df[stat_cols].describe().round(3)
        st.dataframe(stats_df, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">AQI Category Breakdown</div>', unsafe_allow_html=True)
    cat_cols = st.columns(len(AQI_LEVELS))
    for col_w, (lo, hi, label, color) in zip(cat_cols, AQI_LEVELS):
        count = ((df[TARGET] >= lo) & (df[TARGET] <= hi)).sum()
        pct = count / max(len(df), 1) * 100
        with col_w:
            st.markdown(f"""
            <div class="cat-cell" style="--cat-bg:{color}12;--cat-border:{color}30;--cat-color:{color}">
                <div class="cat-name">{label}</div>
                <div class="cat-pct">{pct:.1f}%</div>
                <div class="cat-count">{count:,} records</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-card-title">Hourly AQI Profile</div>', unsafe_allow_html=True)
    fig_h = chart_hourly_profile(df, TARGET, C)
    st.pyplot(fig_h, use_container_width=True)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB: ADMIN
# ─────────────────────────────────────────────────────────────────────────────
def tab_admin(uploaded, C):
    st.markdown('<div class="sec-label">Admin Control Panel</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="aqi-banner" style="--banner-color:{C['accent']}">
        <div class="aqi-banner-title">Admin Mode Active</div>
        <div class="aqi-banner-text">
            Upload a new dataset and retrain the XGBoost model.
            The system will auto-detect the target column, or you can select it manually below.
        </div>
    </div>""", unsafe_allow_html=True)

    col_a, col_b = st.columns([1.5, 1])

    with col_a:
        st.markdown('<div class="admin-card">', unsafe_allow_html=True)
        st.markdown('<div class="admin-card-title">Upload New Training Dataset</div>', unsafe_allow_html=True)
        admin_file = st.file_uploader("New CSV File", type=["csv"],
                                      key="admin_upload_main",
                                      label_visibility="collapsed")
        if admin_file:
            try:
                df_preview = load_csv(admin_file)
                admin_file.seek(0)  # reset after read

                st.success(f"File ready: {admin_file.name}  ({len(df_preview):,} rows)")
                st.markdown(f'<p style="font-size:0.72rem;color:{C["text2"]};margin-top:8px">Detected columns: {", ".join(df_preview.columns.tolist()[:8])}</p>', unsafe_allow_html=True)

                # Target column selector for the new dataset
                candidates = auto_detect_target(df_preview)
                auto_target = find_target(df_preview)

                default_idx = candidates.index(auto_target) if auto_target in candidates else 0
                selected_target = st.selectbox(
                    "Select Target Column",
                    candidates,
                    index=default_idx,
                    help="Choose the column representing the Air Quality Index to forecast."
                )
                st.session_state["admin_selected_target"] = selected_target
                st.session_state["admin_df_preview"] = df_preview

            except Exception as e:
                st.error(f"Could not parse file: {e}")
        else:
            st.markdown(f'<p style="font-size:0.78rem;color:{C["text3"]};margin-top:8px">No new file uploaded. Will retrain on current dataset if available.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="admin-card">', unsafe_allow_html=True)
        st.markdown('<div class="admin-card-title">Retrain Model</div>', unsafe_allow_html=True)

        src_name = admin_file.name if admin_file else (uploaded.name if uploaded else "None")
        tgt_name = st.session_state.get("admin_selected_target", "auto-detect")

        st.markdown(f"""
        <div class="metric-row"><span class="metric-key">Source</span><span class="metric-val">{src_name}</span></div>
        <div class="metric-row"><span class="metric-key">Target column</span><span class="metric-val">{tgt_name}</span></div>""", unsafe_allow_html=True)

        if os.path.exists("aqi_xgboost_model.pkl"):
            fsize = os.path.getsize("aqi_xgboost_model.pkl") / 1024
            st.markdown(f"""
            <div class="metric-row"><span class="metric-key">Current model</span><span class="metric-val">{fsize:.1f} KB</span></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Retrain Model", key="btn_retrain"):
            # Determine source file
            src = admin_file if admin_file else uploaded
            if src is None:
                st.error("No dataset available. Upload a file first.")
            else:
                with st.spinner("Retraining..."):
                    try:
                        src.seek(0)
                        df_new = load_csv(src)

                        # Use manually selected target if available
                        if admin_file and "admin_selected_target" in st.session_state:
                            tgt_new = st.session_state["admin_selected_target"]
                        else:
                            tgt_new = find_target(df_new)

                        if tgt_new is None or tgt_new not in df_new.columns:
                            st.error(f"Target column '{tgt_new}' not found. Please select a valid column above.")
                        else:
                            df_eng2 = engineer_features(df_new.copy(), tgt_new)
                            m2, X2, _, yt2, p2, mae2, rmse2, r2_2 = train_model(df_eng2, tgt_new)
                            joblib.dump(m2, "aqi_xgboost_model.pkl")
                            joblib.dump(list(X2.columns), "aqi_feature_columns.pkl")
                            st.session_state.update({
                                "model": m2, "X": X2,
                                "y_test": yt2, "preds": p2,
                                "metrics": (mae2, rmse2, r2_2),
                                "model_trained": True
                            })
                            st.success(f"Retrained on '{tgt_new}'  |  R2: {r2_2:.4f}  |  MAE: {mae2:.3f}")
                    except Exception as e:
                        st.error(f"Retraining failed: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    # System info
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">System Status</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    info = [
        ("Model File", "aqi_xgboost_model.pkl",
         f"{os.path.getsize('aqi_xgboost_model.pkl')/1024:.1f} KB" if os.path.exists("aqi_xgboost_model.pkl") else "Not found"),
        ("Feature File", "aqi_feature_columns.pkl",
         "Present" if os.path.exists("aqi_feature_columns.pkl") else "Not found"),
        ("Model Status", "Trained" if st.session_state.get("model_trained") else "Not trained",
         f"R2: {st.session_state['metrics'][2]:.4f}" if st.session_state.get("model_trained") else "—"),
    ]
    for col_w, (label, name, val) in zip([c1,c2,c3], info):
        ok = "Not found" not in val and "Not trained" not in val
        color = C["good"] if ok else C["text3"]
        with col_w:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-card-accent" style="--accent-color:{color}"></div>
                <div class="stat-label">{label}</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.72rem;
                            color:{C['text2']};margin:4px 0">{name}</div>
                <div style="font-size:0.72rem;color:{color};font-weight:600">{val}</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────
def render_footer(C):
    st.markdown(f"""
    <div class="footer">
        <p>AirAware — Smart Air Quality Intelligence Platform</p>
        <p>Python · Streamlit · XGBoost · Matplotlib · Seaborn</p>
        <p>Data Sources: CPCB · OpenAQ · Kaggle</p>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    C = theme()
    inject_css(C)

    uploaded, selected_pollutant, alert_threshold, admin_mode = render_sidebar(C)

    if uploaded:
        df_raw = load_csv(uploaded)
        TARGET = find_target(df_raw)
        render_topbar(C, len(df_raw), TARGET)
    else:
        render_topbar(C)

    if uploaded is None:
        render_welcome(C)
        render_footer(C)
        return

    if TARGET is None:
        # Allow manual selection
        st.warning("Could not auto-detect the AQI target column.")
        candidates = auto_detect_target(df_raw)
        if candidates:
            TARGET = st.selectbox("Select target column manually:", candidates)
        else:
            st.error("No suitable numeric column found. Please check your CSV.")
            return

    # Date range
    c1, c2, c_info = st.columns([1, 1, 3])
    with c1:
        start_d = st.date_input("From",
            value=df_raw["Datetime"].min().date(),
            min_value=df_raw["Datetime"].min().date(),
            max_value=df_raw["Datetime"].max().date())
    with c2:
        end_d = st.date_input("To",
            value=df_raw["Datetime"].max().date(),
            min_value=df_raw["Datetime"].min().date(),
            max_value=df_raw["Datetime"].max().date())
    with c_info:
        total_days = (pd.Timestamp(end_d) - pd.Timestamp(start_d)).days
        st.markdown(f"""
        <div style="display:flex;align-items:center;height:100%;gap:20px;padding-top:8px;flex-wrap:wrap">
            <span style="font-size:0.75rem;color:{C['text3']}">
                {total_days} day range
            </span>
            <span style="font-size:0.75rem;color:{C['text3']}">
                {len(df_raw):,} records total
            </span>
            <span style="font-size:0.75rem;color:{C['accent']};font-weight:600">
                {TARGET}
            </span>
        </div>""", unsafe_allow_html=True)

    df = df_raw[
        (df_raw["Datetime"] >= pd.Timestamp(start_d)) &
        (df_raw["Datetime"] <= pd.Timestamp(end_d))
    ].copy()

    if len(df) == 0:
        st.warning("No data in the selected date range.")
        return

    st.markdown("<br>", unsafe_allow_html=True)
    render_kpis(df, TARGET, C)
    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    tabs_list = ["Overview", "Forecast", "Alerts", "Analysis"]
    if admin_mode:
        tabs_list.append("Admin")

    tabs = st.tabs(tabs_list)

    with tabs[0]:
        tab_overview(df, TARGET, selected_pollutant, C)
    with tabs[1]:
        tab_forecast(df, df_raw, TARGET, C)
    with tabs[2]:
        tab_alerts(df, TARGET, alert_threshold, C)
    with tabs[3]:
        tab_analysis(df, TARGET, C)
    if admin_mode and len(tabs) > 4:
        with tabs[4]:
            tab_admin(uploaded, C)

    render_footer(C)


if __name__ == "__main__":
    main()