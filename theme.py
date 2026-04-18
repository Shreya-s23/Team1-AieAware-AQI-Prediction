"""
theme.py — Shared design system for AirAware
"""

def get_theme(dark: bool) -> dict:
    if dark:
        return dict(
            bg="#111827",
            surface="#1f2937",
            card="#1f2937",
            card2="#243044",
            border="#374151",
            border2="#4b5563",
            accent="#3b82f6",
            accent2="#2563eb",
            accent3="#10b981",
            text="#f9fafb",
            text2="#d1d5db",
            text3="#6b7280",
            good="#10b981",
            moderate="#f59e0b",
            bad="#ef4444",
            hazard="#8b5cf6",
            plot_bg="#1f2937",
            mpl_bg="#1f2937",
            is_dark=True,
        )
    else:
        return dict(
            bg="#f8fafc",
            surface="#ffffff",
            card="#ffffff",
            card2="#f1f5f9",
            border="#e2e8f0",
            border2="#cbd5e1",
            accent="#2563eb",
            accent2="#1d4ed8",
            accent3="#059669",
            text="#0f172a",
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


def inject_base_css(C: dict, role: str = "user"):
    """role: 'admin' or 'user' — changes accent color slightly"""
    import streamlit as st

    admin_accent = "#7c3aed" if not C["is_dark"] else "#8b5cf6"
    panel_accent = admin_accent if role == "admin" else C["accent"]

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif !important;
    background: {C['bg']} !important;
    color: {C['text']} !important;
}}

.stApp {{
    background: {C['bg']} !important;
}}

/* Hide Streamlit chrome */
#MainMenu {{ visibility: hidden !important; display: none !important; }}
footer {{ visibility: hidden !important; }}
header {{ visibility: hidden !important; }}
[data-testid="stToolbar"] {{ display: none !important; }}
.stDeployButton {{ display: none !important; }}
[data-testid="stDecoration"] {{ display: none !important; }}

/* ── SIDEBAR ─────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background: {C['surface']} !important;
    border-right: 1px solid {C['border']} !important;
    width: 270px !important;
}}
section[data-testid="stSidebar"] > div {{
    padding: 0 !important;
    background: {C['surface']} !important;
}}
section[data-testid="stSidebar"] * {{ color: {C['text']} !important; }}
section[data-testid="stSidebar"] label {{
    font-size: 0.76rem !important;
    font-weight: 500 !important;
    color: {C['text2']} !important;
}}
section[data-testid="stSidebar"] .stSelectbox > div > div {{
    background: {C['card2']} !important;
    border-color: {C['border']} !important;
    border-radius: 8px !important;
}}
section[data-testid="stSidebar"] .stFileUploader {{
    background: {C['card2']} !important;
    border: 1.5px dashed {C['border2']} !important;
    border-radius: 10px !important;
    padding: 6px !important;
}}

/* ── LAYOUT ─────────────────────────────────── */
.block-container {{
    padding: 0 2rem 3rem 2rem !important;
    max-width: 100% !important;
}}

/* ── TOPBAR ─────────────────────────────────── */
.topbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.4rem 0;
    border-bottom: 1px solid {C['border']};
    margin-bottom: 2rem;
}}
.topbar-brand {{
    font-family: 'Sora', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: {C['text']};
    letter-spacing: -0.04em;
}}
.topbar-brand .accent {{ color: {panel_accent}; }}
.topbar-right {{
    display: flex;
    align-items: center;
    gap: 16px;
    font-size: 0.72rem;
    color: {C['text3']};
    font-family: 'IBM Plex Mono', monospace;
}}
.role-badge {{
    background: {panel_accent}18;
    color: {panel_accent};
    border: 1px solid {panel_accent}40;
    border-radius: 5px;
    padding: 3px 10px;
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'IBM Plex Sans', sans-serif;
}}
.live-dot {{
    width: 6px; height: 6px;
    border-radius: 50%;
    background: {C['good']};
    display: inline-block;
    margin-right: 5px;
    animation: blink 2.5s infinite;
}}
@keyframes blink {{
    0%,100% {{ opacity: 1; }}
    50% {{ opacity: 0.3; }}
}}

/* ── STAT CARDS ─────────────────────────────── */
.stat-card {{
    background: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    transition: box-shadow 0.2s;
}}
.stat-card:hover {{
    box-shadow: 0 2px 16px {'rgba(0,0,0,0.18)' if C['is_dark'] else 'rgba(0,0,0,0.07)'};
}}
.stat-accent-bar {{
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 12px 12px 0 0;
    background: var(--sc, {panel_accent});
}}
.stat-label {{
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {C['text3']};
    margin-bottom: 0.45rem;
}}
.stat-value {{
    font-family: 'Sora', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--sv, {C['text']});
    letter-spacing: -0.04em;
    line-height: 1;
    margin-bottom: 0.3rem;
}}
.stat-sub {{
    font-size: 0.7rem;
    color: {C['text3']};
}}

/* ── CHART WRAPPER ──────────────────────────── */
.cw {{
    background: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 12px;
    padding: 1.3rem;
    margin-bottom: 1.2rem;
}}
.cw-title {{
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {C['text3']};
    margin-bottom: 0.2rem;
}}
.cw-sub {{
    font-size: 0.7rem;
    color: {C['text3']};
    margin-bottom: 0.9rem;
}}

/* ── BANNER ─────────────────────────────────── */
.banner {{
    background: {C['card']};
    border-left: 4px solid var(--bc, {panel_accent});
    border-top: 1px solid {C['border']};
    border-right: 1px solid {C['border']};
    border-bottom: 1px solid {C['border']};
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.3rem;
    margin-bottom: 1.4rem;
}}
.banner-title {{
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--bc, {panel_accent});
    margin-bottom: 0.2rem;
}}
.banner-text {{
    font-size: 0.76rem;
    color: {C['text2']};
}}

/* ── SECTION LABEL ──────────────────────────── */
.sec-lbl {{
    font-size: 0.63rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: {C['text3']};
    padding-bottom: 0.5rem;
    border-bottom: 1px solid {C['border']};
    margin-bottom: 1rem;
}}

/* ── METRIC ROW ─────────────────────────────── */
.mrow {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.55rem 0;
    border-bottom: 1px solid {C['border']};
    font-size: 0.76rem;
}}
.mrow:last-child {{ border-bottom: none; }}
.mrow-k {{ color: {C['text2']}; }}
.mrow-v {{
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
    color: {C['text']};
}}

/* ── SCORE BOX ──────────────────────────────── */
.score-box {{
    background: {C['card2']};
    border: 1px solid {C['border']};
    border-radius: 10px;
    padding: 1.1rem;
    text-align: center;
}}
.score-num {{
    font-family: 'Sora', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: {panel_accent};
}}
.score-key {{
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {C['text3']};
    font-weight: 600;
    margin-top: 3px;
}}
.score-desc {{
    font-size: 0.65rem;
    color: {C['text3']};
    margin-top: 3px;
}}

/* ── PROGRESS ───────────────────────────────── */
.prog {{ margin-bottom: 0.75rem; }}
.prog-header {{
    display: flex;
    justify-content: space-between;
    font-size: 0.7rem;
    margin-bottom: 3px;
    color: {C['text2']};
}}
.prog-val {{
    font-family: 'IBM Plex Mono', monospace;
    color: {panel_accent};
    font-size: 0.68rem;
}}
.prog-track {{
    background: {C['border']};
    border-radius: 3px;
    height: 4px;
    overflow: hidden;
}}
.prog-fill {{
    height: 100%;
    border-radius: 3px;
    background: {panel_accent};
}}

/* ── CATEGORY CELL ──────────────────────────── */
.cat-cell {{
    text-align: center;
    padding: 0.8rem 0.3rem;
    background: var(--cbg);
    border: 1px solid var(--cborder);
    border-radius: 10px;
}}
.cat-name {{
    font-size: 0.6rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--ccolor);
    margin-bottom: 3px;
    line-height: 1.3;
}}
.cat-pct {{
    font-family: 'Sora', sans-serif;
    font-size: 1.2rem;
    font-weight: 800;
    color: var(--ccolor);
}}
.cat-count {{
    font-size: 0.58rem;
    color: {C['text3']};
    margin-top: 2px;
    font-family: 'IBM Plex Mono', monospace;
}}

/* ── BUTTONS ────────────────────────────────── */
.stButton > button {{
    background: {panel_accent} !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 0.55rem 1.3rem !important;
    transition: all 0.15s !important;
}}
.stButton > button:hover {{
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}}

/* ── TABS ───────────────────────────────────── */
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
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
    color: {C['text2']} !important;
    padding: 7px 16px !important;
}}
.stTabs [aria-selected="true"] {{
    background: {panel_accent} !important;
    color: white !important;
    font-weight: 600 !important;
}}

/* ── INPUTS ─────────────────────────────────── */
.stSelectbox label, .stSlider label, .stDateInput label,
.stFileUploader label, .stToggle label, .stTextInput label {{
    color: {C['text2']} !important;
    font-size: 0.76rem !important;
    font-weight: 500 !important;
}}
[data-testid="stMetricValue"] {{
    font-family: 'Sora', sans-serif !important;
    font-weight: 800 !important;
    color: {panel_accent} !important;
}}
div[data-testid="stMarkdownContainer"] p {{ color: {C['text']} !important; }}
.stSpinner > div {{ border-top-color: {panel_accent} !important; }}
.stSuccess, .stError, .stInfo, .stWarning {{ border-radius: 8px !important; }}
.stDataFrame {{ border-radius: 10px !important; overflow: hidden !important; }}

/* ── SIDEBAR INNER ──────────────────────────── */
.sb-brand {{
    padding: 1.5rem 1.3rem 1.1rem;
    border-bottom: 1px solid {C['border']};
}}
.sb-brand-name {{
    font-family: 'Sora', sans-serif;
    font-size: 1.15rem;
    font-weight: 800;
    color: {C['text']};
    letter-spacing: -0.04em;
}}
.sb-brand-name .accent {{ color: {panel_accent}; }}
.sb-brand-sub {{
    font-size: 0.58rem;
    color: {C['text3']};
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-weight: 600;
    margin-top: 2px;
}}
.sb-sec {{ padding: 0.9rem 1.3rem 0.4rem; }}
.sb-sec-lbl {{
    font-size: 0.58rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: {C['text3']};
    display: block;
    margin-bottom: 0.5rem;
}}
.sb-div {{
    height: 1px;
    background: {C['border']};
    margin: 0.5rem 1.3rem;
}}
.sb-info {{
    margin: 0.6rem 1.3rem;
    background: {C['card2']};
    border: 1px solid {C['border']};
    border-radius: 8px;
    padding: 0.75rem;
}}
.sb-info-title {{
    font-size: 0.62rem;
    font-weight: 700;
    color: {C['text3']};
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}}
.sb-info-val {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: {C['text']};
    line-height: 1.7;
}}

/* ── AUTH PAGE ──────────────────────────────── */
.auth-wrap {{
    max-width: 420px;
    margin: 4rem auto;
    background: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 16px;
    padding: 2.5rem 2rem;
}}
.auth-logo {{
    text-align: center;
    margin-bottom: 1.8rem;
}}
.auth-logo-name {{
    font-family: 'Sora', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.05em;
    color: {C['text']};
}}
.auth-logo-name .accent {{ color: {panel_accent}; }}
.auth-logo-sub {{
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: {C['text3']};
    font-weight: 600;
    margin-top: 3px;
}}
.auth-divider {{
    height: 1px;
    background: {C['border']};
    margin: 1.2rem 0;
}}
.auth-switch {{
    text-align: center;
    font-size: 0.76rem;
    color: {C['text3']};
    margin-top: 1.2rem;
}}
.auth-role-tag {{
    display: inline-block;
    background: {panel_accent}15;
    color: {panel_accent};
    border: 1px solid {panel_accent}30;
    border-radius: 5px;
    padding: 2px 9px;
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 1rem;
}}

/* ── FOOTER ─────────────────────────────────── */
.footer {{
    border-top: 1px solid {C['border']};
    margin-top: 3rem;
    padding-top: 1rem;
    text-align: center;
}}
.footer p {{
    font-size: 0.65rem;
    color: {C['text3']};
    line-height: 1.8;
}}

/* ── READONLY BADGE ─────────────────────────── */
.readonly-badge {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: {C['card2']};
    border: 1px solid {C['border']};
    border-radius: 5px;
    padding: 3px 10px;
    font-size: 0.65rem;
    color: {C['text3']};
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}
</style>""", unsafe_allow_html=True)