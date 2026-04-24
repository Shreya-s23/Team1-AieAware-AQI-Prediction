"""
sidebar.py — Shared sidebar component for AirAware
"""

import streamlit as st
from utils import current_user, logout
from theme import get_theme


def render_sidebar(C, role="user", df=None, is_combined=False):
    """
    Render the sidebar for the application.
    role: "user" or "admin"
    is_combined: if True, add admin toggle for combined app
    Returns: uploaded, selected_poll, alert_thr
    """
    st.write("Sidebar rendering...")
    with st.sidebar:
        st.header("AirAware")
        st.subheader("Admin Dashboard" if role == "admin" else "Air Quality Intelligence")
        user = current_user()
        st.write(f"Signed in as {user.get('name','User')}")
        st.write("Admin" if role == "admin" else "Viewer")
        st.markdown("---")
        
        if is_combined and role == "admin":
            st.subheader("Interface")
            view_user = st.toggle("User View", value=st.session_state.admin_view_user)
            if view_user != st.session_state.admin_view_user:
                st.session_state.admin_view_user = view_user
                st.rerun()
            st.markdown("---")
        
        st.subheader("Training Dataset" if role == "admin" else "Load Dataset (View Only)")
        uploaded = st.file_uploader("Upload CSV" if role == "admin" else "Upload CSV to view", type=["csv"], label_visibility="collapsed")
        if role == "admin" and uploaded:
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
        st.write("Full Access Mode" if role == "admin" else "View Only Mode")
        
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
    
    st.markdown("""
<script>
setTimeout(function() {{
    const sidebar = document.querySelector('[data-testid="stSidebar"]');
    if (sidebar) {{
        sidebar.style.transform = 'translateX(0px)';
        sidebar.style.width = '200px';
        sidebar.setAttribute('aria-expanded', 'true');
    }}
    const expandBtn = document.querySelector('button[aria-label="Expand sidebar"]');
    if (expandBtn) {{
        expandBtn.click();
    }}
}}, 1000);
</script>""", unsafe_allow_html=True)
    
    return uploaded, selected_poll, alert_thr