"""
SolarSat AI - Satellite Energy Intelligence
UI Components and Aerospace Mission-Control Styling
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, Dict, Any

# Primary color palette for aerospace dark UI
THEME_COLORS = {
    "bg_dark": "#070D1E",
    "card_bg": "#0D1B36",
    "card_border": "#1A2E56",
    "cyan": "#00F2FE",
    "blue": "#0072FF",
    "solar_orange": "#FF9900",
    "solar_amber": "#FFB703",
    "alert_red": "#FF4D4F",
    "success_green": "#06D6A0",
    "text_main": "#E2E8F0",
    "text_muted": "#8FA0BC",
    "grid_color": "#182848"
}

AEROSPACE_CSS = """
<style>
/* Main app dark aerospace aesthetic */
.stApp {
    background-color: #070D1E;
    color: #E2E8F0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0A1329;
    border-right: 1px solid #1A2E56;
}

/* Card panels */
.sat-card {
    background: linear-gradient(135deg, rgba(13, 27, 54, 0.9) 0%, rgba(10, 19, 41, 0.95) 100%);
    border: 1px solid #1A2E56;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.sat-card:hover {
    border-color: #00F2FE;
    box-shadow: 0 4px 24px rgba(0, 242, 254, 0.12);
}

.sat-kpi-val {
    font-size: 1.85rem;
    font-weight: 700;
    color: #00F2FE;
    font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, monospace;
    letter-spacing: -0.5px;
    line-height: 1.2;
}

.sat-kpi-val-orange {
    font-size: 1.85rem;
    font-weight: 700;
    color: #FF9900;
    font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, monospace;
    letter-spacing: -0.5px;
    line-height: 1.2;
}

.sat-kpi-val-green {
    font-size: 1.85rem;
    font-weight: 700;
    color: #06D6A0;
    font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, monospace;
    letter-spacing: -0.5px;
    line-height: 1.2;
}

.sat-kpi-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #8FA0BC;
    margin-bottom: 6px;
    font-weight: 600;
}

.sat-kpi-sub {
    font-size: 0.75rem;
    color: #64748B;
    margin-top: 6px;
}

/* Provenance badges */
.badge-measured {
    display: inline-block;
    background: rgba(6, 214, 160, 0.15);
    color: #06D6A0;
    border: 1px solid rgba(6, 214, 160, 0.4);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.badge-derived {
    display: inline-block;
    background: rgba(0, 242, 254, 0.15);
    color: #00F2FE;
    border: 1px solid rgba(0, 242, 254, 0.4);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.badge-simulated {
    display: inline-block;
    background: rgba(255, 153, 0, 0.15);
    color: #FFB703;
    border: 1px solid rgba(255, 153, 0, 0.4);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.badge-critical {
    display: inline-block;
    background: rgba(239, 71, 111, 0.2);
    color: #EF476F;
    border: 1px solid rgba(239, 71, 111, 0.5);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
}

/* Mission control header */
.mission-header {
    background: linear-gradient(90deg, #0A1329 0%, #102146 50%, #0A1329 100%);
    border-bottom: 2px solid #00F2FE;
    padding: 16px 24px;
    border-radius: 8px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(0, 242, 254, 0.08);
}

.mission-title {
    font-size: 1.7rem;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -0.5px;
    margin: 0;
}

.mission-sub {
    font-size: 0.88rem;
    color: #00F2FE;
    margin-top: 4px;
    font-weight: 500;
}

/* Tables and dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #1A2E56;
    border-radius: 8px;
    overflow: hidden;
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #070D1E;
}
::-webkit-scrollbar-thumb {
    background: #1A2E56;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #00F2FE;
}
</style>
"""


def apply_aerospace_styling():
    """Injects custom dark aerospace CSS into the current Streamlit app."""
    st.markdown(AEROSPACE_CSS, unsafe_allow_html=True)


def render_header(title: str, subtitle: str, badge: Optional[str] = "LIVE TELEMETRY"):
    """Renders a mission control header banner."""
    badge_html = f'<span class="badge-derived" style="margin-left: 12px; vertical-align: middle;">{badge}</span>' if badge else ""
    st.markdown(f"""
    <div class="mission-header">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <div class="mission-title">🛰️ {title} {badge_html}</div>
                <div class="mission-sub">{subtitle}</div>
            </div>
            <div style="text-align: right; color: #8FA0BC; font-size: 0.78rem;">
                <span style="color: #06D6A0;">●</span> SUBSYSTEM EPS/PV<br>
                <span>NEPALISAT-1 TELEMETRY</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi(
    label: str,
    value: str,
    unit: str = "",
    subtext: str = "",
    color_type: str = "cyan",
    provenance_badge: Optional[str] = None
):
    """
    Renders an aerospace KPI metric card with unit and provenance badge.
    color_type: 'cyan', 'orange', 'green', 'red'
    """
    val_class = "sat-kpi-val"
    if color_type == "orange":
        val_class = "sat-kpi-val-orange"
    elif color_type == "green":
        val_class = "sat-kpi-val-green"

    badge_html = ""
    if provenance_badge == "Measured":
        badge_html = '<span class="badge-measured">Measured</span>'
    elif provenance_badge == "Derived":
        badge_html = '<span class="badge-derived">Derived</span>'
    elif provenance_badge == "Simulated":
        badge_html = '<span class="badge-simulated">Simulated</span>'

    unit_span = f'<span style="font-size: 1.0rem; color: #8FA0BC; font-weight: 500; margin-left: 4px;">{unit}</span>' if unit else ""
    
    st.markdown(f"""
    <div class="sat-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div class="sat-kpi-label">{label}</div>
            {badge_html}
        </div>
        <div class="{val_class}">{value}{unit_span}</div>
        <div class="sat-kpi-sub">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)


def style_plotly_chart(fig: go.Figure, title: Optional[str] = None, height: int = 400) -> go.Figure:
    """
    Applies mission control dark aerospace styling to Plotly charts.
    """
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(13, 27, 54, 0.7)",
        plot_bgcolor="rgba(10, 19, 41, 0.8)",
        font=dict(color="#E2E8F0", family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"),
        margin=dict(l=45, r=25, t=45 if title else 20, b=40),
        height=height,
        title=dict(
            text=title if title else "",
            font=dict(size=14, color="#00F2FE", weight=600),
            x=0.01,
            xanchor="left"
        ) if title else None,
        xaxis=dict(
            gridcolor="#182848",
            zerolinecolor="#1A2E56",
            showline=True,
            linecolor="#1A2E56"
        ),
        yaxis=dict(
            gridcolor="#182848",
            zerolinecolor="#1A2E56",
            showline=True,
            linecolor="#1A2E56"
        ),
        legend=dict(
            bgcolor="rgba(10, 19, 41, 0.8)",
            bordercolor="#1A2E56",
            borderwidth=1,
            font=dict(size=11)
        )
    )
    return fig


def render_sidebar_status(df_len: int, filtered_len: int):
    """Renders persistent telemetry subsystem status in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div style="background: #0D1B36; border: 1px solid #1A2E56; border-radius: 8px; padding: 12px; margin-top: 10px;">
        <div style="font-size: 0.72rem; color: #8FA0BC; text-transform: uppercase; letter-spacing: 0.8px;">Subsystem Status</div>
        <div style="color: #06D6A0; font-size: 0.9rem; font-weight: 600; margin-top: 2px;">● EPS Telemetry Normal</div>
        <div style="font-size: 0.75rem; color: #8FA0BC; margin-top: 6px;">
            Total Records: <b>{df_len:,}</b><br>
            Active Filtered: <b>{filtered_len:,}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)
