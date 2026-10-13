"""
SolarSat AI - Satellite Energy Intelligence
Page 2: Solar Degradation Analysis
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from utils.ui_components import (
    apply_aerospace_styling, render_header, render_kpi,
    style_plotly_chart, render_sidebar_status, THEME_COLORS
)
from utils.data_loader import load_satellite_data, filter_telemetry_data
from utils.degradation_analysis import (
    compute_session_summary, compute_panel_power_breakdown,
    compute_temperature_derating, identify_low_power_episodes
)

st.set_page_config(page_title="Degradation Analysis | SolarSat AI", page_icon="📉", layout="wide")
apply_aerospace_styling()

def main():
    df = load_satellite_data()

    # Sidebar session filter
    st.sidebar.markdown("### 🛰️ Degradation Scope")
    sessions = list(df["Recording_Session"].unique())
    selected_sessions = st.sidebar.multiselect(
        "Evaluate Sessions:",
        options=sessions,
        default=sessions
    )

    drop_threshold = st.sidebar.slider(
        "High Drop Filter Threshold (%):",
        min_value=50.0,
        max_value=99.0,
        value=80.0,
        step=5.0,
        help="Flag telemetry points dropping more than X% below session 95th percentile"
    )

    filtered_df = filter_telemetry_data(df, selected_sessions=selected_sessions)
    render_sidebar_status(len(df), len(filtered_df))

    if filtered_df.empty:
        st.warning("Please select at least one recording session.")
        st.stop()

    render_header(
        title="Solar Power Degradation & Reduction Analysis",
        subtitle="Empirical Evaluation of Session Baselines, Panel Derating, and Relative Power Drops",
        badge="DEGRADATION CONSOLE"
    )

    # Scientific Integrity Banner
    st.info(
        "ℹ️ **Scientific Integrity Note:** A reduction in power within or across recording sessions represents an "
        "*observed power reduction indicator*, influenced by orbital geometry, solar incidence angle (beta angle), "
        "and thermal conditions. The available three discrete downlink sessions do not constitute sufficient continuous "
        "in-orbit exposure data to claim definitive long-term crystalline silicon degradation or radiation fluence damage."
    )

    # Session Summary Table
    st.markdown("### 📋 Multi-Session Performance Comparison")
    session_summary = compute_session_summary(filtered_df)
    st.dataframe(session_summary, use_container_width=True, hide_index=True)

    # Quick Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    overall_mean_drop = filtered_df["session_relative_power_drop_percent"].mean()
    highest_drop_session = session_summary.sort_values("Avg Relative Power Drop (%)", ascending=False).iloc[0]["Session"]
    derating_slope, corr, r2 = compute_temperature_derating(filtered_df)

    with m1:
        render_kpi("Overall Mean Power Drop", f"{overall_mean_drop:.2f}", "%", "Relative to 95th % reference", "orange", "Derived")
    with m2:
        render_kpi("Highest Drop Session", highest_drop_session, "", "Greatest relative drop observed", "orange", "Derived")
    with m3:
        render_kpi("Thermal Derating Slope", f"{derating_slope:.4f}", "W/°C", f"Linear fit (r = {corr:.2f})", "cyan", "Derived")
    with m4:
        render_kpi("Thermal Fit R²", f"{r2:.3f}", "", "Variance explained by temp", "cyan", "Derived")

    st.markdown("---")

    # Interactive Plots
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("#### 1. Session Reference Power (95th %) Comparison")
        fig_ref = px.bar(
            session_summary,
            x="Session",
            y="Session Ref Power (95th %) (W)",
            color="Session",
            text="Session Ref Power (95th %) (W)",
            labels={"Session Ref Power (95th %) (W)": "Reference Power (W)"},
            color_discrete_sequence=["#00F2FE", "#FF9900", "#06D6A0"]
        )
        fig_ref.update_traces(texttemplate='%{text:.3f} W', textposition='outside')
        style_plotly_chart(fig_ref, title="Peak Attainable Power Across Orbital Passes (W)", height=380)
        st.plotly_chart(fig_ref, use_container_width=True)

    with col_g2:
        st.markdown("#### 2. Relative Power Drop Distribution")
        fig_drops = px.violin(
            filtered_df,
            x="Recording_Session",
            y="session_relative_power_drop_percent",
            color="Recording_Session",
            box=True,
            points=False,
            labels={
                "Recording_Session": "Session",
                "session_relative_power_drop_percent": "Power Drop (%)"
            },
            color_discrete_sequence=["#00F2FE", "#FF9900", "#06D6A0"]
        )
        style_plotly_chart(fig_drops, title="Observed Power Reduction Spread per Pass (%)", height=380)
        st.plotly_chart(fig_drops, use_container_width=True)

    st.markdown("---")

    col_p1, col_p2 = st.columns([1, 1])

    with col_p1:
        st.markdown("#### 3. Panel-by-Panel Solar Contribution Breakdown")
        panel_breakdown = compute_panel_power_breakdown(filtered_df)
        st.dataframe(panel_breakdown, use_container_width=True)

        # Donut Chart of Power Shares
        fig_donut = px.pie(
            names=panel_breakdown.index,
            values=panel_breakdown["Total Energy Proxy (W·s)"],
            hole=0.45,
            color_discrete_sequence=["#00F2FE", "#FF9900", "#06D6A0", "#9D4EDD", "#3A86FF"]
        )
        style_plotly_chart(fig_donut, title="Solar Energy Generation Share by Panel Face", height=320)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_p2:
        st.markdown("#### 4. Thermal Derating Profile")
        fig_thermal = px.scatter(
            filtered_df,
            x="mean_panel_temperature_C",
            y="solar_array_power_W",
            color="Recording_Session",
            trendline="ols",
            labels={
                "mean_panel_temperature_C": "Mean Panel Temperature (°C)",
                "solar_array_power_W": "Solar Array Power (W)"
            },
            color_discrete_sequence=["#00F2FE", "#FF9900", "#06D6A0"]
        )
        style_plotly_chart(
            fig_thermal,
            title=f"Solar Power vs Temperature (Slope: {derating_slope:.4f} W/°C)",
            height=430
        )
        st.plotly_chart(fig_thermal, use_container_width=True)

    st.markdown("---")

    # High Drop Episodes Table
    st.markdown(f"### ⚠️ Telemetry Records with Severe Power Drop (≥ {drop_threshold:.0f}%)")
    low_power_episodes = identify_low_power_episodes(filtered_df, threshold_percent=drop_threshold)
    if not low_power_episodes.empty:
        st.write(f"Detected **{len(low_power_episodes):,}** telemetry frames exceeding the {drop_threshold:.0f}% reduction threshold.")
        st.dataframe(low_power_episodes.head(100), use_container_width=True, hide_index=True)
    else:
        st.success(f"No records observed with relative power drop exceeding {drop_threshold:.0f}%.")

if __name__ == "__main__":
    main()
