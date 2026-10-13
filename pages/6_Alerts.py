"""
SolarSat AI - Satellite Energy Intelligence
Page 6: Alerts and Anomaly Detection
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
from utils.anomaly_detection import detect_rule_based_anomalies, detect_isolation_forest_anomalies

st.set_page_config(page_title="Alerts & Anomalies | SolarSat AI", page_icon="🚨", layout="wide")
apply_aerospace_styling()

def main():
    df = load_satellite_data()

    # Sidebar Anomaly Configuration
    st.sidebar.markdown("### 🎛️ Anomaly Bounds")
    vbat_min = st.sidebar.slider("Min Battery Voltage (V):", 3.65, 3.90, 3.80, 0.01)
    vbat_max = st.sidebar.slider("Max Battery Voltage (V):", 4.10, 4.25, 4.15, 0.01)
    panel_temp_max = st.sidebar.slider("Max Panel Temp (°C):", 15.0, 35.0, 25.0, 1.0)
    power_delta_max = st.sidebar.slider("Max Power Step Delta (W):", 0.5, 3.0, 1.5, 0.1)

    iso_contamination = st.sidebar.slider("Isolation Forest Outlier Rate (%):", 0.5, 5.0, 2.0, 0.5) / 100.0

    all_sessions = list(df["Recording_Session"].unique())
    selected_sessions = st.sidebar.multiselect("Filter Sessions:", options=all_sessions, default=all_sessions)

    filtered_df = filter_telemetry_data(df, selected_sessions=selected_sessions)
    render_sidebar_status(len(df), len(filtered_df))

    if filtered_df.empty:
        st.warning("Please select at least one recording session.")
        st.stop()

    render_header(
        title="Mission Telemetry Alerts & Anomaly Detection",
        subtitle="Automated EPS Fault Detection via Physical Flight Rules & Multidimensional Isolation Forest",
        badge="ALERT MONITOR ACTIVE"
    )

    # Run Detection
    with st.spinner("Executing rule-based & ML anomaly detection pipelines..."):
        rule_alerts = detect_rule_based_anomalies(
            filtered_df,
            vbat_min=vbat_min,
            vbat_max=vbat_max,
            panel_temp_max=panel_temp_max,
            power_drop_rate_max=power_delta_max
        )
        enriched_df, ml_alerts = detect_isolation_forest_anomalies(
            filtered_df,
            contamination=iso_contamination
        )

    # Combined summary
    total_rule_alerts = len(rule_alerts)
    critical_alerts = (rule_alerts["Severity"] == "CRITICAL").sum() if not rule_alerts.empty else 0
    warning_alerts = (rule_alerts["Severity"] == "WARNING").sum() if not rule_alerts.empty else 0
    total_ml_anomalies = int(enriched_df["is_ml_anomaly"].sum())

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        render_kpi("Critical Rule Alerts", f"{critical_alerts:,}", "", "Immediate mission intervention required", "red" if critical_alerts > 0 else "green", "Measured")
    with a2:
        render_kpi("Warning Alerts", f"{warning_alerts:,}", "", "Telemetry out-of-nominal advisory", "orange" if warning_alerts > 0 else "green", "Measured")
    with a3:
        render_kpi("Isolation Forest Anomalies", f"{total_ml_anomalies:,}", "", f"Top {iso_contamination*100:.1f}% multidimensional outliers", "orange", "Derived")
    with a4:
        render_kpi("Telemetry Health Score", f"{(1 - (total_rule_alerts + total_ml_anomalies)/(len(filtered_df)*2))*100:.1f}", "%", "Combined EPS operational reliability", "green", "Derived")

    st.markdown("---")

    # Visualizations
    tab_map, tab_rules, tab_ml = st.tabs(["🗺️ Anomaly Timeline Mapping", "📋 Physical Rule Alerts Log", "🤖 Isolation Forest Outlier Analysis"])

    ts_axis = enriched_df["timestamp"] if "timestamp" in enriched_df.columns else enriched_df.index

    with tab_map:
        st.markdown("#### 1. Solar Array Power Timeline with Flagged Outliers")
        fig_anom_p = go.Figure()

        # Normal trace
        fig_anom_p.add_trace(go.Scatter(
            x=ts_axis[~enriched_df["is_ml_anomaly"]],
            y=enriched_df.loc[~enriched_df["is_ml_anomaly"], "solar_array_power_W"],
            mode="lines",
            name="Nominal Telemetry",
            line=dict(color=THEME_COLORS["cyan"], width=1.5)
        ))

        # Outliers highlighted
        if total_ml_anomalies > 0:
            fig_anom_p.add_trace(go.Scatter(
                x=ts_axis[enriched_df["is_ml_anomaly"]],
                y=enriched_df.loc[enriched_df["is_ml_anomaly"], "solar_array_power_W"],
                mode="markers",
                name="Flagged ML Outliers",
                marker=dict(color=THEME_COLORS["alert_red"], size=7, symbol="diamond")
            ))

        fig_anom_p.update_xaxes(title_text="Timestamp")
        fig_anom_p.update_yaxes(title_text="Solar Array Power (W)")
        style_plotly_chart(fig_anom_p, title="Solar Power Trajectory Highlighting Telemetry Anomalies", height=380)
        st.plotly_chart(fig_anom_p, use_container_width=True)

        # Battery Voltage Timeline with Anomalies
        st.markdown("#### 2. Battery Voltage Bus Health & Alert Bands")
        fig_anom_v = go.Figure()
        fig_anom_v.add_trace(go.Scatter(
            x=ts_axis,
            y=enriched_df["Vbat (V)"],
            mode="lines",
            name="Battery Voltage",
            line=dict(color=THEME_COLORS["success_green"], width=1.5)
        ))
        fig_anom_v.add_hline(y=vbat_max, line_dash="dash", line_color=THEME_COLORS["alert_red"], annotation_text=f"Overvoltage Limit ({vbat_max}V)")
        fig_anom_v.add_hline(y=vbat_min, line_dash="dash", line_color=THEME_COLORS["alert_red"], annotation_text=f"Undervoltage Limit ({vbat_min}V)")
        fig_anom_v.update_xaxes(title_text="Timestamp")
        fig_anom_v.update_yaxes(title_text="Battery Voltage (V)")
        style_plotly_chart(fig_anom_v, title="EPS Voltage Bus Outlier Monitoring", height=350)
        st.plotly_chart(fig_anom_v, use_container_width=True)

    with tab_rules:
        st.markdown("#### 📋 Deterministic Physical Rule Alerts Log")
        if not rule_alerts.empty:
            # Add severity filter
            sev_filter = st.multiselect("Filter by Severity:", ["CRITICAL", "WARNING", "INFO"], default=["CRITICAL", "WARNING", "INFO"])
            filtered_rule_alerts = rule_alerts[rule_alerts["Severity"].isin(sev_filter)]
            st.dataframe(filtered_rule_alerts, use_container_width=True, hide_index=True)
        else:
            st.success("✅ Zero physical rule violations detected under current telemetry threshold parameters.")

    with tab_ml:
        st.markdown("#### 🤖 Isolation Forest Multidimensional Outlier Analysis")
        st.caption("Outlier score distribution evaluating coupled panel voltages, thermal readings, and battery telemetry:")

        fig_score = px.histogram(
            enriched_df,
            x="anomaly_score",
            color="is_ml_anomaly",
            labels={"anomaly_score": "Isolation Forest Anomaly Score", "is_ml_anomaly": "Is Outlier"},
            color_discrete_map={True: THEME_COLORS["alert_red"], False: THEME_COLORS["cyan"]},
            nbins=40
        )
        style_plotly_chart(fig_score, title="Telemetry Multidimensional Deviation Score Distribution", height=340)
        st.plotly_chart(fig_score, use_container_width=True)

        if not ml_alerts.empty:
            st.markdown("##### 🔬 Machine Learning Flagged Outlier Records")
            st.dataframe(ml_alerts.head(100), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
