"""
SolarSat AI - Satellite Energy Intelligence
Page 4: Battery and Energy Monitoring
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from utils.ui_components import (
    apply_aerospace_styling, render_header, render_kpi,
    style_plotly_chart, render_sidebar_status, THEME_COLORS
)
from utils.data_loader import load_satellite_data, filter_telemetry_data

st.set_page_config(page_title="Battery & Energy | SolarSat AI", page_icon="🔋", layout="wide")
apply_aerospace_styling()

def main():
    df = load_satellite_data()

    # Sidebar threshold config & filtering
    st.sidebar.markdown("### ⚠️ Battery Threshold Controls")
    vbat_min_thresh = st.sidebar.slider("Undervoltage Alarm (V):", 3.60, 3.90, 3.80, 0.01)
    vbat_max_thresh = st.sidebar.slider("Overvoltage Alarm (V):", 4.05, 4.25, 4.15, 0.01)
    tbatt_max_thresh = st.sidebar.slider("Max Battery Temp (°C):", 15.0, 35.0, 25.0, 0.5)

    all_sessions = list(df["Recording_Session"].unique())
    selected_sessions = st.sidebar.multiselect(
        "Filter Sessions:",
        options=all_sessions,
        default=all_sessions
    )

    filtered_df = filter_telemetry_data(df, selected_sessions=selected_sessions)
    render_sidebar_status(len(df), len(filtered_df))

    if filtered_df.empty:
        st.warning("Please select at least one recording session.")
        st.stop()

    render_header(
        title="EPS Battery & Energy Subsystem Monitoring",
        subtitle="Electrical Bus Health, Thermal Regimes, and Orbit Energy Balance",
        badge="BUS TELEMETRY"
    )

    # Provenance Disclaimer Banner
    st.markdown("""
    <div style="background: rgba(255, 153, 0, 0.1); border-left: 4px solid #FF9900; padding: 10px 16px; border-radius: 4px; margin-bottom: 20px; font-size: 0.85rem;">
        <b>⚠️ Telemetry Provenance Notice:</b><br>
        <code>Vbat (V)</code>, <code>Ibatt(mA)</code>, and <code>Tbatt (℃)</code> represent direct <b>Measured Telemetry</b> downlinked from the spacecraft.<br>
        <code>battery_soc_percent_simulated</code> and <code>available_energy_Wh_simulated</code> represent <b>Simulated Scenario Assumptions</b> (assuming a nominal 10 Wh pack) and are not direct onboard Coulomb-counting telemetry.
    </div>
    """, unsafe_allow_html=True)

    # Battery KPIs
    latest_vbat = filtered_df["Vbat (V)"].iloc[-1]
    mean_vbat = filtered_df["Vbat (V)"].mean()
    latest_ibatt = filtered_df["Ibatt(mA)"].iloc[-1]
    latest_tbatt = filtered_df["Tbatt (℃)"].iloc[-1]
    latest_soc = filtered_df["battery_soc_percent_simulated"].iloc[-1]
    latest_avail_energy = filtered_df["available_energy_Wh_simulated"].iloc[-1]
    total_net_energy = filtered_df["energy_surplus_or_deficit_Wh_simulated"].sum()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        v_color = "green" if vbat_min_thresh <= latest_vbat <= vbat_max_thresh else "orange"
        render_kpi("Battery Voltage", f"{latest_vbat:.3f}", "V", f"Range: {filtered_df['Vbat (V)'].min():.2f}V - {filtered_df['Vbat (V)'].max():.2f}V", v_color, "Measured")
    with k2:
        render_kpi("Battery Current", f"{latest_ibatt:.1f}", "mA", f"Mean: {filtered_df['Ibatt(mA)'].mean():.1f} mA", "cyan", "Measured")
    with k3:
        t_color = "green" if latest_tbatt <= tbatt_max_thresh else "orange"
        render_kpi("Battery Temperature", f"{latest_tbatt:.2f}", "°C", f"Limit: {tbatt_max_thresh:.1f} °C", t_color, "Measured")
    with k4:
        render_kpi("Simulated SOC", f"{latest_soc:.1f}", "%", "Cumulative energy proxy (10Wh pack)", "orange", "Simulated")

    k5, k6, k7, k8 = st.columns(4)
    with k5:
        render_kpi("Available Energy", f"{latest_avail_energy:.2f}", "Wh", "Assumed pack capacity × SOC", "orange", "Simulated")
    with k6:
        render_kpi("Cumulative Net Energy", f"{total_net_energy:+.3f}", "Wh", "Integral of generation minus demand", "cyan", "Simulated")
    with k7:
        vbat_violations = (filtered_df["Vbat (V)"] < vbat_min_thresh) | (filtered_df["Vbat (V)"] > vbat_max_thresh)
        render_kpi("Voltage Threshold Flags", f"{vbat_violations.sum():,}", "", f"Out of {len(filtered_df):,} samples", "orange" if vbat_violations.sum() > 0 else "green", "Measured")
    with k8:
        temp_violations = filtered_df["Tbatt (℃)"] > tbatt_max_thresh
        render_kpi("Temp Threshold Flags", f"{temp_violations.sum():,}", "", f"Exceeding {tbatt_max_thresh}°C", "red" if temp_violations.sum() > 0 else "green", "Measured")

    st.markdown("---")

    # Time Series Charts
    tab1, tab2 = st.tabs(["⚡ Voltage & Current Tracking", "⚖️ Power Balance & Energy Surplus/Deficit"])

    with tab1:
        # Dual-axis chart: Battery Voltage and Current
        st.markdown("#### 1. EPS Battery Voltage & Current Telemetry")
        fig_dual = make_subplots(specs=[[{"secondary_y": True}]])

        ts_axis = filtered_df["timestamp"] if "timestamp" in filtered_df.columns else filtered_df.index

        fig_dual.add_trace(
            go.Scatter(x=ts_axis, y=filtered_df["Vbat (V)"], name="Battery Voltage (V)", line=dict(color=THEME_COLORS["cyan"], width=1.8)),
            secondary_y=False
        )
        fig_dual.add_trace(
            go.Scatter(x=ts_axis, y=filtered_df["Ibatt(mA)"], name="Battery Current (mA)", line=dict(color=THEME_COLORS["solar_orange"], width=1.2, dash="dot")),
            secondary_y=True
        )

        fig_dual.add_hline(y=vbat_max_thresh, line_dash="dash", line_color=THEME_COLORS["alert_red"], annotation_text=f"Max Limit ({vbat_max_thresh}V)", secondary_y=False)
        fig_dual.add_hline(y=vbat_min_thresh, line_dash="dash", line_color=THEME_COLORS["alert_red"], annotation_text=f"Min Limit ({vbat_min_thresh}V)", secondary_y=False)

        fig_dual.update_xaxes(title_text="Timestamp")
        fig_dual.update_yaxes(title_text="Battery Voltage (V)", secondary_y=False, gridcolor="#182848")
        fig_dual.update_yaxes(title_text="Battery Current (mA)", secondary_y=True, gridcolor="#182848")
        style_plotly_chart(fig_dual, title="Battery Bus Voltage (V) & Current Draw (mA)", height=400)
        st.plotly_chart(fig_dual, use_container_width=True)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.markdown("#### 2. Battery Temperature Telemetry")
            fig_tbatt = px.line(
                filtered_df,
                x=ts_axis,
                y="Tbatt (℃)",
                color="Recording_Session",
                labels={"Tbatt (℃)": "Battery Temp (°C)"},
                color_discrete_sequence=["#00F2FE", "#FF9900", "#06D6A0"]
            )
            fig_tbatt.add_hline(y=tbatt_max_thresh, line_dash="dash", line_color=THEME_COLORS["alert_red"], annotation_text=f"Max Temp ({tbatt_max_thresh}°C)")
            style_plotly_chart(fig_tbatt, title="Internal Battery Pack Temperature Profile (°C)", height=350)
            st.plotly_chart(fig_tbatt, use_container_width=True)

        with col_b2:
            st.markdown("#### 3. Battery Voltage vs. Current Correlation")
            fig_v_i = px.scatter(
                filtered_df,
                x="Ibatt(mA)",
                y="Vbat (V)",
                color="Recording_Session",
                labels={"Ibatt(mA)": "Current (mA)", "Vbat (V)": "Voltage (V)"},
                color_discrete_sequence=["#00F2FE", "#FF9900", "#06D6A0"]
            )
            style_plotly_chart(fig_v_i, title="Internal Impedance Characterization (V vs I)", height=350)
            st.plotly_chart(fig_v_i, use_container_width=True)

    with tab2:
        # Solar Generation vs Simulated Demand
        st.markdown("#### 4. Orbit Power Balance: Solar Generation vs. Simulated Demand")
        fig_bal = go.Figure()
        fig_bal.add_trace(go.Scatter(
            x=ts_axis,
            y=filtered_df["solar_array_power_W"],
            mode="lines",
            name="Solar Generation (W)",
            line=dict(color=THEME_COLORS["cyan"], width=2)
        ))
        if "total_power_demand_W_simulated" in filtered_df.columns:
            fig_bal.add_trace(go.Scatter(
                x=ts_axis,
                y=filtered_df["total_power_demand_W_simulated"],
                mode="lines",
                name="Simulated Demand (W)",
                line=dict(color=THEME_COLORS["solar_orange"], width=1.5, dash="dash")
            ))
        fig_bal.update_xaxes(title_text="Timestamp")
        fig_bal.update_yaxes(title_text="Power (Watts)")
        style_plotly_chart(fig_bal, title="Instantaneous Power Balance (Generation vs Demand)", height=380)
        st.plotly_chart(fig_bal, use_container_width=True)

        # Simulated Energy Surplus/Deficit
        st.markdown("#### 5. Sample-Interval Energy Surplus (+) or Deficit (-) [Wh]")
        fig_surp = px.area(
            filtered_df,
            x=ts_axis,
            y="energy_surplus_or_deficit_Wh_simulated",
            color="Recording_Session",
            labels={"energy_surplus_or_deficit_Wh_simulated": "Energy Delta (Wh)"},
            color_discrete_sequence=["#00F2FE", "#FF9900", "#06D6A0"]
        )
        fig_surp.add_hline(y=0.0, line_color="#FFFFFF", line_width=1)
        style_plotly_chart(fig_surp, title="Sample-to-Sample Orbital Energy Net Flow (Wh)", height=350)
        st.plotly_chart(fig_surp, use_container_width=True)

if __name__ == "__main__":
    main()
