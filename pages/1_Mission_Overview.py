"""
SolarSat AI - Satellite Energy Intelligence
Page 1: Mission Overview
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
from utils.prediction_model import load_persisted_model, train_and_evaluate_models

st.set_page_config(page_title="Mission Overview | SolarSat AI", page_icon="📊", layout="wide")
apply_aerospace_styling()

def main():
    df = load_satellite_data()

    # Sidebar Filters
    st.sidebar.markdown("### 🛰️ Mission Filters")
    all_sessions = list(df["Recording_Session"].unique())
    selected_sessions = st.sidebar.multiselect(
        "Recording Sessions:",
        options=all_sessions,
        default=all_sessions,
        help="Select one or more downlinked satellite recording passes"
    )

    all_dates = sorted(list(df["Recording_Date"].unique()))
    selected_dates = st.sidebar.date_input(
        "Recording Dates:",
        value=(all_dates[0], all_dates[-1]),
        min_value=all_dates[0],
        max_value=all_dates[-1]
    )

    # Telemetry range slider
    max_idx = len(df)
    idx_range = st.sidebar.slider(
        "Telemetry Record Range:",
        min_value=0,
        max_value=max_idx,
        value=(0, max_idx),
        help="Slice by sequential downlink frame index"
    )

    filtered_df = filter_telemetry_data(
        df,
        selected_sessions=selected_sessions,
        date_range=selected_dates if isinstance(selected_dates, tuple) and len(selected_dates) == 2 else None,
        index_range=idx_range
    )

    render_sidebar_status(len(df), len(filtered_df))

    if filtered_df.empty:
        st.warning("⚠️ No telemetry records match the current filter selection. Please broaden your filters.")
        st.stop()

    render_header(
        title="Mission Telemetry Overview",
        subtitle="Real-Time Photovoltaic & Electrical Power Subsystem (EPS) Telemetry Monitoring",
        badge="ONLINE"
    )

    # Calculate KPIs
    total_records = len(filtered_df)
    avg_power = filtered_df["solar_array_power_W"].mean()
    latest_power = filtered_df["solar_array_power_W"].iloc[-1]
    avg_temp = filtered_df["mean_panel_temperature_C"].mean()
    latest_vbat = filtered_df["Vbat (V)"].iloc[-1]
    latest_soc = filtered_df["battery_soc_percent_simulated"].iloc[-1] if "battery_soc_percent_simulated" in filtered_df.columns else np.nan
    avg_drop = filtered_df["session_relative_power_drop_percent"].mean() if "session_relative_power_drop_percent" in filtered_df.columns else np.nan

    # Model prediction KPI
    rf_model, features = load_persisted_model("random_forest")
    if rf_model is None:
        # Pre-train baseline if not yet saved
        results = train_and_evaluate_models(df, split_strategy="session")
        rf_model = results["rf_pipeline"]
        features = results["features"]

    # Predict for latest record
    try:
        latest_row = filtered_df[features].iloc[[-1]]
        predicted_power = max(0.0, float(rf_model.predict(latest_row)[0]))
    except Exception:
        predicted_power = avg_power

    # KPI Row 1
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi("Total Filtered Records", f"{total_records:,}", "", f"{total_records/len(df)*100:.1f}% of mission dataset", "cyan", "Measured")
    with k2:
        render_kpi("Average Solar Power", f"{avg_power:.3f}", "W", "Sum across 5 panel faces", "cyan", "Derived")
    with k3:
        render_kpi("Latest Solar Power", f"{latest_power:.3f}", "W", f"At index {filtered_df.index[-1]}", "green" if latest_power > 0.5 else "orange", "Derived")
    with k4:
        render_kpi("Mean Panel Temp", f"{avg_temp:.2f}", "°C", "Averaged across active faces", "cyan", "Derived")

    # KPI Row 2
    k5, k6, k7, k8 = st.columns(4)
    with k5:
        render_kpi("Latest Bus Voltage", f"{latest_vbat:.3f}", "V", "Li-ion battery bus level", "green" if latest_vbat >= 3.85 else "orange", "Measured")
    with k6:
        render_kpi("Battery State of Charge", f"{latest_soc:.1f}", "%", "Simulated energy balance proxy", "orange", "Simulated")
    with k7:
        render_kpi("Avg Power Drop", f"{avg_drop:.1f}", "%", "Drop below 95th % session ref", "orange", "Derived")
    with k8:
        render_kpi("AI Predicted Power", f"{predicted_power:.3f}", "W", "Random Forest model inference", "green", "Derived")

    st.markdown("<br>", unsafe_allow_html=True)

    # Telemetry Charts
    c_tab1, c_tab2 = st.tabs(["📈 Mission Time Series", "📊 Physical & Session Distributions"])

    with c_tab1:
        # 1. Solar array power over time
        st.markdown("#### 1. Solar Array Power Over Time")
        fig_power = go.Figure()
        fig_power.add_trace(go.Scatter(
            x=filtered_df["timestamp"] if "timestamp" in filtered_df.columns else filtered_df.index,
            y=filtered_df["solar_array_power_W"],
            mode="lines",
            name="Solar Array Power (W)",
            line=dict(color=THEME_COLORS["cyan"], width=1.8)
        ))
        if "session_reference_power_W" in filtered_df.columns:
            fig_power.add_trace(go.Scatter(
                x=filtered_df["timestamp"] if "timestamp" in filtered_df.columns else filtered_df.index,
                y=filtered_df["session_reference_power_W"],
                mode="lines",
                name="Session Reference (95th %)",
                line=dict(color=THEME_COLORS["solar_amber"], width=1.5, dash="dash")
            ))
        fig_power.update_xaxes(title_text="Downlink Timestamp / Time")
        fig_power.update_yaxes(title_text="Power Output (Watts)")
        style_plotly_chart(fig_power, title="Total Solar Array Generation vs Reference Baseline (W)", height=380)
        st.plotly_chart(fig_power, use_container_width=True)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            # 2. Solar panel temperature over time
            st.markdown("#### 2. Panel Temperatures Over Time")
            fig_temp = go.Figure()
            # Mean temperature
            fig_temp.add_trace(go.Scatter(
                x=filtered_df["timestamp"] if "timestamp" in filtered_df.columns else filtered_df.index,
                y=filtered_df["mean_panel_temperature_C"],
                mode="lines",
                name="Mean Panel Temp",
                line=dict(color="#FF9900", width=2)
            ))
            # Individual faces
            temp_faces = [("Tpy (°C)", "#00F2FE"), ("Tpx (°C)", "#06D6A0"), ("Tmz (°C)", "#8338EC"), ("Tmx (°C)", "#3A86FF"), ("Tpz (°C)", "#FF006E")]
            for face_col, face_color in temp_faces:
                if face_col in filtered_df.columns:
                    fig_temp.add_trace(go.Scatter(
                        x=filtered_df["timestamp"] if "timestamp" in filtered_df.columns else filtered_df.index,
                        y=filtered_df[face_col],
                        mode="lines",
                        name=face_col.split()[0],
                        opacity=0.6,
                        line=dict(width=1, dash="dot", color=face_color)
                    ))
            fig_temp.update_xaxes(title_text="Timestamp")
            fig_temp.update_yaxes(title_text="Temperature (°C)")
            style_plotly_chart(fig_temp, title="Thermal Gradient Across Solar Panels (°C)", height=350)
            st.plotly_chart(fig_temp, use_container_width=True)

        with col_c2:
            # 3. Battery voltage over time
            st.markdown("#### 3. Battery Bus Voltage Over Time")
            fig_vbat = go.Figure()
            fig_vbat.add_trace(go.Scatter(
                x=filtered_df["timestamp"] if "timestamp" in filtered_df.columns else filtered_df.index,
                y=filtered_df["Vbat (V)"],
                mode="lines",
                name="Vbat (V)",
                line=dict(color=THEME_COLORS["success_green"], width=1.8)
            ))
            # Operational limit lines
            fig_vbat.add_hline(y=4.15, line_dash="dash", line_color=THEME_COLORS["alert_red"], annotation_text="Overcharge (4.15V)")
            fig_vbat.add_hline(y=3.80, line_dash="dash", line_color=THEME_COLORS["solar_orange"], annotation_text="Low Bus (3.80V)")
            fig_vbat.update_xaxes(title_text="Timestamp")
            fig_vbat.update_yaxes(title_text="Bus Voltage (Volts)")
            style_plotly_chart(fig_vbat, title="EPS Battery Bus Voltage (V)", height=350)
            st.plotly_chart(fig_vbat, use_container_width=True)

    with c_tab2:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            # 4. Solar power versus panel temperature
            st.markdown("#### 4. Solar Power vs. Panel Temperature")
            fig_p_vs_t = px.scatter(
                filtered_df,
                x="mean_panel_temperature_C",
                y="solar_array_power_W",
                color="Recording_Session",
                trendline="ols",
                labels={
                    "mean_panel_temperature_C": "Mean Panel Temperature (°C)",
                    "solar_array_power_W": "Solar Array Power (W)",
                    "Recording_Session": "Session"
                },
                color_discrete_sequence=["#00F2FE", "#FF9900", "#06D6A0"]
            )
            style_plotly_chart(fig_p_vs_t, title="Power Output vs. Thermal Environment", height=380)
            st.plotly_chart(fig_p_vs_t, use_container_width=True)

        with col_d2:
            # 5. Solar power distribution
            st.markdown("#### 5. Solar Power Output Distribution")
            fig_dist = px.histogram(
                filtered_df,
                x="solar_array_power_W",
                color="Recording_Session",
                nbins=35,
                marginal="box",
                barmode="overlay",
                opacity=0.75,
                labels={"solar_array_power_W": "Solar Array Power (W)"},
                color_discrete_sequence=["#00F2FE", "#FF9900", "#06D6A0"]
            )
            style_plotly_chart(fig_dist, title="Solar Array Generation Distribution (W)", height=380)
            st.plotly_chart(fig_dist, use_container_width=True)

        # 6. Power drop percentage by recording session
        st.markdown("#### 6. Relative Power Drop Percentage by Recording Session")
        fig_drop = px.box(
            filtered_df,
            x="Recording_Session",
            y="session_relative_power_drop_percent",
            color="Recording_Session",
            points="all",
            labels={
                "Recording_Session": "Downlink Session",
                "session_relative_power_drop_percent": "Relative Power Drop (%)"
            },
            color_discrete_sequence=["#00F2FE", "#FF9900", "#06D6A0"]
        )
        style_plotly_chart(fig_drop, title="Session Power Drop (%) Relative to 95th Percentile Reference", height=360)
        st.plotly_chart(fig_drop, use_container_width=True)

if __name__ == "__main__":
    main()
