"""
SolarSat AI - Satellite Energy Intelligence
Page 5: Energy Optimization Simulator
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
from utils.data_loader import load_satellite_data
from utils.energy_optimizer import run_power_optimization, compare_precomputed_dataset_optimization

st.set_page_config(page_title="Energy Optimization | SolarSat AI", page_icon="⚡", layout="wide")
apply_aerospace_styling()

def main():
    df = load_satellite_data()
    render_sidebar_status(len(df), len(df))

    render_header(
        title="Orbital Energy Allocation & Optimization Simulator",
        subtitle="Priority-Driven Power Dispatch Engine Protecting Battery Reserves and Mission-Critical Loads",
        badge="SIMULATOR ACTIVE"
    )

    st.info(
        "ℹ️ **Simulation Integrity Note:** This simulator models power dispatch policies for space mission planning. "
        "User simulation results are generated interactively in real time and are kept strictly distinct from the dataset's "
        "precomputed scenario columns."
    )

    # Simulator Controls Panel
    st.markdown("### 🎛️ Dynamic Energy Allocation Controls")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. Available Stored Energy**")
        avail_energy = st.slider(
            "Available Stored Energy (Wh):",
            min_value=0.5,
            max_value=20.0,
            value=float(df["available_energy_Wh_simulated"].mean()),
            step=0.25,
            help="Total current energy capacity present in the satellite battery pack"
        )
        bat_reserve = st.slider(
            "Battery Safety Reserve (%):",
            min_value=10.0,
            max_value=40.0,
            value=20.0,
            step=5.0,
            help="Minimum energy margin strictly withheld from active subsystem consumption"
        )

    with c2:
        st.markdown("**2. Subsystem Power Demands**")
        crit_demand = st.slider(
            "Critical Load Demand (W):",
            min_value=0.1,
            max_value=3.0,
            value=float(df["critical_load_W_simulated"].mean()),
            step=0.05,
            help="Power required by essential flight hardware (OBC, ADCS magnetorquers, UHF beacon)"
        )
        opt_demand = st.slider(
            "Optional Load Demand (W):",
            min_value=0.0,
            max_value=2.0,
            value=float(df["optional_load_W_simulated"].mean()),
            step=0.05,
            help="Power demanded by secondary mission payloads (earth imaging sensor, APRS digipeater)"
        )

    with c3:
        st.markdown("**3. Dispatch Strategy**")
        priority_mode = st.selectbox(
            "Allocation Policy Mode:",
            options=[
                "Critical First (Standard Orbit)",
                "Balanced Shedding",
                "Strict Emergency Safe Mode"
            ],
            help="Determines the hierarchical load-shedding rule when usable energy is constrained"
        )
        time_window = st.slider(
            "Dispatch Time Window (Hours):",
            min_value=0.25,
            max_value=3.0,
            value=1.0,
            step=0.25,
            help="Duration over which the simulated load profile must be sustained"
        )

    # Run Optimization
    opt_result = run_power_optimization(
        available_energy_Wh=avail_energy,
        critical_demand_W=crit_demand,
        optional_demand_W=opt_demand,
        battery_reserve_pct=bat_reserve,
        time_window_hours=time_window,
        priority_mode=priority_mode
    )

    st.markdown("---")

    # Status Banner
    st.markdown(f"""
    <div class="sat-card" style="border: 2px solid {'#06D6A0' if opt_result['status_color'] == 'green' else ('#FF9900' if opt_result['status_color'] == 'orange' else '#EF476F')};">
        <div style="font-size: 0.8rem; color: #8FA0BC; text-transform: uppercase;">Optimization Engine Status</div>
        <div style="font-size: 1.3rem; font-weight: 700; color: {'#06D6A0' if opt_result['status_color'] == 'green' else ('#FF9900' if opt_result['status_color'] == 'orange' else '#EF476F')}; margin-top: 4px;">
            {opt_result['status']}
        </div>
        <div style="font-size: 0.82rem; color: #8FA0BC; margin-top: 4px;">
            Usable Power Budget: <b>{opt_result['usable_energy_Wh'] / time_window:.3f} W</b> | Reserved Margin: <b>{opt_result['reserve_energy_Wh']:.2f} Wh</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI Summary Row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi("Critical Load Coverage", f"{opt_result['critical_coverage_pct']:.1f}", "%", f"Allocated {opt_result['allocated_critical_W']:.2f}W / {crit_demand:.2f}W", "green" if opt_result['critical_coverage_pct'] >= 100 else "red", "Simulated")
    with k2:
        render_kpi("Optional Load Coverage", f"{opt_result['optional_coverage_pct']:.1f}", "%", f"Allocated {opt_result['allocated_optional_W']:.2f}W / {opt_demand:.2f}W", "green" if opt_result['optional_coverage_pct'] >= 100 else "orange", "Simulated")
    with k3:
        render_kpi("Energy Saved", f"{opt_result['energy_saved_Wh']:.3f}", "Wh", f"{opt_result['energy_saving_percent']:.1f}% reduction vs baseline", "cyan", "Simulated")
    with k4:
        render_kpi("Surplus / Deficit", f"{opt_result['surplus_deficit_Wh']:+.3f}", "Wh", "Remaining after dispatched loads", "cyan" if opt_result['surplus_deficit_Wh'] >= 0 else "red", "Simulated")

    st.markdown("---")

    # Side-by-side Visual Comparison
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown("#### 1. Baseline vs. Optimized Power Allocation (Watts)")
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Baseline Demand",
            x=["Critical Load", "Optional Load", "Total System Power"],
            y=[crit_demand, opt_demand, crit_demand + opt_demand],
            marker_color=THEME_COLORS["solar_orange"]
        ))
        fig_bar.add_trace(go.Bar(
            name="Optimized Dispatch",
            x=["Critical Load", "Optional Load", "Total System Power"],
            y=[opt_result["allocated_critical_W"], opt_result["allocated_optional_W"], opt_result["total_optimized_power_W"]],
            marker_color=THEME_COLORS["cyan"]
        ))
        fig_bar.update_layout(barmode="group")
        fig_bar.update_yaxes(title_text="Power (Watts)")
        style_plotly_chart(fig_bar, title="Subsystem Load Allocation Comparison (W)", height=380)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_c2:
        st.markdown("#### 2. Energy Budget Partitioning (Wh)")
        fig_pie = px.pie(
            names=["Critical Consumption", "Optional Consumption", "Battery Safety Reserve", "Surplus Margin"],
            values=[
                opt_result["allocated_critical_W"] * time_window,
                opt_result["allocated_optional_W"] * time_window,
                opt_result["reserve_energy_Wh"],
                max(0.0, opt_result["surplus_deficit_Wh"])
            ],
            hole=0.45,
            color_discrete_sequence=["#06D6A0", "#00F2FE", "#FF9900", "#3A86FF"]
        )
        style_plotly_chart(fig_pie, title="Orbital Energy Distribution (Wh)", height=380)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    # Precomputed Dataset Optimization Comparison
    st.markdown("### 📚 Precomputed Dataset Optimization Baseline")
    st.caption("Reference simulation averages across all 3 downlinked satellite sessions recorded in the dataset:")
    precomputed_summary = compare_precomputed_dataset_optimization(df)
    st.dataframe(precomputed_summary, use_container_width=True)

if __name__ == "__main__":
    main()
