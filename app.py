"""
SolarSat AI - Satellite Energy Intelligence
Main Application & Mission Control Command Center
"""

import sys
from pathlib import Path
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="SolarSat AI | Satellite Energy Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from utils.ui_components import apply_aerospace_styling, render_header, render_kpi, render_sidebar_status
from utils.data_loader import load_satellite_data, load_data_dictionary

# Apply custom aerospace theme CSS
apply_aerospace_styling()

def main():
    # Load dataset
    try:
        df = load_satellite_data()
        data_dict = load_data_dictionary()
    except Exception as e:
        st.error(f"Error loading telemetry dataset: {e}")
        st.stop()

    render_sidebar_status(len(df), len(df))

    # Sidebar mission clock and info
    st.sidebar.title("🛰️ SolarSat AI")
    st.sidebar.caption("Satellite Energy Intelligence System")
    st.sidebar.markdown("""
    **Spacecraft:** NepaliSat-1 (BIRD-3)  
    **Bus Architecture:** 1U CubeSat  
    **Payload:** EPS / Photovoltaic Subsystem  
    **Downlink Sessions:** 3 passes (2020–2021)  
    """)

    # Main header
    render_header(
        title="SolarSat AI — Satellite Energy Intelligence",
        subtitle="Autonomous Solar Array Performance Monitoring, Power Degradation Analysis & Energy Optimization",
        badge="MISSION CONTROL ACTIVE"
    )

    # Top KPI row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi("Telemetry Records", f"{len(df):,}", "", "Verified downlink frames", "cyan", "Measured")
    with col2:
        render_kpi("Avg Solar Generation", f"{df['solar_array_power_W'].mean():.2f}", "W", "Sum across 5 panel faces", "cyan", "Derived")
    with col3:
        render_kpi("Battery Bus Voltage", f"{df['Vbat (V)'].mean():.2f}", "V", "Nominal 3.7V - 4.2V EPS bus", "green", "Measured")
    with col4:
        render_kpi("Avg Relative Power Drop", f"{df['session_relative_power_drop_percent'].mean():.1f}", "%", "Drop from 95th percentile ref", "orange", "Derived")

    st.markdown("<br>", unsafe_allow_html=True)

    # System Architecture & Data Integrity Section
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown(r"""
        ### 📡 Executive Overview
        **SolarSat AI** is an advanced satellite energy intelligence and telemetry analytics platform built for the **NepaliSat-1** CubeSat mission. The system continuously evaluates photovoltaic generation efficiency, models empirical thermal degradation, forecasts solar power output using physical machine learning regressors, and executes priority-based orbital energy allocation.
        
        #### 🛡️ Telemetry Provenance & Integrity Standards
        In accordance with stringent aerospace standards, this dashboard establishes strict data segregation:
        - **🟢 Measured Telemetry:** Direct downlinked sensor readings, including panel voltages ($V_{py}, V_{px}, V_{mz}, V_{mx}, V_{pz}$), panel temperatures ($T_{py}, T_{px}, T_{mz}, T_{mx}, T_{pz}$), battery voltage ($V_{bat}$), and battery current ($I_{batt}$).
        - **🔵 Derived Telemetry:** Algebraically calculated physical metrics, including individual face powers ($P = V \times I$), total array power ($P_{array} = \sum P_i$), and session 95th-percentile reference power.
        - **🟠 Simulated Scenarios:** Algorithmic simulation scenarios including baseline power demands, assumed 10 Wh battery capacity, and prioritized load-shedding dispatch calculations.
        """)

    with col_right:
        st.markdown("""
        ### 🛰️ Subsystem Specifications
        <div class="sat-card">
            <div style="font-size: 0.85rem; line-height: 1.8;">
                <b>Satellite Classification:</b> 1U Educational CubeSat<br>
                <b>Solar Panel Layout:</b> 5 faces (+Y, +X, -Z, -X, +Z)<br>
                <b>Primary Bus Voltage:</b> Li-Ion Battery (~3.8V – 4.1V)<br>
                <b>Telemetry Downlink Periods:</b><br>
                &nbsp;&nbsp;• <i>Session 1:</i> 09 Nov 2020 (1,080 samples)<br>
                &nbsp;&nbsp;• <i>Session 2:</i> 25 Nov 2020 (880 samples)<br>
                &nbsp;&nbsp;• <i>Session 3:</i> 10 Apr 2021 (1,080 samples)<br>
                <b>Machine Learning Pipeline:</b> Random Forest Regressor & Linear Baseline<br>
                <b>Data Integrity:</b> Zero missing entries across 3,040 records
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎛️ Mission Consoles & Navigation")

    nav_c1, nav_c2, nav_c3 = st.columns(3)
    with nav_c1:
        st.markdown("""
        <div class="sat-card">
            <h4 style="color: #00F2FE; margin-top:0;">📊 1. Mission Overview</h4>
            <p style="font-size: 0.82rem; color: #8FA0BC;">
                Comprehensive mission control metrics, interactive time series of power, temperature, and bus voltage across all downlinked passes.
            </p>
        </div>
        <div class="sat-card">
            <h4 style="color: #FF9900; margin-top:0;">📉 2. Degradation Analysis</h4>
            <p style="font-size: 0.82rem; color: #8FA0BC;">
                Evaluates solar power reductions across sessions against 95th percentile references, thermal derating slopes, and panel-by-panel distributions.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with nav_c2:
        st.markdown("""
        <div class="sat-card">
            <h4 style="color: #06D6A0; margin-top:0;">🤖 3. AI Power Prediction</h4>
            <p style="font-size: 0.82rem; color: #8FA0BC;">
                Physical ML models (Linear Regression & Random Forest) trained on legitimate telemetry to forecast solar power without data leakage.
            </p>
        </div>
        <div class="sat-card">
            <h4 style="color: #00F2FE; margin-top:0;">🔋 4. Battery Monitoring</h4>
            <p style="font-size: 0.82rem; color: #8FA0BC;">
                Real-time battery voltage, current, and temperature tracking, plus energy surplus/deficit analysis and configurable voltage alarms.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with nav_c3:
        st.markdown("""
        <div class="sat-card">
            <h4 style="color: #FF9900; margin-top:0;">⚡ 5. Energy Optimization</h4>
            <p style="font-size: 0.82rem; color: #8FA0BC;">
                Interactive load-shedding simulator optimizing critical vs. optional power dispatch while preserving configurable battery reserves.
            </p>
        </div>
        <div class="sat-card">
            <h4 style="color: #EF476F; margin-top:0;">🚨 6. Alerts & Anomaly Detection</h4>
            <p style="font-size: 0.82rem; color: #8FA0BC;">
                Dual-layer detection: physical rule-based threshold checks and Scikit-learn Isolation Forest for multidimensional telemetry anomalies.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sat-card" style="text-align: center; margin-top: 10px;">
        <span style="color: #00F2FE; font-weight: 600;">📁 7. Telemetry Data Explorer:</span>
        <span style="color: #8FA0BC; font-size: 0.85rem; margin-left: 8px;">
            Full schema inspection, data dictionary lookup, parameter filtering, and raw/processed CSV export.
        </span>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
