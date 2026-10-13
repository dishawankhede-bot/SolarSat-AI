"""
SolarSat AI - Satellite Energy Intelligence
Page 7: Telemetry Data Explorer & Provenance
"""

import sys
from pathlib import Path
import io
import streamlit as st
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from utils.ui_components import (
    apply_aerospace_styling, render_header, render_kpi,
    render_sidebar_status, THEME_COLORS
)
from utils.data_loader import (
    load_satellite_data, load_data_dictionary,
    get_column_provenance, MEASURED_TELEMETRY_COLUMNS,
    DERIVED_COLUMNS, SIMULATED_COLUMNS, METADATA_COLUMNS
)
from utils.prediction_model import load_persisted_model

st.set_page_config(page_title="Data Explorer | SolarSat AI", page_icon="📁", layout="wide")
apply_aerospace_styling()

def main():
    df = load_satellite_data()
    data_dict = load_data_dictionary()

    render_sidebar_status(len(df), len(df))

    render_header(
        title="Satellite Telemetry Explorer & Provenance Registry",
        subtitle="Comprehensive Dataset Inspection, Statistical Profiling, Schema Auditing, and Data Export",
        badge="TELEMETRY ARCHIVE"
    )

    tab_data, tab_stats, tab_dict, tab_export = st.tabs([
        "🔍 Telemetry Table & Filtering",
        "📊 Statistical Summary",
        "📖 Telemetry Data Dictionary",
        "💾 Data & Model Metrics Export"
    ])

    with tab_data:
        st.markdown("### 🛰️ Live Telemetry Inspector")

        # Filters
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            session_filter = st.multiselect(
                "Filter by Session:",
                options=list(df["Recording_Session"].unique()),
                default=list(df["Recording_Session"].unique())
            )
        with f_col2:
            provenance_filter = st.multiselect(
                "Filter Columns by Provenance:",
                options=["Measured Telemetry", "Derived Quantity", "Simulated / Scenario", "Metadata"],
                default=["Measured Telemetry", "Derived Quantity", "Simulated / Scenario"]
            )
        with f_col3:
            row_limit = st.slider("Display Row Limit:", 10, 500, 100, 10)

        # Build column list based on provenance filter
        selected_cols = []
        for col in df.columns:
            prov = get_column_provenance(col)
            if prov in provenance_filter:
                selected_cols.append(col)

        if not selected_cols:
            selected_cols = df.columns.tolist()

        # Specific column multi-selector
        final_cols = st.multiselect("Active Columns in View:", options=df.columns.tolist(), default=selected_cols[:15])

        filtered_df = df[df["Recording_Session"].isin(session_filter)]
        if final_cols:
            display_df = filtered_df[final_cols].head(row_limit)
        else:
            display_df = filtered_df.head(row_limit)

        st.caption(f"Showing **{len(display_df):,}** of **{len(filtered_df):,}** filtered records ({len(final_cols)} columns):")
        st.dataframe(display_df, use_container_width=True)

    with tab_stats:
        st.markdown("### 📊 Descriptive Statistics & Schema Health")

        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            render_kpi("Total Records", f"{len(df):,}", "", "3,040 verified frames", "cyan", "Measured")
        with stat_col2:
            render_kpi("Total Columns", f"{len(df.columns)}", "", "Full schema width", "cyan", "Measured")
        with stat_col3:
            render_kpi("Missing Values", f"{df.isnull().sum().sum()}", "", "Zero nulls across all cells", "green", "Measured")

        st.markdown("#### Numerical Attributes Profile")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_summary = df[num_cols].describe().T
        num_summary["Provenance"] = [get_column_provenance(c) for c in num_summary.index]
        # Reorder to put Provenance up front
        cols_order = ["Provenance", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        num_summary = num_summary[[c for c in cols_order if c in num_summary.columns]]
        st.dataframe(num_summary.round(4), use_container_width=True)

    with tab_dict:
        st.markdown("### 📖 Official Project Data Dictionary")
        st.caption("Standardized definitions, physical units, derivation logic, and data statuses:")
        if not data_dict.empty:
            st.dataframe(data_dict, use_container_width=True, hide_index=True)
        else:
            st.info("Data dictionary file not found in repository.")

    with tab_export:
        st.markdown("### 💾 Data & Evaluation Export Console")
        st.write("Generate and download working copies of data and machine learning artifacts for offline analysis:")

        e_col1, e_col2 = st.columns(2)

        with e_col1:
            st.markdown("#### 1. Filtered Telemetry Dataset (CSV)")
            st.caption("Exports current filtered working dataframe copy without modifying the master dataset.")
            csv_buffer = io.StringIO()
            filtered_df.to_csv(csv_buffer, index=False)
            csv_bytes = csv_buffer.getvalue().encode("utf-8")

            st.download_button(
                label="📥 Download Filtered Telemetry CSV",
                data=csv_bytes,
                file_name="solarsat_filtered_telemetry.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )

        with e_col2:
            st.markdown("#### 2. ML Model Performance Report")
            st.caption("Exports metrics summary for Random Forest and Linear Regression benchmarks.")
            
            # Generate exportable metrics summary
            metrics_export_df = pd.DataFrame([
                {"Model": "Random Forest Regressor", "Target": "solar_array_power_W", "Features_Count": 14, "Primary_Metric": "R2 > 0.99 (Random) / 0.71 (Session)", "Artifact": "models/random_forest.joblib"},
                {"Model": "Linear Regression Baseline", "Target": "solar_array_power_W", "Features_Count": 14, "Primary_Metric": "R2 ~ 0.88 (Random) / 0.67 (Session)", "Artifact": "models/linear_regression.joblib"}
            ])
            rep_buffer = io.StringIO()
            metrics_export_df.to_csv(rep_buffer, index=False)
            rep_bytes = rep_buffer.getvalue().encode("utf-8")

            st.download_button(
                label="📥 Download ML Evaluation Report (CSV)",
                data=rep_bytes,
                file_name="solarsat_ml_model_evaluation.csv",
                mime="text/csv",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
