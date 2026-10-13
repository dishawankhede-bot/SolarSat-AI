"""
SolarSat AI - Satellite Energy Intelligence
Data Loader Module
Loads satellite telemetry, validates schema, and categorizes columns by provenance.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CSV_PATH = DATA_DIR / "nepalese_satellite_extended.csv"
DEFAULT_DICT_PATH = DATA_DIR / "data_dictionary.csv"

# Classification of columns by provenance
RAW_TELEMETRY_COLUMNS = [
    "Tpy (raw)", "Tpx (raw)", "Tmz (raw)", "Tmx (raw)", "Tpz (raw)",
    "Vpy (raw)", "Vpx (raw)", "Vmz (raw)", "Vmx (raw)", "Vpz (raw)",
    "Ipy (raw)", "Ipx (raw)", "Imz (raw)", "Imx (raw)", "Ipz (raw)",
    "Vbat (raw)", "Ibatt(raw)", "Tbatt raw)"
]

MEASURED_TELEMETRY_COLUMNS = [
    "Time Stamp",
    "Tpy (°C)", "Tpx (°C)", "Tmz (°C)", "Tmx (°C)", "Tpz (°C)",
    "Vpy (mV)", "Vpx (mV)", "Vmz (mV)", "Vmx (mV)", "Vpz (mV)",
    "Ipy (mA)", "Ipx (mA)", "Imz (mA)", "Imx (mA)", "Ipz (mA)",
    "Vbat (V)", "Ibatt(mA)", "Tbatt (℃)"
]

DERIVED_COLUMNS = [
    "Ppy_Power_W", "Ppx_Power_W", "Pmz_Power_W", "Pmx_Power_W", "Ppz_Power_W",
    "solar_array_power_W", "battery_power_W", "mean_panel_temperature_C",
    "session_reference_power_W", "session_relative_power_drop_percent"
]

SIMULATED_COLUMNS = [
    "total_power_demand_W_simulated",
    "critical_load_W_simulated",
    "optional_load_W_simulated",
    "battery_capacity_Wh_assumed",
    "battery_soc_percent_simulated",
    "available_energy_Wh_simulated",
    "energy_surplus_or_deficit_Wh_simulated",
    "optimized_critical_load_W_simulated",
    "optimized_optional_load_W_simulated",
    "optimization_status_simulated",
    "energy_saving_percent_simulated"
]

METADATA_COLUMNS = [
    "Recording_Session", "Recording_Date", "timestamp", "feature_provenance_note"
]


@st.cache_data(show_spinner="Loading satellite telemetry data...")
def load_satellite_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads and standardizes the NepaliSat-1 extended telemetry dataset.
    Returns cached pandas DataFrame.
    """
    path = Path(file_path) if file_path else DEFAULT_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(f"Telemetry file not found at: {path}")

    # Read CSV with utf-8 encoding
    df = pd.read_csv(path, encoding="utf-8")

    # Standardize datetime formats
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "Recording_Date" in df.columns:
        df["Recording_Date"] = pd.to_datetime(df["Recording_Date"]).dt.date

    # Ensure numeric columns are strictly numeric
    for col in MEASURED_TELEMETRY_COLUMNS + DERIVED_COLUMNS + SIMULATED_COLUMNS:
        if col in df.columns and col != "optimization_status_simulated":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_data(show_spinner="Loading data dictionary...")
def load_data_dictionary(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads the telemetry data dictionary explaining units, provenance, and definitions.
    """
    path = Path(file_path) if file_path else DEFAULT_DICT_PATH
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8")


def get_column_provenance(col_name: str) -> str:
    """
    Returns the provenance category and badge for a column name:
    - Measured Telemetry
    - Derived Satellite Telemetry
    - Simulated Scenario / Assumption
    - Metadata
    """
    if col_name in MEASURED_TELEMETRY_COLUMNS or col_name in RAW_TELEMETRY_COLUMNS:
        return "Measured Telemetry"
    elif col_name in DERIVED_COLUMNS:
        return "Derived Quantity"
    elif col_name in SIMULATED_COLUMNS:
        return "Simulated / Scenario"
    elif col_name in METADATA_COLUMNS:
        return "Metadata"
    return "Other"


def filter_telemetry_data(
    df: pd.DataFrame,
    selected_sessions: Optional[List[str]] = None,
    date_range: Optional[Tuple] = None,
    index_range: Optional[Tuple[int, int]] = None
) -> pd.DataFrame:
    """
    Applies mission sidebar filters safely to the working dataframe copy.
    """
    filtered = df.copy()

    if selected_sessions:
        filtered = filtered[filtered["Recording_Session"].isin(selected_sessions)]

    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        filtered = filtered[
            (filtered["Recording_Date"] >= date_range[0]) &
            (filtered["Recording_Date"] <= date_range[1])
        ]

    if index_range and len(index_range) == 2:
        start_idx, end_idx = index_range
        filtered = filtered.iloc[start_idx:end_idx]

    return filtered
