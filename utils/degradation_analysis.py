"""
SolarSat AI - Satellite Energy Intelligence
Solar Degradation & Power Reduction Analysis Module
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def compute_session_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes rigorous statistical metrics for each recording session to analyze
    observed solar power drops and reference power baselines.
    """
    records = []
    for session, grp in df.groupby("Recording_Session"):
        date_val = grp["Recording_Date"].iloc[0] if "Recording_Date" in grp.columns else "N/A"
        count = len(grp)
        mean_p = grp["solar_array_power_W"].mean()
        max_p = grp["solar_array_power_W"].max()
        ref_p = grp["session_reference_power_W"].iloc[0] if "session_reference_power_W" in grp.columns else grp["solar_array_power_W"].quantile(0.95)
        mean_drop = grp["session_relative_power_drop_percent"].mean() if "session_relative_power_drop_percent" in grp.columns else np.nan
        max_drop = grp["session_relative_power_drop_percent"].max() if "session_relative_power_drop_percent" in grp.columns else np.nan
        mean_temp = grp["mean_panel_temperature_C"].mean() if "mean_panel_temperature_C" in grp.columns else np.nan
        mean_vbat = grp["Vbat (V)"].mean() if "Vbat (V)" in grp.columns else np.nan

        records.append({
            "Session": session,
            "Date": str(date_val),
            "Samples": count,
            "Mean Solar Power (W)": round(mean_p, 4),
            "Max Solar Power (W)": round(max_p, 4),
            "Session Ref Power (95th %) (W)": round(ref_p, 4),
            "Avg Relative Power Drop (%)": round(mean_drop, 2),
            "Peak Power Drop (%)": round(max_drop, 2),
            "Mean Panel Temp (°C)": round(mean_temp, 2),
            "Mean Vbat (V)": round(mean_vbat, 3)
        })

    summary_df = pd.DataFrame(records)
    # Sort chronologically if possible
    if "Date" in summary_df.columns:
        summary_df = summary_df.sort_values("Date")
    return summary_df


def compute_panel_power_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the individual power contributions of each satellite panel face:
    +Y (Ppy), +X (Ppx), -Z (Pmz), -X (Pmx), +Z (Ppz).
    """
    panel_cols = {
        "+Y Panel (Ppy)": "Ppy_Power_W",
        "+X Panel (Ppx)": "Ppx_Power_W",
        "-Z Panel (Pmz)": "Pmz_Power_W",
        "-X Panel (Pmx)": "Pmx_Power_W",
        "+Z Panel (Ppz)": "Ppz_Power_W"
    }

    breakdown = {}
    total_power_sum = df["solar_array_power_W"].sum()
    if total_power_sum == 0:
        total_power_sum = 1e-6

    for label, col in panel_cols.items():
        if col in df.columns:
            panel_total = df[col].sum()
            panel_mean = df[col].mean()
            breakdown[label] = {
                "Mean Power (W)": round(panel_mean, 4),
                "Total Energy Proxy (W·s)": round(panel_total, 2),
                "Share of Total (%)": round((panel_total / total_power_sum) * 100, 2)
            }

    return pd.DataFrame(breakdown).T


def compute_temperature_derating(df: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Analyzes the correlation and empirical derating coefficient between
    solar panel temperature and array power output.
    Returns: (derating_slope_W_per_C, correlation, r_squared)
    """
    valid = df.dropna(subset=["mean_panel_temperature_C", "solar_array_power_W"])
    if len(valid) < 10:
        return 0.0, 0.0, 0.0

    X = valid[["mean_panel_temperature_C"]].values
    y = valid["solar_array_power_W"].values

    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]
    r_squared = reg.score(X, y)
    corr = valid["mean_panel_temperature_C"].corr(valid["solar_array_power_W"])

    return slope, corr, r_squared


def identify_low_power_episodes(df: pd.DataFrame, threshold_percent: float = 75.0) -> pd.DataFrame:
    """
    Identifies telemetry records exhibiting substantial power drop (> threshold_percent)
    compared to the session reference power.
    """
    if "session_relative_power_drop_percent" not in df.columns:
        return pd.DataFrame()

    drops = df[df["session_relative_power_drop_percent"] >= threshold_percent].copy()
    display_cols = [
        "timestamp", "Recording_Session", "solar_array_power_W",
        "session_reference_power_W", "session_relative_power_drop_percent",
        "mean_panel_temperature_C", "Vbat (V)"
    ]
    avail_cols = [c for c in display_cols if c in drops.columns]
    return drops[avail_cols].sort_values("session_relative_power_drop_percent", ascending=False)
