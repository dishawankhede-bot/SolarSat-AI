"""
SolarSat AI - Satellite Energy Intelligence
Anomaly Detection and Mission Alerting Module
Combines aerospace telemetry physical threshold rules with scikit-learn Isolation Forest.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_rule_based_anomalies(
    df: pd.DataFrame,
    vbat_min: float = 3.80,
    vbat_max: float = 4.15,
    panel_temp_max: float = 25.0,
    panel_temp_min: float = -20.0,
    power_drop_rate_max: float = 1.5
) -> pd.DataFrame:
    """
    Evaluates physical satellite telemetry threshold bounds and generates structured alerts.
    """
    alerts = []

    # Calculate step-to-step power change rate if possible
    df_eval = df.copy()
    if "solar_array_power_W" in df_eval.columns:
        df_eval["power_delta_W"] = df_eval["solar_array_power_W"].diff().abs()

    for idx, row in df_eval.iterrows():
        ts = row.get("timestamp", f"Idx {idx}")
        session = row.get("Recording_Session", "Unknown")

        # 1. Battery Undervoltage Check
        if "Vbat (V)" in row and pd.notnull(row["Vbat (V)"]):
            vbat = row["Vbat (V)"]
            if vbat < vbat_min:
                alerts.append({
                    "Timestamp": ts,
                    "Session": session,
                    "Severity": "CRITICAL",
                    "Parameter": "Battery Voltage",
                    "Observed Value": f"{vbat:.3f} V",
                    "Threshold": f"< {vbat_min:.2f} V",
                    "Reason": "Subsystem undervoltage detected; risk of bus collapse.",
                    "Recommended Action": "Initiate EPS emergency load shed; inhibit high-current transmitters."
                })
            elif vbat > vbat_max:
                alerts.append({
                    "Timestamp": ts,
                    "Session": session,
                    "Severity": "WARNING",
                    "Parameter": "Battery Voltage",
                    "Observed Value": f"{vbat:.3f} V",
                    "Threshold": f"> {vbat_max:.2f} V",
                    "Reason": "Bus voltage approaching overcharge limit.",
                    "Recommended Action": "Verify shunt regulator dissipation and battery charge controller status."
                })

        # 2. Panel Thermal Check
        if "mean_panel_temperature_C" in row and pd.notnull(row["mean_panel_temperature_C"]):
            temp = row["mean_panel_temperature_C"]
            if temp > panel_temp_max:
                alerts.append({
                    "Timestamp": ts,
                    "Session": session,
                    "Severity": "WARNING",
                    "Parameter": "Solar Panel Temp",
                    "Observed Value": f"{temp:.2f} °C",
                    "Threshold": f"> {panel_temp_max:.1f} °C",
                    "Reason": "Solar cell temperature high; causes photovoltaic efficiency derating.",
                    "Recommended Action": "Monitor attitude to reduce direct solar thermal accumulation."
                })
            elif temp < panel_temp_min:
                alerts.append({
                    "Timestamp": ts,
                    "Session": session,
                    "Severity": "INFO",
                    "Parameter": "Solar Panel Temp",
                    "Observed Value": f"{temp:.2f} °C",
                    "Threshold": f"< {panel_temp_min:.1f} °C",
                    "Reason": "Cryogenic or eclipse thermal regime detected.",
                    "Recommended Action": "Check battery survival heater activation status."
                })

        # 3. Sudden Power Step Change
        if "power_delta_W" in row and pd.notnull(row["power_delta_W"]):
            delta = row["power_delta_W"]
            if delta > power_drop_rate_max:
                alerts.append({
                    "Timestamp": ts,
                    "Session": session,
                    "Severity": "WARNING",
                    "Parameter": "Solar Array Power Step",
                    "Observed Value": f"Δ {delta:.3f} W/sample",
                    "Threshold": f"> {power_drop_rate_max:.2f} W",
                    "Reason": "Rapid solar generation step drop; likely fast attitude tumble or shadow entry.",
                    "Recommended Action": "Correlate with ADCS gyro attitude rates and orbital position."
                })

    return pd.DataFrame(alerts)


def detect_isolation_forest_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.02,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fits Scikit-Learn IsolationForest on multidimensional telemetry features
    to uncover non-linear telemetry outliers.
    Returns:
    - enriched_df: copy of df with 'is_ml_anomaly' (bool) and 'anomaly_score' (float)
    - ml_alerts_df: structured alerts table for anomalies
    """
    feature_candidates = [
        "Vbat (V)", "Ibatt(mA)", "Tbatt (℃)",
        "mean_panel_temperature_C", "solar_array_power_W",
        "Vpy (mV)", "Vpx (mV)", "Vmz (mV)", "Vmx (mV)", "Vpz (mV)"
    ]
    features = [f for f in feature_candidates if f in df.columns]

    valid = df.dropna(subset=features).copy()
    if len(valid) < 30:
        df_ret = df.copy()
        df_ret["is_ml_anomaly"] = False
        df_ret["anomaly_score"] = 0.0
        return df_ret, pd.DataFrame()

    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100
    )
    preds = iso.fit_predict(valid[features])  # -1 for anomaly, 1 for normal
    scores = iso.decision_function(valid[features])

    valid["is_ml_anomaly"] = preds == -1
    valid["anomaly_score"] = -scores  # higher means more anomalous

    # Generate ML alerts
    ml_alerts = []
    anom_rows = valid[valid["is_ml_anomaly"]]
    for idx, row in anom_rows.iterrows():
        ts = row.get("timestamp", f"Idx {idx}")
        session = row.get("Recording_Session", "Unknown")
        ml_alerts.append({
            "Timestamp": ts,
            "Session": session,
            "Severity": "WARNING",
            "Parameter": "Multidimensional Telemetry Vector",
            "Observed Value": f"Anomaly Score: {row['anomaly_score']:.3f}",
            "Threshold": f"Top {contamination*100:.1f}% Outlier",
            "Reason": "Statistical deviation across coupled thermal, electrical, and generation telemetry.",
            "Recommended Action": "Investigate coupled telemetry trends for unexpected transient behavior."
        })

    enriched_df = df.copy()
    enriched_df["is_ml_anomaly"] = False
    enriched_df["anomaly_score"] = 0.0
    enriched_df.loc[valid.index, "is_ml_anomaly"] = valid["is_ml_anomaly"]
    enriched_df.loc[valid.index, "anomaly_score"] = valid["anomaly_score"]

    return enriched_df, pd.DataFrame(ml_alerts)
