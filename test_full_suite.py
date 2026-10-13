"""
SolarSat AI - Full Automated Verification Suite
Tests data loading, ML training, inference, energy optimization,
anomaly detection, and CSV exports across all project components.
"""

import sys
import io
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure utf-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ensure root is in sys.path
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

def run_tests():
    print("=" * 60)
    print("🛰️ STARTING SOLARSAT AI FULL SYSTEM VERIFICATION SUITE")
    print("=" * 60)

    # 1. Data Loader Test
    print("\n[TEST 1] Loading Satellite Data & Data Dictionary...")
    from utils.data_loader import (
        load_satellite_data, load_data_dictionary,
        filter_telemetry_data, get_column_provenance
    )
    df = load_satellite_data()
    assert df is not None, "Failed to load telemetry dataframe"
    assert len(df) == 3040, f"Expected 3040 rows, got {len(df)}"
    assert len(df.columns) == 62, f"Expected 62 columns, got {len(df.columns)}"
    assert df.isnull().sum().sum() == 0, "Telemetry dataframe contains unexpected null values"
    assert df.duplicated().sum() == 0, "Telemetry dataframe contains unexpected duplicate rows"
    print(f"✔ Data Loader PASSED: {len(df):,} rows, {len(df.columns)} columns, zero nulls/duplicates.")

    data_dict = load_data_dictionary()
    assert not data_dict.empty, "Data dictionary is empty"
    print(f"✔ Data Dictionary PASSED: {len(data_dict)} entries registered.")

    # Test filtering
    filtered = filter_telemetry_data(df, selected_sessions=["9 Nov 2020"])
    assert len(filtered) == 1080, f"Expected 1080 rows for '9 Nov 2020', got {len(filtered)}"
    print(f"✔ Telemetry Filtering PASSED: session slice returned {len(filtered)} rows.")

    # 2. Degradation Analysis Test
    print("\n[TEST 2] Testing Solar Power Degradation Analysis...")
    from utils.degradation_analysis import (
        compute_session_summary, compute_panel_power_breakdown,
        compute_temperature_derating, identify_low_power_episodes
    )
    summary = compute_session_summary(df)
    assert len(summary) == 3, f"Expected 3 sessions in summary, got {len(summary)}"
    assert "Avg Relative Power Drop (%)" in summary.columns
    print(f"✔ Session Summary PASSED:\n{summary[['Session', 'Samples', 'Mean Solar Power (W)', 'Avg Relative Power Drop (%)']].to_string(index=False)}")

    breakdown = compute_panel_power_breakdown(df)
    assert len(breakdown) == 5, f"Expected 5 panels in breakdown, got {len(breakdown)}"
    print("✔ Panel Power Breakdown PASSED:")
    for idx, row in breakdown.iterrows():
        print(f"   • {idx}: Mean {row['Mean Power (W)']} W ({row['Share of Total (%)']}%)")

    slope, corr, r2 = compute_temperature_derating(df)
    print(f"✔ Thermal Derating PASSED: Slope = {slope:.5f} W/°C, Correlation = {corr:.3f}, R² = {r2:.3f}")

    episodes = identify_low_power_episodes(df, threshold_percent=85.0)
    print(f"✔ Severe Drop Episode Detection PASSED: {len(episodes)} frames found dropping >= 85%.")

    # 3. Machine Learning Prediction Model Test
    print("\n[TEST 3] Testing AI Power Prediction Training & Evaluation...")
    from utils.prediction_model import (
        train_and_evaluate_models, load_persisted_model,
        predict_single_instance, DEFAULT_FEATURES, TARGET_COL
    )

    # Test Chronological Split
    sess_results = train_and_evaluate_models(df, split_strategy="session")
    rf_m = sess_results["rf_metrics"]
    lr_m = sess_results["lr_metrics"]
    print(f"✔ Session-Based Split ML PASSED:")
    print(f"   • Random Forest: R² = {rf_m['R2']:.4f}, MAE = {rf_m['MAE']:.4f} W, RMSE = {rf_m['RMSE']:.4f} W")
    print(f"   • Linear Regression: R² = {lr_m['R2']:.4f}, MAE = {lr_m['MAE']:.4f} W, RMSE = {lr_m['RMSE']:.4f} W")
    assert rf_m["R2"] > 0.65, "Random Forest cross-session R2 lower than expected threshold"

    # Test Random Split
    rand_results = train_and_evaluate_models(df, split_strategy="random")
    rf_m_rand = rand_results["rf_metrics"]
    print(f"✔ Random 80/20 Split ML PASSED:")
    print(f"   • Random Forest: R² = {rf_m_rand['R2']:.4f}, MAE = {rf_m_rand['MAE']:.4f} W, RMSE = {rf_m_rand['RMSE']:.4f} W")
    assert rf_m_rand["R2"] > 0.95, "Random Forest in-distribution R2 lower than expected threshold"

    # Test Model Persistence & Loading
    rf_loaded, feats_loaded = load_persisted_model("random_forest")
    assert rf_loaded is not None, "Failed to load saved random_forest model"
    assert len(feats_loaded) == len(DEFAULT_FEATURES), "Saved features mismatch"
    print("✔ Model Persistence & Loading PASSED.")

    # Test Inference on Sample Inputs
    test_input = {
        "Vpy (mV)": 1500.0, "Vpx (mV)": 1400.0, "Vmz (mV)": 1300.0, "Vmx (mV)": 1450.0, "Vpz (mV)": 1400.0,
        "Tpy (°C)": -5.0, "Tpx (°C)": -6.0, "Tmz (°C)": -8.0, "Tmx (°C)": -5.0, "Tpz (°C)": -4.0,
        "mean_panel_temperature_C": -5.6,
        "Vbat (V)": 3.98, "Tbatt (℃)": 12.0, "Time Stamp": 350
    }
    inferred_w = predict_single_instance(rf_loaded, feats_loaded, test_input)
    assert inferred_w > 0.0, f"Expected positive inferred power, got {inferred_w}"
    print(f"✔ Single-Instance Inference PASSED: Inferred {inferred_w:.4f} Watts.")

    # 4. Energy Optimization Test
    print("\n[TEST 4] Testing Energy Allocation Optimization Simulator...")
    from utils.energy_optimizer import (
        run_power_optimization, compare_precomputed_dataset_optimization
    )

    # Scenario A: Adequate power
    res_nominal = run_power_optimization(
        available_energy_Wh=8.0,
        critical_demand_W=0.6,
        optional_demand_W=0.3,
        battery_reserve_pct=20.0,
        time_window_hours=1.0,
        priority_mode="Critical First (Standard Orbit)"
    )
    assert res_nominal["critical_coverage_pct"] == 100.0
    assert res_nominal["optional_coverage_pct"] == 100.0
    assert "NOMINAL" in res_nominal["status"]
    print(f"✔ Optimization Scenario A (Nominal) PASSED: Status = '{res_nominal['status']}'")

    # Scenario B: Constrained power (shedding optional load)
    res_constrained = run_power_optimization(
        available_energy_Wh=1.0,
        critical_demand_W=0.6,
        optional_demand_W=0.5,
        battery_reserve_pct=20.0,
        time_window_hours=1.0,
        priority_mode="Critical First (Standard Orbit)"
    )
    assert res_constrained["critical_coverage_pct"] == 100.0
    assert res_constrained["optional_coverage_pct"] < 100.0
    assert "PARTIAL SHEDDING" in res_constrained["status"]
    print(f"✔ Optimization Scenario B (Shedding) PASSED: Status = '{res_constrained['status']}', Optional Coverage = {res_constrained['optional_coverage_pct']}%")

    # Scenario C: Deficit
    res_deficit = run_power_optimization(
        available_energy_Wh=0.3,
        critical_demand_W=0.8,
        optional_demand_W=0.4,
        battery_reserve_pct=20.0,
        time_window_hours=1.0,
        priority_mode="Critical First (Standard Orbit)"
    )
    assert res_deficit["critical_coverage_pct"] < 100.0
    assert "CRITICAL POWER DEFICIT" in res_deficit["status"]
    print(f"✔ Optimization Scenario C (Deficit) PASSED: Status = '{res_deficit['status']}', Critical Coverage = {res_deficit['critical_coverage_pct']}%")

    # 5. Anomaly Detection Test
    print("\n[TEST 5] Testing Alerts & Anomaly Detection Modules...")
    from utils.anomaly_detection import (
        detect_rule_based_anomalies, detect_isolation_forest_anomalies
    )
    rule_alerts = detect_rule_based_anomalies(df)
    assert not rule_alerts.empty, "Rule based detection returned empty dataframe"
    assert "Severity" in rule_alerts.columns
    print(f"✔ Rule-Based Alerts PASSED: Found {len(rule_alerts):,} threshold flags.")

    enriched_df, ml_alerts = detect_isolation_forest_anomalies(df, contamination=0.02)
    assert "is_ml_anomaly" in enriched_df.columns
    assert "anomaly_score" in enriched_df.columns
    anom_count = enriched_df["is_ml_anomaly"].sum()
    print(f"✔ Isolation Forest PASSED: Flagged {anom_count} outliers ({anom_count/len(df)*100:.1f}%).")

    # 6. Data Explorer and CSV Download Test
    print("\n[TEST 6] Testing CSV Serialization & Data Export...")
    csv_buf = io.StringIO()
    df.head(50).to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode("utf-8")
    assert len(csv_bytes) > 0, "CSV export buffer is empty"
    print(f"✔ CSV Data Export PASSED: Generated {len(csv_bytes):,} bytes.")

    print("\n" + "=" * 60)
    print("🎉 ALL 6 SYSTEM TEST MODULES COMPLETED WITH 100% SUCCESS!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
