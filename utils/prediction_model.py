"""
SolarSat AI - Satellite Energy Intelligence
Machine Learning Prediction Module
Trains, evaluates, and persists physical ML regression models to predict solar_array_power_W.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_FEATURES = [
    "Vpy (mV)", "Vpx (mV)", "Vmz (mV)", "Vmx (mV)", "Vpz (mV)",
    "Tpy (°C)", "Tpx (°C)", "Tmz (°C)", "Tmx (°C)", "Tpz (°C)",
    "mean_panel_temperature_C", "Vbat (V)", "Tbatt (℃)", "Time Stamp"
]

TARGET_COL = "solar_array_power_W"

LEAKAGE_EXCLUDED_COLUMNS = [
    "Ppy_Power_W", "Ppx_Power_W", "Pmz_Power_W", "Pmx_Power_W", "Ppz_Power_W",
    "solar_array_power_W", "session_reference_power_W", "session_relative_power_drop_percent",
    "total_power_demand_W_simulated", "critical_load_W_simulated", "optional_load_W_simulated",
    "optimized_critical_load_W_simulated", "optimized_optional_load_W_simulated",
    "energy_saving_percent_simulated", "available_energy_Wh_simulated", "energy_surplus_or_deficit_Wh_simulated"
]


def train_and_evaluate_models(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    split_strategy: str = "session",  # 'session' or 'random'
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains Linear Regression and Random Forest Regressor on legitimate telemetry features,
    calculates MAE, RMSE, and R2 on held-out test data, and saves models to disk.
    """
    if features is None:
        features = [f for f in DEFAULT_FEATURES if f in df.columns]

    valid_df = df.dropna(subset=features + [TARGET_COL]).copy()
    X = valid_df[features]
    y = valid_df[TARGET_COL]

    if split_strategy == "session":
        # Chronological split: Train on Nov 2020 sessions, test on April 2021 session
        train_mask = valid_df["Recording_Session"].isin(["9 Nov 2020", "25 Nov 2020"])
        test_mask = valid_df["Recording_Session"] == "10 April 2021"

        if train_mask.sum() > 50 and test_mask.sum() > 50:
            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]
            split_desc = "Chronological split (Train: Nov 2020 sessions | Test: April 2021 session)"
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_state
            )
            split_desc = "Random 80/20 train/test split (Fallback)"
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        split_desc = f"Random 80/20 train/test split (seed={random_state})"

    # Model 1: Baseline Linear Regression Pipeline
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_preds = lr_pipeline.predict(X_test)

    lr_metrics = {
        "MAE": float(mean_absolute_error(y_test, lr_preds)),
        "RMSE": float(root_mean_squared_error(y_test, lr_preds)),
        "R2": float(r2_score(y_test, lr_preds))
    }

    # Model 2: Random Forest Regressor Pipeline
    rf_pipeline = Pipeline([
        ("model", RandomForestRegressor(n_estimators=100, max_depth=12, random_state=random_state, n_jobs=-1))
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_preds = rf_pipeline.predict(X_test)

    rf_metrics = {
        "MAE": float(mean_absolute_error(y_test, rf_preds)),
        "RMSE": float(root_mean_squared_error(y_test, rf_preds)),
        "R2": float(r2_score(y_test, rf_preds))
    }

    # Extract Feature Importances from Random Forest
    rf_model: RandomForestRegressor = rf_pipeline.named_steps["model"]
    feature_importances = pd.Series(
        rf_model.feature_importances_, index=features
    ).sort_values(ascending=False).to_dict()

    # Save models locally
    joblib.dump(lr_pipeline, MODELS_DIR / "linear_regression.joblib")
    joblib.dump(rf_pipeline, MODELS_DIR / "random_forest.joblib")
    joblib.dump(features, MODELS_DIR / "trained_features.joblib")

    return {
        "split_desc": split_desc,
        "features": features,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "lr_pipeline": lr_pipeline,
        "rf_pipeline": rf_pipeline,
        "lr_metrics": lr_metrics,
        "rf_metrics": rf_metrics,
        "feature_importances": feature_importances,
        "test_data": {
            "y_test": y_test.values,
            "lr_preds": lr_preds,
            "rf_preds": rf_preds,
            "indices": X_test.index.values
        }
    }


def load_persisted_model(model_name: str = "random_forest"):
    """Loads a previously saved model pipeline from the models directory."""
    path = MODELS_DIR / f"{model_name}.joblib"
    feat_path = MODELS_DIR / "trained_features.joblib"
    if path.exists() and feat_path.exists():
        model = joblib.load(path)
        features = joblib.load(feat_path)
        return model, features
    return None, None


def predict_single_instance(
    model_pipeline,
    feature_names: List[str],
    input_values: Dict[str, float]
) -> float:
    """Predicts solar array power in Watts for user-specified telemetry inputs."""
    row = [input_values.get(f, 0.0) for f in feature_names]
    df_row = pd.DataFrame([row], columns=feature_names)
    prediction = model_pipeline.predict(df_row)[0]
    return max(0.0, float(prediction))
