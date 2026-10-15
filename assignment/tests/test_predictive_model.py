"""
Unit tests for Requirement 2: Predictive Wear Detection Model
"""

import pytest
import pandas as pd
from pathlib import Path
from assignment.src.config import HISTORICAL_CSV
from assignment.src.predictive_model import PredictiveWearModel, FEATURES, TARGET


def test_synthetic_dataset_has_at_least_50_rows():
    assert HISTORICAL_CSV.exists(), "Historical dataset CSV must exist."
    df = pd.read_csv(HISTORICAL_CSV)
    assert len(df) >= 50, f"Expected at least 50 labeled rows, got {len(df)}"
    for feat in FEATURES + [TARGET]:
        assert feat in df.columns, f"Expected column '{feat}' in dataset"


def test_model_training_and_accuracy(tmp_path):
    test_model_file = tmp_path / "test_model.joblib"
    model = PredictiveWearModel(model_file=test_model_file)
    results = model.train(HISTORICAL_CSV)

    assert results["sample_count"] >= 50
    assert results["test_accuracy"] >= 0.85
    assert results["cv_mean_accuracy"] >= 0.80
    assert test_model_file.exists()


def test_model_prediction_single():
    model = PredictiveWearModel()
    model.load()

    # Nominal healthy reading
    nominal = {
        "torque_nm": 55.0,
        "temperature_c": 42.0,
        "vibration_mms": 10.0,
        "motor_current_a": 11.0,
        "speed_rpm": 25.0,
        "operating_hours": 300,
    }
    pred = model.predict_single(nominal)
    assert pred["wear_state"] in ["NORMAL", "EARLY_WEAR"]
    assert 0.0 <= pred["wear_risk_score"] <= 0.45
    assert "probabilities" in pred

    # Severely degraded reading
    degraded = {
        "torque_nm": 140.0,
        "temperature_c": 78.0,
        "vibration_mms": 42.0,
        "motor_current_a": 28.0,
        "speed_rpm": 25.0,
        "operating_hours": 7500,
    }
    pred_deg = model.predict_single(degraded)
    assert pred_deg["wear_state"] in ["ACCELERATED_WEAR", "CRITICAL_WEAR"]
    assert pred_deg["wear_risk_score"] >= 0.50


def test_graceful_fallback_when_model_unloaded(tmp_path):
    empty_model_file = tmp_path / "non_existent.joblib"
    model = PredictiveWearModel(model_file=empty_model_file)
    # predict without training/loading
    pred = model.predict_single({"torque_nm": 50.0})
    assert pred["wear_state"] == "UNKNOWN"
    assert pred["wear_risk_score"] == 0.0
