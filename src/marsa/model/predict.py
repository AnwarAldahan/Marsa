"""Runtime contract for the final XGBoost cargo-congestion proxy."""

from __future__ import annotations

import pandas as pd

from marsa.data.load import parse_exact_utc_hour
from marsa.model.congestion_proxy import (
    MODEL_DIR,
    MODEL_VERSION,
    PRIMARY_HORIZON_HOURS,
    TARGET,
    FeatureParityError,
    predict,
)
from marsa.provenance.types import DataProvenance


def _unavailable(ts, reason: str) -> dict:
    return {
        "status": "ml_unavailable",
        "timestamp_utc": ts.isoformat(),
        "target": TARGET,
        "target_definition": "Retrospective operational cargo-congestion proxy",
        "horizon_hours": PRIMARY_HORIZON_HOURS,
        "target_timestamp_utc": (ts + pd.Timedelta(hours=6)).isoformat(),
        "congestion_score": None,
        "classification": None,
        "model_version": MODEL_VERSION,
        "threshold_metadata": None,
        "provenance": None,
        "is_real_model_prediction": False,
        "probability_is_calibrated": False,
        "artifact_available": MODEL_DIR.is_dir(),
        "reason": reason,
    }


def forecast(df, timestamp, cfg=None, feature_set="final_proxy", horizons=None) -> dict:
    """Run exact +6h XGBoost inference or return an honest unavailable result."""
    ts = parse_exact_utc_hour(timestamp)
    try:
        result = predict(df, ts)
    except (FeatureParityError, FileNotFoundError, KeyError, ValueError) as error:
        return _unavailable(ts, str(error))
    spec = result["spec"]
    return {
        "status": "success",
        "timestamp_utc": ts.isoformat(),
        "target": TARGET,
        "target_definition": "Retrospective operational cargo-congestion proxy",
        "horizon_hours": PRIMARY_HORIZON_HOURS,
        "target_timestamp_utc": (ts + pd.Timedelta(hours=6)).isoformat(),
        "congestion_score": result["score"],
        "classification": result["classification"],
        "model_version": MODEL_VERSION,
        "threshold_metadata": {
            "alert": float(spec["alert_threshold"]),
            "medium": float(spec["risk_thresholds"]["medium"]),
            "high": float(spec["risk_thresholds"]["high"]),
        },
        "alert_active": result["score"] >= float(spec["alert_threshold"]),
        "provenance": DataProvenance.PREDICTED.value,
        "is_real_model_prediction": True,
        "probability_is_calibrated": False,
        "feature_contract": {
            "source_features": result["source_feature_count"],
            "preprocessor_inputs": result["preprocessor_input_count"],
            "model_features": result["model_feature_count"],
            "history_hours": 24,
        },
        "limitations": [
            "The score is not a calibrated probability.",
            "The target is an operational proxy, not independently observed congestion ground truth.",
            "This is a retrospective predictive prototype using finalized 2025 Marsa data.",
            "Some Cargo inputs and weather-pressure thresholds have full-period construction concerns.",
        ],
    }


def validate_test_forecast(value: dict, timestamp) -> dict:
    """Validate an internal-only forecast context used by deterministic tests."""
    ts = parse_exact_utc_hour(timestamp)
    if value.get("status") != "supplied_test_context" or not value.get("test_only"):
        raise ValueError("forecast override must be explicitly marked supplied_test_context/test_only")
    score = value.get("congestion_score")
    if score is not None and not 0.0 <= float(score) <= 1.0:
        raise ValueError("test forecast congestion_score must be between 0 and 1")
    return {
        "status": "supplied_test_context",
        "timestamp_utc": ts.isoformat(),
        "target": TARGET,
        "horizon_hours": PRIMARY_HORIZON_HOURS,
        "congestion_score": score,
        "classification": str(value.get("classification", "UNKNOWN")).upper(),
        "provenance": None,
        "is_real_model_prediction": False,
        "probability_is_calibrated": False,
        "test_only": True,
        "reason": "Caller-supplied deterministic test context; not a model prediction.",
    }
