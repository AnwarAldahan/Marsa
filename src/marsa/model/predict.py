"""Runtime contract for the preserved XGBoost C1 forecast artifact.

The target merged table does not contain every trained feature under its approved
meaning. This module reports an explicit unavailable state instead of renaming or
approximating features.
"""
from __future__ import annotations

import json
from pathlib import Path

from marsa.data.load import parse_exact_utc_hour

APPROVED_FEATURES = (
    "vessels_in_area", "vessels_waiting", "waiting_ratio",
    "cargo_vessels_in_area", "tanker_vessels_in_area", "avg_sog",
    "median_sog", "ship_density", "avg_port_speed", "port_throughput_lag1",
    "wind_speed_10m", "wave_height", "weather_code", "hour_sin", "hour_cos",
    "day_of_week", "month",
)

ARTIFACT_PATH = Path(__file__).resolve().parents[3] / "artifacts" / "models" / "marsa_xgboost_c1.joblib"
METADATA_PATH = ARTIFACT_PATH.with_name("marsa_xgboost_c1_metadata.json")


def forecast(df, timestamp, cfg=None, feature_set="ais_only", horizons=None) -> dict:
    ts = parse_exact_utc_hour(timestamp)
    missing = [feature for feature in APPROVED_FEATURES if feature not in set(df.columns)]
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {
        "status": "ml_unavailable",
        "timestamp_utc": ts.isoformat(),
        "target": "c1_congestion_next_6h",
        "horizon_hours": 6,
        "window": "(t, t + 6h]",
        "probability": None,
        "classification": None,
        "provenance": None,
        "is_real_model_prediction": False,
        "artifact_available": ARTIFACT_PATH.is_file(),
        "model_version": metadata["model_version"],
        "selected_probability_threshold": metadata["selected_probability_threshold"],
        "missing_required_features": missing,
        "reason": (
            "Exact approved runtime feature parity is unavailable; no model inference "
            "was fabricated."
        ),
    }


def validate_test_forecast(value: dict, timestamp) -> dict:
    """Validate an internal-only forecast context used by deterministic tests."""
    ts = parse_exact_utc_hour(timestamp)
    if value.get("status") != "supplied_test_context" or not value.get("test_only"):
        raise ValueError("forecast override must be explicitly marked supplied_test_context/test_only")
    probability = value.get("probability")
    if probability is not None and not 0.0 <= float(probability) <= 1.0:
        raise ValueError("test forecast probability must be between 0 and 1")
    return {
        "status": "supplied_test_context",
        "timestamp_utc": ts.isoformat(),
        "target": "c1_congestion_next_6h",
        "horizon_hours": 6,
        "window": "(t, t + 6h]",
        "probability": probability,
        "classification": str(value.get("classification", "UNKNOWN")).upper(),
        "provenance": None,
        "is_real_model_prediction": False,
        "test_only": True,
        "reason": "Caller-supplied deterministic test context; not a model prediction.",
    }
