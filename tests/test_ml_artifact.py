import json

import pandas as pd
import pytest
import xgboost as xgb

from marsa.data.load import load_dataset
from marsa.model.congestion_proxy import (
    MODEL_DIR,
    FittedPreprocessor,
    build_feature_row,
    classify_score,
    horizon_spec,
    load_manifest,
)
from marsa.model.predict import forecast

VALID_TIMESTAMP = pd.Timestamp("2025-01-02T08:00:00Z")


def _spec() -> dict:
    return horizon_spec(load_manifest())


def test_final_xgboost_artifact_and_exact_feature_contract():
    spec = _spec()
    assert spec["model_type"] == "XGBoost"
    assert spec["model_identifier"] == "XGBoost_6h"
    assert len(spec["source_columns"]) == 32
    assert len(spec["feature_columns"]) == 448
    assert spec["feature_columns"][:14] == [
        "vessel_count__current",
        "vessel_count__lag_1h",
        "vessel_count__lag_3h",
        "vessel_count__lag_6h",
        "vessel_count__lag_12h",
        "vessel_count__lag_24h",
        "vessel_count__roll_mean_3h",
        "vessel_count__roll_std_3h",
        "vessel_count__roll_mean_6h",
        "vessel_count__roll_std_6h",
        "vessel_count__roll_mean_12h",
        "vessel_count__roll_std_12h",
        "vessel_count__roll_mean_24h",
        "vessel_count__roll_std_24h",
    ]
    booster = xgb.Booster()
    booster.load_model(str(MODEL_DIR / spec["model_path"]))
    assert booster.num_features() == 604


def test_native_features_and_prediction_match_final_teammate_artifact():
    frame = load_dataset()
    spec = _spec()
    row = build_feature_row(frame, VALID_TIMESTAMP, spec)
    assert list(row.columns) == spec["feature_columns"]
    assert row.at[0, "cargo_flow_imbalance__current"] == (
        row.at[0, "container_arrivals__current"]
        - row.at[0, "container_departures__current"]
    )
    preprocessor = FittedPreprocessor.load(MODEL_DIR / spec["preprocessor_path"])
    transformed = preprocessor.transform(row)
    assert transformed.shape == (1, 604)
    booster = xgb.Booster()
    booster.load_model(str(MODEL_DIR / spec["model_path"]))
    reference_score = float(booster.predict(xgb.DMatrix(transformed))[0])
    result = forecast(frame, VALID_TIMESTAMP)
    assert result["congestion_score"] == pytest.approx(reference_score, abs=1e-12)
    assert result["congestion_score"] == pytest.approx(0.008864540606737137, abs=1e-12)
    assert result["classification"] == "LOW"
    assert result["provenance"] == "PREDICTED"
    assert result["is_real_model_prediction"] is True
    assert "probability" not in result


def test_required_history_is_exact_and_missing_hours_are_not_interpolated():
    frame = load_dataset()
    result = forecast(frame, "2025-04-27T06:00:00Z")
    assert result["status"] == "ml_unavailable"
    assert result["congestion_score"] is None
    assert result["is_real_model_prediction"] is False
    assert "2025-04-26T08:00:00+00:00" in result["reason"]

    removed = frame[frame["hour_key"] != pd.Timestamp("2025-01-02T07:00:00Z")]
    result = forecast(removed, VALID_TIMESTAMP)
    assert result["status"] == "ml_unavailable"
    assert "2025-01-02T07:00:00+00:00" in result["reason"]


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, "LOW"),
        (0.10340226814150809, "LOW"),
        (0.1034022681415081, "MEDIUM"),
        (0.5280332624912266, "MEDIUM"),
        (0.5280332624912267, "HIGH"),
    ],
)
def test_approved_score_boundaries(score, expected):
    assert classify_score(score, _spec()) == expected


def test_manifest_preserves_proxy_and_uncalibrated_score_metadata():
    manifest = json.loads((MODEL_DIR / "deployment_manifest.json").read_text())
    assert manifest["model_version"] == "layer1-cargo-proxy-v1"
    assert manifest["target_definition"]["name"] == "operational_cargo_congestion_proxy"
    assert manifest["score_calibration"]["probability_is_calibrated"] is False
