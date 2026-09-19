"""Agent 1-compatible multi-horizon inference interface."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import pandas as pd

from prediction_model.models.logistic_model import NumpyLogisticRegression
from prediction_model.src.calibration import PlattCalibrator
from prediction_model.src.data_audit import load_dataset
from prediction_model.src.features import (
    TabularPreprocessor,
    build_single_tabular_feature_row,
    ensure_time_features,
)


def _load_artifact(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_deployment_manifest(model_dir: Path) -> dict[str, Any]:
    manifest_path = model_dir / "deployment_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Deployment manifest not found: {manifest_path}")
    return _load_artifact(manifest_path)


def predict_future_congestion(
    history_csv: Path,
    model_dir: Path,
    prediction_timestamp: str | None = None,
) -> dict[str, Any]:
    df = ensure_time_features(load_dataset(history_csv))
    manifest = load_deployment_manifest(model_dir)
    timestamp = pd.Timestamp(prediction_timestamp) if prediction_timestamp else df["hour_key"].max()
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    predictions = []
    for horizon_spec in manifest["horizons"]:
        horizon = int(horizon_spec["horizon_hours"])
        feature_columns = list(horizon_spec["feature_columns"])
        source_columns = list(horizon_spec["source_columns"])
        feature_row = build_single_tabular_feature_row(df, source_columns, timestamp, feature_columns)
        preprocessor = TabularPreprocessor.load(model_dir / horizon_spec["preprocessor_path"])
        model = NumpyLogisticRegression.load(model_dir / horizon_spec["model_path"])
        x = preprocessor.transform(feature_row)
        probability = float(model.predict_proba(x)[0])
        calibration_method = horizon_spec["calibration_method"]
        if calibration_method == "platt_sigmoid":
            calibrator = PlattCalibrator.from_dict(horizon_spec["calibrator"])
            probability = float(calibrator.predict([probability])[0])
        threshold = float(horizon_spec["alert_threshold"])
        predictions.append(
            {
                "target_timestamp": (timestamp + pd.Timedelta(hours=horizon)).isoformat(),
                "horizon_hours": horizon,
                "congestion_probability": probability,
                "alert_threshold": threshold,
                "predicted_congestion_state": int(probability >= threshold),
                "model_identifier": horizon_spec["model_identifier"],
                "calibration_method": calibration_method,
            }
        )

    return {
        "prediction_timestamp": timestamp.isoformat(),
        "model_version": manifest["model_version"],
        "feature_group": manifest["feature_group"],
        "predictions": predictions,
    }


def save_prediction_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
