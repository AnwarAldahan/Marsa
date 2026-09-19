"""Marsa-native inference for the final retrospective cargo-congestion proxy."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from marsa.common.paths import project_root

TIMESTAMP_COLUMN = "hour_key"
MODEL_VERSION = "layer1-cargo-proxy-v1"
TARGET = "operational_cargo_congestion_proxy"
PRIMARY_HORIZON_HOURS = 6
LAG_HOURS = (1, 3, 6, 12, 24)
ROLLING_HOURS = (3, 6, 12, 24)
MODEL_DIR = project_root() / "prediction_model" / "final_proxy" / "artifacts"


class FeatureParityError(ValueError):
    """Raised when an exact model input cannot be constructed."""


class FittedPreprocessor:
    """Exact loader for the teammate model's fitted tabular preprocessor."""

    def __init__(self, payload: dict) -> None:
        self.numeric_columns = list(payload["numeric_columns"])
        self.categorical_columns = list(payload["categorical_columns"])
        self.medians = {key: float(value) for key, value in payload["medians"].items()}
        self.means = {key: float(value) for key, value in payload["means"].items()}
        self.stds = {key: float(value) for key, value in payload["stds"].items()}
        self.categories = {key: list(value) for key, value in payload["categories"].items()}
        self.feature_names = list(payload["feature_names"])

    @classmethod
    def load(cls, path: Path) -> FittedPreprocessor:
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        parts: list[np.ndarray] = []
        if self.numeric_columns:
            numeric = []
            for column in self.numeric_columns:
                series = pd.to_numeric(frame[column], errors="coerce").astype(float)
                median = self.medians[column]
                filled = series.fillna(median).replace([np.inf, -np.inf], median)
                numeric.append(
                    (filled.to_numpy(dtype=float) - self.means[column]) / self.stds[column]
                )
            parts.append(np.vstack(numeric).T)

        for column in self.categorical_columns:
            values = frame[column].astype("object").where(frame[column].notna(), "MISSING").astype(str)
            categories = self.categories[column]
            encoded = np.zeros((len(frame), len(categories)), dtype=float)
            category_index = {category: index for index, category in enumerate(categories)}
            for row_index, value in enumerate(values):
                index = category_index.get(value)
                if index is not None:
                    encoded[row_index, index] = 1.0
            parts.append(encoded)
        return np.hstack(parts).astype(float)


def load_manifest(model_dir: Path = MODEL_DIR) -> dict:
    return json.loads((model_dir / "deployment_manifest.json").read_text(encoding="utf-8"))


def horizon_spec(manifest: dict, horizon_hours: int = PRIMARY_HORIZON_HOURS) -> dict:
    try:
        return next(
            spec for spec in manifest["horizons"]
            if int(spec["horizon_hours"]) == horizon_hours
        )
    except StopIteration as error:
        raise FeatureParityError(f"No model artifact exists for horizon {horizon_hours}h") from error


def prepare_source_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy().sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    work[TIMESTAMP_COLUMN] = pd.to_datetime(work[TIMESTAMP_COLUMN], utc=True, errors="raise")
    required = {"container_arrivals", "container_departures"}
    missing = sorted(required - set(work.columns))
    if missing:
        raise FeatureParityError(f"Missing source columns: {', '.join(missing)}")
    work["cargo_flow_imbalance"] = work["container_arrivals"] - work["container_departures"]
    if "dow_sin" not in work:
        work["dow_sin"] = np.sin(2 * np.pi * work["day_of_week"] / 7)
    if "dow_cos" not in work:
        work["dow_cos"] = np.cos(2 * np.pi * work["day_of_week"] / 7)
    return work


def build_feature_row(df: pd.DataFrame, timestamp: pd.Timestamp, spec: dict) -> pd.DataFrame:
    work = prepare_source_frame(df)
    source_columns = list(spec["source_columns"])
    missing_sources = [column for column in source_columns if column not in work.columns]
    if missing_sources:
        raise FeatureParityError(f"Missing source columns: {', '.join(missing_sources)}")

    indexed = work.set_index(TIMESTAMP_COLUMN, drop=False)
    if timestamp not in indexed.index:
        raise FeatureParityError(f"Prediction timestamp {timestamp.isoformat()} is not present")
    missing_history = [
        timestamp - pd.Timedelta(hours=offset)
        for offset in range(25)
        if timestamp - pd.Timedelta(hours=offset) not in indexed.index
    ]
    if missing_history:
        raise FeatureParityError(
            "Required exact 24-hour history is incomplete; first missing timestamp: "
            f"{missing_history[0].isoformat()}"
        )

    values: dict[str, object] = {}
    for column in source_columns:
        series = work[column]
        if not (pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)):
            value = indexed.at[timestamp, column]
            values[f"{column}__current"] = "MISSING" if pd.isna(value) else str(value)
            continue
        values[f"{column}__current"] = indexed.at[timestamp, column]
        for lag in LAG_HOURS:
            values[f"{column}__lag_{lag}h"] = indexed.at[
                timestamp - pd.Timedelta(hours=lag), column
            ]
        for window in ROLLING_HOURS:
            window_values = [
                indexed.at[timestamp - pd.Timedelta(hours=offset), column]
                for offset in range(window)
            ]
            numeric = pd.to_numeric(pd.Series(window_values), errors="coerce")
            values[f"{column}__roll_mean_{window}h"] = float(numeric.mean(skipna=True))
            std = float(numeric.std(skipna=True))
            values[f"{column}__roll_std_{window}h"] = 0.0 if math.isnan(std) else std

    expected = list(spec["feature_columns"])
    missing_features = [column for column in expected if column not in values]
    if missing_features:
        raise FeatureParityError(f"Could not construct features: {', '.join(missing_features)}")
    return pd.DataFrame([values], columns=expected)


def classify_score(score: float, spec: dict) -> str:
    thresholds = spec["risk_thresholds"]
    if score >= float(thresholds["high"]):
        return "HIGH"
    if score >= float(thresholds["medium"]):
        return "MEDIUM"
    return "LOW"


def predict(df: pd.DataFrame, timestamp: pd.Timestamp, model_dir: Path = MODEL_DIR) -> dict:
    manifest = load_manifest(model_dir)
    spec = horizon_spec(manifest)
    if spec["model_type"] != "XGBoost":
        raise FeatureParityError("Primary model artifact is not XGBoost")
    row = build_feature_row(df, timestamp, spec)
    preprocessor = FittedPreprocessor.load(model_dir / spec["preprocessor_path"])
    matrix = preprocessor.transform(row)
    if row.shape[1] != 448 or matrix.shape[1] != 604:
        raise FeatureParityError(
            f"Feature parity failed: expected 448/604, got {row.shape[1]}/{matrix.shape[1]}"
        )
    booster = xgb.Booster()
    booster.load_model(str(model_dir / spec["model_path"]))
    score = float(booster.predict(xgb.DMatrix(matrix))[0])
    return {
        "score": score,
        "classification": classify_score(score, spec),
        "spec": spec,
        "source_feature_count": len(spec["source_columns"]),
        "preprocessor_input_count": row.shape[1],
        "model_feature_count": matrix.shape[1],
    }
