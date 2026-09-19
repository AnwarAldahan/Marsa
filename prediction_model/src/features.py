"""Leakage-safe time and tabular historical feature engineering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
from typing import Iterable

import numpy as np
import pandas as pd

from prediction_model.src.schema import TIMESTAMP_COLUMN
from prediction_model.src.splits import assign_split_by_target_timestamp


LAG_HOURS = (1, 3, 6, 12, 24)
ROLLING_HOURS = (3, 6, 12, 24)


@dataclass
class FeatureBuildMetadata:
    horizon_hours: int
    feature_group: str
    candidate_rows: int
    usable_rows: int
    rows_removed_missing_future_target: int
    rows_removed_incomplete_history: int
    feature_count: int
    numeric_source_features: int
    categorical_source_features: int


def ensure_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = out[TIMESTAMP_COLUMN]
    if "hour" not in out.columns:
        out["hour"] = ts.dt.hour
    if "day_of_week" not in out.columns:
        out["day_of_week"] = ts.dt.dayofweek
    if "month" not in out.columns:
        out["month"] = ts.dt.month
    if "is_weekend" not in out.columns:
        out["is_weekend"] = out["day_of_week"].isin([5, 6])
    if "hour_sin" not in out.columns:
        out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    if "hour_cos" not in out.columns:
        out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    if "dow_sin" not in out.columns:
        out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    if "dow_cos" not in out.columns:
        out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    return out


def split_feature_types(df: pd.DataFrame, columns: Iterable[str]) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for column in columns:
        if column not in df.columns or column == TIMESTAMP_COLUMN:
            continue
        if pd.api.types.is_bool_dtype(df[column]):
            numeric.append(column)
        elif pd.api.types.is_numeric_dtype(df[column]):
            numeric.append(column)
        else:
            categorical.append(column)
    return numeric, categorical


def continuous_history_mask(df: pd.DataFrame, max_back_hours: int) -> pd.Series:
    timestamps = pd.DatetimeIndex(df[TIMESTAMP_COLUMN])
    present = set(timestamps)
    mask = []
    for ts in timestamps:
        ok = True
        for offset in range(0, max_back_hours + 1):
            if ts - pd.Timedelta(hours=offset) not in present:
                ok = False
                break
        mask.append(ok)
    return pd.Series(mask, index=df.index)


def _timestamp_mapped_series(
    df: pd.DataFrame,
    value_column: str,
    target_timestamps: pd.Series,
) -> pd.Series:
    lookup = df.set_index(TIMESTAMP_COLUMN)[value_column]
    return target_timestamps.map(lookup)


def build_tabular_lag_features(
    df: pd.DataFrame,
    source_columns: list[str],
    horizon_hours: int,
    feature_group: str,
) -> tuple[pd.DataFrame, FeatureBuildMetadata]:
    target_col = f"target_{horizon_hours}h"
    target_ts_col = f"target_timestamp_{horizon_hours}h"
    if target_col not in df.columns or target_ts_col not in df.columns:
        raise ValueError(f"Future target columns for {horizon_hours}h are missing.")

    numeric_cols, categorical_cols = split_feature_types(df, source_columns)
    max_back = max(max(LAG_HOURS), max(ROLLING_HOURS) - 1)
    history_ok = continuous_history_mask(df, max_back)
    target_ok = df[target_col].notna()

    parts: list[pd.DataFrame] = [
        pd.DataFrame(
            {
                TIMESTAMP_COLUMN: df[TIMESTAMP_COLUMN],
                target_ts_col: df[target_ts_col],
                target_col: df[target_col],
                "split": assign_split_by_target_timestamp(df[target_ts_col]),
                "history_window_valid": history_ok,
            }
        )
    ]

    for column in numeric_cols:
        column_features: dict[str, pd.Series] = {}
        values = df[column]
        if pd.api.types.is_bool_dtype(values):
            values = values.astype(float)
        column_features[f"{column}__current"] = values
        for lag in LAG_HOURS:
            column_features[f"{column}__lag_{lag}h"] = _timestamp_mapped_series(
                df, column, df[TIMESTAMP_COLUMN] - pd.Timedelta(hours=lag)
            )
        for window in ROLLING_HOURS:
            lagged = []
            for offset in range(window):
                lagged.append(
                    _timestamp_mapped_series(df, column, df[TIMESTAMP_COLUMN] - pd.Timedelta(hours=offset))
                )
            stacked = pd.concat(lagged, axis=1)
            column_features[f"{column}__roll_mean_{window}h"] = stacked.mean(axis=1, skipna=True)
            column_features[f"{column}__roll_std_{window}h"] = stacked.std(axis=1, skipna=True).fillna(0.0)
        parts.append(pd.DataFrame(column_features))

    categorical_features: dict[str, pd.Series] = {}
    for column in categorical_cols:
        categorical_features[f"{column}__current"] = df[column].astype("object").where(df[column].notna(), "MISSING")
    if categorical_features:
        parts.append(pd.DataFrame(categorical_features))

    out = pd.concat(parts, axis=1)

    candidate = out[target_ok].copy()
    usable = candidate[candidate["history_window_valid"] & (candidate["split"] != "outside")].copy()
    usable = usable.drop(columns=["history_window_valid"])
    feature_count = len([c for c in usable.columns if c not in {TIMESTAMP_COLUMN, target_ts_col, target_col, "split"}])
    metadata = FeatureBuildMetadata(
        horizon_hours=horizon_hours,
        feature_group=feature_group,
        candidate_rows=int(len(out)),
        usable_rows=int(len(usable)),
        rows_removed_missing_future_target=int((~target_ok).sum()),
        rows_removed_incomplete_history=int((target_ok & ~history_ok).sum()),
        feature_count=int(feature_count),
        numeric_source_features=int(len(numeric_cols)),
        categorical_source_features=int(len(categorical_cols)),
    )
    return usable.reset_index(drop=True), metadata


class TabularPreprocessor:
    """Train-only imputation, scaling, and one-hot encoding."""

    def __init__(self) -> None:
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.medians: dict[str, float] = {}
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.categories: dict[str, list[str]] = {}
        self.feature_names: list[str] = []

    def fit(self, frame: pd.DataFrame) -> "TabularPreprocessor":
        self.numeric_columns = [
            c
            for c in frame.columns
            if pd.api.types.is_numeric_dtype(frame[c]) or pd.api.types.is_bool_dtype(frame[c])
        ]
        self.categorical_columns = [c for c in frame.columns if c not in self.numeric_columns]

        self.medians = {}
        self.means = {}
        self.stds = {}
        for column in self.numeric_columns:
            series = pd.to_numeric(frame[column], errors="coerce").astype(float)
            finite = series[np.isfinite(series)]
            median = float(finite.median()) if len(finite) else 0.0
            filled = series.fillna(median).replace([np.inf, -np.inf], median)
            mean = float(filled.mean()) if len(filled) else 0.0
            std = float(filled.std(ddof=0)) if len(filled) else 1.0
            if not math.isfinite(std) or std == 0:
                std = 1.0
            self.medians[column] = median
            self.means[column] = mean
            self.stds[column] = std

        self.categories = {}
        for column in self.categorical_columns:
            values = frame[column].astype("object").where(frame[column].notna(), "MISSING").astype(str)
            self.categories[column] = sorted(values.unique().tolist())

        self.feature_names = []
        self.feature_names.extend(self.numeric_columns)
        for column in self.categorical_columns:
            self.feature_names.extend([f"{column}=={category}" for category in self.categories[column]])
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        parts: list[np.ndarray] = []
        if self.numeric_columns:
            numeric = []
            for column in self.numeric_columns:
                series = pd.to_numeric(frame[column], errors="coerce").astype(float)
                median = self.medians[column]
                filled = series.fillna(median).replace([np.inf, -np.inf], median)
                scaled = (filled.to_numpy(dtype=float) - self.means[column]) / self.stds[column]
                numeric.append(scaled)
            parts.append(np.vstack(numeric).T)

        for column in self.categorical_columns:
            values = frame[column].astype("object").where(frame[column].notna(), "MISSING").astype(str)
            categories = self.categories[column]
            encoded = np.zeros((len(frame), len(categories)), dtype=float)
            category_index = {category: idx for idx, category in enumerate(categories)}
            for row_idx, value in enumerate(values):
                idx = category_index.get(value)
                if idx is not None:
                    encoded[row_idx, idx] = 1.0
            parts.append(encoded)

        if not parts:
            return np.empty((len(frame), 0), dtype=float)
        return np.hstack(parts).astype(float)

    def to_dict(self) -> dict:
        return {
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "medians": self.medians,
            "means": self.means,
            "stds": self.stds,
            "categories": self.categories,
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TabularPreprocessor":
        obj = cls()
        obj.numeric_columns = list(payload["numeric_columns"])
        obj.categorical_columns = list(payload["categorical_columns"])
        obj.medians = {k: float(v) for k, v in payload["medians"].items()}
        obj.means = {k: float(v) for k, v in payload["means"].items()}
        obj.stds = {k: float(v) for k, v in payload["stds"].items()}
        obj.categories = {k: list(v) for k, v in payload["categories"].items()}
        obj.feature_names = list(payload["feature_names"])
        return obj

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TabularPreprocessor":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def feature_columns_from_frame(frame: pd.DataFrame, target_col: str, target_ts_col: str) -> list[str]:
    excluded = {TIMESTAMP_COLUMN, target_ts_col, target_col, "split"}
    return [column for column in frame.columns if column not in excluded]


def build_single_tabular_feature_row(
    df: pd.DataFrame,
    source_columns: list[str],
    prediction_timestamp: pd.Timestamp | str | None,
    expected_feature_columns: list[str],
) -> pd.DataFrame:
    """Build one inference row using only data at or before prediction_timestamp."""
    work = ensure_time_features(df).sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    if prediction_timestamp is None:
        ts = work[TIMESTAMP_COLUMN].max()
    else:
        ts = pd.Timestamp(prediction_timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

    if ts not in set(work[TIMESTAMP_COLUMN]):
        raise ValueError(f"Prediction timestamp {ts.isoformat()} is not present in the supplied history.")

    max_back = max(max(LAG_HOURS), max(ROLLING_HOURS) - 1)
    present = set(pd.DatetimeIndex(work[TIMESTAMP_COLUMN]))
    missing_history = [
        (ts - pd.Timedelta(hours=offset)).isoformat()
        for offset in range(0, max_back + 1)
        if ts - pd.Timedelta(hours=offset) not in present
    ]
    if missing_history:
        raise ValueError(
            "Cannot build leakage-safe inference features because the required "
            f"{max_back}h continuous history is incomplete. First missing timestamp: {missing_history[0]}"
        )

    row = work[work[TIMESTAMP_COLUMN] == ts].iloc[0]
    numeric_cols, categorical_cols = split_feature_types(work, source_columns)
    values: dict[str, object] = {}
    for column in numeric_cols:
        values[f"{column}__current"] = row[column]
        for lag in LAG_HOURS:
            lag_ts = ts - pd.Timedelta(hours=lag)
            values[f"{column}__lag_{lag}h"] = work.loc[work[TIMESTAMP_COLUMN] == lag_ts, column].iloc[0]
        for window in ROLLING_HOURS:
            window_values = []
            for offset in range(window):
                sample_ts = ts - pd.Timedelta(hours=offset)
                window_values.append(work.loc[work[TIMESTAMP_COLUMN] == sample_ts, column].iloc[0])
            numeric_window = pd.to_numeric(pd.Series(window_values), errors="coerce")
            values[f"{column}__roll_mean_{window}h"] = float(numeric_window.mean(skipna=True))
            std = float(numeric_window.std(skipna=True))
            values[f"{column}__roll_std_{window}h"] = 0.0 if np.isnan(std) else std

    for column in categorical_cols:
        value = row[column]
        values[f"{column}__current"] = "MISSING" if pd.isna(value) else str(value)

    output = pd.DataFrame([values])
    for column in expected_feature_columns:
        if column not in output.columns:
            output[column] = np.nan
    return output[expected_feature_columns]
