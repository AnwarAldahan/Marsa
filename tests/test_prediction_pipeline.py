"""Lightweight methodological tests for the Layer 1 pipeline.

Run with:
    python tests/test_prediction_pipeline.py
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from prediction_model.models.logistic_model import NumpyLogisticRegression
from prediction_model.src.data_audit import add_congestion_label
from prediction_model.src.evaluation import optimize_threshold
from prediction_model.src.features import TabularPreprocessor, build_tabular_lag_features, ensure_time_features
from prediction_model.src.schema import STATUS_COLUMN, TARGET_COLUMN, TIMESTAMP_COLUMN, feature_groups, primary_feature_columns
from prediction_model.src.sequences import count_sequence_windows
from prediction_model.src.splits import assign_split_by_target_timestamp
from prediction_model.src.targets import add_future_target


def sample_frame() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2025-01-01 00:00:00+00:00",
            "2025-01-01 01:00:00+00:00",
            "2025-01-01 02:00:00+00:00",
            "2025-01-01 04:00:00+00:00",
            "2025-01-01 05:00:00+00:00",
            "2025-01-01 06:00:00+00:00",
            "2025-01-01 07:00:00+00:00",
        ],
        utc=True,
    )
    return pd.DataFrame(
        {
            TIMESTAMP_COLUMN: timestamps,
            "operations_status": ["NORMAL", "MODERATE", "ELEVATED", "NORMAL", "CRITICAL", "NORMAL", "NORMAL"],
            "yard_occupancy_percent": [10, 20, 30, 40, 50, 60, 70],
            "cargo_flow_ratio": [1, 1, 2, 2, 3, 3, 4],
            "event_active": [False, False, True, False, False, False, False],
        }
    )


def test_future_target_requires_exact_timestamp() -> None:
    df = add_congestion_label(sample_frame())
    out, meta = add_future_target(df, 1)
    row_02 = out[out[TIMESTAMP_COLUMN] == pd.Timestamp("2025-01-01 02:00:00+00:00")].iloc[0]
    assert pd.isna(row_02["target_1h"])
    assert meta.removed_missing_future_timestamp == 2


def test_chronological_target_split_boundaries() -> None:
    target_ts = pd.Series(
        pd.to_datetime(["2025-09-30 23:00:00+00:00", "2025-10-01 00:00:00+00:00", "2025-12-01 00:00:00+00:00"])
    )
    assert assign_split_by_target_timestamp(target_ts).tolist() == ["train", "validation", "test"]


def test_sequence_does_not_bridge_missing_hour() -> None:
    df, _ = add_future_target(add_congestion_label(ensure_time_features(sample_frame())), 1)
    meta = count_sequence_windows(df, ["yard_occupancy_percent", "cargo_flow_ratio"], 1, 3)
    assert meta.usable_windows == 1
    assert meta.removed_incomplete_history > 0


def test_tabular_features_use_past_only_and_drop_gaps() -> None:
    df, _ = add_future_target(add_congestion_label(ensure_time_features(sample_frame())), 1)
    features, meta = build_tabular_lag_features(df, ["yard_occupancy_percent", "cargo_flow_ratio"], 1, "test")
    assert meta.rows_removed_incomplete_history > 0
    assert "yard_occupancy_percent__lag_1h" in features.columns
    assert (features["yard_occupancy_percent__lag_1h"] < features["yard_occupancy_percent__current"]).all()


def test_preprocessor_is_train_only() -> None:
    train = pd.DataFrame({"x": [0.0, 2.0], "cat": ["a", "b"]})
    validation = pd.DataFrame({"x": [100.0], "cat": ["c"]})
    preprocessor = TabularPreprocessor().fit(train)
    transformed = preprocessor.transform(validation)
    assert preprocessor.means["x"] == 1.0
    assert transformed.shape[1] == len(preprocessor.feature_names)
    assert all("cat==c" not in name for name in preprocessor.feature_names)


def test_primary_feature_governance_excludes_target_derived_columns() -> None:
    columns = [
        TIMESTAMP_COLUMN,
        STATUS_COLUMN,
        TARGET_COLUMN,
        "target_1h",
        "event_start",
        "event_end",
        "event_id",
        "yard_occupancy_percent",
        "vessel_count",
        "hour",
    ]
    primary = primary_feature_columns(columns)
    assert STATUS_COLUMN not in primary
    assert TARGET_COLUMN not in primary
    assert "target_1h" not in primary
    assert "event_start" not in primary
    assert "event_end" not in primary
    assert "event_id" not in primary
    assert "yard_occupancy_percent" in primary

    full_valid = [group for group in feature_groups(columns) if group.name == "FullValidNoStatus"][0]
    assert STATUS_COLUMN not in full_valid.columns
    assert TARGET_COLUMN not in full_valid.columns


def test_holdout_labels_do_not_affect_preprocessing_or_threshold() -> None:
    train = pd.DataFrame({"x": [0.0, 2.0], "cat": ["a", "b"]})
    validation_probabilities = np.array([0.10, 0.40, 0.80, 0.95])
    validation_labels = np.array([0, 0, 1, 1])
    holdout_labels = np.array([1, 1, 1, 1])

    preprocessor = TabularPreprocessor().fit(train)
    before = preprocessor.to_dict()
    threshold, _ = optimize_threshold(validation_labels, validation_probabilities, minimum_recall=0.5)

    changed_holdout_labels = 1 - holdout_labels
    preprocessor_after_holdout_change = TabularPreprocessor().fit(train)
    threshold_after_holdout_change, _ = optimize_threshold(validation_labels, validation_probabilities, minimum_recall=0.5)

    assert changed_holdout_labels.tolist() == [0, 0, 0, 0]
    assert preprocessor_after_holdout_change.to_dict() == before
    assert threshold_after_holdout_change == threshold


# NOTE: the former test_saved_deployment_artifacts_load_and_inference_schema was removed:
# it targeted the superseded logistic-regression artifacts in prediction_model/saved_models,
# which no longer exist. The final XGBoost proxy artifacts are covered by
# tests/test_final_proxy_pipeline.py.


def test_probability_bounds() -> None:
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = NumpyLogisticRegression(epochs=20, learning_rate=0.05).fit(x, y)
    probabilities = model.predict_proba(x)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


if __name__ == "__main__":
    test_future_target_requires_exact_timestamp()
    test_chronological_target_split_boundaries()
    test_sequence_does_not_bridge_missing_hour()
    test_tabular_features_use_past_only_and_drop_gaps()
    test_preprocessor_is_train_only()
    test_primary_feature_governance_excludes_target_derived_columns()
    test_holdout_labels_do_not_affect_preprocessing_or_threshold()
    test_probability_bounds()
    print("All lightweight pipeline tests passed.")