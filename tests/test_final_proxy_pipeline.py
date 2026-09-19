"""Lightweight tests for the final cargo-congestion proxy pipeline.

Run with:
    python tests/test_final_proxy_pipeline.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prediction_model.src.final_proxy import (
    FINAL_MODEL_VERSION,
    FINAL_PROXY_COLUMN,
    ProxyThresholds,
    add_future_proxy_targets,
    apply_proxy_rule,
    final_source_columns,
    predict_final_proxy,
    proxy_episodes,
)
from prediction_model.src.schema import TIMESTAMP_COLUMN


def proxy_sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            TIMESTAMP_COLUMN: pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T01:00:00Z",
                    "2025-01-01T02:00:00Z",
                    "2025-01-01T04:00:00Z",
                    "2025-01-01T05:00:00Z",
                ],
                utc=True,
            ),
            "container_arrivals": [40, 140, 150, 160, 170],
            "container_departures": [40, 10, 20, 30, 40],
            "containers_in_yard": [100, 900, 920, 950, 980],
            "yard_occupancy_percent": [20, 90, 91, 92, 93],
            "average_dwell_time_hours": [3, 12, 13, 14, 15],
            "truck_waiting_time_minutes": [5, 40, 42, 44, 46],
            "gate_throughput": [100, 20, 25, 25, 25],
        }
    )


def sample_thresholds() -> ProxyThresholds:
    return ProxyThresholds(
        training_window_end="2025-01-01T00:00:00+00:00",
        yard_occupancy_q80=80.0,
        containers_in_yard_q80=800.0,
        average_dwell_time_q80=10.0,
        truck_waiting_time_q80=30.0,
        cargo_flow_imbalance_q75=100.0,
        container_arrivals_q75=100.0,
        gate_throughput_q25=50.0,
        persistence_hours=2,
    )


def test_final_proxy_requires_composite_stress_and_consecutive_persistence() -> None:
    proxied = apply_proxy_rule(proxy_sample_frame(), sample_thresholds())
    assert proxied["cargo_stress_raw"].tolist() == [0, 1, 1, 1, 1]
    assert proxied[FINAL_PROXY_COLUMN].tolist() == [0, 0, 1, 0, 1]


def test_proxy_episodes_respect_hourly_gaps() -> None:
    proxied = apply_proxy_rule(proxy_sample_frame(), sample_thresholds())
    episodes = proxy_episodes(proxied)
    assert len(episodes) == 2
    assert episodes["duration_hours"].tolist() == [1, 1]


def test_future_proxy_targets_require_exact_future_timestamp() -> None:
    proxied = apply_proxy_rule(proxy_sample_frame(), sample_thresholds())
    out, meta = add_future_proxy_targets(proxied, (1,))
    row_02 = out[out[TIMESTAMP_COLUMN] == pd.Timestamp("2025-01-01T02:00:00Z")].iloc[0]
    assert pd.isna(row_02["target_1h"])
    assert int(meta.loc[0, "removed_missing_future_timestamp"]) == 2


def test_final_source_governance_excludes_target_and_status_columns() -> None:
    columns = [
        "operations_status",
        FINAL_PROXY_COLUMN,
        "cargo_stress_raw",
        "target_1h",
        "event_id",
        "event_start",
        "yard_occupancy_percent",
        "container_arrivals",
        "gate_throughput",
        "hour",
    ]
    selected = final_source_columns(columns)
    assert "operations_status" not in selected
    assert FINAL_PROXY_COLUMN not in selected
    assert "cargo_stress_raw" not in selected
    assert not any(column.startswith("target_") for column in selected)
    assert "event_id" not in selected
    assert "event_start" not in selected
    assert {"yard_occupancy_percent", "container_arrivals", "gate_throughput", "hour"}.issubset(selected)


def test_saved_final_artifacts_load_and_inference_schema() -> None:
    data_path = ROOT / "data" / "processed" / "merged_port_dataset_2025_v2.csv"
    model_dir = ROOT / "prediction_model" / "final_proxy" / "artifacts"
    assert (model_dir / "deployment_manifest.json").exists()
    payload = predict_final_proxy(data_path, model_dir)
    assert payload["model_version"] == FINAL_MODEL_VERSION
    assert payload["target_definition"] == "operational_cargo_congestion_proxy"

    prediction_timestamp = pd.Timestamp(payload["prediction_timestamp"])
    horizons = [int(row["horizon_hours"]) for row in payload["predictions"]]
    assert horizons == sorted(horizons)

    for prediction in payload["predictions"]:
        horizon = int(prediction["horizon_hours"])
        assert pd.Timestamp(prediction["target_time"]) == prediction_timestamp + pd.Timedelta(hours=horizon)
        assert 0.0 <= float(prediction["congestion_score"]) <= 1.0
        assert prediction["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
        assert prediction["probability_is_calibrated"] is False


if __name__ == "__main__":
    test_final_proxy_requires_composite_stress_and_consecutive_persistence()
    test_proxy_episodes_respect_hourly_gaps()
    test_future_proxy_targets_require_exact_future_timestamp()
    test_final_source_governance_excludes_target_and_status_columns()
    test_saved_final_artifacts_load_and_inference_schema()
    print("All final proxy pipeline tests passed.")
