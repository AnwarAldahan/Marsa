"""Final cargo-side congestion proxy and integration inference helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Any

import numpy as np
import pandas as pd

from prediction_model.src.data_audit import load_dataset
from prediction_model.src.features import (
    TabularPreprocessor,
    build_single_tabular_feature_row,
    ensure_time_features,
    split_feature_types,
)
from prediction_model.src.schema import TIMESTAMP_COLUMN


FINAL_PROXY_COLUMN = "cargo_congestion_proxy"
RAW_STRESS_COLUMN = "cargo_stress_raw"
FLOW_IMBALANCE_COLUMN = "cargo_flow_imbalance"
FINAL_MODEL_VERSION = "layer1-cargo-proxy-v1"


@dataclass
class ProxyThresholds:
    training_window_end: str
    yard_occupancy_q80: float
    containers_in_yard_q80: float
    average_dwell_time_q80: float
    truck_waiting_time_q80: float
    cargo_flow_imbalance_q75: float
    container_arrivals_q75: float
    gate_throughput_q25: float
    persistence_hours: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProxyThresholds":
        return cls(
            training_window_end=str(payload["training_window_end"]),
            yard_occupancy_q80=float(payload["yard_occupancy_q80"]),
            containers_in_yard_q80=float(payload["containers_in_yard_q80"]),
            average_dwell_time_q80=float(payload["average_dwell_time_q80"]),
            truck_waiting_time_q80=float(payload["truck_waiting_time_q80"]),
            cargo_flow_imbalance_q75=float(payload["cargo_flow_imbalance_q75"]),
            container_arrivals_q75=float(payload["container_arrivals_q75"]),
            gate_throughput_q25=float(payload["gate_throughput_q25"]),
            persistence_hours=int(payload.get("persistence_hours", 2)),
        )


def add_cargo_flow_imbalance(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[FLOW_IMBALANCE_COLUMN] = out["container_arrivals"] - out["container_departures"]
    return out


def final_source_columns(columns: list[str]) -> list[str]:
    """Prediction-time feature families for the final proxy model.

    Event outcome columns are intentionally excluded unless operational
    availability is later proven.
    """
    requested = [
        "vessel_count",
        "waiting_vessel_count",
        "approaching_vessel_count",
        "departing_vessel_count",
        "average_speed",
        "port_throughput",
        "import_teu",
        "export_teu",
        "container_arrivals",
        "container_departures",
        "containers_in_yard",
        "yard_capacity",
        "average_dwell_time_hours",
        "truck_arrivals",
        "truck_departures",
        "truck_waiting_time_minutes",
        "gate_throughput",
        "yard_occupancy_percent",
        FLOW_IMBALANCE_COLUMN,
        "wind_speed_10m",
        "wave_height",
        "weather_code",
        "weather_pressure_index",
        "hour",
        "hour_sin",
        "hour_cos",
        "day_of_week",
        "dow_sin",
        "dow_cos",
        "month",
        "is_weekend",
        "is_public_holiday",
    ]
    present = set(columns)
    return [column for column in requested if column in present]


def compute_proxy_thresholds(train_df: pd.DataFrame, training_window_end: pd.Timestamp | str) -> ProxyThresholds:
    work = add_cargo_flow_imbalance(train_df)
    return ProxyThresholds(
        training_window_end=pd.Timestamp(training_window_end).isoformat(),
        yard_occupancy_q80=float(work["yard_occupancy_percent"].quantile(0.80)),
        containers_in_yard_q80=float(work["containers_in_yard"].quantile(0.80)),
        average_dwell_time_q80=float(work["average_dwell_time_hours"].quantile(0.80)),
        truck_waiting_time_q80=float(work["truck_waiting_time_minutes"].quantile(0.80)),
        cargo_flow_imbalance_q75=float(max(0.0, work[FLOW_IMBALANCE_COLUMN].quantile(0.75))),
        container_arrivals_q75=float(work["container_arrivals"].quantile(0.75)),
        gate_throughput_q25=float(work["gate_throughput"].quantile(0.25)),
        persistence_hours=2,
    )


def apply_proxy_rule(df: pd.DataFrame, thresholds: ProxyThresholds) -> pd.DataFrame:
    out = add_cargo_flow_imbalance(df)
    yard_stress = (
        (out["yard_occupancy_percent"] >= thresholds.yard_occupancy_q80)
        | (out["containers_in_yard"] >= thresholds.containers_in_yard_q80)
    )
    delay_stress = (
        (out["average_dwell_time_hours"] >= thresholds.average_dwell_time_q80)
        | (out["truck_waiting_time_minutes"] >= thresholds.truck_waiting_time_q80)
    )
    flow_stress = (
        (out[FLOW_IMBALANCE_COLUMN] >= thresholds.cargo_flow_imbalance_q75)
        | (
            (out["gate_throughput"] <= thresholds.gate_throughput_q25)
            & (
                (out["container_arrivals"] >= thresholds.container_arrivals_q75)
                | (out[FLOW_IMBALANCE_COLUMN] > 0)
            )
        )
    )
    raw = yard_stress & delay_stress & flow_stress
    timestamps = pd.DatetimeIndex(out[TIMESTAMP_COLUMN])
    previous_is_consecutive = pd.Series(timestamps).diff().dt.total_seconds().fillna(0).to_numpy() == 3600
    persisted = raw & raw.shift(1, fill_value=False) & previous_is_consecutive

    out["yard_stress_signal"] = yard_stress.astype(int)
    out["delay_stress_signal"] = delay_stress.astype(int)
    out["flow_stress_signal"] = flow_stress.astype(int)
    out[RAW_STRESS_COLUMN] = raw.astype(int)
    out[FINAL_PROXY_COLUMN] = persisted.astype(int)
    return out


def add_future_proxy_targets(df: pd.DataFrame, horizons: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    lookup = out.set_index(TIMESTAMP_COLUMN)[FINAL_PROXY_COLUMN]
    rows = []
    for horizon in horizons:
        target_col = f"target_{horizon}h"
        target_ts_col = f"target_timestamp_{horizon}h"
        out[target_ts_col] = out[TIMESTAMP_COLUMN] + pd.Timedelta(hours=horizon)
        out[target_col] = out[target_ts_col].map(lookup)
        usable = out[target_col].notna()
        positives = int((out.loc[usable, target_col] == 1).sum())
        rows.append(
            {
                "horizon_hours": horizon,
                "usable_samples": int(usable.sum()),
                "positive_targets": positives,
                "negative_targets": int(usable.sum() - positives),
                "positive_prevalence": float(positives / usable.sum()) if usable.sum() else 0.0,
                "removed_missing_future_timestamp": int((~usable).sum()),
            }
        )
    return out, pd.DataFrame(rows)


def proxy_episodes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    in_episode = False
    start = None
    end = None
    episode_id = 0
    previous_ts = None
    for _, row in df[[TIMESTAMP_COLUMN, FINAL_PROXY_COLUMN]].iterrows():
        ts = row[TIMESTAMP_COLUMN]
        active = int(row[FINAL_PROXY_COLUMN]) == 1
        consecutive = previous_ts is not None and ts - previous_ts == pd.Timedelta(hours=1)
        if active and not in_episode:
            episode_id += 1
            start = ts
            end = ts
            in_episode = True
        elif active and consecutive:
            end = ts
        elif active:
            rows.append((episode_id, start, end))
            episode_id += 1
            start = ts
            end = ts
        elif in_episode:
            rows.append((episode_id, start, end))
            in_episode = False
        previous_ts = ts
    if in_episode:
        rows.append((episode_id, start, end))

    return pd.DataFrame(
        [
            {
                "episode_id": int(eid),
                "start_timestamp": start,
                "end_timestamp": end,
                "duration_hours": int((end - start).total_seconds() / 3600) + 1,
            }
            for eid, start, end in rows
        ]
    )


def sequence_feature_frame(
    df: pd.DataFrame,
    source_columns: list[str],
    timestamp: pd.Timestamp,
    context_length: int,
) -> np.ndarray:
    numeric_cols, _ = split_feature_types(df, source_columns)
    indexed = df.set_index(TIMESTAMP_COLUMN)
    window_times = [timestamp - pd.Timedelta(hours=offset) for offset in range(context_length - 1, -1, -1)]
    missing = [ts for ts in window_times if ts not in indexed.index]
    if missing:
        raise ValueError(f"Missing required sequence history before {timestamp.isoformat()}: {missing[0].isoformat()}")
    return indexed.loc[window_times, numeric_cols].to_numpy(dtype=float)[None, :, :]


def transform_sequence(values: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    clean = np.where(np.isfinite(values), values, means.reshape(1, 1, -1))
    return ((clean - means.reshape(1, 1, -1)) / stds.reshape(1, 1, -1)).astype("float32")


def risk_level(score: float, thresholds: dict[str, float]) -> str:
    if score >= float(thresholds["high"]):
        return "HIGH"
    if score >= float(thresholds["medium"]):
        return "MEDIUM"
    return "LOW"


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_final_manifest(model_dir: Path) -> dict[str, Any]:
    return json.loads((model_dir / "deployment_manifest.json").read_text(encoding="utf-8"))


def predict_final_proxy(
    history_csv: Path,
    model_dir: Path,
    prediction_timestamp: str | None = None,
) -> dict[str, Any]:
    manifest = load_final_manifest(model_dir)
    df = ensure_time_features(load_dataset(history_csv))
    df = add_cargo_flow_imbalance(df)
    timestamp = pd.Timestamp(prediction_timestamp) if prediction_timestamp else df[TIMESTAMP_COLUMN].max()
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    predictions = []
    for spec in manifest["horizons"]:
        horizon = int(spec["horizon_hours"])
        model_type = spec["model_type"]
        feature_columns = list(spec["feature_columns"])
        source_columns = list(spec["source_columns"])

        if model_type == "XGBoost":
            import xgboost as xgb

            row = build_single_tabular_feature_row(df, source_columns, timestamp, feature_columns)
            preprocessor = TabularPreprocessor.load(model_dir / spec["preprocessor_path"])
            matrix = xgb.DMatrix(preprocessor.transform(row))
            booster = xgb.Booster()
            booster.load_model(str(model_dir / spec["model_path"]))
            score = float(booster.predict(matrix)[0])
        else:
            import torch
            from transformers import PatchTSMixerConfig, PatchTSMixerForTimeSeriesClassification
            from transformers import PatchTSTConfig, PatchTSTForClassification

            values = sequence_feature_frame(df, source_columns, timestamp, int(spec["context_length"]))
            scaler = json.loads((model_dir / spec["sequence_scaler_path"]).read_text(encoding="utf-8"))
            x = transform_sequence(values, np.asarray(scaler["means"]), np.asarray(scaler["stds"]))
            if model_type == "GRU":
                class CompactGRUClassifier(torch.nn.Module):
                    def __init__(self, input_size: int) -> None:
                        super().__init__()
                        self.gru = torch.nn.GRU(input_size=input_size, hidden_size=16, num_layers=1, batch_first=True)
                        self.head = torch.nn.Sequential(torch.nn.LayerNorm(16), torch.nn.Linear(16, 1))

                    def forward(self, vals: Any) -> Any:
                        _, hidden = self.gru(vals)
                        return self.head(hidden[-1]).squeeze(-1)

                model = CompactGRUClassifier(x.shape[-1])
                model.load_state_dict(torch.load(model_dir / spec["model_path"], map_location="cpu"))
                model.eval()
                with torch.no_grad():
                    score = float(torch.sigmoid(model(torch.tensor(x, dtype=torch.float32))).item())
            elif model_type == "PatchTSMixer":
                config = PatchTSMixerConfig(**spec["transformer_config"])
                model = PatchTSMixerForTimeSeriesClassification(config)
                model.load_state_dict(torch.load(model_dir / spec["model_path"], map_location="cpu"))
                model.eval()
                with torch.no_grad():
                    output = model(past_values=torch.tensor(x, dtype=torch.float32))
                    score = float(torch.softmax(output.prediction_outputs, dim=1)[0, 1].item())
            elif model_type == "PatchTST":
                config = PatchTSTConfig(**spec["transformer_config"])
                model = PatchTSTForClassification(config)
                model.load_state_dict(torch.load(model_dir / spec["model_path"], map_location="cpu"))
                model.eval()
                with torch.no_grad():
                    output = model(past_values=torch.tensor(x, dtype=torch.float32))
                    score = float(torch.softmax(output.prediction_logits, dim=1)[0, 1].item())
            else:
                raise ValueError(f"Unsupported final model type: {model_type}")

        predictions.append(
            {
                "horizon_hours": horizon,
                "target_time": (timestamp + pd.Timedelta(hours=horizon)).isoformat(),
                "congestion_score": score,
                "risk_level": risk_level(score, spec["risk_thresholds"]),
                "probability_is_calibrated": False,
            }
        )

    return {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "prediction_timestamp": timestamp.isoformat(),
        "model_version": manifest["model_version"],
        "target_definition": manifest["target_definition"]["name"],
        "predictions": predictions,
    }
