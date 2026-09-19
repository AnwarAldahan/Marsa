#!/usr/bin/env python3
"""Final cargo-congestion proxy benchmark and deployment artifacts.

This script preserves Stage 2 artifacts and writes final proxy outputs under
prediction_model/final_proxy/.
"""

from __future__ import annotations

from pathlib import Path
import gc
import json
import time
from typing import Any

import numpy as np
import pandas as pd

from prediction_model.src.data_audit import load_dataset
from prediction_model.src.evaluation import (
    average_precision,
    balanced_accuracy,
    brier_score,
    log_loss,
    optimize_threshold,
    precision_recall_f1,
    roc_auc,
)
from prediction_model.src.features import (
    TabularPreprocessor,
    build_tabular_lag_features,
    ensure_time_features,
    feature_columns_from_frame,
)
from prediction_model.src.final_proxy import (
    FINAL_MODEL_VERSION,
    FINAL_PROXY_COLUMN,
    add_cargo_flow_imbalance,
    add_future_proxy_targets,
    apply_proxy_rule,
    compute_proxy_thresholds,
    final_source_columns,
    predict_final_proxy,
    proxy_episodes,
    save_json,
)
from prediction_model.src.schema import DEFAULT_DATA_PATH, HORIZONS, PRIMARY_RANDOM_SEED, TIMESTAMP_COLUMN
from prediction_model.src.sequences import create_numeric_sequence_windows
from run_prediction_experiments import markdown_table, write_markdown
from run_stage2_benchmark import (
    fit_gru_sequence,
    fit_sequence_scaler,
    fit_transformer_sequence,
    fit_xgboost,
    transform_sequence_values,
)


OUTPUT_DIR = Path("prediction_model/final_proxy")
RESULTS_DIR = OUTPUT_DIR / "results"
ARTIFACT_DIR = OUTPUT_DIR / "artifacts"
CONTEXT_LENGTH = 24
MAX_SEQUENCE_TRAIN_ROWS = 2048
LEARNED_MODELS = ["XGBoost", "GRU", "PatchTSMixer", "PatchTST"]
ALL_MODELS = ["CurrentStatePersistence", *LEARNED_MODELS]
FINAL_TRAIN_END = pd.Timestamp("2025-09-30T23:59:59+00:00")

WALK_FORWARD_FOLDS = [
    ("wf_2025_06", "2025-05-31T23:59:59+00:00", "2025-06-01T00:00:00+00:00", "2025-06-30T23:59:59+00:00"),
    ("wf_2025_07", "2025-06-30T23:59:59+00:00", "2025-07-01T00:00:00+00:00", "2025-07-31T23:59:59+00:00"),
    ("wf_2025_08", "2025-07-31T23:59:59+00:00", "2025-08-01T00:00:00+00:00", "2025-08-31T23:59:59+00:00"),
    ("wf_2025_09", "2025-08-31T23:59:59+00:00", "2025-09-01T00:00:00+00:00", "2025-09-30T23:59:59+00:00"),
]


def timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value)


def base_frame() -> pd.DataFrame:
    return add_cargo_flow_imbalance(ensure_time_features(load_dataset(DEFAULT_DATA_PATH)))


def fold_proxy_frame(base: pd.DataFrame, train_end: str | pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_source = base[base[TIMESTAMP_COLUMN] <= timestamp(train_end)].copy()
    thresholds = compute_proxy_thresholds(train_source, timestamp(train_end))
    proxied = apply_proxy_rule(base, thresholds)
    proxied, target_meta = add_future_proxy_targets(proxied, HORIZONS)
    threshold_meta = pd.DataFrame([thresholds.to_dict()])
    return proxied, target_meta, threshold_meta


def evaluate_with_thresholds(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    pred = (y_prob >= thresholds).astype(int)
    cm = {
        "tn": int(((pred == 0) & (y_true == 0)).sum()),
        "fp": int(((pred == 1) & (y_true == 0)).sum()),
        "fn": int(((pred == 0) & (y_true == 1)).sum()),
        "tp": int(((pred == 1) & (y_true == 1)).sum()),
    }
    precision, recall, f1 = precision_recall_f1(cm)
    return {
        "PR_AUC": average_precision(y_true, y_prob),
        "ROC_AUC": roc_auc(y_true, y_prob),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Balanced_Accuracy": balanced_accuracy(cm),
        "Brier_Score": brier_score(y_true, y_prob),
        "Log_Loss": log_loss(y_true, y_prob),
        "TP": cm["tp"],
        "FP": cm["fp"],
        "TN": cm["tn"],
        "FN": cm["fn"],
        "Positive_Count": int(y_true.sum()),
        "Negative_Count": int((y_true == 0).sum()),
        "Positive_Prevalence": float(y_true.mean()) if len(y_true) else 0.0,
    }


def cap_sequence_training(
    x_train: np.ndarray,
    y_train: np.ndarray,
    max_rows: int = MAX_SEQUENCE_TRAIN_ROWS,
) -> tuple[np.ndarray, np.ndarray]:
    if len(y_train) <= max_rows:
        return x_train, y_train
    y_train = np.asarray(y_train, dtype=int)
    positive_idx = np.where(y_train == 1)[0]
    negative_idx = np.where(y_train == 0)[0]
    max_positive_rows = min(len(positive_idx), max_rows // 2)
    keep_positive = positive_idx[-max_positive_rows:] if max_positive_rows else np.empty((0,), dtype=int)
    negative_slots = max_rows - len(keep_positive)
    keep_negative = negative_idx[-negative_slots:] if negative_slots else np.empty((0,), dtype=int)
    keep = np.sort(np.concatenate([keep_positive, keep_negative]))
    return x_train[keep], y_train[keep]


def attach_target_episode(frame: pd.DataFrame, episodes: pd.DataFrame, target_ts_col: str) -> pd.Series:
    ids = []
    for target_ts in frame[target_ts_col]:
        match = episodes[(episodes["start_timestamp"] <= target_ts) & (episodes["end_timestamp"] >= target_ts)]
        ids.append(int(match["episode_id"].iloc[0]) if len(match) else np.nan)
    return pd.Series(ids, index=frame.index)


def evaluate_named(
    model: str,
    horizon: int,
    fold_id: str,
    eval_frame: pd.DataFrame,
    target_col: str,
    target_ts_col: str,
    prob: np.ndarray,
    threshold: float,
    episodes: pd.DataFrame,
    current_proxy: pd.Series,
    notes: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    y = eval_frame[target_col].astype(int).to_numpy()
    thresholds = np.repeat(float(threshold), len(y))
    metrics = evaluate_with_thresholds(y, prob, thresholds)
    episode_ids = attach_target_episode(eval_frame, episodes, target_ts_col)
    current_state = eval_frame[TIMESTAMP_COLUMN].map(current_proxy).fillna(0).astype(int)
    row = {
        "Model": model,
        "Horizon": f"+{horizon}h",
        "Fold": fold_id,
        "Evaluation_Role": "walk_forward",
        "Decision_Threshold": float(threshold),
        "Calibration_Method": "none_score_not_calibrated",
        "Distinct_Positive_Episodes": int(pd.Series(episode_ids).dropna().nunique()),
        "Notes": notes,
        **metrics,
    }
    pred_frame = eval_frame[[TIMESTAMP_COLUMN, target_ts_col, target_col]].copy()
    pred_frame["model"] = model
    pred_frame["horizon_hours"] = horizon
    pred_frame["fold"] = fold_id
    pred_frame["evaluation_role"] = "walk_forward"
    pred_frame["score"] = prob
    pred_frame["threshold"] = threshold
    pred_frame["predicted_label"] = (prob >= threshold).astype(int)
    pred_frame["target_episode_id"] = episode_ids
    pred_frame["current_proxy_at_t"] = current_state.to_numpy()
    return row, pred_frame


def run_walk_forward(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_cols = final_source_columns(list(base.columns))
    result_rows = []
    prediction_frames = []
    threshold_rows = []
    target_meta_rows = []

    for fold_id, train_end, eval_start, eval_end in WALK_FORWARD_FOLDS:
        print(f"[{fold_id}] preparing proxy labels", flush=True)
        proxied, target_meta, threshold_meta = fold_proxy_frame(base, train_end)
        threshold_meta["fold"] = fold_id
        threshold_rows.append(threshold_meta)
        target_meta["fold"] = fold_id
        target_meta_rows.append(target_meta)
        episodes = proxy_episodes(proxied)
        current_proxy = proxied.set_index(TIMESTAMP_COLUMN)[FINAL_PROXY_COLUMN]

        for horizon in HORIZONS:
            print(f"[{fold_id}] +{horizon}h tabular models", flush=True)
            target_col = f"target_{horizon}h"
            target_ts_col = f"target_timestamp_{horizon}h"
            features, _ = build_tabular_lag_features(proxied, source_cols, horizon, "FinalCargoProxyFeatures")
            feature_cols = feature_columns_from_frame(features, target_col, target_ts_col)
            train = features[features[target_ts_col] <= timestamp(train_end)].copy()
            eval_frame = features[
                (features[target_ts_col] >= timestamp(eval_start))
                & (features[target_ts_col] <= timestamp(eval_end))
            ].copy()
            if len(train) == 0 or len(eval_frame) == 0 or train[target_col].nunique() < 2:
                continue

            persistence_prob = eval_frame[TIMESTAMP_COLUMN].map(current_proxy).fillna(0).to_numpy(dtype=float)
            row, preds = evaluate_named(
                "CurrentStatePersistence",
                horizon,
                fold_id,
                eval_frame,
                target_col,
                target_ts_col,
                persistence_prob,
                0.5,
                episodes,
                current_proxy,
                "Persistence baseline: current proxy state at t.",
            )
            result_rows.append(row)
            prediction_frames.append(preds)

            prob, threshold, train_time = fit_xgboost(train, eval_frame, feature_cols, target_col)
            row, preds = evaluate_named(
                "XGBoost",
                horizon,
                fold_id,
                eval_frame,
                target_col,
                target_ts_col,
                prob,
                threshold,
                episodes,
                current_proxy,
                f"Compact XGBoost; train-only class weighting and preprocessing; train_time_seconds={train_time:.3f}.",
            )
            result_rows.append(row)
            prediction_frames.append(preds)

            seq_x, seq_y, seq_meta, _ = create_numeric_sequence_windows(proxied, source_cols, horizon, CONTEXT_LENGTH)
            seq_meta[TIMESTAMP_COLUMN] = pd.to_datetime(seq_meta[TIMESTAMP_COLUMN], utc=True)
            seq_meta[target_ts_col] = pd.to_datetime(seq_meta[target_ts_col], utc=True)
            seq_meta[target_col] = seq_y.astype(int)
            seq_train_mask = seq_meta[target_ts_col] <= timestamp(train_end)
            seq_eval_mask = (seq_meta[target_ts_col] >= timestamp(eval_start)) & (seq_meta[target_ts_col] <= timestamp(eval_end))
            if seq_train_mask.any() and seq_eval_mask.any():
                y_train = seq_meta.loc[seq_train_mask, target_col].astype(int).to_numpy()
                if len(np.unique(y_train)) >= 2:
                    raw_train_full = seq_x[seq_train_mask.to_numpy()]
                    raw_eval = seq_x[seq_eval_mask.to_numpy()]
                    means, stds = fit_sequence_scaler(raw_train_full)
                    raw_train, y_train = cap_sequence_training(raw_train_full, y_train)
                    x_train = transform_sequence_values(raw_train, means, stds)
                    x_eval = transform_sequence_values(raw_eval, means, stds)
                    eval_seq_frame = seq_meta.loc[seq_eval_mask, [TIMESTAMP_COLUMN, target_ts_col, target_col]].copy()
                    for model_name in ["GRU", "PatchTSMixer", "PatchTST"]:
                        print(f"[{fold_id}] +{horizon}h {model_name} rows={len(y_train)}", flush=True)
                        if model_name == "GRU":
                            prob, threshold, train_time = fit_gru_sequence(x_train, y_train, x_eval)
                        else:
                            prob, threshold, train_time = fit_transformer_sequence(model_name, x_train, y_train, x_eval)
                        row, preds = evaluate_named(
                            model_name,
                            horizon,
                            fold_id,
                            eval_seq_frame,
                            target_col,
                            target_ts_col,
                            prob,
                            threshold,
                            episodes,
                            current_proxy,
                            f"Compact {model_name}; 24-hour history; train_time_seconds={train_time:.3f}.",
                        )
                        result_rows.append(row)
                        prediction_frames.append(preds)

    return (
        pd.DataFrame(result_rows),
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(threshold_rows, ignore_index=True),
        pd.concat(target_meta_rows, ignore_index=True),
    )


def aggregate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, horizon), group in predictions.groupby(["model", "horizon_hours"]):
        horizon = int(horizon)
        target_col = f"target_{horizon}h"
        y = group[target_col].astype(int).to_numpy()
        metrics = evaluate_with_thresholds(y, group["score"].to_numpy(float), group["threshold"].to_numpy(float))
        positive_episodes = group[group["target_episode_id"].notna()][["fold", "target_episode_id"]].drop_duplicates()
        rows.append(
            {
                "Model": model,
                "Horizon": f"+{horizon}h",
                "Evaluation_Scope": "walk_forward",
                "Context_Length": CONTEXT_LENGTH if model != "XGBoost" else np.nan,
                "Calibration_Method": "none_score_not_calibrated",
                "Decision_Threshold": "per_fold_train_only",
                "Distinct_Positive_Episodes": int(len(positive_episodes)),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def onset_only_results(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, horizon), group in predictions.groupby(["model", "horizon_hours"]):
        horizon = int(horizon)
        target_col = f"target_{horizon}h"
        subset = group[group["current_proxy_at_t"] == 0].copy()
        if len(subset) == 0:
            continue
        y = subset[target_col].astype(int).to_numpy()
        metrics = evaluate_with_thresholds(y, subset["score"].to_numpy(float), subset["threshold"].to_numpy(float))
        positive_episodes = subset[subset["target_episode_id"].notna()][["fold", "target_episode_id"]].drop_duplicates()
        rows.append(
            {
                "Model": model,
                "Horizon": f"+{horizon}h",
                "Samples": int(len(subset)),
                "Distinct_Positive_Episodes": int(len(positive_episodes)),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def episode_early_warning(predictions: pd.DataFrame) -> pd.DataFrame:
    fold_rows = []
    for (model, horizon, fold), group in predictions.groupby(["model", "horizon_hours", "fold"]):
        horizon = int(horizon)
        target_ts_col = f"target_timestamp_{horizon}h"
        candidate = group[(group["current_proxy_at_t"] == 0) & group["target_episode_id"].notna()].copy()
        detected = candidate[candidate["score"] >= candidate["threshold"]].copy()
        episode_count = int(candidate["target_episode_id"].nunique())
        detected_episode_count = int(detected["target_episode_id"].nunique())
        leads = []
        for episode_id, detected_rows in detected.groupby("target_episode_id"):
            episode_rows = candidate[candidate["target_episode_id"] == episode_id]
            onset = episode_rows[target_ts_col].min()
            warning_leads = (onset - detected_rows[TIMESTAMP_COLUMN]).dt.total_seconds() / 3600
            leads.append(float(warning_leads.max()))
        negatives = group[(group["current_proxy_at_t"] == 0) & (group[f"target_{horizon}h"] == 0)]
        false_alerts = int(((negatives["score"] >= negatives["threshold"])).sum())
        fold_rows.append(
            {
                "Model": model,
                "Horizon": f"+{horizon}h",
                "Fold": fold,
                "Congestion_Episodes": episode_count,
                "Episodes_Detected_Before_Onset": detected_episode_count,
                "Warning_Lead_Times": leads,
                "False_Alerts": false_alerts,
                "Onset_Negatives": int(len(negatives)),
            }
        )
    rows = []
    for (model, horizon), group in pd.DataFrame(fold_rows).groupby(["Model", "Horizon"]):
        leads = [lead for leads in group["Warning_Lead_Times"] for lead in leads]
        episode_count = int(group["Congestion_Episodes"].sum())
        detected_episode_count = int(group["Episodes_Detected_Before_Onset"].sum())
        false_alerts = int(group["False_Alerts"].sum())
        negatives = int(group["Onset_Negatives"].sum())
        rows.append(
            {
                "Model": model,
                "Horizon": horizon,
                "Congestion_Episodes": episode_count,
                "Episodes_Detected_Before_Onset": detected_episode_count,
                "Detection_Rate": detected_episode_count / episode_count if episode_count else 0.0,
                "Median_Warning_Lead_Time_Hours": float(np.median(leads)) if leads else 0.0,
                "Max_Warning_Lead_Time_Hours": float(np.max(leads)) if leads else 0.0,
                "False_Alerts": false_alerts,
                "Onset_Negatives": negatives,
                "False_Alert_Rate": false_alerts / negatives if negatives else 0.0,
            }
        )
    return pd.DataFrame(rows)


def select_models(aggregate: pd.DataFrame, onset: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    learned_onset = onset[onset["Model"].isin(LEARNED_MODELS)].copy()
    for horizon in [f"+{h}h" for h in HORIZONS]:
        candidates = learned_onset[learned_onset["Horizon"] == horizon].copy()
        if len(candidates) == 0:
            continue
        candidates = candidates.merge(
            episodes[["Model", "Horizon", "Detection_Rate", "False_Alert_Rate", "Median_Warning_Lead_Time_Hours"]],
            on=["Model", "Horizon"],
            how="left",
        ).merge(
            aggregate[["Model", "Horizon", "PR_AUC", "F1", "Brier_Score"]].rename(
                columns={"PR_AUC": "Aggregate_PR_AUC", "F1": "Aggregate_F1", "Brier_Score": "Aggregate_Brier_Score"}
            ),
            on=["Model", "Horizon"],
            how="left",
        )
        candidates = candidates.sort_values(
            ["PR_AUC", "Detection_Rate", "False_Alert_Rate", "Aggregate_PR_AUC"],
            ascending=[False, False, True, False],
        )
        selected = candidates.iloc[0]
        rows.append(
            {
                "Horizon": horizon,
                "Selected_Model": selected["Model"],
                "Selection_Basis": "onset_PR_AUC_then_episode_detection_then_false_alert_rate",
                "Onset_PR_AUC": selected["PR_AUC"],
                "Onset_F1": selected["F1"],
                "Detection_Rate": selected["Detection_Rate"],
                "False_Alert_Rate": selected["False_Alert_Rate"],
                "Aggregate_PR_AUC": selected["Aggregate_PR_AUC"],
                "Aggregate_F1": selected["Aggregate_F1"],
                "Aggregate_Brier_Score": selected["Aggregate_Brier_Score"],
            }
        )
    return pd.DataFrame(rows)


def train_xgboost_artifact(train: pd.DataFrame, feature_cols: list[str], target_col: str, artifact_dir: Path) -> dict[str, Any]:
    import xgboost as xgb

    preprocessor = TabularPreprocessor().fit(train[feature_cols])
    x_train = preprocessor.transform(train[feature_cols])
    y_train = train[target_col].astype(int).to_numpy()
    positives = max(int(y_train.sum()), 1)
    negatives = max(int((y_train == 0).sum()), 1)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "eta": 0.05,
        "max_depth": 3,
        "min_child_weight": 2.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
        "tree_method": "hist",
        "seed": PRIMARY_RANDOM_SEED,
        "nthread": 1,
        "scale_pos_weight": negatives / positives,
    }
    dtrain = xgb.DMatrix(x_train, label=y_train)
    booster = xgb.train(params, dtrain, num_boost_round=90, verbose_eval=False)
    train_prob = booster.predict(dtrain)
    alert_threshold, _ = optimize_threshold(y_train, train_prob, minimum_recall=0.7)
    preprocessor_path = artifact_dir / "preprocessor.json"
    model_path = artifact_dir / "model.json"
    preprocessor.save(preprocessor_path)
    booster.save_model(str(model_path))
    return {
        "model_path": model_path.name,
        "preprocessor_path": preprocessor_path.name,
        "alert_threshold": float(alert_threshold),
        "risk_thresholds": {
            "medium": float(np.quantile(train_prob, 0.75)),
            "high": float(np.quantile(train_prob, 0.90)),
        },
        "training_positive_count": int(y_train.sum()),
        "training_negative_count": int((y_train == 0).sum()),
    }


def train_sequence_artifact(
    model_name: str,
    proxied: pd.DataFrame,
    source_cols: list[str],
    horizon: int,
    artifact_dir: Path,
) -> dict[str, Any]:
    import torch
    from transformers import PatchTSMixerConfig, PatchTSMixerForTimeSeriesClassification
    from transformers import PatchTSTConfig, PatchTSTForClassification

    target_col = f"target_{horizon}h"
    target_ts_col = f"target_timestamp_{horizon}h"
    seq_x, seq_y, seq_meta, _ = create_numeric_sequence_windows(proxied, source_cols, horizon, CONTEXT_LENGTH)
    seq_meta[target_ts_col] = pd.to_datetime(seq_meta[target_ts_col], utc=True)
    train_mask = seq_meta[target_ts_col] <= FINAL_TRAIN_END
    y_train = seq_y[train_mask.to_numpy()].astype(int)
    raw_train_full = seq_x[train_mask.to_numpy()]
    means, stds = fit_sequence_scaler(raw_train_full)
    raw_train, y_train = cap_sequence_training(raw_train_full, y_train)
    x_train = transform_sequence_values(raw_train, means, stds)
    scaler_path = artifact_dir / "sequence_scaler.json"
    save_json(scaler_path, {"means": means.tolist(), "stds": stds.tolist()})
    model_path = artifact_dir / "model.pt"

    if model_name == "GRU":
        torch.manual_seed(PRIMARY_RANDOM_SEED)

        class CompactGRUClassifier(torch.nn.Module):
            def __init__(self, input_size: int) -> None:
                super().__init__()
                self.gru = torch.nn.GRU(input_size=input_size, hidden_size=16, num_layers=1, batch_first=True)
                self.head = torch.nn.Sequential(torch.nn.LayerNorm(16), torch.nn.Linear(16, 1))

            def forward(self, values: Any) -> Any:
                _, hidden = self.gru(values)
                return self.head(hidden[-1]).squeeze(-1)

        model = CompactGRUClassifier(x_train.shape[-1])
        positives = max(float(y_train.sum()), 1.0)
        negatives = max(float((y_train == 0).sum()), 1.0)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], dtype=torch.float32))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=1e-4)
        dataset = torch.utils.data.TensorDataset(torch.tensor(x_train), torch.tensor(y_train.astype("float32")))
        loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(PRIMARY_RANDOM_SEED))
        model.train()
        for _ in range(10):
            for xb, yb in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        model.eval()
        with torch.no_grad():
            train_prob = torch.sigmoid(model(torch.tensor(x_train))).detach().cpu().numpy()
        torch.save(model.state_dict(), model_path)
        transformer_config = None
    else:
        torch.manual_seed(PRIMARY_RANDOM_SEED)
        positives = max(float(y_train.sum()), 1.0)
        negatives = max(float((y_train == 0).sum()), 1.0)
        if model_name == "PatchTSMixer":
            transformer_config = {
                "context_length": x_train.shape[1],
                "patch_length": 4,
                "patch_stride": 4,
                "num_input_channels": x_train.shape[2],
                "d_model": 8,
                "num_layers": 1,
                "expansion_factor": 2,
                "dropout": 0.1,
                "head_dropout": 0.1,
                "scaling": None,
                "loss": "cross_entropy",
                "num_targets": 2,
                "prediction_length": 1,
            }
            model = PatchTSMixerForTimeSeriesClassification(PatchTSMixerConfig(**transformer_config))
        elif model_name == "PatchTST":
            transformer_config = {
                "context_length": x_train.shape[1],
                "patch_length": 4,
                "patch_stride": 4,
                "num_input_channels": x_train.shape[2],
                "d_model": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "ffn_dim": 32,
                "head_dropout": 0.1,
                "scaling": None,
                "loss": "cross_entropy",
                "num_targets": 2,
            }
            model = PatchTSTForClassification(PatchTSTConfig(**transformer_config))
        else:
            raise ValueError(model_name)
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, negatives / positives], dtype=torch.float32))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
        dataset = torch.utils.data.TensorDataset(torch.tensor(x_train), torch.tensor(y_train.astype("int64")))
        loader = torch.utils.data.DataLoader(dataset, batch_size=96, shuffle=True, generator=torch.Generator().manual_seed(PRIMARY_RANDOM_SEED))
        model.train()
        for _ in range(5):
            for xb, yb in loader:
                optimizer.zero_grad(set_to_none=True)
                output = model(past_values=xb)
                logits = output.prediction_outputs if model_name == "PatchTSMixer" else output.prediction_logits
                loss = loss_fn(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        model.eval()
        with torch.no_grad():
            output = model(past_values=torch.tensor(x_train))
            logits = output.prediction_outputs if model_name == "PatchTSMixer" else output.prediction_logits
            train_prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        torch.save(model.state_dict(), model_path)

    alert_threshold, _ = optimize_threshold(y_train, train_prob, minimum_recall=0.7)
    gc.collect()
    return {
        "model_path": model_path.name,
        "sequence_scaler_path": scaler_path.name,
        "alert_threshold": float(alert_threshold),
        "risk_thresholds": {
            "medium": float(np.quantile(train_prob, 0.75)),
            "high": float(np.quantile(train_prob, 0.90)),
        },
        "transformer_config": transformer_config,
        "training_positive_count": int(y_train.sum()),
        "training_negative_count": int((y_train == 0).sum()),
    }


def train_final_artifacts(base: pd.DataFrame, selections: pd.DataFrame, thresholds: dict[str, Any]) -> dict[str, Any]:
    proxied = apply_proxy_rule(base, compute_proxy_thresholds(base[base[TIMESTAMP_COLUMN] <= FINAL_TRAIN_END], FINAL_TRAIN_END))
    proxied, _ = add_future_proxy_targets(proxied, HORIZONS)
    source_cols = final_source_columns(list(base.columns))
    horizon_specs = []
    for _, selection in selections.iterrows():
        horizon = int(str(selection["Horizon"]).replace("+", "").replace("h", ""))
        model_name = str(selection["Selected_Model"])
        target_col = f"target_{horizon}h"
        target_ts_col = f"target_timestamp_{horizon}h"
        artifact_dir = ARTIFACT_DIR / f"{model_name}_{horizon}h"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if model_name == "XGBoost":
            features, _ = build_tabular_lag_features(proxied, source_cols, horizon, "FinalCargoProxyFeatures")
            feature_cols = feature_columns_from_frame(features, target_col, target_ts_col)
            train = features[features[target_ts_col] <= FINAL_TRAIN_END].copy()
            artifact = train_xgboost_artifact(train, feature_cols, target_col, artifact_dir)
            spec = {
                "horizon_hours": horizon,
                "model_identifier": f"{model_name}_{horizon}h",
                "model_type": model_name,
                "context_length": None,
                "feature_columns": feature_cols,
                "source_columns": source_cols,
                "preprocessor_path": f"{model_name}_{horizon}h/{artifact['preprocessor_path']}",
                "model_path": f"{model_name}_{horizon}h/{artifact['model_path']}",
                "sequence_scaler_path": None,
                "transformer_config": None,
                **{k: artifact[k] for k in ["alert_threshold", "risk_thresholds", "training_positive_count", "training_negative_count"]},
            }
        else:
            artifact = train_sequence_artifact(model_name, proxied, source_cols, horizon, artifact_dir)
            numeric_cols, _ = create_numeric_sequence_feature_names(proxied, source_cols)
            spec = {
                "horizon_hours": horizon,
                "model_identifier": f"{model_name}_{horizon}h",
                "model_type": model_name,
                "context_length": CONTEXT_LENGTH,
                "feature_columns": numeric_cols,
                "source_columns": source_cols,
                "preprocessor_path": None,
                "model_path": f"{model_name}_{horizon}h/{artifact['model_path']}",
                "sequence_scaler_path": f"{model_name}_{horizon}h/{artifact['sequence_scaler_path']}",
                "transformer_config": artifact["transformer_config"],
                **{k: artifact[k] for k in ["alert_threshold", "risk_thresholds", "training_positive_count", "training_negative_count"]},
            }
        horizon_specs.append(spec)

    manifest = {
        "model_version": FINAL_MODEL_VERSION,
        "created_from_data": str(DEFAULT_DATA_PATH),
        "training_window_end": FINAL_TRAIN_END.isoformat(),
        "target_definition": {
            "name": "operational_cargo_congestion_proxy",
            "description": (
                "Operational proxy constructed from cargo/yard/gate stress indicators because "
                "independently observed congestion ground-truth labels were unavailable."
            ),
            "official_thresholds": False,
            "proxy_thresholds": thresholds,
            "persistence_requirement": "raw composite stress must be true for 2 consecutive hourly observations ending at t",
        },
        "score_calibration": {
            "probability_is_calibrated": False,
            "reason": "Proxy labels are synthetic/derived and positive episodes remain clustered; scores are uncalibrated rankings.",
        },
        "risk_level_policy": "LOW/MEDIUM/HIGH are relative score bands from final training score quantiles: medium=q75, high=q90.",
        "horizons": sorted(horizon_specs, key=lambda row: row["horizon_hours"]),
    }
    save_json(ARTIFACT_DIR / "deployment_manifest.json", manifest)
    return manifest


def create_numeric_sequence_feature_names(df: pd.DataFrame, source_cols: list[str]) -> tuple[list[str], list[str]]:
    from prediction_model.src.features import split_feature_types

    numeric_cols, categorical_cols = split_feature_types(df, source_cols)
    return numeric_cols, categorical_cols


def write_report(
    aggregate: pd.DataFrame,
    onset: pd.DataFrame,
    episode: pd.DataFrame,
    selections: pd.DataFrame,
    threshold_meta: pd.DataFrame,
    label_summary: pd.DataFrame,
    manifest: dict[str, Any],
    example: dict[str, Any],
) -> None:
    report = f"""# Final Layer 1 Cargo-Congestion Proxy Report

The final cargo congestion label is an operational proxy constructed from cargo/yard/gate stress indicators because independently observed congestion ground-truth labels were unavailable.

These proxy thresholds are not official Port of Los Angeles or industry-standard congestion thresholds. They are training-only robust distribution thresholds for this project dataset. The previous Stage 2 `operations_status` benchmark remains preserved as historical evidence and is not treated as final ground truth.

## Proxy Definition
For each training fold, thresholds are computed from historical training data only:

- yard stress: `yard_occupancy_percent >= q80` OR `containers_in_yard >= q80`
- delay stress: `average_dwell_time_hours >= q80` OR `truck_waiting_time_minutes >= q80`
- flow stress: `cargo_flow_imbalance >= max(0, q75)` OR (`gate_throughput <= q25` AND (`container_arrivals >= q75` OR `cargo_flow_imbalance > 0`))
- raw composite stress: yard stress AND delay stress AND flow stress
- final proxy state: raw composite stress persists for 2 consecutive hourly observations ending at t

## Label Summary
{markdown_table(label_summary)}

## Threshold Metadata
{markdown_table(threshold_meta)}

## Aggregate Walk-Forward Results
{markdown_table(aggregate[['Model','Horizon','PR_AUC','Precision','Recall','F1','ROC_AUC','Brier_Score','Positive_Count','Negative_Count','Distinct_Positive_Episodes']])}

## Onset-Only Results
{markdown_table(onset[['Model','Horizon','PR_AUC','Precision','Recall','F1','Positive_Count','Negative_Count','Distinct_Positive_Episodes']])}

## Episode Early-Warning Results
{markdown_table(episode)}

## Selected Learned Models
{markdown_table(selections)}

## Calibration
Scores are not calibrated probabilities. `probability_is_calibrated=false` in the inference payload. Calibration should wait for independent, non-clustered positive episodes and target validation.

## Inference Example
```json
{json.dumps(example, indent=2)}
```

## Artifact Manifest
`prediction_model/final_proxy/artifacts/deployment_manifest.json`

"""
    write_markdown(OUTPUT_DIR / "final_proxy_technical_report.md", report)


def main() -> None:
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    base = base_frame()
    source_cols = final_source_columns(list(base.columns))
    (RESULTS_DIR / "final_feature_columns.json").write_text(json.dumps(source_cols, indent=2), encoding="utf-8")

    print("Running final proxy walk-forward benchmark...")
    fold_results, fold_predictions, threshold_meta, target_meta = run_walk_forward(base)
    fold_results.to_csv(RESULTS_DIR / "fold_results.csv", index=False)
    fold_predictions.to_csv(RESULTS_DIR / "fold_predictions.csv", index=False)
    threshold_meta.to_csv(RESULTS_DIR / "proxy_thresholds_by_fold.csv", index=False)
    target_meta.to_csv(RESULTS_DIR / "target_generation_by_fold.csv", index=False)

    aggregate = aggregate_predictions(fold_predictions)
    onset = onset_only_results(fold_predictions)
    episode = episode_early_warning(fold_predictions)
    selections = select_models(aggregate, onset, episode)
    aggregate.to_csv(RESULTS_DIR / "aggregate_results.csv", index=False)
    onset.to_csv(RESULTS_DIR / "onset_only_results.csv", index=False)
    episode.to_csv(RESULTS_DIR / "episode_early_warning_results.csv", index=False)
    selections.to_csv(RESULTS_DIR / "selected_models.csv", index=False)

    final_thresholds = compute_proxy_thresholds(base[base[TIMESTAMP_COLUMN] <= FINAL_TRAIN_END], FINAL_TRAIN_END)
    final_proxied = apply_proxy_rule(base, final_thresholds)
    final_episodes = proxy_episodes(final_proxied)
    final_label_summary = pd.DataFrame(
        [
            {
                "training_window_end": FINAL_TRAIN_END.isoformat(),
                "samples": int(len(final_proxied)),
                "positive_labels": int(final_proxied[FINAL_PROXY_COLUMN].sum()),
                "negative_labels": int((final_proxied[FINAL_PROXY_COLUMN] == 0).sum()),
                "positive_prevalence": float(final_proxied[FINAL_PROXY_COLUMN].mean()),
                "congestion_episodes": int(len(final_episodes)),
            }
        ]
    )
    final_label_summary.to_csv(RESULTS_DIR / "final_proxy_label_summary.csv", index=False)
    final_episodes.to_csv(RESULTS_DIR / "final_proxy_episodes.csv", index=False)

    print("Training selected final artifacts...")
    manifest = train_final_artifacts(base, selections, final_thresholds.to_dict())
    example = predict_final_proxy(DEFAULT_DATA_PATH, ARTIFACT_DIR)
    save_json(RESULTS_DIR / "latest_final_proxy_prediction.json", example)
    write_report(aggregate, onset, episode, selections, threshold_meta, final_label_summary, manifest, example)

    save_json(
        RESULTS_DIR / "run_summary.json",
        {
            "elapsed_seconds": time.perf_counter() - start,
            "models_trained": LEARNED_MODELS,
            "stage2_preserved": True,
            "output_dir": str(OUTPUT_DIR),
        },
    )
    print("Final proxy benchmark complete.")
    print(f"Results: {RESULTS_DIR}")
    print(f"Artifacts: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
