"""Classification metrics, threshold search, and curve utilities."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


EPS = 1e-12


def clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), EPS, 1.0 - EPS)


def confusion_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, int]:
    y_true = np.asarray(y_true, dtype=int)
    pred = (np.asarray(y_prob) >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def precision_recall_f1(cm: dict[str, int]) -> tuple[float, float, float]:
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def balanced_accuracy(cm: dict[str, int]) -> float:
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return 0.5 * (tpr + tnr)


def log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p = clip_probabilities(y_prob)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    return float(np.mean((p - y_true) ** 2))


def average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    positives = int(y_true.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-y_prob, kind="mergesort")
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true == 1)
    rank = np.arange(1, len(sorted_true) + 1)
    precision = tp / rank
    return float(precision[sorted_true == 1].sum() / positives)


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    scores = np.concatenate([pos, neg])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average tied ranks.
    unique_scores, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if len(unique_scores) != len(scores):
        for idx, count in enumerate(counts):
            if count > 1:
                tied = np.where(inverse == idx)[0]
                ranks[tied] = ranks[tied].mean()
    sum_pos_ranks = ranks[: len(pos)].sum()
    auc = (sum_pos_ranks - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def pr_curve(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    thresholds = np.unique(np.asarray(y_prob, dtype=float))
    rows = []
    for threshold in thresholds:
        cm = confusion_at_threshold(y_true, y_prob, float(threshold))
        precision, recall, f1 = precision_recall_f1(cm)
        rows.append({"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1})
    rows.append({"threshold": 1.0 + EPS, "precision": 1.0, "recall": 0.0, "f1": 0.0})
    return pd.DataFrame(rows).sort_values("recall")


def roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    thresholds = np.unique(np.asarray(y_prob, dtype=float))
    rows = []
    for threshold in thresholds:
        cm = confusion_at_threshold(y_true, y_prob, float(threshold))
        tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        rows.append({"threshold": float(threshold), "fpr": fpr, "tpr": tpr})
    rows.extend(
        [
            {"threshold": -EPS, "fpr": 1.0, "tpr": 1.0},
            {"threshold": 1.0 + EPS, "fpr": 0.0, "tpr": 0.0},
        ]
    )
    return pd.DataFrame(rows).sort_values("fpr")


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for i in range(bins):
        low, high = edges[i], edges[i + 1]
        if i == bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)
        if mask.sum():
            rows.append(
                {
                    "bin_low": low,
                    "bin_high": high,
                    "count": int(mask.sum()),
                    "mean_predicted_probability": float(y_prob[mask].mean()),
                    "observed_frequency": float(y_true[mask].mean()),
                }
            )
    return pd.DataFrame(rows)


def optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray, minimum_recall: float = 0.7) -> tuple[float, pd.DataFrame]:
    thresholds = np.unique(np.quantile(np.asarray(y_prob, dtype=float), np.linspace(0, 1, 201)))
    rows = []
    for threshold in thresholds:
        cm = confusion_at_threshold(y_true, y_prob, float(threshold))
        precision, recall, f1 = precision_recall_f1(cm)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positives": cm["fp"],
                "false_negatives": cm["fn"],
            }
        )
    table = pd.DataFrame(rows)
    candidates = table[table["recall"] >= minimum_recall]
    if len(candidates):
        best = candidates.sort_values(["f1", "precision", "threshold"], ascending=[False, False, False]).iloc[0]
    else:
        best = table.sort_values(["f1", "recall", "precision"], ascending=[False, False, False]).iloc[0]
    return float(best["threshold"]), table


def evaluate_classifier(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = clip_probabilities(y_prob)
    cm = confusion_at_threshold(y_true, y_prob, threshold)
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
        "TN": cm["tn"],
        "FP": cm["fp"],
        "FN": cm["fn"],
        "TP": cm["tp"],
        "Positive_Count": int(y_true.sum()),
        "Negative_Count": int((y_true == 0).sum()),
        "Positive_Prevalence": float(y_true.mean()) if len(y_true) else 0.0,
    }


def nan_metrics() -> dict[str, Any]:
    names = [
        "PR_AUC",
        "ROC_AUC",
        "Precision",
        "Recall",
        "F1",
        "Balanced_Accuracy",
        "Brier_Score",
        "Log_Loss",
    ]
    return {name: math.nan for name in names}
