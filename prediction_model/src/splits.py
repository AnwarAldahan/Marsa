"""Chronological split helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SplitBoundaries:
    train_start: str = "2025-01-01T00:00:00+00:00"
    train_end: str = "2025-09-30T23:59:59+00:00"
    validation_start: str = "2025-10-01T00:00:00+00:00"
    validation_end: str = "2025-11-30T23:59:59+00:00"
    test_start: str = "2025-12-01T00:00:00+00:00"
    test_end: str = "2025-12-31T23:59:59+00:00"


def assign_split_by_target_timestamp(target_ts: pd.Series, boundaries: SplitBoundaries | None = None) -> pd.Series:
    bounds = boundaries or SplitBoundaries()
    split = pd.Series("outside", index=target_ts.index, dtype="object")
    train_start = pd.Timestamp(bounds.train_start)
    train_end = pd.Timestamp(bounds.train_end)
    val_start = pd.Timestamp(bounds.validation_start)
    val_end = pd.Timestamp(bounds.validation_end)
    test_start = pd.Timestamp(bounds.test_start)
    test_end = pd.Timestamp(bounds.test_end)

    split[(target_ts >= train_start) & (target_ts <= train_end)] = "train"
    split[(target_ts >= val_start) & (target_ts <= val_end)] = "validation"
    split[(target_ts >= test_start) & (target_ts <= test_end)] = "test"
    return split


def split_class_counts(df: pd.DataFrame, target_col: str, target_ts_col: str) -> pd.DataFrame:
    work = df[df[target_col].notna()].copy()
    work["split"] = assign_split_by_target_timestamp(work[target_ts_col])
    rows = []
    for split in ["train", "validation", "test", "outside"]:
        subset = work[work["split"] == split]
        positives = int((subset[target_col] == 1).sum())
        negatives = int((subset[target_col] == 0).sum())
        rows.append(
            {
                "split": split,
                "samples": int(len(subset)),
                "positive": positives,
                "negative": negatives,
                "positive_prevalence": positives / len(subset) if len(subset) else 0.0,
            }
        )
    return pd.DataFrame(rows)
