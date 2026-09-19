"""Sequence-window construction with explicit gap checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from prediction_model.src.features import split_feature_types
from prediction_model.src.schema import TIMESTAMP_COLUMN
from prediction_model.src.splits import assign_split_by_target_timestamp


@dataclass
class SequenceWindowMetadata:
    horizon_hours: int
    context_length: int
    candidate_rows: int
    usable_windows: int
    removed_missing_future_target: int
    removed_incomplete_history: int
    feature_count: int


def count_sequence_windows(
    df: pd.DataFrame,
    source_columns: list[str],
    horizon_hours: int,
    context_length: int,
) -> SequenceWindowMetadata:
    target_col = f"target_{horizon_hours}h"
    target_ts_col = f"target_timestamp_{horizon_hours}h"
    timestamps = pd.DatetimeIndex(df[TIMESTAMP_COLUMN])
    present = set(timestamps)
    target_ok = df[target_col].notna()
    split_ok = assign_split_by_target_timestamp(df[target_ts_col]) != "outside"
    history_ok = []
    for ts in timestamps:
        ok = True
        for offset in range(context_length):
            if ts - pd.Timedelta(hours=offset) not in present:
                ok = False
                break
        history_ok.append(ok)
    history_ok_series = pd.Series(history_ok, index=df.index)
    usable = target_ok & split_ok & history_ok_series
    numeric, categorical = split_feature_types(df, source_columns)
    feature_count = len(numeric) + len(categorical)
    return SequenceWindowMetadata(
        horizon_hours=horizon_hours,
        context_length=context_length,
        candidate_rows=int(len(df)),
        usable_windows=int(usable.sum()),
        removed_missing_future_target=int((~target_ok).sum()),
        removed_incomplete_history=int((target_ok & ~history_ok_series).sum()),
        feature_count=int(feature_count),
    )


def create_numeric_sequence_windows(
    df: pd.DataFrame,
    source_columns: list[str],
    horizon_hours: int,
    context_length: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, SequenceWindowMetadata]:
    target_col = f"target_{horizon_hours}h"
    target_ts_col = f"target_timestamp_{horizon_hours}h"
    numeric_cols, _ = split_feature_types(df, source_columns)
    index = df.set_index(TIMESTAMP_COLUMN)
    present = set(index.index)
    windows = []
    labels = []
    metadata_rows = []
    for _, row in df.iterrows():
        ts = row[TIMESTAMP_COLUMN]
        if pd.isna(row[target_col]):
            continue
        window_times = [ts - pd.Timedelta(hours=offset) for offset in range(context_length - 1, -1, -1)]
        if any(t not in present for t in window_times):
            continue
        target_split = assign_split_by_target_timestamp(pd.Series([row[target_ts_col]])).iloc[0]
        if target_split == "outside":
            continue
        window = index.loc[window_times, numeric_cols].to_numpy(dtype=float)
        windows.append(window)
        labels.append(float(row[target_col]))
        metadata_rows.append(
            {
                TIMESTAMP_COLUMN: ts,
                target_ts_col: row[target_ts_col],
                "split": target_split,
            }
        )
    meta = count_sequence_windows(df, source_columns, horizon_hours, context_length)
    if not windows:
        return (
            np.empty((0, context_length, len(numeric_cols)), dtype=float),
            np.empty((0,), dtype=float),
            pd.DataFrame(metadata_rows),
            meta,
        )
    return np.stack(windows), np.asarray(labels), pd.DataFrame(metadata_rows), meta
