"""Data loading and audit utilities for the merged 2025 port dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prediction_model.src.schema import (
    NEGATIVE_STATUS,
    POSITIVE_STATUS,
    STATUS_COLUMN,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    build_feature_governance,
)


@dataclass
class AuditSummary:
    dataset_shape: tuple[int, int]
    timestamp_column: str
    timestamp_range_start: str
    timestamp_range_end: str
    timezone: str
    unique_timestamps: int
    duplicate_timestamps: int
    expected_full_year_hours: int
    missing_hours: int
    min_interval_hours: float | None
    max_interval_hours: float | None
    modal_interval_hours: float | None


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if TIMESTAMP_COLUMN not in df.columns:
        raise ValueError(f"Expected timestamp column {TIMESTAMP_COLUMN!r} not found.")
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], utc=True, errors="raise")
    df = df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    return df


def add_congestion_label(df: pd.DataFrame) -> pd.DataFrame:
    if STATUS_COLUMN not in df.columns:
        raise ValueError(f"Expected status column {STATUS_COLUMN!r} not found.")
    observed = set(df[STATUS_COLUMN].dropna().astype(str).unique())
    valid = NEGATIVE_STATUS | POSITIVE_STATUS
    unexpected = observed - valid
    if unexpected:
        raise ValueError(f"Unexpected operations_status values: {sorted(unexpected)}")
    out = df.copy()
    out[TARGET_COLUMN] = out[STATUS_COLUMN].isin(POSITIVE_STATUS).astype(int)
    return out


def status_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[STATUS_COLUMN].value_counts(dropna=False).rename_axis("operations_status")
    total = len(df)
    status = counts.reset_index(name="count")
    status["percentage"] = status["count"] / total
    label_counts = df[TARGET_COLUMN].value_counts().rename_axis(TARGET_COLUMN).reset_index(name="count")
    label_counts["percentage"] = label_counts["count"] / total
    label_counts[TARGET_COLUMN] = label_counts[TARGET_COLUMN].astype(int)
    label_counts["operations_status"] = label_counts[TARGET_COLUMN].map(
        {0: "NORMAL_OR_MODERATE", 1: "ELEVATED_OR_CRITICAL"}
    )
    return pd.concat(
        [
            status[["operations_status", "count", "percentage"]],
            label_counts[["operations_status", "count", "percentage"]],
        ],
        ignore_index=True,
    )


def temporal_gap_table(df: pd.DataFrame) -> pd.DataFrame:
    ts = df[TIMESTAMP_COLUMN]
    if ts.empty:
        return pd.DataFrame()
    full_range = pd.date_range(ts.min(), ts.max(), freq="h", tz="UTC")
    missing = full_range.difference(pd.DatetimeIndex(ts))
    gaps = pd.DataFrame({"missing_timestamp": missing})
    return gaps


def sampling_intervals(df: pd.DataFrame) -> pd.DataFrame:
    intervals = df[TIMESTAMP_COLUMN].diff().dropna().dt.total_seconds() / 3600.0
    return intervals.value_counts().rename_axis("interval_hours").reset_index(name="count")


def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        count = int(df[column].isna().sum())
        rows.append({"column": column, "missing_count": count, "missing_percentage": count / len(df)})
    return pd.DataFrame(rows).sort_values(["missing_count", "column"], ascending=[False, True])


def infinity_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    numeric = df.select_dtypes(include=[np.number])
    for column in numeric.columns:
        count = int(np.isinf(numeric[column].to_numpy(dtype=float, na_value=np.nan)).sum())
        rows.append({"column": column, "infinity_count": count})
    return pd.DataFrame(rows).sort_values(["infinity_count", "column"], ascending=[False, True])


def categorical_variables(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        if df[column].dtype == "object" or str(df[column].dtype) == "bool":
            values = df[column].dropna().astype(str).value_counts().head(20)
            rows.append(
                {
                    "column": column,
                    "dtype": str(df[column].dtype),
                    "unique_count": int(df[column].nunique(dropna=True)),
                    "top_values": "; ".join(f"{idx}={val}" for idx, val in values.items()),
                }
            )
    return pd.DataFrame(rows)


def constant_variables(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        nunique = int(df[column].nunique(dropna=False))
        top_pct = float(df[column].value_counts(dropna=False, normalize=True).iloc[0])
        if nunique <= 1 or top_pct >= 0.995:
            rows.append({"column": column, "unique_count": nunique, "top_value_percentage": top_pct})
    return pd.DataFrame(rows)


def suspicious_values(df: pd.DataFrame) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []
    non_negative_columns = [
        column
        for column in df.select_dtypes(include=[np.number]).columns
        if column
        not in {
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
        }
    ]
    for column in non_negative_columns:
        negative_count = int((df[column] < 0).sum())
        if negative_count:
            checks.append({"column": column, "issue": "negative_values", "count": negative_count})

    bounded_checks = {
        "yard_occupancy_percent": (0, 100),
        "cargo_flow_ratio": (0, None),
        "hour": (0, 23),
        "day_of_week": (0, 6),
        "month": (1, 12),
    }
    for column, (low, high) in bounded_checks.items():
        if column in df.columns:
            series = df[column]
            bad = series < low
            if high is not None:
                bad = bad | (series > high)
            count = int(bad.sum())
            if count:
                checks.append({"column": column, "issue": f"outside_expected_range_{low}_{high}", "count": count})
    return pd.DataFrame(checks)


def audit_dataset(df: pd.DataFrame) -> dict[str, pd.DataFrame | dict[str, Any]]:
    ts = df[TIMESTAMP_COLUMN]
    full_range = pd.date_range(ts.min(), ts.max(), freq="h", tz="UTC")
    intervals = ts.diff().dropna().dt.total_seconds() / 3600.0
    mode = intervals.mode()
    summary = AuditSummary(
        dataset_shape=tuple(df.shape),
        timestamp_column=TIMESTAMP_COLUMN,
        timestamp_range_start=ts.min().isoformat(),
        timestamp_range_end=ts.max().isoformat(),
        timezone=str(ts.dt.tz),
        unique_timestamps=int(ts.nunique()),
        duplicate_timestamps=int(ts.duplicated().sum()),
        expected_full_year_hours=int(len(full_range)),
        missing_hours=int(len(full_range.difference(pd.DatetimeIndex(ts)))),
        min_interval_hours=float(intervals.min()) if not intervals.empty else None,
        max_interval_hours=float(intervals.max()) if not intervals.empty else None,
        modal_interval_hours=float(mode.iloc[0]) if not mode.empty else None,
    )

    return {
        "summary": asdict(summary),
        "columns": pd.DataFrame({"column": df.columns, "dtype": [str(df[c].dtype) for c in df.columns]}),
        "missing_values": missing_values(df),
        "infinities": infinity_counts(df),
        "descriptive_statistics": df.describe(include="all").transpose().reset_index(names="column"),
        "categorical_variables": categorical_variables(df),
        "constant_or_near_constant_variables": constant_variables(df),
        "suspicious_values": suspicious_values(df),
        "temporal_gaps": temporal_gap_table(df),
        "sampling_intervals": sampling_intervals(df),
        "feature_governance": pd.DataFrame(build_feature_governance(list(df.columns))),
    }


def save_audit(audit: dict[str, pd.DataFrame | dict[str, Any]], output_dir: Path) -> None:
    audit_dir = output_dir / "results" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    for name, value in audit.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(audit_dir / f"{name}.csv", index=False)
        else:
            pd.Series(value).to_json(audit_dir / f"{name}.json", indent=2)

