"""What is "usual" for each number in section ١ of the dashboard, computed from the data files.

Read-only: uses the same dataset the pipeline reads (marsa.data.load) and the twin's vessel-stay file
through the twin's own loader (marsa.twin.calls.load_calls). Nothing in the twin is changed.

  in, wait, dep, yard, out -> hourly columns of merged_port_dataset_2025_v2.csv (as in port_state)
  berth                    -> vessels at berth at each hour with the twin's rule
                              berth_start <= hour < berth_end  (same count as vessels_at_berth_now)

For each: the 10th, 25th, 50th, 75th and 90th percentile over every hour in the dataset (pandas
quantile, linear). Also the first/last hour and the hours missing from the dataset, for the hour picker.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from marsa.data.load import load_dataset
from marsa.twin.calls import load_calls

COLUMNS = {
    "in": "approaching_vessel_count",
    "wait": "waiting_vessel_count",
    "dep": "departing_vessel_count",
    "yard": "yard_occupancy_percent",
    "out": "container_departures",
}
QUANTILES = (10, 25, 50, 75, 90)


def berth_counts(hours: pd.DatetimeIndex, calls: pd.DataFrame) -> pd.Series:
    """Vessels at berth at each hour (berth_start <= hour < berth_end), as the twin counts them."""
    c = calls.dropna(subset=["berth_start", "berth_end"])
    starts = np.sort(c["berth_start"].to_numpy(dtype="datetime64[ns]"))
    ends = np.sort(c["berth_end"].to_numpy(dtype="datetime64[ns]"))
    h = hours.tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")
    started = np.searchsorted(starts, h, side="right")   # berth_start <= hour
    ended = np.searchsorted(ends, h, side="right")       # berth_end <= hour
    return pd.Series(started - ended, index=hours)


def _summary(values: pd.Series, source: str) -> dict:
    v = values.dropna().astype(float)
    return {"source": source, "hours": int(v.size),
            **{f"p{q}": round(float(v.quantile(q / 100)), 2) for q in QUANTILES}}


@lru_cache(maxsize=1)
def flow_norms() -> dict:
    df = load_dataset()
    hours = pd.DatetimeIndex(df["hour_key"])
    norms = {key: _summary(df[col], f"merged_port_dataset_2025_v2.csv:{col}") for key, col in COLUMNS.items()}
    norms["berth"] = _summary(berth_counts(hours, load_calls()),
                              "la_cargo_calls_2023_2025.csv: berth_start <= hour < berth_end")
    full = pd.date_range(hours.min(), hours.max(), freq="h")
    missing = full.difference(hours)
    return {
        "first_hour": hours.min().strftime("%Y-%m-%dT%H:00:00Z"),
        "last_hour": hours.max().strftime("%Y-%m-%dT%H:00:00Z"),
        "missing_hours": [t.strftime("%Y-%m-%dT%H") for t in missing],
        "quantiles": list(QUANTILES),
        "norms": norms,
    }
