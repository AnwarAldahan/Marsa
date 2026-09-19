"""Feature and target construction for the congestion model.

Rules (see docs/DATA_DICTIONARY.md):
- the target is the *future* value of `target_column`, shifted by wall-clock hours (not rows),
  because 72 source hours are missing;
- calendar features are recomputed in the port's operational timezone;
- excluded columns are dropped by name, never silently.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

WEEKEND_DAYS = (5, 6)  # Sat, Sun in the US


def add_local_calendar(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    local = df["hour_key"].dt.tz_convert(tz)
    out = df.copy()
    out["local_hour"] = local.dt.hour
    out["local_dow"] = local.dt.dayofweek
    out["local_is_weekend"] = local.dt.dayofweek.isin(WEEKEND_DAYS).astype(int)
    out["local_hour_sin"] = np.sin(2 * np.pi * out["local_hour"] / 24)
    out["local_hour_cos"] = np.cos(2 * np.pi * out["local_hour"] / 24)
    return out


def add_lags(df: pd.DataFrame, cols, lags=(1, 3, 6, 24)) -> pd.DataFrame:
    """Time-based lags: value at t-h hours (NaN if that hour is missing)."""
    out = df.copy()
    base = df.set_index("hour_key")
    for c in cols:
        for h in lags:
            shifted = base[c].copy()
            shifted.index = shifted.index + pd.Timedelta(hours=h)
            out[f"{c}_lag{h}h"] = out["hour_key"].map(shifted)
    return out


def add_target(df: pd.DataFrame, target_col: str, horizon_h: int) -> pd.DataFrame:
    """target_<h>h = target_col at t + h hours. Rows whose future hour is missing get NaN."""
    out = df.copy()
    future = df.set_index("hour_key")[target_col].copy()
    future.index = future.index - pd.Timedelta(hours=horizon_h)
    out[f"target_{horizon_h}h"] = out["hour_key"].map(future)
    return out


def build_feature_frame(df: pd.DataFrame, cfg: dict, horizon_h: int):
    mcfg = cfg["model"]
    tz = cfg["port"]["timezone_operational"]
    d = add_local_calendar(df, tz)
    lag_cols = [c for c in ["waiting_vessel_count", "vessel_count", "port_throughput",
                            "containers_in_yard", "truck_waiting_time_minutes",
                            "pola_berthed_cargo_vessels"] if c in d.columns]
    d = add_lags(d, lag_cols)
    d = add_target(d, mcfg["target_column"], horizon_h)
    excluded = set(mcfg["excluded_features"]) | {c for c in d.columns if c.startswith("target_")}
    # original UTC calendar columns are replaced by local ones
    excluded |= {"hour", "hour_sin", "hour_cos", "day_of_week", "month"}
    feature_cols = [c for c in d.columns if c not in excluded and d[c].dtype != object]
    return d, feature_cols, f"target_{horizon_h}h"
