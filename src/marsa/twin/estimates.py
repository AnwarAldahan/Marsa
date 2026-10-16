"""Estimate how long a vessel will stay at berth, using only stays completed before the chosen hour.

Used for vessels the user adds to a what-if test (the twin knows real stays for vessels in AIS).
Order of use:
  1. the same vessel's earlier LA stays (median)            -> "own_history"
  2. the k vessels closest in length (median of their stays) -> "similar_length"
     k is not chosen by hand: scripts/select_similar_k.py tests several k on 2025 stays using only
     earlier stays, and the k with the smallest error is stored in config/port_config.yaml
  3. a duration typed by the user                            -> "user_input"
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from marsa.twin.calls import load_calls


# Stays that had already ended at hour ts (nothing from the future is used).
def completed_before(ts: pd.Timestamp, calls: pd.DataFrame | None = None) -> pd.DataFrame:
    calls = load_calls() if calls is None else calls
    return calls[calls["berth_end"] <= ts]


# Median stay of the k vessels whose length is closest to length_m, one stay per vessel
# (its median), so a frequent visitor does not count more than once. Vessels tied with the k-th
# closest length are all kept, so the choice never depends on row order.
def similar_length_hours(history: pd.DataFrame, length_m: float, k: int,
                         exclude_vessel: str | None = None) -> tuple[float, int]:
    per_vessel = history.groupby("vessel_id").agg(length_m=("length_m", "median"),
                                                  hours=("berth_hours", "median")).dropna()
    if exclude_vessel is not None:
        per_vessel = per_vessel.drop(index=exclude_vessel, errors="ignore")
    if per_vessel.empty:
        raise ValueError("no earlier stays to compare with")
    distance = np.abs(per_vessel["length_m"].to_numpy() - float(length_m))
    cutoff = np.sort(distance)[min(k, len(distance)) - 1]
    nearest = per_vessel[distance <= cutoff]
    return float(nearest["hours"].median()), len(nearest)


# Estimate one added vessel's stay; returns hours plus how it was obtained, for the interface.
def estimate_stay(ts: pd.Timestamp, k: int, vessel_id: str | None = None,
                  length_m: float | None = None, service_hours: float | None = None,
                  calls: pd.DataFrame | None = None) -> dict:
    if service_hours is not None:
        if not service_hours > 0:
            raise ValueError("service_hours must be greater than 0")
        return {"hours": float(service_hours), "method": "user_input", "based_on": 1}
    history = completed_before(ts, calls)
    if vessel_id:
        own = history[history["vessel_id"] == vessel_id]
        if not own.empty:
            return {"hours": float(own["berth_hours"].median()), "method": "own_history",
                    "based_on": len(own), "length_m": float(own["length_m"].median())}
        known = (load_calls() if calls is None else calls)
        known = known[known["vessel_id"] == vessel_id]
        if length_m is None and not known.empty:
            length_m = float(known["length_m"].median())
    if length_m is None:
        raise ValueError("an added vessel needs a known vessel_id, a length_m or service_hours")
    hours, n = similar_length_hours(history, length_m, k, exclude_vessel=vessel_id)
    return {"hours": hours, "method": "similar_length", "based_on": n, "length_m": float(length_m)}
