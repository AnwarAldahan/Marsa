"""Port of Los Angeles cargo-vessel calls from AIS, and the Digital Twin starting state at an hour.

Reads data/processed/la_cargo_calls_2023_2025.csv (built by scripts/build_la_cargo_calls.py).
Every vessel, time and duration below comes from that file; nothing is sampled or assumed.
Future arrivals inside the horizon are the vessels that really arrived next: a port knows them in
advance from arrival notices, so the twin uses them as the known arrival schedule.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from marsa.config import DATA_DIR
from marsa.twin.engine import Scenario, Vessel

CALLS_PATH = DATA_DIR / "processed" / "la_cargo_calls_2023_2025.csv"
TIMELINE_PATH = DATA_DIR / "processed" / "merged_port_dataset_2025_v2.csv"
TIME_COLUMNS = ["arrival", "wait_start", "wait_end", "berth_start", "berth_end"]


# Load the call table once; times are UTC.
@lru_cache(maxsize=1)
def load_calls(path=CALLS_PATH) -> pd.DataFrame:
    calls = pd.read_csv(path)
    for column in TIME_COLUMNS:
        calls[column] = pd.to_datetime(calls[column], utc=True)
    return calls


# The 2025 hours that exist in the AIS file (same timeline as the merged dataset).
@lru_cache(maxsize=1)
def ais_hours() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(pd.read_csv(TIMELINE_PATH, usecols=["hour_key"])["hour_key"], utc=True))


# 2025 hours between start and end that are missing from the AIS file, grouped into ranges.
def missing_ranges(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    lo = max(start.floor("h"), pd.Timestamp("2025-01-01", tz="UTC"))
    hi = min(end.floor("h"), pd.Timestamp("2025-12-31 23:00", tz="UTC"))
    if lo > hi:
        return []
    missing = pd.date_range(lo, hi, freq="h").difference(ais_hours())
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for hour in missing:
        if ranges and hour - ranges[-1][1] == pd.Timedelta(hours=1):
            ranges[-1] = (ranges[-1][0], hour)
        else:
            ranges.append((hour, hour))
    return ranges


# Hours between two timestamps as a float.
def _hours(later, earlier) -> float:
    return (later - earlier).total_seconds() / 3600


# Build the twin starting state at hour ts: vessels at berth, waiting, and arriving in the horizon.
# arrival_multiplier is a user-chosen stress scenario: x2 means the real vessels of the next 2*H
# hours all arrive within H hours (twice as many real vessels, each keeps its own service time).
# x1 is the real schedule; below 1 stretches it (fewer vessels in the horizon).
def scenario_at(ts: pd.Timestamp, berths: int, horizon_hours: int,
                berths_closed: int = 0, arrival_multiplier: float = 1.0,
                calls: pd.DataFrame | None = None) -> Scenario:
    calls = load_calls() if calls is None else calls
    surge = arrival_multiplier
    window_end = ts + pd.Timedelta(hours=horizon_hours * surge)

    at_berth = calls[(calls["berth_start"] <= ts) & (calls["berth_end"] > ts)]
    waiting = calls[(calls["arrival"] <= ts) & (calls["berth_start"] > ts)]
    coming = calls[(calls["arrival"] > ts) & (calls["arrival"] <= window_end)]

    berthed = [Vessel(r.call_id, mmsi=int(r.mmsi), arrival_h=_hours(r.arrival, ts),
                      ready_h=_hours(r.berth_start, ts), service_h=r.berth_hours,
                      berth_h=_hours(r.berth_start, ts))
               for r in at_berth.itertuples()]
    queue = [Vessel(r.call_id, mmsi=int(r.mmsi), arrival_h=_hours(r.arrival, ts),
                    ready_h=_hours(r.berth_start, ts), service_h=r.berth_hours)
             for r in waiting.itertuples()]
    future = []
    for r in coming.itertuples():
        eta = _hours(r.arrival, ts) / surge
        # the vessel keeps its own real scheduled wait (terminal window), if it had one
        future.append(Vessel(r.call_id, mmsi=int(r.mmsi), arrival_h=eta,
                             ready_h=eta + r.wait_hours, service_h=r.berth_hours))
    return Scenario(
        berths=berths, berths_closed=berths_closed, horizon_hours=horizon_hours,
        berthed=berthed, vessels=queue + future,
        arrival_multiplier=arrival_multiplier,
        data_end=_hours(calls["arrival"].max(), ts),
    )
