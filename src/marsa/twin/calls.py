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
TIME_COLUMNS = ["arrival", "wait_start", "wait_end", "berth_start", "berth_end"]


# Load the call table once; times are UTC.
@lru_cache(maxsize=1)
def load_calls(path=CALLS_PATH) -> pd.DataFrame:
    calls = pd.read_csv(path)
    for column in TIME_COLUMNS:
        calls[column] = pd.to_datetime(calls[column], utc=True)
    return calls


# Hours between two timestamps as a float.
def _hours(later, earlier) -> float:
    return (later - earlier).total_seconds() / 3600


# Build the twin starting state at hour ts: vessels at berth, waiting, and arriving in the horizon.
# arrival_surge_pct compresses the real arrival schedule (vessels of the next H*(1+s) hours arrive
# within H hours); it is a user-chosen stress scenario, 0 means the real schedule.
def scenario_at(ts: pd.Timestamp, berths: int, horizon_hours: int,
                berths_closed: int = 0, arrival_surge_pct: float = 0.0,
                calls: pd.DataFrame | None = None) -> Scenario:
    calls = load_calls() if calls is None else calls
    surge = 1.0 + arrival_surge_pct / 100.0
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
        arrival_surge_pct=arrival_surge_pct,
        data_end=_hours(calls["arrival"].max(), ts),
    )
