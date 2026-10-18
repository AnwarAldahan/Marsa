"""Forecast mode: the twin treats the chosen hour as "today" and uses only what is known by then.

Replay mode (default) takes every stay and berth time from the AIS calls, so it knows the future.
Forecast mode keeps what a port manager really knows at hour ts and estimates the rest from the past:

  known at ts (from AIS):   which vessels are at berth, which are waiting, and how long they have
                            been there; the vessels arriving in the next H hours (ports know them in
                            advance from arrival notices / ETA)
  estimated from the past: how long each vessel will stay at its berth, and how much longer a
                            waiting vessel will wait for its berth time

Only stays that ended (berth_end <= ts) and waits that ended (berth_start <= ts) are used.
Stay pool of a vessel: its own earlier LA stays; if it has none, the stays of the k vessels closest
in length (k from config, chosen by scripts/select_similar_k.py). Wait pool: its own earlier waits;
if none, every earlier wait at LA.
A vessel already there for e hours gets the median of (x - e) over the pool values x > e: the usual
remaining time of similar stays that lasted at least as long. If no earlier value is that long, the
estimate is 0 (it is expected to finish now) and this is reported in the estimate.
Arriving vessels are taken as ready on arrival: whether they will wait for a terminal window is not
known in advance.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from marsa.twin.calls import load_calls
from marsa.twin.engine import Scenario


# Values from a pool that already passed `elapsed`; the median remaining time, or 0 if none.
def remaining_median(pool: np.ndarray, elapsed: float) -> tuple[float, int]:
    longer = pool[pool > elapsed]
    if longer.size == 0:
        return 0.0, 0
    return float(np.median(longer - elapsed)), int(longer.size)


# Everything known about earlier LA calls at ts, prepared once per scenario.
class History:
    def __init__(self, ts: pd.Timestamp, calls: pd.DataFrame | None = None):
        calls = load_calls() if calls is None else calls
        self.stays = calls[calls["berth_end"] <= ts]
        self.waits = calls[(calls["waited"]) & (calls["berth_start"] <= ts)]
        self.per_vessel = (self.stays.groupby("vessel_id")["length_m"].median().dropna())

    # Earlier stay lengths of this vessel, else of the k vessels closest in length (ties kept).
    def stay_pool(self, vessel_id: str, length_m: float | None, k: int) -> tuple[np.ndarray, str]:
        own = self.stays.loc[self.stays["vessel_id"] == vessel_id, "berth_hours"].to_numpy()
        if own.size:
            return own, "own_history"
        if length_m is None or np.isnan(length_m) or self.per_vessel.empty:
            return self.stays["berth_hours"].to_numpy(), "all_earlier_stays"
        distance = np.abs(self.per_vessel.to_numpy() - float(length_m))
        cutoff = np.sort(distance)[min(k, distance.size) - 1]
        nearest = self.per_vessel.index[distance <= cutoff]
        return (self.stays.loc[self.stays["vessel_id"].isin(nearest), "berth_hours"].to_numpy(),
                "similar_length")

    # Earlier waits of this vessel, else every earlier wait at LA.
    def wait_pool(self, vessel_id: str) -> tuple[np.ndarray, str]:
        own = self.waits.loc[self.waits["vessel_id"] == vessel_id, "wait_hours"].to_numpy()
        if own.size:
            return own, "own_history"
        return self.waits["wait_hours"].to_numpy(), "all_earlier_waits"


# Replace every future-only value in a replay scenario with an estimate from the past.
def to_forecast(sc: Scenario, ts: pd.Timestamp, k: int, calls: pd.DataFrame | None = None) -> Scenario:
    calls = load_calls() if calls is None else calls
    info = calls.set_index("call_id")
    hist = History(ts, calls)

    def stay(vessel, elapsed: float) -> tuple[float, dict]:
        row = info.loc[vessel.call_id]
        pool, method = hist.stay_pool(row.vessel_id, row.length_m, k)
        hours, n = remaining_median(pool, elapsed)
        return hours, {"method": method, "based_on": n, "pool_size": int(pool.size),
                       "elapsed_h": round(elapsed, 2), "vessel_id": row.vessel_id,
                       "length_m": None if pd.isna(row.length_m) else float(row.length_m),
                       "finished_by_estimate": n == 0}

    berthed = []
    for v in sc.berthed:   # at berth for -berth_h hours; stays the usual remaining time
        remaining, est = stay(v, -v.berth_h)
        berthed.append(replace(v, service_h=-v.berth_h + remaining, ready_h=v.berth_h,
                               estimate={**est, "stay_remaining_h": round(remaining, 2)}))
    vessels = []
    for v in sc.vessels:
        if v.source != "ais":
            vessels.append(v)
            continue
        full, est = stay(v, 0.0)
        if v.arrival_h <= 0:   # waiting now: usual remaining wait of similar waits
            row = info.loc[v.call_id]
            pool, method = hist.wait_pool(row.vessel_id)
            wait_left, n = remaining_median(pool, -v.arrival_h)
            est = {**est, "wait_method": method, "wait_based_on": n, "wait_remaining_h": round(wait_left, 2)}
            vessels.append(replace(v, ready_h=wait_left, service_h=full, estimate=est))
        else:                  # arriving: ready on arrival (a terminal window is not known ahead)
            vessels.append(replace(v, ready_h=v.arrival_h, service_h=full, estimate=est))
    return replace(sc, berthed=berthed, vessels=vessels)
