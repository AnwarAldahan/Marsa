"""Yard and gate part of the Digital Twin, read from the Cargo agent's file (synthetic data).

Source: data/agent2_cargo_operations_2025.csv (the Cargo agent's own file; calibrated synthetic).
In that file every hour satisfies  yard[t] = yard[t-1] + container_arrivals[t] - container_departures[t]
(checked: largest difference 0.01), so the twin replays it exactly when nothing changes.

Gate decision: the gate clears up to a chosen number of containers per hour. The data-based choice is
the highest hourly container_departures the file recorded up to the chosen hour (no future values);
the user may type another number. Extra containers cleared each hour lower the yard from then on.
The yard does not change vessel waiting: there is no data on how a full yard slows unloading.

User stress on the yard (assumptions typed by the user, shown as such in the interface):
  gate_closed_hours: the gate clears nothing during the first N hours of the horizon
  container_inflow_multiplier: containers arriving in the yard are multiplied (x1 = the file)
The occupancy is not capped at 100 %: a value above 100 shows the yard would overflow its scale.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from marsa.config import DATA_DIR

CARGO_PATH = DATA_DIR / "agent2_cargo_operations_2025.csv"


# Load the Cargo agent's hourly file once, indexed by UTC hour.
@lru_cache(maxsize=1)
def load_cargo() -> pd.DataFrame:
    cargo = pd.read_csv(CARGO_PATH)
    cargo["timestamp"] = pd.to_datetime(cargo["timestamp"], utc=True)
    return cargo.set_index("timestamp").sort_index()


# Yard state at ts and the recorded flows for the next H hours (None where the hour is missing).
def yard_state(ts: pd.Timestamp, horizon_hours: int) -> dict | None:
    cargo = load_cargo()
    if ts not in cargo.index:
        return None
    hours = [ts + pd.Timedelta(hours=h) for h in range(horizon_hours + 1)]
    row = lambda t: cargo.loc[t] if t in cargo.index else None  # noqa: E731
    rows = [row(t) for t in hours]
    return {
        "capacity": float(rows[0]["yard_capacity"]),
        "yard": [None if r is None else float(r["containers_in_yard"]) for r in rows],
        "arrivals": [None if r is None else float(r["container_arrivals"]) for r in rows],
        "departures": [None if r is None else float(r["container_departures"]) for r in rows],
        "max_recorded_departures": float(cargo.loc[:ts, "container_departures"].max()),
        "max_recorded_at": cargo.loc[:ts, "container_departures"].idxmax().isoformat(),
    }


# Yard path for one candidate: replay of the file, adjusted by the gate decision and the user's
# yard stress. The adjustment to the recorded yard accumulates hour by hour:
#   + (multiplier - 1) x recorded arrivals      (more or fewer containers coming in)
#   + recorded departures while the gate is closed (they stay in the yard)
#   - extra containers a faster gate clears      (never more than the yard holds)
def simulate_yard(state: dict | None, gate_per_hour: float | None,
                  gate_closed_hours: int = 0, inflow_multiplier: float = 1.0) -> dict | None:
    if state is None:
        return None
    path, adjust, cleared = [], 0.0, 0.0
    for h, (yard, arr, out) in enumerate(zip(state["yard"], state["arrivals"],
                                             state["departures"], strict=True)):
        if yard is None:
            path.append(None)
            continue
        if h > 0 and arr is not None and out is not None:
            adjust += (inflow_multiplier - 1.0) * arr
            if h <= gate_closed_hours:
                adjust += out
            elif gate_per_hour is not None:
                extra = min(max(0.0, gate_per_hour - out), max(0.0, yard + adjust))
                adjust -= extra
                cleared += extra
        path.append(max(0.0, yard + adjust))
    known = [p for p in path if p is not None]
    cap = state["capacity"]
    return {
        "occupancy_pct": [None if p is None else round(p / cap * 100, 2) for p in path],
        "kpis": {
            "yard_peak_occupancy_pct": round(max(known) / cap * 100, 2) if known else None,
            "yard_end_occupancy_pct": round(known[-1] / cap * 100, 2) if known else None,
            "extra_containers_cleared": round(cleared, 1),
            "gate_per_hour": gate_per_hour,
        },
    }
