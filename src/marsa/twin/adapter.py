"""Bridge between the pipeline and the Digital Twin: build the starting state and list its sources.

The starting state comes only from AIS (data/processed/la_cargo_calls_2023_2025.csv) plus the
number of container berths (config/port_config.yaml, with its source). Stress values are what the
user chose in the interface; with none chosen the twin replays the real AIS schedule.
"""
from __future__ import annotations

import pandas as pd

from marsa.twin.calls import CALLS_PATH, scenario_at
from marsa.twin.engine import Scenario

MAX_SURGE_PCT = 200.0


# Check the user's stress inputs; anything outside the allowed range is rejected, never clipped.
def validate_stress(stress: dict | None, berths: int) -> dict:
    stress = dict(stress or {})
    unknown = set(stress) - {"berths_closed", "arrival_surge_pct"}
    if unknown:
        raise ValueError(f"unknown stress inputs: {sorted(unknown)}")
    closed = int(stress.get("berths_closed", 0))
    surge = float(stress.get("arrival_surge_pct", 0.0))
    if not 0 <= closed <= berths:
        raise ValueError(f"berths_closed must be between 0 and {berths}")
    if not 0.0 <= surge <= MAX_SURGE_PCT:
        raise ValueError(f"arrival_surge_pct must be between 0 and {MAX_SURGE_PCT:g}")
    return {"berths_closed": closed, "arrival_surge_pct": surge}


# Build the scenario at ts and the list of sources the interface shows next to the results.
def build_scenario(ts: pd.Timestamp, cfg: dict, stress: dict | None = None) -> tuple[Scenario, list[dict]]:
    berths = int(cfg["capacity"]["container_berths"])
    stress = validate_stress(stress, berths)
    sc = scenario_at(ts, berths=berths, horizon_hours=int(cfg["twin"]["horizon_hours"]), **stress)
    sources = [
        {"input": "vessels at berth, waiting and arriving; their arrival, berth and service times",
         "source": str(CALLS_PATH.name), "provenance": "OBSERVED (AIS)"},
        {"input": "container berths", "value": berths,
         "source": cfg["capacity"]["container_berths_source"], "provenance": "PUBLIC SOURCE"},
    ]
    if stress["berths_closed"] or stress["arrival_surge_pct"]:
        sources.append({"input": "stress scenario", "value": stress,
                        "source": "chosen by the user in the interface", "provenance": "USER INPUT"})
    if sc.data_end < sc.horizon_hours:
        sources.append({"input": "arrival schedule",
                        "source": "AIS data ends inside the horizon; later arrivals are unknown",
                        "provenance": "LIMITATION"})
    return sc, sources
