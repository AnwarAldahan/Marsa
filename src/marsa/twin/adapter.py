"""Bridge between the pipeline and the Digital Twin: build the starting state and list its sources.

The starting state comes from AIS (data/processed/la_cargo_calls_2023_2025.csv), the Cargo agent's
file (yard and gate) and the number of container berths (config/port_config.yaml, with its source).
Stress values and added vessels are what the user chose in the interface; with none chosen the twin
replays the recorded data.
"""
from __future__ import annotations

import pandas as pd

from marsa.twin.calls import CALLS_PATH, missing_ranges, scenario_at
from marsa.twin.engine import Scenario, Vessel
from marsa.twin.estimates import estimate_stay
from marsa.twin.workloads import attach_workloads
from marsa.twin.yard import CARGO_PATH, yard_state

ADDED_KEYS = {"arrival_in_hours", "vessel_id", "length_m", "service_hours"}


# Check the user's stress inputs; invalid values are rejected, never clipped. There is no upper
# limit on the arrival multiplier: the user is free to test any scenario.
YARD_STRESS = {"gate_closed_hours", "container_inflow_multiplier"}


def validate_stress(stress: dict | None, berths: int, horizon: int | None = None) -> dict:
    stress = dict(stress or {})
    unknown = set(stress) - {"berths_closed", "arrival_multiplier"} - YARD_STRESS
    if unknown:
        raise ValueError(f"unknown stress inputs: {sorted(unknown)}")
    closed = int(stress.get("berths_closed", 0))
    multiplier = float(stress.get("arrival_multiplier", 1.0))
    gate_closed = int(stress.get("gate_closed_hours", 0))
    inflow = float(stress.get("container_inflow_multiplier", 1.0))
    if not 0 <= closed <= berths:
        raise ValueError(f"berths_closed must be between 0 and {berths}")
    if not multiplier > 0.0:
        raise ValueError("arrival_multiplier must be greater than 0 (1 = real schedule)")
    if not 0 <= gate_closed <= (horizon if horizon is not None else gate_closed):
        raise ValueError(f"gate_closed_hours must be between 0 and {horizon}")
    if not inflow > 0.0:
        raise ValueError("container_inflow_multiplier must be greater than 0 (1 = the Cargo file)")
    return {"berths_closed": closed, "arrival_multiplier": multiplier,
            "gate_closed_hours": gate_closed, "container_inflow_multiplier": inflow}


# Turn the user's added vessels into twin vessels; each stay is estimated from history (or typed).
def added_vessels(ts: pd.Timestamp, specs: list[dict] | None, horizon: int, k: int) -> list[Vessel]:
    out = []
    for i, spec in enumerate(specs or []):
        unknown = set(spec) - ADDED_KEYS
        if unknown:
            raise ValueError(f"unknown added-vessel fields: {sorted(unknown)}")
        eta = float(spec.get("arrival_in_hours", 0.0))
        if not 0.0 <= eta <= horizon:
            raise ValueError(f"arrival_in_hours must be between 0 and {horizon}")
        est = estimate_stay(ts, k, vessel_id=spec.get("vessel_id") or None,
                            length_m=spec.get("length_m"), service_hours=spec.get("service_hours"))
        out.append(Vessel(f"ADD{i + 1}", mmsi=0, arrival_h=eta, ready_h=eta,
                          service_h=est["hours"], source="added",
                          estimate={**est, "vessel_id": spec.get("vessel_id")}))
    return out


# Short description of the starting state, shown in the interface next to the results.
def scenario_summary(sc: Scenario) -> dict:
    return {
        "open_berths": sc.berths - sc.berths_closed,
        "container_berths": sc.berths,
        "berths_closed": sc.berths_closed,
        "arrival_multiplier": sc.arrival_multiplier,
        "vessels_at_berth_now": len(sc.berthed),
        "vessels_at_berth": [
            {"call_id": v.call_id, "mmsi": v.mmsi, "berthed_since_h": round(v.berth_h, 2),
             "leaves_in_h": round(v.berth_h + v.service_h, 2) if sc.cargo is None else None,
             **({"reference_service_remaining_h": round(v.berth_h + v.service_h, 2)} if sc.cargo is not None else {})}
            for v in sc.berthed],
        "vessels_waiting_now": sum(1 for v in sc.vessels if v.arrival_h <= 0),
        "vessels_arriving_in_horizon": sum(
            1 for v in sc.vessels if 0 < v.arrival_h <= sc.horizon_hours),
        "stress_applied": bool(sc.berths_closed or sc.arrival_multiplier != 1.0
                               or sc.gate_closed_hours or sc.container_inflow_multiplier != 1.0),
        "gate_closed_hours": sc.gate_closed_hours,
        "container_inflow_multiplier": sc.container_inflow_multiplier,
        "yard_capacity": None if sc.yard is None else sc.yard["capacity"],
        "gate_max_recorded": None if sc.yard is None else sc.yard["max_recorded_departures"],
        "vessels_added": sum(1 for v in sc.vessels if v.source == "added"),
        **({"cargo_coupling": sc.cargo, "vessel_workloads": sc.workloads} if sc.cargo is not None else {}),
    }


# Build the scenario at ts and the list of sources the interface shows next to the results.
def build_scenario(ts: pd.Timestamp, cfg: dict, stress: dict | None = None,
                   added: list[dict] | None = None) -> tuple[Scenario, list[dict]]:
    berths = int(cfg["capacity"]["container_berths"])
    horizon = int(cfg["twin"]["horizon_hours"])
    stress = validate_stress(stress, berths, horizon)
    sc = scenario_at(ts, berths=berths, horizon_hours=horizon,
                     berths_closed=stress["berths_closed"],
                     arrival_multiplier=stress["arrival_multiplier"])
    sc.yard = yard_state(ts, horizon)
    if cfg["twin"].get("cargo_coupling", False):
        if sc.yard is None:
            raise ValueError("cargo coupling requires an exact-hour Cargo snapshot")
        from math import isfinite

        buffer_hours = float(cfg["twin"].get("cargo_buffer_hours", 24))
        source_capacity = sc.yard["capacity"]
        occupancy = sc.yard["yard"][0] / source_capacity if source_capacity > 0 else float("nan")
        departures = sc.yard["departures"][0]
        if (not isfinite(buffer_hours) or buffer_hours <= 0 or not isfinite(occupancy)
                or not 0 <= occupancy <= 1 or departures is None or not isfinite(departures) or departures < 0):
            raise ValueError("invalid Cargo proxy initialization")
        capacity = berths * buffer_hours
        sc.cargo = {"capacity": capacity, "initial_inventory": occupancy * capacity,
                    "gate_rate": departures / source_capacity * capacity, "source_capacity": source_capacity,
                    "units": "normalized_service_effort", "buffer_hours": buffer_hours,
                    "near_capacity_pct": float(cfg["twin"].get("cargo_near_capacity_pct", 90.)),
                    "assumption": "Synthetic Cargo occupancy and clearance/capacity ratios proxy an abstract yard; "
                                  "one nominal berth-service hour is one workload unit; each occupied slot provides one unit/hour. Not TEU, measured unloading, "
                                  "or verified terminal membership. Historical service durations are retrospective references."}
        workload_source = attach_workloads(sc, ts)
    sc.gate_closed_hours = stress["gate_closed_hours"]
    sc.container_inflow_multiplier = stress["container_inflow_multiplier"]
    extra = added_vessels(ts, added, horizon, int(cfg["twin"]["similar_vessels_k"]))
    sc.vessels = sc.vessels + extra
    if sc.workloads is not None:
        for vessel in extra:
            sc.workloads[vessel.call_id] = {"total_workload": vessel.service_h,
                "initial_workload": vessel.service_h, "method": vessel.estimate["method"],
                "provenance": "USER_INPUT_OR_EARLIER_HISTORY_ESTIMATE", "available_at_utc": ts.isoformat()}
    sources = [
        {"input": "vessels at berth, waiting and arriving; their arrival, berth and service times",
         "source": str(CALLS_PATH.name), "provenance": "OBSERVED (AIS)"},
        {"input": "container berths", "value": berths,
         "source": cfg["capacity"]["container_berths_source"], "provenance": "PUBLIC SOURCE"},
    ]
    if sc.yard is not None:
        if sc.cargo is not None:
            sources.append(workload_source)
            sources.append({"input": "connected cargo scenario", "value": sc.cargo,
                            "source": "Explicit normalized service-effort proxy using t0 Cargo ratios only",
                            "provenance": "SYNTHETIC SCENARIO ASSUMPTION"})
        sources.append({"input": "yard and gate", "kind": "yard",
                        "value": {"capacity": sc.yard["capacity"],
                                  "max_recorded_departures": sc.yard["max_recorded_departures"],
                                  "max_recorded_at": sc.yard["max_recorded_at"]},
                        "source": str(CARGO_PATH.name), "provenance": "SYNTHETIC (Cargo agent)"})
    if extra:
        sources.append({"input": "added vessels", "kind": "added",
                        "value": [v.estimate for v in extra],
                        "source": "added by the user; stays estimated from earlier stays",
                        "provenance": "USER INPUT"})
    if (stress["berths_closed"] or stress["arrival_multiplier"] != 1.0
            or stress["gate_closed_hours"] or stress["container_inflow_multiplier"] != 1.0):
        sources.append({"input": "stress scenario", "value": stress,
                        "source": "chosen by the user in the interface", "provenance": "USER INPUT"})
    if sc.data_end < sc.horizon_hours:
        sources.append({"input": "arrival schedule", "kind": "data_end",
                        "source": "AIS data ends inside the horizon; later arrivals are unknown",
                        "provenance": "LIMITATION"})
    # hours missing from the AIS file inside the horizon or inside a current vessel's stay:
    # a vessel seen at berth/waiting on both sides of such a gap is kept as one stay
    earliest = min([v.arrival_h for v in sc.berthed + sc.vessels if v.arrival_h <= 0], default=0.0)
    gaps = missing_ranges(ts + pd.Timedelta(hours=earliest), ts + pd.Timedelta(hours=sc.horizon_hours))
    if gaps:
        sources.append({"input": "missing AIS hours", "kind": "data_gap",
                        "value": [[a.isoformat(), b.isoformat()] for a, b in gaps],
                        "source": "These hours are missing from the AIS file; a vessel seen on both "
                                  "sides of a gap is treated as the same stay.",
                        "provenance": "LIMITATION"})
    return sc, sources
