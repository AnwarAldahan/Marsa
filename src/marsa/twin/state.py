"""Build a twin Scenario from one hour of the merged dataset + config."""
from __future__ import annotations
from marsa.twin.engine import Scenario


def scenario_from_snapshot(snap: dict, cfg: dict, context: dict | None = None) -> Scenario:
    cap, vs, tw = cfg["capacity"], cfg["vessel_service"], cfg["twin"]
    berthed = int(round(snap.get("pola_berthed_cargo_vessels") or 0))
    waiting_all = float(snap.get("waiting_vessel_count") or 0)
    waiting = int(round(waiting_all * vs["la_share_of_waiting"]))
    berthed = min(berthed, cap["container_berths"])

    discharge_units = cap["discharge_units_per_vessel_hour"]
    gate_units = cap["gate_units_per_hour_capacity"]

    lam = vs["arrival_rate_per_hour"]
    if lam is None:                                   # Little's law: L = lambda * W
        lam = max(0.05, (berthed or 10) / vs["mean_service_hours"])

    ctx = context or {}
    return Scenario(
        berths=cap["container_berths"],
        berthed_now=berthed,
        waiting_now=waiting,
        yard_inventory=float(snap.get("containers_in_yard") or 0.6 * cap["yard_capacity_units"]),
        yard_capacity=float(cap["yard_capacity_units"]),
        discharge_units_per_hour=discharge_units,
        gate_units_per_hour=gate_units,
        arrival_rate_per_hour=lam,
        mean_service_hours=vs["mean_service_hours"],
        service_bounds=(vs["min_service_hours"], vs["max_service_hours"]),
        horizon_hours=tw["horizon_hours"],
        crane_multiplier=ctx.get("crane_multiplier", 1.0),
        gate_multiplier=ctx.get("gate_multiplier", 1.0),
        arrival_multiplier=ctx.get("arrival_multiplier", 1.0),
        late_threshold_hours=tw["late_threshold_hours"],
        yard_slowdown_occupancy=cap["yard_slowdown_occupancy"],
    )
