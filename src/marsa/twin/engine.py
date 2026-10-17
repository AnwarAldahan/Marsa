"""Marsa Digital Twin - Port of Los Angeles cargo berths and yard (NOT an agent).

Sea side: replays the real AIS state forward in time and asks: are there enough berths?

    vessel arrives -> waits until its own real berth time (terminal window from AIS)
                   -> if no berth is free at that moment it waits longer (capacity wait)
                   -> occupies a berth for its own real service duration

With no stress scenario and no action, the twin reproduces what AIS recorded (tested). A stress
scenario (fewer berths, a compressed arrival schedule) makes berths scarce, and the candidate
actions change how the scarce berths are used. The simulation is deterministic: there is no
random sampling, so there are no Monte Carlo runs.

Without a cargo profile the yard replays the Cargo file independently. With an explicit
normalized cargo profile, service progress feeds yard stock and full-yard blocking delays
completion and berth release. Ranking remains outside this engine.

Vessels added by the user get a stay estimated from history (src/marsa/twin/estimates.py).

Actions understood (candidate["actions"]):
    queue_policy   value: fcfs | shortest_service_first   (who gets the next free berth)
    add_berths     value: int >= 0                         (temporary extra berths)
    gate_capacity  value: "max_recorded" | number > 0      (containers the gate clears per hour)
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field, replace

from marsa.twin.yard import simulate_yard

SUPPORTED_ACTIONS = {"queue_policy", "add_berths", "gate_capacity"}
QUEUE_POLICIES = {"fcfs", "shortest_service_first"}


@dataclass
class Vessel:
    call_id: str
    mmsi: int
    arrival_h: float           # hours from t0 (<= 0: already in port)
    ready_h: float             # real berth time from AIS (earliest the terminal takes the vessel)
    service_h: float           # real time spent at the berth
    berth_h: float | None = None
    source: str = "ais"        # "ais" or "added" (added by the user for a what-if test)
    estimate: dict | None = None   # how an added vessel's stay was obtained


@dataclass
class Scenario:
    berths: int
    horizon_hours: int
    berthed: list[Vessel]      # at a berth at t0
    vessels: list[Vessel]      # waiting at t0 or arriving inside the horizon
    berths_closed: int = 0
    arrival_multiplier: float = 1.0
    data_end: float = float("inf")   # hours from t0 to the last arrival in the AIS file
    extra_berths: int = 0
    policy: str = "fcfs"
    yard: dict | None = None         # from yard.yard_state (Cargo agent file)
    gate_per_hour: float | None = None
    gate_closed_hours: int = 0           # user yard stress (see yard.simulate_yard)
    container_inflow_multiplier: float = 1.0
    notes: list[str] = field(default_factory=list)
    cargo: dict | None = None  # Explicit normalized service-effort yard proxy; None preserves replay.
    workloads: dict | None = None  # Per-call derived reference effort, frozen across candidates.


# Apply a candidate's actions to a copy of the scenario.
def apply_actions(sc: Scenario, actions: list[dict]) -> Scenario:
    sc = replace(sc)
    for action in actions:
        kind, value = action.get("type"), action.get("value")
        if kind == "queue_policy":
            if value not in QUEUE_POLICIES:
                raise ValueError(f"unknown queue policy: {value}")
            sc.policy = value
        elif kind == "add_berths":
            if int(value) < 0:
                raise ValueError("add_berths must be 0 or more")
            sc.extra_berths += int(value)
        elif kind == "gate_capacity":
            if sc.yard is None:
                raise ValueError("no yard data for this hour")
            if value == "max_recorded":
                sc.gate_per_hour = sc.yard["max_recorded_departures"]
            elif float(value) > 0:
                sc.gate_per_hour = float(value)
            else:
                raise ValueError("gate_capacity must be 'max_recorded' or a number greater than 0")
        else:
            raise ValueError(f"unknown action type: {kind}")
    return sc


# Times at which a berth slot becomes usable, given vessels already at berth.
def _slot_times(sc: Scenario) -> list[float]:
    slots = max(0, sc.berths - sc.berths_closed + sc.extra_berths)
    leaving = sorted(max(0.0, v.berth_h + v.service_h) for v in sc.berthed)
    # if more vessels are at berth than open slots, the first departures only clear the excess
    usable = leaving[len(leaving) - slots:] if len(leaving) > slots else leaving
    return sorted([0.0] * max(0, slots - len(leaving)) + usable) if slots else []


# Assign berths in time order; returns each vessel's berth time (None if never within the data).
def _assign(sc: Scenario) -> dict[str, float | None]:
    free = _slot_times(sc)
    heapq.heapify(free)
    pending = sorted(sc.vessels, key=lambda v: (max(v.arrival_h, v.ready_h), v.arrival_h))
    order = (lambda v: (v.arrival_h, v.call_id)) if sc.policy == "fcfs" else \
            (lambda v: (v.service_h, v.arrival_h, v.call_id))
    berth: dict[str, float | None] = {v.call_id: None for v in sc.vessels}
    while pending and free:
        t = heapq.heappop(free)
        ready = [v for v in pending if v.ready_h <= t]
        if not ready:
            t = min(v.ready_h for v in pending)
            ready = [v for v in pending if v.ready_h <= t]
        chosen = min(ready, key=order)
        pending.remove(chosen)
        berth[chosen.call_id] = t
        heapq.heappush(free, t + chosen.service_h)
    return berth


# Run one candidate and return hourly timeline + KPIs (all hours are within [0, horizon]).
def simulate_candidate(sc: Scenario, candidate: dict) -> dict:
    s = apply_actions(sc, candidate.get("actions", []))
    if s.cargo is not None:
        return _simulate_cargo(s, candidate)
    H = s.horizon_hours
    berth = _assign(s)

    def in_horizon(a: float, b: float) -> float:
        return max(0.0, min(b, H) - max(a, 0.0))

    total_wait = capacity_wait = 0.0
    waits, arrived, berthed_in_h = [], 0, 0
    for v in s.vessels:
        b = berth[v.call_id]
        end = b if b is not None else float("inf")
        if v.arrival_h <= H:
            arrived += v.arrival_h > 0
            total_wait += in_horizon(v.arrival_h, end)
            capacity_wait += in_horizon(max(v.arrival_h, v.ready_h), end)
            if end > max(v.arrival_h, 0.0) and v.arrival_h < H:
                waits.append(min(end, H) - v.arrival_h)
        berthed_in_h += b is not None and 0 <= b < H

    timeline = []
    for h in range(H + 1):
        at_berth = sum(1 for v in s.berthed if v.berth_h + v.service_h > h)
        at_berth += sum(1 for v in s.vessels if berth[v.call_id] is not None
                        and berth[v.call_id] <= h < berth[v.call_id] + v.service_h)
        queue = sum(1 for v in s.vessels if v.arrival_h <= h
                    and (berth[v.call_id] is None or berth[v.call_id] > h))
        timeline.append({"hour": h, "at_berth": at_berth, "waiting": queue})

    yard = simulate_yard(s.yard, s.gate_per_hour, s.gate_closed_hours, s.container_inflow_multiplier)
    if yard is not None:
        for point, occ in zip(timeline, yard["occupancy_pct"], strict=True):
            point["yard_occupancy_pct"] = occ

    waited = [w for w in waits if w > 0]
    kpis = {
        "total_wait_vessel_hours": round(total_wait, 2),
        "capacity_wait_vessel_hours": round(capacity_wait, 2),
        "vessels_waited": len(waited),
        "avg_wait_hours": round(sum(waited) / len(waited), 2) if waited else 0.0,
        "max_wait_hours": round(max(waited), 2) if waited else 0.0,
        "vessels_arrived": arrived,
        "vessels_berthed": berthed_in_h,
        "waiting_end": timeline[-1]["waiting"],
        "peak_berths_used": max(p["at_berth"] for p in timeline),
        "open_berths": s.berths - s.berths_closed + s.extra_berths,
        **(yard["kpis"] if yard is not None else {}),
    }
    vessels = [{"call_id": v.call_id, "mmsi": v.mmsi, "arrival_h": round(v.arrival_h, 2),
                "scheduled_berth_h": round(v.ready_h, 2), "service_h": round(v.service_h, 2),
                "berth_h": None if berth[v.call_id] is None else round(berth[v.call_id], 2),
                "source": v.source, "estimate": v.estimate}
               for v in s.vessels if v.arrival_h <= H]
    return {"candidate_id": candidate["id"], "actions": candidate.get("actions", []),
            "kpis": kpis, "timeline": timeline, "vessels": vessels}


def _simulate_cargo(s: Scenario, candidate: dict) -> dict:
    """Joint service/yard progression within this Twin, in service-equivalent units.

    One occupied berth can discharge one nominal service-hour of work per hour.
    Full-yard headroom shares clearance across occupied berths. Historical service
    duration is only a retrospective effort proxy, never a measured cargo quantity.
    """
    from math import isfinite

    H, slots = s.horizon_hours, max(0, s.berths - s.berths_closed + s.extra_berths)
    capacity, initial, gate = (s.cargo[k] for k in ("capacity", "initial_inventory", "gate_rate"))
    if s.gate_per_hour is not None:
        gate = s.gate_per_hour / s.cargo["source_capacity"] * capacity
    if (not all(isfinite(x) for x in (capacity, initial, gate, H)) or capacity <= 0
            or not 0 <= initial <= capacity or gate < 0 or H <= 0):
        raise ValueError("invalid normalized cargo scenario")
    if not isfinite(s.container_inflow_multiplier) or s.container_inflow_multiplier <= 0:
        raise ValueError("invalid vessel workload multiplier")
    all_vessels = s.berthed + s.vessels
    if (len({v.call_id for v in all_vessels}) != len(all_vessels)
            or any(not isfinite(v.service_h) or v.service_h <= 0 for v in all_vessels)
            or any(not all(isfinite(x) for x in (v.arrival_h, v.ready_h)) for v in all_vessels)):
        raise ValueError("invalid vessel service effort or identity")
    near_pct = float(s.cargo.get("near_capacity_pct", 90.))
    if not isfinite(near_pct) or not 0 < near_pct <= 100:
        raise ValueError("near-capacity threshold must be between 0 and 100 percent")
    near_inventory = capacity * near_pct / 100

    def congestion(stock):
        return "FULL" if stock >= capacity - 1e-9 else "NEAR_CAPACITY" if stock >= near_inventory - 1e-9 else "BELOW_THRESHOLD"

    if s.workloads is not None and set(s.workloads) != {v.call_id for v in all_vessels}:
        raise ValueError("per-call cargo workloads must cover exactly the simulated cohort")
    rows = []
    for v in all_vessels:
        start = v.berth_h if v in s.berthed else None
        reference = None if s.workloads is None else s.workloads[v.call_id]
        effort = (max(0.0, start + v.service_h) if start is not None else v.service_h) if reference is None else reference["initial_workload"]
        if not isfinite(effort) or effort < 0 or (start is None and effort == 0):
            raise ValueError("invalid per-call cargo workload")
        work = effort * s.container_inflow_multiplier
        rows.append({"call_id": v.call_id, "mmsi": v.mmsi, "arrival_h": v.arrival_h,
                     "scheduled_berth_h": v.ready_h, "service_h": v.service_h,
                     "berth_h": start, "completion_h": None,
                     "initial_workload": work, "remaining_workload": work,
                     "source": v.source, "estimate": v.estimate,
                     **({"workload_reference": dict(reference)} if reference is not None else {})})
    pending = [r for r in rows if r["berth_h"] is None]
    active = [r for r in rows if r["berth_h"] is not None]
    total_work = sum(r["initial_workload"] for r in rows)
    inventory, unloaded, cleared, blocked, total_wait, capacity_wait, t = initial, 0., 0., 0., 0., 0., 0.
    peak_inventory, peak_berths = initial, len(active)
    timeline, flows = [], []
    near_hours = full_hours = 0.
    eps = 1e-9
    order = (lambda r: (r["arrival_h"], r["call_id"])) if s.policy == "fcfs" else (
        lambda r: (r["service_h"], r["arrival_h"], r["call_id"]))
    for h in range(H + 1):
        # Release only completed work; never use the recorded departure as a release override.
        active = [r for r in active if r["completion_h"] is None]
        timeline.append({"hour": h, "at_berth": len(active),
                         "waiting": sum(r["arrival_h"] <= h for r in pending),
                         "yard_inventory": inventory, "yard_occupancy_pct": inventory / capacity * 100,
                         "completed": sum(r["completion_h"] is not None for r in rows),
                         "cargo_congestion_status": congestion(inventory), "cargo_unloaded": unloaded,
                         "cargo_cleared": cleared, "blocked_unloading": blocked,
                         "remaining_vessel_workload": sum(r["remaining_workload"] for r in rows),
                         "yard_near_capacity_hours": near_hours, "yard_full_hours": full_hours})
        if h == H:
            break
        opening, before_unload, before_clear, before_block = inventory, unloaded, cleared, blocked
        before_near, before_full = near_hours, full_hours
        while t < h + 1 - eps:
            for r in active[:]:
                if r["remaining_workload"] <= eps:
                    r["remaining_workload"] = 0.
                    r["completion_h"] = t
                    active.remove(r)
            ready = sorted((r for r in pending if max(r["arrival_h"], r["scheduled_berth_h"]) <= t + eps), key=order)
            for r in ready[:max(0, slots - len(active))]:
                r["berth_h"] = t
                pending.remove(r)
                active.append(r)
            peak_berths = max(peak_berths, len(active))
            if abs(t - h) <= eps:
                timeline[-1].update(at_berth=len(active),
                                    waiting=sum(r["arrival_h"] <= h for r in pending),
                                    completed=sum(r["completion_h"] is not None for r in rows))
            clearance = 0. if t < s.gate_closed_hours else gate
            productivity = min(1., clearance / len(active)) if active and inventory >= capacity - eps else 1.
            unload_rate = len(active) * productivity
            clear_rate = clearance if inventory > eps else min(clearance, unload_rate)
            limits = [h + 1 - t]
            limits += [r["arrival_h"] - t for r in pending if r["arrival_h"] > t + eps]
            limits += [max(r["arrival_h"], r["scheduled_berth_h"]) - t for r in pending
                       if max(r["arrival_h"], r["scheduled_berth_h"]) > t + eps]
            if productivity > 0:
                limits += [r["remaining_workload"] / productivity for r in active]
            delta = unload_rate - clear_rate
            if delta > eps and inventory < capacity - eps:
                limits.append((capacity - inventory) / delta)
            elif delta < -eps and inventory > eps:
                limits.append(inventory / -delta)
            # Split only for measurement at the threshold; service/clearance physics is unchanged.
            if abs(delta) > eps:
                crossing = (near_inventory - inventory) / delta
                if crossing > eps:
                    limits.append(crossing)
            dt = min(limits)
            midpoint_inventory = inventory + delta * dt / 2
            near_hours += dt if midpoint_inventory >= near_inventory - eps else 0.
            full_hours += dt if midpoint_inventory >= capacity - eps else 0.
            total_wait += dt * sum(r["arrival_h"] <= t + eps for r in pending)
            capacity_wait += dt * sum(max(r["arrival_h"], r["scheduled_berth_h"]) <= t + eps for r in pending)
            for r in active:
                r["remaining_workload"] = max(0., r["remaining_workload"] - dt * productivity)
                if r["remaining_workload"] <= eps:
                    r["remaining_workload"] = 0.
                    r["completion_h"] = t + dt
            unloaded += unload_rate * dt
            cleared += clear_rate * dt
            blocked += (len(active) - unload_rate) * dt
            inventory += delta * dt
            peak_inventory = max(peak_inventory, inventory)
            t += dt
            active = [r for r in active if r["completion_h"] is None]
        residual = initial + unloaded - cleared - inventory
        work_residual = total_work - unloaded - sum(r["remaining_workload"] for r in rows)
        if (abs(residual) > 1e-7 or abs(work_residual) > 1e-7
                or inventory < -eps or inventory > capacity + eps):
            raise ArithmeticError("cargo conservation or capacity invariant failed")
        flows.append({"hour": h, "opening_inventory": opening, "unloaded": unloaded - before_unload,
                      "gate_cleared": cleared - before_clear, "blocked_unloading": blocked - before_block,
                      "closing_inventory": inventory, "yard_conservation_residual": residual,
                      "work_conservation_residual": work_residual,
                      "yard_near_capacity_hours": near_hours - before_near,
                      "yard_full_hours": full_hours - before_full})
    initial_ids = {v.call_id for v in s.berthed}
    waits = [min(r["berth_h"] if r["berth_h"] is not None else H, H) - r["arrival_h"]
             for r in rows if r["arrival_h"] < H and r["call_id"] not in initial_ids
             and (r["berth_h"] is None or r["berth_h"] > max(0, r["arrival_h"]))]
    kpis = {"total_wait_vessel_hours": round(total_wait, 2), "capacity_wait_vessel_hours": round(capacity_wait, 2),
            "vessels_waited": len(waits), "avg_wait_hours": round(sum(waits) / len(waits), 2) if waits else 0.,
            "max_wait_hours": round(max(waits), 2) if waits else 0.,
            "vessels_arrived": sum(0 < r["arrival_h"] <= H for r in rows),
            "vessels_berthed": sum(r["berth_h"] is not None and 0 <= r["berth_h"] < H for r in rows),
            "waiting_end": timeline[-1]["waiting"], "open_berths": slots,
            "peak_berths_used": peak_berths,
            "yard_peak_occupancy_pct": peak_inventory / capacity * 100,
            "yard_end_occupancy_pct": inventory / capacity * 100, "gate_per_hour": s.gate_per_hour,
            "completed_vessels": sum(r["completion_h"] is not None for r in rows),
            "remaining_vessel_workload": sum(r["remaining_workload"] for r in rows),
            "yard_near_capacity_hours": near_hours, "yard_full_hours": full_hours,
            "cargo_congestion_peak_status": congestion(peak_inventory),
            "cargo_unloaded": unloaded, "cargo_cleared": cleared, "blocked_unloading": blocked,
            "yard_conservation_residual": initial + unloaded - cleared - inventory,
            "work_conservation_residual": total_work - unloaded - sum(r["remaining_workload"] for r in rows)}
    return {"candidate_id": candidate["id"], "actions": candidate.get("actions", []), "kpis": kpis,
            "timeline": timeline, "vessels": rows, "flows": flows,
            "cargo_congestion_definition": {"near_capacity_pct": near_pct, "full_capacity_pct": 100.,
                "status_scope": "simulated yard occupancy; independent of XGBoost +6h proxy",
                "threshold_provenance": "DEMO SCENARIO ASSUMPTION; not a validated operational threshold",
                "duration_method": "continuous simulated time, including fractional crossings"},
            "cargo_assumptions": dict(s.cargo), "units": "normalized_service_effort", "provenance": "SIMULATED"}
