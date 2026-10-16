"""Marsa Digital Twin - Port of Los Angeles cargo berths and yard (NOT an agent).

Sea side: replays the real AIS state forward in time and asks: are there enough berths?

    vessel arrives -> waits until its own real berth time (terminal window from AIS)
                   -> if no berth is free at that moment it waits longer (capacity wait)
                   -> occupies a berth for its own real service duration

With no stress scenario and no action, the twin reproduces what AIS recorded (tested). A stress
scenario (fewer berths, a compressed arrival schedule) makes berths scarce, and the candidate
actions change how the scarce berths are used. The simulation is deterministic: there is no
random sampling, so there are no Monte Carlo runs.

Yard side (src/marsa/twin/yard.py): replays the Cargo agent's file; a gate decision clears more
containers per hour. The twin returns vessel and yard measures side by side and does not rank them.

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
    notes: list[str] = field(default_factory=list)


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

    yard = simulate_yard(s.yard, s.gate_per_hour)
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
