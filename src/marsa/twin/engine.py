"""Marsa digital twin - simulation environment (NOT an agent).

Simulates the container chain for a horizon starting from the CURRENT port state:

    vessels arrive -> wait if no berth -> berth (queue rule) -> cranes discharge TEU
    -> yard inventory (stock-flow) -> gate/trucks clear the yard

It never chooses a strategy. It takes a scenario (current state) and one candidate
(actions) and returns raw KPIs. Monte Carlo repetition gives mean + spread.

Action types understood (candidate["actions"]):
    queue_policy        value: fcfs | shortest_service_first | longest_service_first
    add_berths          value: int  (temporary extra berths, e.g. open a lay berth)
    crane_boost         value: float multiplier on discharge rate (e.g. 1.25 = +25%)
    gate_boost          value: float multiplier on gate clearance rate
    delay_arrivals      value: hours to push back arrivals of vessels not yet in port ("virtual arrival")
    prioritise_vessel   value: vessel_id  (jumps the queue)
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field, asdict


@dataclass
class Vessel:
    id: str
    eta: float                 # hours from t0 (<=0 means already in port / at berth)
    service_hours: float
    teu: float
    at_berth_since: float | None = None
    berth_time: float | None = None
    depart_time: float | None = None
    priority: float = 0.0


@dataclass
class Scenario:
    berths: int
    berthed_now: int               # vessels currently at berth
    waiting_now: int               # vessels currently waiting for a berth
    yard_inventory: float          # yard units now
    yard_capacity: float
    discharge_units_per_hour: float   # per vessel at berth (yard units)
    gate_units_per_hour: float        # yard clearance capacity (yard units)
    arrival_rate_per_hour: float
    mean_service_hours: float
    service_bounds: tuple = (8.0, 48.0)
    horizon_hours: int = 24
    crane_multiplier: float = 1.0     # from weather / events
    gate_multiplier: float = 1.0
    arrival_multiplier: float = 1.0
    late_threshold_hours: float = 6.0
    yard_slowdown_occupancy: float = 0.95


@dataclass
class RunResult:
    avg_wait_hours: float
    max_wait_hours: float
    delayed_vessels: int
    vessels_served: int
    vessels_arrived: int
    berth_utilization: float
    yard_peak_occupancy: float
    yard_end_occupancy: float
    queue_end: int


def _sample_service(rng, sc: Scenario):
    lo, hi = sc.service_bounds
    # log-normal around the mean, clipped to bounds
    mu, sigma = math.log(sc.mean_service_hours) - 0.125, 0.5
    return min(hi, max(lo, rng.lognormvariate(mu, sigma)))


def build_vessels(rng: random.Random, sc: Scenario, forecast: list | None = None) -> list[Vessel]:
    """Initial vessels: those at berth (partially served), those waiting, and future arrivals."""
    vessels = []
    for i in range(sc.berthed_now):
        s = _sample_service(rng, sc)
        done = rng.uniform(0, 0.9) * s
        vessels.append(Vessel(f"B{i}", eta=-done, service_hours=s, teu=s * 200, at_berth_since=-done))
    for i in range(sc.waiting_now):
        w = rng.uniform(0, sc.late_threshold_hours)
        vessels.append(Vessel(f"W{i}", eta=-w, service_hours=_sample_service(rng, sc), teu=0))
    if forecast:
        for v in forecast:
            vessels.append(Vessel(str(v["vessel_id"]), eta=float(v["eta_hours"]),
                                  service_hours=float(v.get("service_hours", sc.mean_service_hours)),
                                  teu=float(v.get("teu", 0))))
    else:
        t, lam = 0.0, sc.arrival_rate_per_hour * sc.arrival_multiplier
        while lam > 0:
            t += rng.expovariate(lam)
            if t > sc.horizon_hours:
                break
            vessels.append(Vessel(f"A{len(vessels)}", eta=t, service_hours=_sample_service(rng, sc), teu=0))
    return vessels


def apply_actions(sc: Scenario, vessels: list[Vessel], actions: list[dict]) -> tuple[Scenario, str]:
    policy = "fcfs"
    sc = Scenario(**asdict(sc))
    for a in actions:
        t, v = a.get("type"), a.get("value")
        if t == "queue_policy":
            policy = v
        elif t == "add_berths":
            sc.berths += int(v)
        elif t == "crane_boost":
            sc.crane_multiplier *= float(v)
        elif t == "gate_boost":
            sc.gate_multiplier *= float(v)
        elif t == "delay_arrivals":
            for ves in vessels:
                if ves.eta > 0:
                    ves.eta += float(v)
        elif t == "prioritise_vessel":
            for ves in vessels:
                if ves.id == str(v):
                    ves.priority = -1e6
        else:
            raise ValueError(f"unknown action type: {t}")
    return sc, policy


def run_once(sc: Scenario, vessels: list[Vessel], policy: str) -> RunResult:
    """Hour-stepped simulation (1h resolution is what the data supports)."""
    H = sc.horizon_hours
    key = {"fcfs": lambda v: (v.priority, v.eta),
           "shortest_service_first": lambda v: (v.priority, v.service_hours),
           "longest_service_first": lambda v: (v.priority, -v.service_hours)}[policy]
    at_berth = [v for v in vessels if v.at_berth_since is not None]
    for v in at_berth:
        v.berth_time = v.at_berth_since
        v.depart_time = v.at_berth_since + v.service_hours
    waiting = sorted([v for v in vessels if v.at_berth_since is None and v.eta <= 0], key=key)
    future = sorted([v for v in vessels if v.eta > 0], key=lambda v: v.eta)
    yard, peak = sc.yard_inventory, sc.yard_inventory
    busy_hours, waits = 0.0, []

    for h in range(H):
        t = float(h)
        # departures
        at_berth = [v for v in at_berth if v.depart_time > t]
        # arrivals this hour
        while future and future[0].eta <= t + 1:
            waiting.append(future.pop(0))
        waiting.sort(key=key)
        # a congested yard slows discharge (cranes wait for yard space)
        rate = sc.crane_multiplier * (0.5 if yard >= sc.yard_slowdown_occupancy * sc.yard_capacity else 1.0)
        # assign free berths
        while waiting and len(at_berth) < sc.berths:
            v = waiting.pop(0)
            v.berth_time = max(t, v.eta)
            v.depart_time = v.berth_time + v.service_hours / max(rate, 0.1)
            waits.append(v.berth_time - v.eta)
            at_berth.append(v)
        # cargo flow this hour
        inflow = len(at_berth) * sc.discharge_units_per_hour * rate
        outflow = sc.gate_units_per_hour * sc.gate_multiplier
        yard = min(sc.yard_capacity, max(0.0, yard + inflow - outflow))
        peak = max(peak, yard)
        busy_hours += len(at_berth)

    # vessels still waiting at the end count their wait so far
    for v in waiting:
        waits.append(H - v.eta)
    served = len([v for v in vessels if v.berth_time is not None and v.at_berth_since is None])
    arrived = len([v for v in vessels if v.eta <= H])
    return RunResult(
        avg_wait_hours=float(sum(waits) / len(waits)) if waits else 0.0,
        max_wait_hours=float(max(waits)) if waits else 0.0,
        delayed_vessels=int(sum(w > sc.late_threshold_hours for w in waits)),
        vessels_served=served, vessels_arrived=arrived,
        berth_utilization=float(busy_hours / (sc.berths * H)) if sc.berths else 0.0,
        yard_peak_occupancy=float(peak / sc.yard_capacity),
        yard_end_occupancy=float(yard / sc.yard_capacity),
        queue_end=len(waiting),
    )


def simulate_candidate(sc: Scenario, candidate: dict, runs: int = 30, seed: int = 42,
                       forecast: list | None = None) -> dict:
    """Monte Carlo over `runs` seeds -> mean and 10/90 percentiles of each KPI."""
    rows = []
    for r in range(runs):
        rng = random.Random(seed + r)                 # same seeds for every candidate -> paired comparison
        vessels = build_vessels(rng, sc, forecast)
        sc_r, policy = apply_actions(sc, vessels, candidate.get("actions", []))
        rows.append(asdict(run_once(sc_r, vessels, policy)))
    keys = rows[0].keys()
    summary = {}
    for k in keys:
        vals = sorted(r[k] for r in rows)
        n = len(vals)
        summary[k] = {"mean": round(sum(vals) / n, 3),
                      "p10": round(vals[int(0.1 * (n - 1))], 3),
                      "p90": round(vals[int(0.9 * (n - 1))], 3)}
    return {"candidate_id": candidate["id"], "actions": candidate.get("actions", []),
            "runs": runs, "kpis": summary}
