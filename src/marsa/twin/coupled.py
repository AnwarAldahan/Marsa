"""Opt-in, import-only normalized vessel/yard simulation; no AIS or Cargo file reads.

Intervals are [t,t+1). Arrivals are on the hourly grid. Gate service uses eligible
opening inventory; unloading follows, sharing one synthetic productivity pool.
Completions release berths at interval end; no mid-hour berth reuse. At H, arrivals
join the queue but receive no berth or service. See docs/COUPLED_TWIN.md.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

UNIT = "normalized_units"


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("timestamps must be timezone-aware UTC")
    return value


class CoupledParameters(Contract):
    """All numeric inputs are synthetic scenario assumptions, never physical POLA values."""

    yard_capacity: float = Field(default=10000.0, gt=0)
    initial_inventory: float = Field(ge=0)
    berth_slots: int = Field(ge=0, strict=True)
    unloading_units_per_hour: float = Field(ge=0)
    gate_units_per_hour: float = Field(ge=0)
    # Cargo released at interval end t becomes eligible at t + delay.
    release_delay_hours: int = Field(default=0, ge=0, strict=True)
    initial_inventory_eligible_in_hours: int = Field(default=0, ge=0, strict=True)
    units: Literal["normalized_units"] = UNIT
    provenance: Literal["SYNTHETIC_SCENARIO_ASSUMPTION"] = "SYNTHETIC_SCENARIO_ASSUMPTION"

    @model_validator(mode="after")
    def within_capacity(self):
        if self.initial_inventory > self.yard_capacity:
            raise ValueError("initial_inventory exceeds yard_capacity")
        return self


class WorkloadVessel(Contract):
    simulation_id: str = Field(min_length=1)
    call_id: str | None = Field(default=None, min_length=1)
    arrival_timestamp: datetime
    initial_import_workload: float = Field(gt=0)
    estimated_workload_for_priority: float | None = Field(default=None, gt=0)
    arrival_basis: Literal["synthetic_scenario", "user_supplied", "historical_replay", "matched_wait_start", "berth_start_fallback"] = "synthetic_scenario"
    arrival_provenance: Literal["SYNTHETIC", "USER_INPUT", "RETROSPECTIVE_AIS_DERIVED", "AS_OF_USER_ASSERTED"] = "SYNTHETIC"
    workload_units: Literal["normalized_units"] = UNIT
    workload_method: Literal["explicit_synthetic", "constant_per_call_v1"] = "explicit_synthetic"
    workload_provenance: Literal["SYNTHETIC", "SYNTHETIC_ESTIMATED"] = "SYNTHETIC"
    initial_state: Literal["future", "waiting", "berthed"] = "future"
    initial_berth_start_timestamp: datetime | None = None
    source_vessel_id: str | None = None
    source_mmsi: int | None = None
    source_imo: str | None = None

    _utc = field_validator("arrival_timestamp")(utc)

    @field_validator("initial_berth_start_timestamp")
    @classmethod
    def berth_utc(cls, value):
        return None if value is None else utc(value)


class ScenarioInputs(Contract):
    mode: Literal["coupled_normalized"] = "coupled_normalized"
    timestamp_utc: datetime
    horizon_hours: int = Field(default=24, gt=0, le=720, strict=True)
    cohort_id: str = Field(min_length=1)
    resource_pool_id: str = Field(min_length=1)
    geographic_scope: Literal["ABSTRACT_POLA_INSPIRED_UNVERIFIED_MEMBERSHIP"] = "ABSTRACT_POLA_INSPIRED_UNVERIFIED_MEMBERSHIP"
    queue_policy: Literal["fcfs", "shortest_estimated_workload_first"] = "fcfs"
    vessels: list[WorkloadVessel] = Field(max_length=1000)

    _utc = field_validator("timestamp_utc")(utc)

    @model_validator(mode="after")
    def consistent_population(self):
        ids = [v.simulation_id for v in self.vessels]
        calls = [v.call_id for v in self.vessels if v.call_id is not None]
        if len(set(ids)) != len(ids) or len(set(calls)) != len(calls):
            raise ValueError("simulation_id and supplied call_id must be unique")
        for v in self.vessels:
            hours = (v.arrival_timestamp - self.timestamp_utc).total_seconds() / 3600
            if not hours.is_integer() or (hours < 0 and v.initial_state == "future"):
                raise ValueError("arrivals must be hourly; pre-t0 arrivals require an initial waiting/berthed state")
            if v.initial_state != "future" and hours > 0:
                raise ValueError("initial waiting/berthed vessels must have arrived by t0")
            berth = v.initial_berth_start_timestamp
            if v.initial_state == "berthed":
                if berth is None or not v.arrival_timestamp <= berth <= self.timestamp_utc:
                    raise ValueError("initial berthed state requires arrival <= berth start <= t0")
                if not ((berth - self.timestamp_utc).total_seconds() / 3600).is_integer():
                    raise ValueError("initial berth start must be on the hourly grid")
            elif berth is not None:
                raise ValueError("only initial berthed vessels can supply a berth start")
            if self.queue_policy == "shortest_estimated_workload_first" and v.estimated_workload_for_priority is None:
                raise ValueError("shortest policy requires an explicit estimated workload for every vessel")
        return self


class CoupledScenario(ScenarioInputs):
    parameters: CoupledParameters

    @model_validator(mode="after")
    def finite_totals(self):
        if sum(v.initial_state == "berthed" for v in self.vessels) > self.parameters.berth_slots:
            raise ValueError("berth_slots cannot be less than initial berth occupancy")
        total = sum(v.initial_import_workload for v in self.vessels) + self.parameters.initial_inventory
        if not all(isfinite(value) for value in (
            total, self.parameters.unloading_units_per_hour * self.horizon_hours,
            self.parameters.gate_units_per_hour * self.horizon_hours,
        )):
            raise ValueError("scenario totals and horizon resource budgets must be finite")
        try:
            self.timestamp_utc + timedelta(hours=self.horizon_hours)
        except OverflowError as error:
            raise ValueError("horizon exceeds supported timestamp range") from error
        return self


class CoupledSettings(Contract):
    enabled: bool = False
    mode: Literal["coupled_normalized"] = "coupled_normalized"
    parameters: CoupledParameters
    storage_sensitivity: tuple[float, float, float] = (9090.91, 10000.0, 10909.09)
    reference_note: str = "Physical POLA references are not runtime inputs or unit conversions."

    @field_validator("storage_sensitivity")
    @classmethod
    def ordered_storage(cls, values):
        if not 0 < values[0] <= values[1] <= values[2]:
            raise ValueError("storage sensitivity must be positive and ordered")
        return values


def _share(work: list[float], budget: float) -> list[float]:
    """Deterministic max-min fair allocation, redistributing unused shares."""
    result = [0.0] * len(work)
    active = list(range(len(work)))
    while active and budget > 0:
        share = budget / len(active)
        capped = [i for i in active if work[i] <= share]
        if not capped:
            for i in active:
                result[i] = share
            break
        for i in capped:
            result[i] = work[i]
            budget -= work[i]
        active = [i for i in active if i not in capped]
    return result


def simulate_coupled(scenario: CoupledScenario) -> dict:
    """Run a standalone explicit scenario. Never changes inputs or strategy rankings."""
    p, H = scenario.parameters, scenario.horizon_hours
    rows = []
    for v in scenario.vessels:
        berth = v.initial_berth_start_timestamp
        rows.append({**v.model_dump(mode="json"), "state": v.initial_state,
                     "remaining_import_workload": v.initial_import_workload,
                     "berth_start_timestamp": None if berth is None else berth.isoformat(), "completion_timestamp": None,
                     "berth_start_hour": None if berth is None else (berth - scenario.timestamp_utc).total_seconds() / 3600,
                     "completion_hour": None,
                     "arrival_hour": (v.arrival_timestamp - scenario.timestamp_utc).total_seconds() / 3600})
    # FIFO cargo lots: [eligibility hour, quantity]. Eligibility never removes cargo.
    lots = [[p.initial_inventory_eligible_in_hours, p.initial_inventory]] if p.initial_inventory else []
    inventory = p.initial_inventory
    unloaded_total = cleared_total = queue_exposure = berth_hours = 0.0
    blocked_units = blocked_hours = blocked_vessel_hours = 0.0
    timeline, flows = [], []
    initial_work = sum(v.initial_import_workload for v in scenario.vessels)

    def stamp(hour):
        return (scenario.timestamp_utc + timedelta(hours=hour)).isoformat()

    def priority(v):
        order = (v["arrival_hour"], v["simulation_id"])
        return order if scenario.queue_policy == "fcfs" else (v["estimated_workload_for_priority"], *order)

    for h in range(H + 1):
        for v in rows:
            if v["state"] == "future" and v["arrival_hour"] <= h:
                v["state"] = "waiting"
        berthed = [v for v in rows if v["state"] == "berthed"]
        if h < H:
            waiting = sorted((v for v in rows if v["state"] == "waiting"), key=priority)
            for v in waiting[:p.berth_slots - len(berthed)]:
                v.update(state="berthed", berth_start_hour=h, berth_start_timestamp=stamp(h))
                berthed.append(v)
        waiting_count = sum(v["state"] == "waiting" for v in rows)
        timeline.append({"hour": h, "timestamp_utc": stamp(h), "waiting": waiting_count,
                         "at_berth": len(berthed), "yard_inventory": inventory,
                         "yard_occupancy_pct": inventory / p.yard_capacity * 100,
                         "completed": sum(v["state"] == "completed" for v in rows)})
        if h == H:
            break
        opening = inventory
        queue_exposure += waiting_count
        berth_hours += len(berthed)
        gate_budget, cleared = p.gate_units_per_hour, 0.0
        for lot in lots:
            if lot[0] <= h:
                amount = min(lot[1], gate_budget)
                lot[1] -= amount
                gate_budget -= amount
                cleared += amount
        lots = [lot for lot in lots if lot[1] > 0]
        inventory -= cleared
        potential = _share([v["remaining_import_workload"] for v in berthed], p.unloading_units_per_hour)
        accepted = _share(potential, max(0.0, p.yard_capacity - inventory))
        blocked = sum(potential) - sum(accepted)
        blocked_count = sum(a < b for a, b in zip(accepted, potential, strict=True))
        blocked_units += blocked
        blocked_hours += bool(blocked_count)
        blocked_vessel_hours += blocked_count
        for v, amount in zip(berthed, accepted, strict=True):
            v["remaining_import_workload"] -= amount
            if v["remaining_import_workload"] == 0:
                v.update(state="completed", completion_hour=h + 1, completion_timestamp=stamp(h + 1))
        unloaded = sum(accepted)
        inventory += unloaded
        if unloaded:
            lots.append([h + 1 + p.release_delay_hours, unloaded])
        unloaded_total += unloaded
        cleared_total += cleared
        remaining = sum(v["remaining_import_workload"] for v in rows)
        vessel_residual = initial_work - remaining - unloaded_total
        yard_residual = p.initial_inventory + unloaded_total - cleared_total - inventory
        tolerance = 1e-9 * max(1.0, initial_work + p.initial_inventory)
        if (abs(vessel_residual) > tolerance or abs(yard_residual) > tolerance
                or inventory < -tolerance or inventory > p.yard_capacity + tolerance
                or any(v["remaining_import_workload"] < -tolerance for v in rows)
                or abs(sum(lot[1] for lot in lots) - inventory) > tolerance):
            raise ArithmeticError("coupled conservation or capacity invariant failed")
        flows.append({"start_hour": h, "end_hour": h + 1, "opening_inventory": opening,
                      "gate_cleared": cleared, "unloaded": unloaded, "closing_inventory": inventory,
                      "yard_blocked_unloading": blocked, "yard_blocked_vessels": blocked_count,
                      "vessel_conservation_residual": vessel_residual,
                      "yard_conservation_residual": yard_residual})

    admitted_waits = [v["berth_start_hour"] - max(0.0, v["arrival_hour"]) for v in rows
                      if v["berth_start_hour"] is not None and v["initial_state"] != "berthed"]
    unfinished_waits = [{"simulation_id": v["simulation_id"], "wait_hours": H - max(0.0, v["arrival_hour"])}
                        for v in rows if v["state"] == "waiting"]
    remaining = sum(v["remaining_import_workload"] for v in rows)
    kpis = {
        "waiting_vessels": timeline[-1]["waiting"],
        "queue_exposure_vessel_hours": queue_exposure,
        "avg_completed_queue_wait_hours": sum(admitted_waits) / len(admitted_waits) if admitted_waits else None,
        "completed_queue_wait_count": len(admitted_waits),
        "unfinished_queue_waits": unfinished_waits,
        "occupied_berth_hours": berth_hours,
        "berth_utilization": berth_hours / (p.berth_slots * H) if p.berth_slots else None,
        "yard_inventory_end": inventory,
        "yard_peak_occupancy_pct": max(t["yard_occupancy_pct"] for t in timeline),
        "yard_end_occupancy_pct": inventory / p.yard_capacity * 100,
        "cargo_unloaded": unloaded_total,
        "cargo_remaining_aboard": remaining,
        "cargo_cleared": cleared_total,
        "yard_blocked_unloading": blocked_units,
        "yard_blocked_hours": blocked_hours,
        "blocked_service_vessel_hours": blocked_vessel_hours,
        "vessels_berthed": len(admitted_waits),
        "completed_vessels": sum(v["state"] == "completed" for v in rows),
        "unfinished_vessels": sum(v["state"] in {"waiting", "berthed"} for v in rows),
        "future_vessels": sum(v["state"] == "future" for v in rows),
        "vessel_conservation_residual": initial_work - remaining - unloaded_total,
        "yard_conservation_residual": p.initial_inventory + unloaded_total - cleared_total - inventory,
        "system_conservation_residual": initial_work + p.initial_inventory - remaining - inventory - cleared_total,
    }
    return {"mode": scenario.mode, "scenario": scenario.model_dump(mode="json"),
            "provenance": "SIMULATED", "units": UNIT, "kpis": kpis, "vessels": rows,
            "timeline": timeline, "flows": flows,
            "yard_lots": [{"eligible_hour": lot[0], "quantity": lot[1]} for lot in lots],
            "limitations": ["Abstract aggregated yard; container-terminal membership unverified.",
                            "Synthetic import workloads; no observed TEU or physical crane allocation.",
                            "Hourly arrivals and end-of-hour completion; no mid-hour berth reuse.",
                            "No operational predictive validation; source availability is the adapter/caller's responsibility."],
            "human_approval_required": True}
