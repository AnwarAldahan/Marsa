"""Opt-in deterministic experiments; no Strategy ranking or simulation changes.

Build the hybrid population once. Vary seven declared assumption factors OAT and
two named interactions; pair interventions with identical work within each case.
Diagnostics reconstruct accounting from the existing output ledger, not by
re-running the engine or using the intervention name.
"""
from __future__ import annotations

import hashlib
import json

from pydantic import Field, field_validator, model_validator

from marsa.twin.coupled import Contract, CoupledParameters, CoupledScenario, simulate_coupled
from marsa.twin.hybrid import HybridRequest, WorkloadSettings, build_hybrid


class SensitivityPlan(Contract):
    storage_factors: tuple[float, float, float] = (0.909091, 1.0, 1.090909)
    rate_workload_factors: tuple[float, float, float] = (0.5, 1.0, 1.5)
    occupancy_low: float = Field(default=0.5, ge=0, le=1)
    occupancy_high: float = Field(default=1.0, ge=0, le=1)
    remaining_low: float = Field(default=0.25, gt=0, le=1)
    remaining_high: float = Field(default=0.75, gt=0, le=1)
    modest_increase: float = Field(default=0.2, gt=0, le=0.2)
    large_gate_factor: float = Field(default=3.0, gt=1.2, le=10)
    include_interactions: bool = True

    @model_validator(mode="after")
    def ordered_bounds(self):
        if self.occupancy_low > self.occupancy_high or self.remaining_low > self.remaining_high:
            raise ValueError("low assumptions must not exceed high assumptions")
        return self

    @field_validator("storage_factors", "rate_workload_factors")
    @classmethod
    def ordered_factors(cls, values):
        if not 0 < values[0] < values[1] == 1.0 < values[2] <= 10:
            raise ValueError("factors must satisfy 0 < low < base=1 < high <= 10")
        return values


class SensitivitySettings(Contract):
    enabled: bool = False
    plan: SensitivityPlan = Field(default_factory=SensitivityPlan)


class SensitivityRequest(Contract):
    scenario: HybridRequest
    plan: SensitivityPlan | None = None


def _signature(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def diagnose_bottlenecks(result: dict) -> dict:
    """Binding over this finite horizon, not empirical causal identification.

    Gate: eligible stock exceeds service, with service saturated (zero included).
    Yard: requested unloading exceeds headroom, or full yard with unfinished work.
    Unloading: work exceeds the pool budget and the budget is fully consumed;
    zero productivity also binds with positive berth work, even at a full yard.
    Independent berth: a full pool has a queue and spare productivity/headroom.
    Full berths due to other binding resources are separately reported pressure.
    """
    p = result["scenario"]["parameters"]
    vessels = result["vessels"]
    initial_work = sum(v["initial_import_workload"] for v in vessels)
    tolerance = 1e-9 * max(1.0, initial_work + p["initial_inventory"])
    names = ["yard_capacity_blocking", "gate_clearance_limitation",
             "unloading_productivity_limitation", "berth_slot_limitation"]
    hours = dict.fromkeys(names, 0)
    queue_pressure = demand_limited = eligibility_blocked = gate_yard_link = 0
    productive_hours = busy_hours = 0
    cumulative_unloaded = cumulative_cleared = 0.0
    evidence = []
    flows = result["flows"]
    for index, flow in enumerate(flows):
        h = flow["start_hour"]
        point = result["timeline"][index]
        admitted_work = sum(v["initial_import_workload"] for v in vessels
                            if v["berth_start_hour"] is not None and v["berth_start_hour"] <= h)
        berth_work = max(0.0, admitted_work - cumulative_unloaded)
        waiting_work = sum(v["initial_import_workload"] for v in vessels
                           if v["arrival_hour"] <= h and (v["berth_start_hour"] is None or v["berth_start_hour"] > h))
        eligible = p["initial_inventory"] if p["initial_inventory_eligible_in_hours"] <= h else 0.0
        eligible += sum(f["unloaded"] for f in flows[:index]
                        if f["end_hour"] + p["release_delay_hours"] <= h)
        eligible = max(0.0, eligible - cumulative_cleared)
        headroom = p["yard_capacity"] - (flow["opening_inventory"] - flow["gate_cleared"])
        yard = flow["yard_blocked_unloading"] > tolerance or (berth_work > tolerance and headroom <= tolerance)
        gate = eligible > flow["gate_cleared"] + tolerance and flow["gate_cleared"] >= p["gate_units_per_hour"] - tolerance
        unloading = berth_work > p["unloading_units_per_hour"] + tolerance and flow["unloaded"] >= p["unloading_units_per_hour"] - tolerance
        unloading = unloading or (p["unloading_units_per_hour"] == 0 and berth_work + waiting_work > tolerance)
        pressure = point["waiting"] > 0 and point["at_berth"] >= p["berth_slots"]
        berth = pressure and not yard and not unloading and headroom > tolerance
        # Spare pool capacity is needed to attribute lost throughput to berth slots.
        berth = (berth and flow["unloaded"] < p["unloading_units_per_hour"] - tolerance) or (pressure and p["berth_slots"] == 0)
        flags = [yard, gate, unloading, berth]
        for name, active in zip(names, flags, strict=True):
            hours[name] += int(active)
        queue_pressure += int(pressure)
        gate_yard_link += int(gate and yard)
        eligibility_blocked += int(yard and not gate and eligible <= tolerance and flow["opening_inventory"] > tolerance)
        productive_hours += int(flow["unloaded"] > tolerance)
        busy_hours += int(berth_work > tolerance)
        demand_limited += int(not yard and not pressure and berth_work <= p["unloading_units_per_hour"] + tolerance)
        evidence.append({"hour": h, "berth_work_remaining_open": berth_work,
                         "waiting_work_open": waiting_work,
                         "eligible_yard_stock_open": eligible, "headroom_after_gate": headroom,
                         "active_constraints": [name for name, active in zip(names, flags, strict=True) if active],
                         "berth_queue_pressure": pressure})
        cumulative_unloaded += flow["unloaded"]
        cumulative_cleared += flow["gate_cleared"]
    active = [name for name in names if hours[name]]
    classification = "mixed" if len(active) > 1 else active[0] if active else "inconclusive" if queue_pressure else "insufficient_demand"
    return {"method": "ledger_binding_constraints_v1", "classification": classification,
            "active_constraints": active, "constraint_hours": hours,
            "berth_queue_pressure_hours": queue_pressure, "demand_limited_hours": demand_limited,
            "gate_and_yard_binding_hours": gate_yard_link,
            "yard_release_eligibility_blocking_hours": eligibility_blocked,
            "productive_hours": productive_hours, "berth_work_present_hours": busy_hours,
            "unloading_pool_utilization": result["kpis"]["cargo_unloaded"] / (p["unloading_units_per_hour"] * len(flows)) if p["unloading_units_per_hour"] else None,
            "hourly_evidence": evidence,
            "limitations": ["Constraints can bind at different hours; mixed does not imply simultaneous binding.",
                            "Unattributed queue pressure is inconclusive, never insufficient demand; budget/slot co-saturation may need finer time resolution.",
                            "Gate service can bind yard clearance without improving vessel outcomes when raised.",
                            "Berth-slot attribution requires spare shared productivity; full berths alone are not proof.",
                            "Hourly completion/berth reuse can cause slot limitations; not physical POLA capacity estimates."]}


def _benefit(base: dict, tried: dict) -> dict:
    """Queue/completion Pareto comparison; cargo-only progress stays separate."""
    keys = {"queue_exposure_vessel_hours": -1, "completed_vessels": 1}
    deltas = {key: tried[key] - base[key] for key in keys}
    improvements = worsening = 0
    for key, direction in keys.items():
        tolerance = 1e-9 * max(1.0, abs(base[key]), abs(tried[key]))
        signed = direction * deltas[key]
        improvements += signed > tolerance
        worsening += signed < -tolerance
    status = "tradeoff" if improvements and worsening else "beneficial" if improvements else "harmful" if worsening else "ineffective"
    cargo_delta = tried["cargo_unloaded"] - base["cargo_unloaded"]
    cargo_tolerance = 1e-9 * max(1.0, base["cargo_unloaded"], tried["cargo_unloaded"])
    return {"status": status, "vessel_metric_deltas": deltas,
            "cargo_unloaded_delta": cargo_delta,
            "cargo_progress_status": "improved" if cargo_delta > cargo_tolerance else "worse" if cargo_delta < -cargo_tolerance else "unchanged",
            "cargo_clearance_delta": tried["cargo_cleared"] - base["cargo_cleared"],
            "yard_end_inventory_delta": tried["yard_inventory_end"] - base["yard_inventory_end"],
            "definition": "Queue/completion Pareto benefit: queue exposure down or completed vessels up, neither worsens. Cargo-only progress and yard effects are separate; no weights or recommendation ranking."}


def summarize_robustness(contexts: list[dict], excluded: list[dict]) -> dict:
    summary = {}
    for intervention in ["gate_modest", "gate_large", "unloading_modest", "berth_plus_one"]:
        observations = []
        for context in contexts:
            row = next(r for r in context["comparisons"] if r["intervention"] == intervention)
            observations.append({"context": context["context"], "axis": context["axis"],
                                 "status": row["vs_baseline"]["status"]})
        statuses = [o["status"] for o in observations]
        if excluded or len(statuses) < 2 or "tradeoff" in statuses:
            classification = "inconclusive"
        elif all(s == "beneficial" for s in statuses):
            classification = "robust_across_tested_assumptions"
        elif "beneficial" in statuses:
            classification = "sensitive_to_specific_parameters"
        else:
            classification = "ineffective_under_tested_conditions"
        summary[intervention] = {
            "classification": classification, "tested_contexts": len(statuses),
            "status_counts": {s: statuses.count(s) for s in ["beneficial", "ineffective", "harmful", "tradeoff"]},
            "useful_to_ineffective": "beneficial" in statuses and "ineffective" in statuses,
            "contexts": observations,
            "interpretation": "Finite deterministic scenario coverage; not empirical confidence or a Strategy recommendation.",
        }
    return summary


def _context_scenario(base: CoupledScenario, workload: dict, knobs: dict) -> CoupledScenario:
    parameters = {**base.parameters.model_dump()}
    parameters.update({key: value for key, value in knobs.items() if key in parameters})
    # Capacity sensitivity holds the fill fraction fixed, not the absolute stock.
    if "occupancy_fraction" in knobs or "yard_capacity" in knobs:
        fraction = knobs.get("occupancy_fraction", base.parameters.initial_inventory / base.parameters.yard_capacity)
        parameters["initial_inventory"] = fraction * parameters["yard_capacity"]
    settings = WorkloadSettings.model_validate({**workload, **{key: value for key, value in knobs.items() if key in workload}})
    vessels = []
    for vessel in base.vessels:
        work = settings.full_import_units * (settings.berthed_remaining_fraction if vessel.initial_state == "berthed" else 1.0)
        vessels.append({**vessel.model_dump(), "initial_import_workload": work, "estimated_workload_for_priority": work})
    return CoupledScenario.model_validate({**base.model_dump(), "parameters": parameters, "vessels": vessels})


def analyze_sensitivity(request: HybridRequest, *, plan: SensitivityPlan | None = None, **inputs) -> dict:
    plan = plan or SensitivityPlan()
    base, metadata = build_hybrid(request, **inputs)
    p = base.parameters
    workload = metadata["workload_estimation"]["settings"]
    occupancy = p.initial_inventory / p.yard_capacity
    minimum_slots = sum(v.initial_state == "berthed" for v in base.vessels)
    factors = plan.rate_workload_factors
    ranges = {
        "occupancy_fraction": [min(plan.occupancy_low, occupancy), occupancy, max(plan.occupancy_high, occupancy)],
        "yard_capacity": [p.yard_capacity * f for f in plan.storage_factors],
        "full_import_units": [workload["full_import_units"] * f for f in factors],
        "berthed_remaining_fraction": [min(plan.remaining_low, workload["berthed_remaining_fraction"]), workload["berthed_remaining_fraction"], max(plan.remaining_high, workload["berthed_remaining_fraction"])],
        "gate_units_per_hour": [p.gate_units_per_hour * f for f in factors],
        "unloading_units_per_hour": [p.unloading_units_per_hour * f for f in factors],
        "berth_slots": [max(minimum_slots, p.berth_slots - 1), p.berth_slots, p.berth_slots + 1],
    }
    definitions = [("base", "base", {})]
    for axis, values in ranges.items():
        for label, value in [("low", values[0]), ("high", values[2])]:
            if value != values[1]:
                definitions.append((f"{axis}_{label}", axis, {axis: value}))
    if plan.include_interactions:
        definitions.extend([
            ("high_occupancy_low_gate", "interaction", {"occupancy_fraction": ranges["occupancy_fraction"][2], "gate_units_per_hour": ranges["gate_units_per_hour"][0]}),
            ("high_workload_low_productivity", "interaction", {"full_import_units": ranges["full_import_units"][2], "unloading_units_per_hour": ranges["unloading_units_per_hour"][0]}),
        ])
    contexts, excluded, duplicates = [], [], []
    seen = {}
    population = [{key: vessel.model_dump(mode="json")[key] for key in (
        "simulation_id", "call_id", "arrival_timestamp", "initial_state", "initial_berth_start_timestamp",
    )} for vessel in base.vessels]
    for name, axis, knobs in definitions:
        try:
            scenario = _context_scenario(base, workload, knobs)
        except ValueError as error:
            excluded.append({"context": name, "reason": str(error)})
            continue
        runtime_signature = _signature(scenario.model_dump(mode="json"))
        if runtime_signature in seen:
            duplicates.append({"context": name, "duplicate_of": seen[runtime_signature], "axis": axis})
            continue
        seen[runtime_signature] = name
        mutations = [
            ("baseline", {}),
            ("gate_modest", {"gate_units_per_hour": scenario.parameters.gate_units_per_hour * (1 + plan.modest_increase)}),
            ("gate_large", {"gate_units_per_hour": scenario.parameters.gate_units_per_hour * plan.large_gate_factor}),
            ("unloading_modest", {"unloading_units_per_hour": scenario.parameters.unloading_units_per_hour * (1 + plan.modest_increase)}),
            ("berth_plus_one", {"berth_slots": scenario.parameters.berth_slots + 1}),
        ]
        rows = []
        reference = None
        work_signature = _signature([v.model_dump(mode="json") for v in scenario.vessels])
        for intervention, change in mutations:
            parameters = CoupledParameters.model_validate({**scenario.parameters.model_dump(), **change})
            changed = CoupledScenario.model_validate({**scenario.model_dump(), "parameters": parameters})
            result = simulate_coupled(changed)
            diagnostics = diagnose_bottlenecks(result)
            row = {"intervention": intervention, "parameters": parameters.model_dump(),
                   "workload_signature": work_signature, "kpis": result["kpis"], "bottleneck": diagnostics}
            if reference is None:
                reference = row
            else:
                row["vs_baseline"] = _benefit(reference["kpis"], row["kpis"])
                row["bottleneck_changed"] = reference["bottleneck"]["active_constraints"] != diagnostics["active_constraints"]
            rows.append(row)
        contexts.append({"context": name, "axis": axis, "knobs": knobs,
                         "runtime_signature": runtime_signature,
                         "workload_signature": work_signature, "comparisons": rows})
    return {"mode": "coupled_normalized", "analysis": "sensitivity_bottleneck_v1", "metadata": metadata,
            "population": population, "population_signature": _signature(population), "units": "normalized_units",
            "plan": plan.model_dump(), "ranges_low_base_high": ranges, "contexts": contexts,
            "excluded_contexts": excluded, "run_count": sum(len(c["comparisons"]) for c in contexts),
            "skipped_duplicate_contexts": duplicates,
            "robustness": summarize_robustness(contexts, excluded),
            "range_provenance": "EXPLORATORY_SYNTHETIC_ASSUMPTIONS; storage ratios preserve the approved relative anchors, not physical capacity.",
            "capacity_policy": "Hold initial occupancy fraction fixed when storage capacity varies; initial stock is therefore derived, not the unchanged Cargo snapshot.",
            "pairing": "Same call IDs, arrival times, initial states and workload assignments within each context; no RNG; request gate comparison setting is superseded by this explicit plan.",
            "strategy_integration": False, "provenance": "SIMULATED", "human_approval_required": True}
