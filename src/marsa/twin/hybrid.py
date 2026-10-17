"""Historical-call adapter for the normalized engine, with an explicit as-of boundary.

The finalized CSV is retrospective only. As-of experiments require a separately
supplied observation snapshot; they never read that CSV or the finalized Cargo file.
No realized stay duration or vessel length is used to estimate synthetic cargo.
"""
from __future__ import annotations

from datetime import datetime
from math import isfinite
from typing import Literal

import pandas as pd
from pydantic import Field, field_validator, model_validator

from marsa.twin.calls import CALLS_PATH, load_calls
from marsa.twin.coupled import (
    Contract,
    CoupledParameters,
    CoupledScenario,
    WorkloadVessel,
    simulate_coupled,
    utc,
)
from marsa.twin.yard import CARGO_PATH, load_cargo

REQUIRED_COLUMNS = {
    "call_id", "vessel_id", "mmsi", "imo", "length_m", "arrival", "waited",
    "wait_start", "wait_end", "berth_start", "berth_end", "wait_hours", "berth_hours",
}
TIME_COLUMNS = ["arrival", "wait_start", "wait_end", "berth_start", "berth_end"]


class WorkloadSettings(Contract):
    method: Literal["constant_per_call_v1"] = "constant_per_call_v1"
    full_import_units: float = Field(default=1000.0, gt=0)
    berthed_remaining_fraction: float = Field(default=0.5, gt=0, le=1)
    remaining_fraction_sensitivity: tuple[float, float, float] = (0.25, 0.5, 0.75)

    @field_validator("remaining_fraction_sensitivity")
    @classmethod
    def ordered_fractions(cls, values):
        if not 0 < values[0] <= values[1] <= values[2] <= 1:
            raise ValueError("remaining fractions must be ordered within (0,1]")
        return values


class AsOfObservation(Contract):
    """Caller assertion of an active call at t0, not reconstruction from final berth_end."""

    call_id: str = Field(min_length=1)
    vessel_id: str = Field(min_length=1)
    mmsi: int = Field(gt=0, strict=True)
    imo: str | None = None
    available_at: datetime
    state: Literal["waiting", "berthed"]
    arrival: datetime
    arrival_basis: Literal["matched_wait_start", "berth_start_fallback"]
    berth_start: datetime | None = None

    _utc = field_validator("available_at", "arrival")(utc)

    @field_validator("berth_start")
    @classmethod
    def utc_berth(cls, value):
        return None if value is None else utc(value)


class HybridRequest(Contract):
    timestamp_utc: datetime
    horizon_hours: int = Field(default=24, gt=0, le=720, strict=True)
    time_information_mode: Literal["retrospective", "as_of"]
    cohort_id: str = Field(min_length=1)
    resource_pool_id: str = Field(min_length=1)
    call_ids: list[str] | None = Field(default=None, max_length=1000)
    parameters: CoupledParameters
    workload: WorkloadSettings | None = None
    inventory_source: Literal["explicit", "cargo_snapshot"] = "explicit"
    queue_policy: Literal["fcfs", "shortest_estimated_workload_first"] = "fcfs"
    intervention_gate_units_per_hour: float | None = Field(default=None, ge=0)
    as_of_observations: list[AsOfObservation] | None = Field(default=None, max_length=1000)

    _utc = field_validator("timestamp_utc")(utc)

    @model_validator(mode="after")
    def temporal_contract(self):
        if self.timestamp_utc.minute or self.timestamp_utc.second or self.timestamp_utc.microsecond:
            raise ValueError("decision timestamp must be an exact UTC hour")
        if self.call_ids is not None and len(set(self.call_ids)) != len(self.call_ids):
            raise ValueError("requested call_ids must be unique")
        if self.time_information_mode == "as_of":
            if self.as_of_observations is None:
                raise ValueError("as_of requires an independently supplied active-call snapshot; final CSV is not an as-of feed")
            if self.inventory_source != "explicit":
                raise ValueError("as_of cannot use retrospectively constructed Cargo CSV inventory")
        elif self.as_of_observations is not None:
            raise ValueError("as_of_observations are only accepted in as_of mode")
        return self


class HybridSettings(Contract):
    enabled: bool = False
    workload: WorkloadSettings = Field(default_factory=WorkloadSettings)


def validate_calls(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a copy without altering source records; fail rather than silently drop."""
    if not set(frame.columns) >= REQUIRED_COLUMNS:
        raise ValueError(f"missing call columns: {sorted(REQUIRED_COLUMNS - set(frame.columns))}")
    calls = frame.copy(deep=True)
    for field in ["call_id", "vessel_id", "mmsi", "length_m", "waited", "wait_hours", "berth_hours"]:
        if calls[field].isna().any():
            raise ValueError(f"missing required call field: {field}")
    if calls.call_id.duplicated().any() or calls.call_id.astype(str).str.strip().eq("").any():
        raise ValueError("call_id must be nonempty and unique")
    if calls.vessel_id.astype(str).str.strip().eq("").any():
        raise ValueError("vessel_id must be nonempty")
    for field in ["length_m", "mmsi"]:
        values = pd.to_numeric(calls[field], errors="raise")
        if not values.map(lambda x: isfinite(x) and x > 0).all():
            raise ValueError(f"invalid {field}")
        if field == "mmsi" and not (values % 1 == 0).all():
            raise ValueError("mmsi must be an integer")
    if not pd.api.types.is_bool_dtype(calls.waited):
        raise ValueError("waited must be boolean")
    for column in TIME_COLUMNS:
        calls[column] = pd.to_datetime(calls[column], utc=True, errors="raise")
    if calls[["arrival", "berth_start", "berth_end"]].isna().any().any():
        raise ValueError("missing required call timestamp")
    if not ((calls.arrival <= calls.berth_start) & (calls.berth_start < calls.berth_end)).all():
        raise ValueError("call ordering requires arrival <= berth_start < berth_end")
    matched = calls[calls.waited]
    if (matched[["wait_start", "wait_end"]].isna().any().any()
            or not ((matched.wait_start == matched.arrival) & (matched.wait_start <= matched.wait_end)
                    & (matched.wait_end <= matched.berth_start)).all()):
        raise ValueError("matched waiting timestamps are inconsistent")
    if not calls.loc[~calls.waited, "arrival"].eq(calls.loc[~calls.waited, "berth_start"]).all():
        raise ValueError("unmatched arrivals must be explicitly identified berth-start fallbacks")
    for field, start, end in [("wait_hours", "arrival", "berth_start"), ("berth_hours", "berth_start", "berth_end")]:
        delta = (calls[end] - calls[start]).dt.total_seconds() / 3600
        if not (pd.to_numeric(calls[field], errors="raise") - delta).abs().le(1e-9).all():
            raise ValueError(f"inconsistent {field}")
    return calls


def _retrospective(request: HybridRequest, frame: pd.DataFrame | None):
    calls = validate_calls(load_calls() if frame is None else frame)
    ts = pd.Timestamp(request.timestamp_utc)
    end = ts + pd.Timedelta(hours=request.horizon_hours)
    if calls.empty or not calls.arrival.min() <= ts < calls.berth_end.max():
        raise ValueError("decision timestamp is outside call dataset coverage")
    source_start, source_end, source_rows = calls.berth_start.min(), calls.berth_end.max(), len(calls)
    if request.call_ids is not None:
        unknown = set(request.call_ids) - set(calls.call_id)
        if unknown:
            raise ValueError(f"unknown requested call_ids: {sorted(unknown)}")
        calls = calls[calls.call_id.isin(request.call_ids)]
    active = calls[(calls.berth_start <= ts) & (calls.berth_end > ts)]
    waiting = calls[calls.waited & (calls.arrival <= ts) & (calls.berth_start > ts)]
    future = calls[(calls.arrival > ts) & (calls.arrival <= end)]
    records = []
    for state, part in [("berthed", active), ("waiting", waiting), ("future", future)]:
        for r in part.sort_values(["arrival", "call_id"]).itertuples():
            records.append({"call_id": r.call_id, "vessel_id": r.vessel_id, "mmsi": int(r.mmsi),
                            "imo": None if pd.isna(r.imo) else str(r.imo), "arrival": r.arrival.to_pydatetime(),
                            "arrival_basis": "matched_wait_start" if r.waited else "berth_start_fallback",
                            "state": state, "berth_start": r.berth_start.to_pydatetime() if state == "berthed" else None,
                            "boundary_censoring_possible": bool(r.berth_start == source_start or r.berth_end == source_end)})
    # Detect overlaps against the cohort history, not only the selected initial slice.
    selected = {r["call_id"] for r in records}
    conflicts = {}
    for identity in ["vessel_id", "mmsi"]:
        for _, group in calls.groupby(identity):
            ordered = list(group.sort_values("berth_start").itertuples())
            for i, left in enumerate(ordered):
                for right in ordered[i + 1:]:
                    if right.berth_start >= left.berth_end:
                        break
                    if left.call_id in selected or right.call_id in selected:
                        key = tuple(sorted([left.call_id, right.call_id]))
                        conflict = conflicts.setdefault(key, {"vessel_id": str(left.vessel_id),
                                                              "call_ids": list(key), "identity_fields": []})
                        conflict["identity_fields"].append(identity)
    overlaps = [conflicts[key] for key in sorted(conflicts)]
    return records, overlaps, {
        "call_source": str(CALLS_PATH.name), "source_rows": source_rows, "cohort_source_rows": len(calls),
        "future_arrivals": "ACTUAL_FUTURE_ARRIVALS_HINDSIGHT",
        "coverage_end": str(source_end),
        "horizon_exceeds_source": bool(end > source_end),
        "availability": "Finalized AIS-derived stays; historical publication/availability times are unknown.",
    }


def _as_of(request: HybridRequest):
    observations = request.as_of_observations
    ids = [r.call_id for r in observations]
    if len(ids) != len(set(ids)):
        raise ValueError("as-of call_id must be unique")
    if request.call_ids is not None and set(request.call_ids) - set(ids):
        raise ValueError("requested call_ids are absent from supplied as-of snapshot")
    records = []
    for r in observations:
        if r.available_at > request.timestamp_utc or r.arrival > r.available_at:
            raise ValueError("as-of observation uses information unavailable by the decision timestamp")
        if r.state == "waiting" and (r.arrival_basis != "matched_wait_start" or r.berth_start is not None):
            raise ValueError("initial waiting needs waiting evidence and no known future berth start")
        if r.state == "berthed" and (r.berth_start is None or not r.arrival <= r.berth_start <= r.available_at):
            raise ValueError("as-of berth start must have been observed by available_at")
        if request.call_ids is None or r.call_id in request.call_ids:
            records.append(r.model_dump())
    records.sort(key=lambda r: (r["arrival"], r["call_id"]))
    overlaps = []
    for i, left in enumerate(records):
        for right in records[i + 1:]:
            if left["vessel_id"] == right["vessel_id"] or left["mmsi"] == right["mmsi"]:
                overlaps.append({"vessel_id": left["vessel_id"], "call_ids": [left["call_id"], right["call_id"]]})
    return records, overlaps, {
        "call_source": "CALLER_SUPPLIED_AS_OF_OBSERVATIONS", "source_rows": len(observations),
        "future_arrivals": "UNSPECIFIED_NO_ARRIVAL_FEED", "coverage_end": None,
        "horizon_exceeds_source": False,
        "availability": "Caller asserts active membership and available_at; no independent live AIS feed is present.",
    }


def build_hybrid(request: HybridRequest, *, calls: pd.DataFrame | None = None,
                 cargo: pd.DataFrame | None = None, defaults: WorkloadSettings | None = None) -> tuple[CoupledScenario, dict]:
    """Build once; paired interventions reuse the exact cohort and workloads."""
    retrospective = request.time_information_mode == "retrospective"
    records, overlaps, source = _retrospective(request, calls) if retrospective else _as_of(request)
    workload = request.workload or defaults or WorkloadSettings()
    parameters = request.parameters
    inventory_metadata = {"source": "EXPLICIT_SCENARIO_INPUT", "provenance": "SYNTHETIC_SCENARIO_ASSUMPTION"}
    if request.inventory_source == "cargo_snapshot":
        frame = load_cargo() if cargo is None else cargo.copy(deep=True)
        if "timestamp" in frame.columns:
            frame = frame.set_index(pd.to_datetime(frame["timestamp"], utc=True))
        if frame.index.duplicated().any() or request.timestamp_utc not in frame.index:
            raise ValueError("Cargo requires a unique exact-hour inventory snapshot")
        inventory = float(frame.loc[request.timestamp_utc, "containers_in_yard"])
        parameters = CoupledParameters.model_validate({**parameters.model_dump(), "initial_inventory": inventory})
        inventory_metadata = {"source": CARGO_PATH.name, "provenance": "RETROSPECTIVE_SYNTHETIC_CARGO_SNAPSHOT",
                              "limitation": "Finalized mixed-scope synthetic inventory, not an observed terminal stock."}
    vessels, estimates = [], []
    for r in records:
        fraction = workload.berthed_remaining_fraction if r["state"] == "berthed" else 1.0
        remaining = workload.full_import_units * fraction
        vessels.append(WorkloadVessel(
            simulation_id=r["call_id"], call_id=r["call_id"], arrival_timestamp=r["arrival"],
            initial_state=r["state"], initial_berth_start_timestamp=r["berth_start"],
            source_vessel_id=r["vessel_id"], source_mmsi=r["mmsi"], source_imo=r["imo"],
            arrival_basis=r["arrival_basis"],
            arrival_provenance="RETROSPECTIVE_AIS_DERIVED" if retrospective else "AS_OF_USER_ASSERTED",
            initial_import_workload=remaining, estimated_workload_for_priority=remaining,
            workload_method=workload.method, workload_provenance="SYNTHETIC_ESTIMATED",
        ))
        estimates.append({"call_id": r["call_id"], "method": workload.method,
                          "estimated_original_import_workload": workload.full_import_units,
                          "remaining_fraction_assumption": fraction, "initial_workload_at_t0_or_arrival": remaining,
                          "provenance": "SYNTHETIC_ESTIMATED", "units": "normalized_units"})
    scenario = CoupledScenario(timestamp_utc=request.timestamp_utc, horizon_hours=request.horizon_hours,
                               cohort_id=request.cohort_id, resource_pool_id=request.resource_pool_id,
                               parameters=parameters, vessels=vessels, queue_policy=request.queue_policy)
    counts = {state: sum(r["state"] == state for r in records) for state in ("berthed", "waiting", "future")}
    unmatched = sum(r["arrival_basis"] == "berth_start_fallback" for r in records)
    warnings = [
        "POLA-oriented geographic heuristic; broad cargo AIS types; container-terminal membership unverified.",
        "Unmatched waiting is unknown, not observed zero; waiting-only calls without a later qualifying stay are absent from the CSV.",
        "Workloads and remaining fractions are synthetic assumptions, not observed cargo or TEU.",
        "No historical Cargo inflows/outflows are replayed; initial inventory is included once.",
    ]
    if retrospective:
        warnings.append("Retrospective and hindsight-assisted: final berth_end selects active calls and final waiting linkage selects arrivals; not a future prediction.")
        warnings.append("Hourly gap-bridged stays may be incomplete or censored; no ping-level continuity verification is available.")
    else:
        warnings.append("As-of snapshot membership/availability is caller-asserted; future arrivals are unspecified, not forecast as zero.")
    if overlaps:
        warnings.append("Possible overlapping physical identities retained; review identity_overlaps before interpreting vessel counts.")
    if source["horizon_exceeds_source"]:
        warnings.append("Requested horizon extends past finalized source coverage; later arrivals are unknown.")
    return scenario, {
        "time_information_mode": request.time_information_mode, "hindsight_assisted": retrospective,
        "cohort_id": scenario.cohort_id, "resource_pool_id": scenario.resource_pool_id,
        "geographic_scope": scenario.geographic_scope, "source": source,
        "initial_counts": {"eligible_calls": len(records), **counts},
        "waiting_coverage": {"matched_calls": len(records) - unmatched, "unmatched_calls": unmatched,
                             "unmatched_call_ids": [r["call_id"] for r in records if r["arrival_basis"] == "berth_start_fallback"]},
        "initial_inventory": {**inventory_metadata, "value": parameters.initial_inventory, "units": parameters.units},
        "workload_estimation": {"settings": workload.model_dump(), "assignments": estimates,
                                "initial_total": sum(v.initial_import_workload for v in vessels),
                                "formula": "full_import_units * (berthed_remaining_fraction if initially berthed else 1)"},
        "identity_overlaps": overlaps, "data_quality_warnings": warnings,
        "call_quality": [{"call_id": r["call_id"], "arrival_basis": r["arrival_basis"],
                          "waiting_history": "MATCHED" if r["arrival_basis"] == "matched_wait_start" else "UNKNOWN",
                          "available_at": r["available_at"].isoformat() if "available_at" in r else None,
                          "boundary_censoring_possible": r.get("boundary_censoring_possible", "UNKNOWN_AS_OF")}
                         for r in records],
        "assumptions": ["Initial berthed remaining progress is unobserved; prior unloaded cargo is not added to the yard again.",
                        "Only remaining work enters the t0 conservation ledger; historical inventory already represents prior cargo.",
                        "No length, final stay duration, Poisson arrivals, physical cranes or annual TEU allocation are used.",
                        "Resource slots and rates are explicit synthetic settings; reject insufficient slots for initial occupancy."],
    }


def simulate_hybrid(request: HybridRequest, **inputs) -> dict:
    scenario, metadata = build_hybrid(request, **inputs)
    baseline = simulate_coupled(scenario)
    intervention = None
    if request.intervention_gate_units_per_hour is not None:
        parameters = CoupledParameters.model_validate({**scenario.parameters.model_dump(),
                                                       "gate_units_per_hour": request.intervention_gate_units_per_hour})
        changed = CoupledScenario.model_validate({**scenario.model_dump(), "parameters": parameters})
        intervention = simulate_coupled(changed)
    return {"mode": "coupled_normalized", "initialization": "hybrid_vessel_calls", **metadata,
            "baseline": baseline, "intervention": intervention, "provenance": "SIMULATED",
            "human_approval_required": True}
