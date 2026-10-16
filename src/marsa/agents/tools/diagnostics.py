"""Read-only, exact-hour Strategy diagnostics over the existing domain sources.

Endpoint changes use t-6h and t; flow sums use the six bins (t-6h, t].
Missing hours/values are never interpolated. These are descriptions of finalized
historical source data, not independently verified POLA operational measurements.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from marsa.common.paths import resolve_project_path
from marsa.data.load import parse_exact_utc_hour

ToolName = Literal["cargo_balance", "cargo_delay_profile", "maritime_history"]
Direction = Literal["rising", "falling", "stable", "mixed", "unknown"]


class DiagnosticArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    window_hours: int = Field(strict=True, ge=6, le=6)


class DiagnosticSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    identifier: str
    geographic_scope: Literal["REGIONAL_LA_LB_UNVERIFIED_MEMBERSHIP", "MIXED_SCOPE_SYNTHETIC"]
    provenance: list[str]
    limitations: list[str]


class DiagnosticCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    expected_hours: int = 7
    observed_hours: int = Field(ge=0, le=7)
    missing_timestamps: list[str]
    missing_values: dict[str, list[str]]


class DiagnosticResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    evidence_id: str
    tool_name: ToolName
    timestamp_utc: str
    window_hours: int = 6
    status: Literal["complete", "incomplete", "error"]
    source: DiagnosticSource
    coverage: DiagnosticCoverage
    metrics: dict[str, float | None]
    units: dict[str, str]
    source_fields: list[str]
    current_evidence: dict[str, float | None]
    direction: Direction = "unknown"
    limitations: list[str]


TOOL_FIELDS = {
    "cargo_balance": ["containers_in_yard", "container_arrivals", "container_departures",
                      "yard_occupancy_percent"],
    "cargo_delay_profile": ["truck_waiting_time_minutes", "truck_arrivals",
                            "truck_departures", "gate_throughput"],
    "maritime_history": ["waiting_vessel_count", "vessel_count"],
}
TOOL_DESCRIPTIONS = {
    "cargo_balance": "Exact six-hour synthetic inventory/occupancy changes and net cargo flow.",
    "cargo_delay_profile": "Exact six-hour synthetic truck-wait change and truck/gate flow totals.",
    "maritime_history": "Exact six-hour regional waiting-count/ratio change and vessel-count context.",
}


def _direction(changes: list[float | None]) -> Direction:
    if any(value is None for value in changes):
        return "unknown"
    signs = {0 if abs(value) < 1e-9 else (1 if value > 0 else -1) for value in changes}
    if signs == {0}:
        return "stable"
    if signs <= {0, 1}:
        return "rising"
    if signs <= {0, -1}:
        return "falling"
    return "mixed"


class DiagnosticTools:
    """Paths are application-owned; a planner can select only a tool and window.

    Alternate paths support isolated tests and explicitly injected domain sources.
    They are never accepted from Gemini or the public HTTP request.
    """

    def __init__(self, *, cargo_path: str | Path = "data/agent2_cargo_operations_2025.csv",
                 maritime_path: str | Path = "data/port_operations_agent_dataset.csv") -> None:
        self.cargo_path = resolve_project_path(cargo_path)
        self.maritime_path = resolve_project_path(maritime_path)

    def source(self, tool: ToolName) -> DiagnosticSource:
        maritime = tool == "maritime_history"
        return DiagnosticSource(
            identifier=str(self.maritime_path if maritime else self.cargo_path),
            geographic_scope=("REGIONAL_LA_LB_UNVERIFIED_MEMBERSHIP" if maritime
                              else "MIXED_SCOPE_SYNTHETIC"),
            provenance=["DERIVED"] if maritime else ["SYNTHETIC", "DERIVED"],
            limitations=([
                "Maritime traffic is documented as LA/LB regional; POLA-only membership is unverified.",
                "Counts cannot establish individual vessel destinations or operational causes.",
            ] if maritime else [
                "Cargo is finalized synthetic/calibrated synthetic state, not observed terminal operations.",
                "Construction combines POLA calibration and regional AIS influence; not verified POLA-only.",
                "Reading a historical prefix does not remove retrospective source-construction limitations.",
            ]),
        )

    def execute(self, tool: ToolName, timestamp, arguments: DiagnosticArguments) -> DiagnosticResult:
        if tool not in TOOL_FIELDS:
            raise ValueError("Unknown diagnostic tool")
        args = DiagnosticArguments.model_validate(arguments)
        ts = parse_exact_utc_hour(timestamp)
        fields = TOOL_FIELDS[tool]
        source = self.source(tool)
        key = "hour_key" if tool == "maritime_history" else "timestamp"
        path = self.maritime_path if tool == "maritime_history" else self.cargo_path
        expected = pd.date_range(ts - pd.Timedelta(hours=args.window_hours), ts, freq="h")
        try:
            # Read only the declared columns, never labels or other source features.
            # CSV scanning is unavoidable; numeric analysis/type conversion is prefix-only.
            frame = pd.read_csv(path, usecols=[key, *fields], dtype="string")
            times = pd.to_datetime(frame[key], utc=True, errors="coerce", format="mixed")
            window = frame.loc[times.between(expected[0], ts)].copy()
            window.index = pd.DatetimeIndex(times.loc[window.index])
            if window.index.duplicated().any() or not window.index.isin(expected).all():
                raise ValueError("Diagnostic source has duplicate or misaligned window timestamps")
            observed = window.index
            window = window.reindex(expected)
            for field in fields:
                values = pd.to_numeric(window[field], errors="coerce").astype(float)
                window[field] = values.where(values.ge(0) & values.lt(float("inf")))
            if tool == "maritime_history":
                for field in fields:
                    window[field] = window[field].where(window[field].mod(1).eq(0))
                invalid = window["waiting_vessel_count"].gt(window["vessel_count"])
                window.loc[invalid, "waiting_vessel_count"] = float("nan")
            missing = {
                field: [value.isoformat() for value in expected[window[field].isna()]]
                for field in fields if window[field].isna().any()
            }
            coverage = DiagnosticCoverage(
                observed_hours=len(observed),
                missing_timestamps=[value.isoformat() for value in expected.difference(observed)],
                missing_values=missing,
            )
            metrics, units, current, changes = self._calculate(tool, window)
            complete = not coverage.missing_timestamps and not missing
            if any(value is None for value in metrics.values()):
                complete = False
            limitations = []
            if not complete:
                limitations.append("Incomplete exact history or required values; no interpolation or substitute rows used.")
            if tool == "cargo_balance":
                limitations.append("Inventory is normalized modeling units; flow sums cover (t-6h,t], endpoint changes t-6h to t.")
            if tool == "cargo_delay_profile":
                limitations.append("Truck/gate totals are synthetic indicators, not measured queue length or causal capacity evidence.")
            return DiagnosticResult(
                evidence_id=tool, tool_name=tool, timestamp_utc=ts.isoformat(), source=source,
                status="complete" if complete else "incomplete", coverage=coverage,
                metrics=metrics, units=units, current_evidence=current,
                source_fields=fields, direction=_direction(changes) if complete else "unknown",
                limitations=limitations,
            )
        except (OSError, ValueError, TypeError, KeyError):
            # Do not expose raw exceptions, paths from parser errors, or source contents.
            return self.error(tool, ts, "Diagnostic source unavailable or invalid.")

    def error(self, tool: ToolName, timestamp, reason: str) -> DiagnosticResult:
        ts = parse_exact_utc_hour(timestamp)
        return DiagnosticResult(
            evidence_id=tool, tool_name=tool, timestamp_utc=ts.isoformat(), status="error",
            source=self.source(tool), source_fields=TOOL_FIELDS[tool],
            coverage=DiagnosticCoverage(observed_hours=0, missing_timestamps=[
                t.isoformat() for t in pd.date_range(ts - pd.Timedelta(hours=6), ts, freq="h")
            ], missing_values={}), metrics={}, units={}, current_evidence={}, limitations=[reason],
        )

    @staticmethod
    def _calculate(tool: ToolName, window: pd.DataFrame):
        def value(field, index=-1):
            v = window[field].iloc[index]
            return None if pd.isna(v) else float(v)

        def change(field):
            start, end = value(field, 0), value(field)
            return None if start is None or end is None else end - start

        def total(field):
            values = window[field].iloc[1:]
            return float(values.sum()) if values.notna().all() else None

        if tool == "cargo_balance":
            arrivals, departures = total("container_arrivals"), total("container_departures")
            net = None if arrivals is None or departures is None else arrivals - departures
            metrics = {"inventory_change": change("containers_in_yard"), "net_cargo_flow": net,
                       "yard_occupancy_change": change("yard_occupancy_percent")}
            units = {"inventory_change": "normalized container-equivalent units",
                     "net_cargo_flow": "normalized container-equivalent units",
                     "yard_occupancy_change": "percentage points"}
            current = {field: value(field) for field in TOOL_FIELDS[tool]}
            return metrics, units, current, [metrics["inventory_change"], net]
        if tool == "cargo_delay_profile":
            metrics = {"truck_waiting_time_change": change("truck_waiting_time_minutes"),
                       "truck_arrivals": total("truck_arrivals"),
                       "truck_departures": total("truck_departures"),
                       "gate_throughput": total("gate_throughput")}
            units = {"truck_waiting_time_change": "minutes", "truck_arrivals": "trucks over (t-6h,t]",
                     "truck_departures": "trucks over (t-6h,t]",
                     "gate_throughput": "source gate-throughput units over (t-6h,t]"}
            current = {"truck_waiting_time_minutes": value("truck_waiting_time_minutes"),
                       "gate_throughput_last_1h": value("gate_throughput")}
            return metrics, units, current, [metrics["truck_waiting_time_change"]]

        def ratio(index):
            waiting, vessels = value("waiting_vessel_count", index), value("vessel_count", index)
            if waiting is None or vessels is None or waiting > vessels:
                return None
            return waiting / vessels if vessels else 0.0

        first, last = ratio(0), ratio(-1)
        metrics = {"waiting_vessel_count_change": change("waiting_vessel_count"),
                   "waiting_ratio_change": None if first is None or last is None else last - first,
                   "vessel_count_start": value("vessel_count", 0),
                   "vessel_count_end": value("vessel_count"),
                   "waiting_vessels_start": value("waiting_vessel_count", 0),
                   "waiting_vessels_end": value("waiting_vessel_count"),
                   "waiting_ratio_start": first, "waiting_ratio_end": last}
        units = {field: "ratio" if "ratio" in field else "vessels" for field in metrics}
        current = {"vessels": value("vessel_count"), "waiting_vessels": value("waiting_vessel_count")}
        return metrics, units, current, [metrics["waiting_vessel_count_change"], metrics["waiting_ratio_change"]]
