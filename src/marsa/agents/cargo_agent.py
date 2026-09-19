"""Evidence-first Cargo investigation over frozen synthetic 2025 state."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from marsa.agents.events_weather_agent import ReasoningMode
from marsa.agents.llm import StructuredLLMProvider
from marsa.agents.tools.cargo_status import SOURCE_FIELDS, CargoEvidence, CargoStatusSource
from marsa.provenance.types import DataProvenance


class CargoStatus(StrEnum):
    NORMAL = "NORMAL"
    MODERATE = "MODERATE"
    ELEVATED = "ELEVATED"
    CRITICAL = "CRITICAL"


class CargoForecastContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    horizon_hours: int = Field(ge=1)
    risk_level: str


class CargoInvestigationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp_utc: datetime
    investigation_reason: str | None = None
    forecast_context: CargoForecastContext | None = None

    @field_validator("timestamp_utc")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp_utc must be timezone-aware")
        return value.astimezone(UTC)


class CargoFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    finding: str
    evidence_fields: tuple[str, ...]
    provenance: tuple[DataProvenance, ...]


class CargoEvidenceProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    field: str
    provenance: DataProvenance
    source_field: str | None = None
    semantic_limit: str | None = None


class CargoResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    agent: Literal["cargo"] = "cargo"
    timestamp_utc: datetime
    reasoning_mode: ReasoningMode
    cargo_status: CargoStatus
    evidence: CargoEvidence
    findings: tuple[CargoFinding, ...]
    possible_operational_contribution: str
    limitations: tuple[str, ...]
    evidence_provenance: tuple[CargoEvidenceProvenance, ...]
    forecast_context: CargoForecastContext | None = None


EvidenceField = Literal[
    "cargo_status",
    "containers_in_yard",
    "yard_capacity",
    "yard_occupancy_percent",
    "import_teu",
    "export_teu",
    "container_arrivals_last_1h",
    "container_departures_last_1h",
    "average_dwell_time_hours",
    "truck_waiting_time_minutes",
    "gate_throughput_last_1h",
    "cargo_flow_ratio",
]


class CargoNarrative(BaseModel):
    """Gemini may supply only a cited finding and constrained contribution text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    finding: str
    evidence_fields: tuple[EvidenceField, ...] = Field(min_length=1)
    possible_operational_contribution: str


CONTRIBUTION = (
    "Current synthetic Cargo evidence may provide context for assessing cargo-side "
    "operational pressure, but does not establish causality or independently predict "
    "future congestion."
)

_UNSAFE = re.compile(
    r"\b(?:observed|measured|official|actual|real(?:-world)?|live|sensor|"
    r"caus\w*|led\s+to|resulted\s+in|due\s+to|responsible\s+for|"
    r"predict\w*|forecast\w*|future|will\b|likely\s+to|probability|"
    r"recommend\w*|should|must|strategy|intervention|reroute|divert|deploy|"
    r"open\s+(?:more\s+)?gates?|incident\w*|bottleneck\w*|delay\w*|"
    r"berth\w*|crane\w*|capacity\s+utilization|efficien\w*|"
    r"vessels?\b|ships?\b|maritime|AIS|"
    r"congest\w*|high\s+risk|low\s+risk)\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"(?<![\w.])-?(?:\d+(?:\.\d+)?|\.\d+)\s*%?")
_UNIT_AFTER_NUMBER = re.compile(
    r"^\s*(?P<unit>hours?|hrs?|minutes?|mins?|TEU|transactions?|"
    r"container[- ]equivalent\s+units?|units?)\b",
    re.IGNORECASE,
)
_UNIT_FIELDS = {
    "hour": {"average_dwell_time_hours"},
    "hr": {"average_dwell_time_hours"},
    "minute": {"truck_waiting_time_minutes"},
    "min": {"truck_waiting_time_minutes"},
    "teu": {"import_teu", "export_teu"},
    "transaction": {"gate_throughput_last_1h"},
    "unit": {"containers_in_yard", "yard_capacity", "container_arrivals_last_1h",
             "container_departures_last_1h"},
}
_WORD_NUMBER = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"hundred|thousand|million)\b",
    re.IGNORECASE,
)
_STATUS = re.compile(r"\b(?:NORMAL|MODERATE|ELEVATED|CRITICAL)\b", re.IGNORECASE)
_CONCEPT_FIELDS = (
    (re.compile(r"\boccupan\w*\b", re.I), "yard_occupancy_percent"),
    (re.compile(r"\b(?:yard\s+capacity|modeling\s+capacity)\b", re.I), "yard_capacity"),
    (re.compile(r"\b(?:in\s+(?:the\s+)?yard|yard\s+stock|inventory)\b", re.I), "containers_in_yard"),
    (re.compile(r"\bimport\w*\b", re.I), "import_teu"),
    (re.compile(r"\bexport\w*\b", re.I), "export_teu"),
    (re.compile(r"\barriv\w*\b", re.I), "container_arrivals_last_1h"),
    (re.compile(r"\bdepart\w*\b", re.I), "container_departures_last_1h"),
    (re.compile(r"\bdwell\b", re.I), "average_dwell_time_hours"),
    (re.compile(r"\b(?:truck\s+wait\w*|waiting\s+time)\b", re.I), "truck_waiting_time_minutes"),
    (re.compile(r"\b(?:gate\s+throughput|gate\s+transactions?)\b", re.I), "gate_throughput_last_1h"),
    (re.compile(r"\b(?:flow\s+ratio|arrivals?\s+to\s+departures?\s+ratio)\b", re.I), "cargo_flow_ratio"),
)


def _safe_narrative(value: CargoNarrative, *, status: CargoStatus, evidence: CargoEvidence) -> bool:
    finding = value.finding.strip()
    cited = set(value.evidence_fields)
    if not finding or "synthetic" not in finding.lower():
        return False
    if value.possible_operational_contribution != CONTRIBUTION:
        return False
    if _UNSAFE.search(finding) or _WORD_NUMBER.search(finding):
        return False
    if any(label.group().upper() != status.value or "cargo_status" not in cited
           for label in _STATUS.finditer(finding)):
        return False
    if any(pattern.search(finding) and field not in cited for pattern, field in _CONCEPT_FIELDS):
        return False
    for match in _NUMBER.finditer(finding):
        literal = match.group().strip()
        number = float(literal.removesuffix("%"))
        fields = {"yard_occupancy_percent"} if literal.endswith("%") else cited - {"cargo_status"}
        unit_match = _UNIT_AFTER_NUMBER.match(finding[match.end():])
        if unit_match is not None:
            unit = unit_match.group("unit").lower().rstrip("s")
            if unit.startswith("container-equivalent") or unit.startswith("container equivalent"):
                unit = "unit"
            fields &= _UNIT_FIELDS[unit]
        if not any(abs(number - float(getattr(evidence, field))) < 1e-9 for field in fields):
            return False
    return True


class CargoInvestigationAgent:
    name = "cargo"

    def __init__(
        self,
        source: CargoStatusSource | None = None,
        llm_provider: StructuredLLMProvider | None = None,
    ) -> None:
        self.source = source or CargoStatusSource()
        self.llm_provider = llm_provider

    def investigate(self, request: CargoInvestigationRequest) -> CargoResult:
        stored_status, evidence = self.source.get_status(request.timestamp_utc)
        result = self._deterministic_result(request, CargoStatus(stored_status), evidence)
        if self.llm_provider is None:
            return result
        try:
            prompt = (Path(__file__).with_name("prompts") / "cargo.md").read_text(
                encoding="utf-8"
            )
            raw = self.llm_provider.generate_structured(
                system_prompt=prompt,
                payload={
                    "cargo_status": result.cargo_status.value,
                    "evidence": evidence.model_dump(),
                    "limitations": result.limitations,
                    "forecast_context": request.forecast_context.model_dump()
                    if request.forecast_context else None,
                },
                response_model=CargoNarrative,
            )
            if isinstance(raw, str):
                narrative = CargoNarrative.model_validate_json(raw)
            elif isinstance(raw, BaseModel):
                narrative = CargoNarrative.model_validate(raw.model_dump())
            else:
                narrative = CargoNarrative.model_validate(raw)
            if not _safe_narrative(narrative, status=result.cargo_status, evidence=evidence):
                raise ValueError("Unsafe Cargo narrative")
            return result.model_copy(update={
                "reasoning_mode": ReasoningMode.LLM_ASSISTED,
                "findings": result.findings + (
                    CargoFinding(
                        finding=narrative.finding,
                        evidence_fields=narrative.evidence_fields,
                        provenance=(DataProvenance.AGENT_GENERATED,),
                    ),
                ),
                "possible_operational_contribution": narrative.possible_operational_contribution,
            })
        except Exception:
            return result.model_copy(update={"reasoning_mode": ReasoningMode.DETERMINISTIC_FALLBACK})

    @staticmethod
    def _deterministic_result(
        request: CargoInvestigationRequest, status: CargoStatus, evidence: CargoEvidence
    ) -> CargoResult:
        findings = (
            CargoFinding(
                finding=(
                    f"The supplied synthetic Cargo state reports yard occupancy of "
                    f"{evidence.yard_occupancy_percent}% and a stored current operations "
                    f"status of {status.value}."
                ),
                evidence_fields=("yard_occupancy_percent", "cargo_status"),
                provenance=(DataProvenance.DERIVED,),
            ),
            CargoFinding(
                finding=(
                    f"The synthetic model reports {evidence.containers_in_yard} normalized "
                    f"container-equivalent units in yard, with "
                    f"{evidence.container_arrivals_last_1h} arrivals and "
                    f"{evidence.container_departures_last_1h} departures in the supplied hour."
                ),
                evidence_fields=("containers_in_yard", "container_arrivals_last_1h",
                                 "container_departures_last_1h"),
                provenance=(DataProvenance.SYNTHETIC,),
            ),
            CargoFinding(
                finding=(
                    f"The synthetic gate state reports a truck waiting indicator of "
                    f"{evidence.truck_waiting_time_minutes} minutes and gate throughput of "
                    f"{evidence.gate_throughput_last_1h} transactions for the supplied hour."
                ),
                evidence_fields=("truck_waiting_time_minutes", "gate_throughput_last_1h"),
                provenance=(DataProvenance.SYNTHETIC,),
            ),
        )
        provenance = tuple(
            CargoEvidenceProvenance(
                field=field,
                provenance=(DataProvenance.DERIVED if field in {
                    "yard_occupancy_percent", "cargo_flow_ratio"
                } else DataProvenance.SYNTHETIC),
                source_field=source_field,
                semantic_limit=(
                    "Derived from synthetic state" if field in {
                        "yard_occupancy_percent", "cargo_flow_ratio"
                    } else "Calibrated synthetic hourly allocation, not an observed terminal measurement"
                    if field in {"import_teu", "export_teu"}
                    else "Normalized modeling capacity, not physical POLA capacity"
                    if field == "yard_capacity" else "Synthetic operational indicator"
                ),
            )
            for field, source_field in SOURCE_FIELDS.items()
        ) + (
            CargoEvidenceProvenance(
                field="cargo_status", provenance=DataProvenance.DERIVED,
                source_field="operations_status",
                semantic_limit="Stored current rule-derived status from unrounded synthetic state",
            ),
        )
        if request.forecast_context is not None:
            provenance += (
                CargoEvidenceProvenance(
                    field="forecast_context", provenance=DataProvenance.PREDICTED
                ),
            )
        return CargoResult(
            timestamp_utc=request.timestamp_utc,
            reasoning_mode=ReasoningMode.DETERMINISTIC,
            cargo_status=status,
            evidence=evidence,
            findings=findings,
            possible_operational_contribution=CONTRIBUTION,
            limitations=(
                "Cargo values are synthetic/calibrated synthetic Digital Twin state, not observed Port of Los Angeles terminal measurements.",
                "Yard capacity is 10,000 normalized modeling units, not physical port capacity; hourly import/export TEU are synthetic allocations calibrated to supplied monthly targets.",
                "Dwell and truck waiting indicators are synthetic, not official observed dwell or truck-turn-time series.",
                "Stored cargo status is a current rule-derived synthetic condition, not official ground truth or an ML forecast.",
                "The frozen 2025 state used AIS-derived maritime context and retrospective offline construction; it is not live point-in-time inference.",
                "Evidence does not establish causality or independently justify an operational strategy.",
            ),
            evidence_provenance=provenance,
            forecast_context=request.forecast_context,
        )
