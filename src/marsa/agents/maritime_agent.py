"""Thin Maritime investigation wrapper with optional narrative-only LLM support."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from marsa.agents.events_weather_agent import ReasoningMode
from marsa.agents.llm import StructuredLLMProvider
from marsa.agents.tools.maritime_status import MaritimeEvidence, MaritimeStatusSource
from marsa.provenance.types import DataProvenance


class MaritimeStatus(StrEnum):
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    CONGESTED = "CONGESTED"


class MaritimeForecastContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    horizon_hours: int = Field(ge=1)
    risk_level: str


class MaritimeInvestigationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp_utc: datetime
    investigation_reason: str | None = None
    forecast_context: MaritimeForecastContext | None = None

    @field_validator("timestamp_utc")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp_utc must be timezone-aware")
        return value.astimezone(UTC)


class MaritimeFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    finding: str
    evidence_fields: tuple[str, ...]
    provenance: DataProvenance


class MaritimeEvidenceProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    field: str
    provenance: DataProvenance
    source_field: str | None = None
    semantic_limit: str | None = None


class MaritimeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    agent: Literal["maritime"] = "maritime"
    timestamp_utc: datetime
    reasoning_mode: ReasoningMode
    maritime_status: MaritimeStatus
    evidence: MaritimeEvidence
    findings: tuple[MaritimeFinding, ...]
    possible_operational_contribution: str
    limitations: tuple[str, ...]
    evidence_provenance: tuple[MaritimeEvidenceProvenance, ...]
    forecast_context: MaritimeForecastContext | None = None


class MaritimeNarrative(BaseModel):
    """Only free text may be generated; facts and limitations are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    finding: str
    evidence_fields: tuple[
        Literal["maritime_status", "waiting_ratio", "waiting_vessels", "vessels"], ...
    ] = Field(min_length=1)
    possible_operational_contribution: str


_UNSAFE = re.compile(
    r"\b(?:knots?|mph|m/s|km/h|meters?|metres?|feet|ft|last\s+four\s+hours|"
    r"previous\s+four\s+hours|waiting\s*(?:to|->|→)\s*moored|"
    r"arriv(?:e|al|ing)|leav(?:e|ing)|heading\s+(?:toward|away)|"
    r"means|defined\s+as|refers\s+to|measures|"
    r"incident|collision|accident|strike|breakdown|closure|"
    r"no|none|all|every|only|inbound|outbound|low|high|"
    r"caus(?:e|ed|es|ing)|resulted\s+in|led\s+to|"
    r"will\s+be\s+congested|predict(?:s|ed)?\s+congestion|"
    r"recommend(?:ed|ation)?|should|must|strategy|reroute|divert|deploy)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_OPERATIONAL_CONCEPT = re.compile(
    r"\b(?:observed\s+activity|stationar\w*|stopp?\w*|hold\w*|anchor\w*|"
    r"queu\w*|berth\w*|delay\w*|bottleneck\w*|efficien\w*|"
    r"allocat\w*|resource\w*|maneuv\w*|manoeuv\w*|mov\w*|"
    r"motion|traffic\s+flow|flow\s+of\s+traffic|drift\w*|sail\w*|"
    r"idling|idle|slow\w*)\b",
    re.IGNORECASE,
)
_UNDECLARED_FINDING_EVIDENCE = re.compile(
    r"\b(?:approach\w*|depart\w*|speed|throughput|capac\w*|"
    r"threshold\w*|duration\w*|rates?|hours?|minutes?|days?)\b",
    re.IGNORECASE,
)
_STATUS_LABEL = re.compile(r"\b(?:normal|elevated|congested)\b", re.IGNORECASE)
_UNSUPPORTED_STATUS_ROLE = re.compile(
    r"\b(?:official|ground truth|forecast|predict\w*|ML)\b", re.IGNORECASE
)
_NUMBER = re.compile(r"(?<![\w.])(?:\d+(?:\.\d+)?|\.\d+)\s*%?")
_COUNT_CLAIM = re.compile(r"\b(?P<number>\d+)\s+(?P<kind>waiting|total)?\s*vessels?\b", re.IGNORECASE)
_WORDED_QUANTITY = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:vessels?|ships?|units?|percent|hours?)\b",
    re.IGNORECASE,
)
_WORDED_FRACTION = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten)[\s-]+"
    r"(?:quarter|half|third|fourth|fifth|percent)\b",
    re.IGNORECASE,
)


def _safe_narrative(
    value: MaritimeNarrative, *, status: MaritimeStatus, evidence: MaritimeEvidence
) -> bool:
    cited = set(value.evidence_fields)
    texts = (value.finding, value.possible_operational_contribution)
    for text in texts:
        if any(
            label.group().upper() != status.value or "maritime_status" not in cited
            for label in _STATUS_LABEL.finditer(text)
        ):
            return False
        if _UNSUPPORTED_STATUS_ROLE.search(text) or _WORDED_QUANTITY.search(text):
            return False
    count_claims = list(_COUNT_CLAIM.finditer(value.finding))
    number_matches = list(_NUMBER.finditer(value.finding))
    if any(
        not any(claim.start("number") == number.start() for number in number_matches)
        for claim in count_claims
    ):
        return False
    for match in number_matches:
        literal = match.group().strip()
        count_claim = next(
            (claim for claim in count_claims if claim.start("number") == match.start()), None
        )
        if count_claim is not None:
            is_waiting = (count_claim.group("kind") or "").lower() == "waiting" or re.match(
                r"\s+(?:are|were)\s+waiting\b", value.finding[count_claim.end():], re.IGNORECASE
            ) is not None
            field = "waiting_vessels" if is_waiting else "vessels"
            if field not in cited or int(count_claim.group("number")) != getattr(evidence, field):
                return False
            continue
        number = float(literal.removesuffix("%"))
        if literal.endswith("%"):
            number /= 100
        if "waiting_ratio" not in cited or abs(number - evidence.waiting_ratio) > 1e-9:
            return False
    if _NUMBER.search(value.possible_operational_contribution):
        return False
    for match in _WORDED_FRACTION.finditer(value.finding):
        if (
            "waiting_ratio" not in cited
            or match.group().lower().replace("-", " ") != "one quarter"
            or abs(evidence.waiting_ratio - 0.25) > 1e-9
        ):
            return False
    if _WORDED_FRACTION.search(value.possible_operational_contribution):
        return False
    contribution = value.possible_operational_contribution
    return (
        bool(value.finding.strip() and contribution.strip())
        and not any(
            _UNSAFE.search(text) or _UNSUPPORTED_OPERATIONAL_CONCEPT.search(text)
            for text in (value.finding, contribution)
        )
        and not _UNDECLARED_FINDING_EVIDENCE.search(value.finding)
        and re.search(r"\bmaritime evidence\b", contribution, re.IGNORECASE) is not None
        and re.search(r"\b(?:current|present)\b", contribution, re.IGNORECASE) is not None
        and re.search(r"\b(?:operational|port-side) pressure\b", contribution, re.IGNORECASE)
        is not None
    )


class MaritimeInvestigationAgent:
    name = "maritime"

    def __init__(
        self,
        source: MaritimeStatusSource | None = None,
        llm_provider: StructuredLLMProvider | None = None,
    ) -> None:
        self.source = source or MaritimeStatusSource()
        self.llm_provider = llm_provider

    def investigate(self, request: MaritimeInvestigationRequest) -> MaritimeResult:
        status, evidence = self.source.get_status(request.timestamp_utc)
        result = self._deterministic_result(request, MaritimeStatus(status), evidence)
        if self.llm_provider is None:
            return result
        try:
            prompt = (Path(__file__).with_name("prompts") / "maritime.md").read_text(
                encoding="utf-8"
            )
            raw = self.llm_provider.generate_structured(
                system_prompt=prompt,
                payload={
                    "status": result.maritime_status.value,
                    "evidence": evidence.model_dump(),
                    "limitations": result.limitations,
                    "forecast_context": request.forecast_context.model_dump()
                    if request.forecast_context
                    else None,
                },
                response_model=MaritimeNarrative,
            )
            if isinstance(raw, str):
                narrative = MaritimeNarrative.model_validate_json(raw)
            elif isinstance(raw, BaseModel):
                narrative = MaritimeNarrative.model_validate(raw.model_dump())
            else:
                narrative = MaritimeNarrative.model_validate(raw)
            if not _safe_narrative(
                narrative, status=result.maritime_status, evidence=evidence
            ):
                raise ValueError("Unsafe Maritime narrative")
            return result.model_copy(
                update={
                    "reasoning_mode": ReasoningMode.LLM_ASSISTED,
                    "findings": result.findings
                    + (
                        MaritimeFinding(
                            finding=narrative.finding,
                            evidence_fields=narrative.evidence_fields,
                            provenance=DataProvenance.AGENT_GENERATED,
                        ),
                    ),
                    "possible_operational_contribution": narrative.possible_operational_contribution,
                }
            )
        except Exception:
            return result.model_copy(update={"reasoning_mode": ReasoningMode.DETERMINISTIC_FALLBACK})

    @staticmethod
    def _deterministic_result(
        request: MaritimeInvestigationRequest, status: MaritimeStatus, evidence: MaritimeEvidence
    ) -> MaritimeResult:
        findings = (
            MaritimeFinding(
                finding=(
                    f"Current team heuristic status is {status.value}; {evidence.waiting_vessels} "
                    f"of {evidence.vessels} vessels are waiting "
                    f"(teammate waiting ratio {evidence.waiting_ratio})."
                ),
                evidence_fields=("vessels", "waiting_vessels", "waiting_ratio", "maritime_status"),
                provenance=DataProvenance.DERIVED,
            ),
            MaritimeFinding(
                finding=(
                    f"Source reports {evidence.approaching_vessels} approaching and "
                    f"{evidence.departing_vessels} departing vessels, average speed value "
                    f"{evidence.average_speed_value}, and throughput value "
                    f"{evidence.port_throughput_value}; definitions and units remain unverified."
                ),
                evidence_fields=(
                    "approaching_vessels", "departing_vessels", "average_speed_value",
                    "port_throughput_value",
                ),
                provenance=DataProvenance.DERIVED,
            ),
        )
        provenance = tuple(
            MaritimeEvidenceProvenance(field=field, provenance=DataProvenance.DERIVED, source_field=source,
                                        semantic_limit=limit)
            for field, source, limit in (
                ("vessels", "vessel_count", None),
                ("waiting_vessels", "waiting_vessel_count", None),
                ("waiting_ratio", "waiting_ratio", "Computed by teammate component"),
                ("approaching_vessels", "approaching_vessel_count", "Predicate unverified"),
                ("departing_vessels", "departing_vessel_count", "Predicate unverified"),
                ("average_speed_value", "average_speed_knots", "Source unit unverified"),
                ("port_throughput_value", "port_throughput_last_4h", "Definition/window unverified"),
                ("maritime_status", "port_status", "Current deterministic team heuristic"),
            )
        )
        if request.forecast_context is not None:
            provenance += (
                MaritimeEvidenceProvenance(
                    field="forecast_context", provenance=DataProvenance.PREDICTED
                ),
            )
        return MaritimeResult(
            timestamp_utc=request.timestamp_utc,
            reasoning_mode=ReasoningMode.DETERMINISTIC,
            maritime_status=status,
            evidence=evidence,
            findings=findings,
            possible_operational_contribution=(
                "Current Maritime evidence may indicate operational pressure, but does not "
                "establish causality or predict future congestion."
                if status is not MaritimeStatus.NORMAL
                else "Current Maritime evidence does not indicate elevated pressure under the team heuristic."
            ),
            limitations=(
                "Status is a current team heuristic, not official port ground truth or an ML forecast.",
                "Average speed unit is unverified; teammate source key is average_speed_knots.",
                "Throughput definition and window are unverified; teammate source key is port_throughput_last_4h.",
                "Approaching and departing vessel predicates are unverified.",
                "Evidence does not establish cause or justify an operational strategy.",
            ),
            evidence_provenance=provenance,
            forecast_context=request.forecast_context,
        )
