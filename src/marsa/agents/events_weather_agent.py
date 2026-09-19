"""Evidence-first Events & Weather Agent with deterministic safe fallback."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from marsa.agents.events_weather import EventsWeatherAgentInput
from marsa.agents.llm import StructuredLLMProvider
from marsa.agents.tools.events_weather import EventsWeatherContextSource
from marsa.provenance.types import DataProvenance

logger = logging.getLogger("uvicorn.error")


class PressureLevel(StrEnum):
    LOW = "low"
    MODERATE = "moderate"
    ELEVATED = "elevated"
    UNKNOWN = "unknown"


class ReasoningMode(StrEnum):
    DETERMINISTIC = "deterministic"
    LLM_ASSISTED = "llm_assisted"
    DETERMINISTIC_FALLBACK = "deterministic_fallback"


class ForecastContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    horizon_hours: int = Field(ge=1)
    risk_level: str


class EventsWeatherInvestigationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp_utc: datetime
    investigation_reason: str | None = None
    forecast_context: ForecastContext | None = None

    @field_validator("timestamp_utc")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp_utc must be timezone-aware")
        return value.astimezone(UTC)


class PressureAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    level: PressureLevel
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_semantics: Literal["evidence_completeness"] = "evidence_completeness"


class WeatherAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    level: PressureLevel
    summary: str


class ExternalEventAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    active: bool
    type: str | None = None
    severity: str | None = None
    summary: str
    provenance: DataProvenance = DataProvenance.SYNTHETIC


class CalendarAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operational_local_time: datetime
    is_weekend_local: bool
    is_public_holiday_local: bool
    summary: str
    provenance: DataProvenance = DataProvenance.DERIVED


class DomainFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    finding: str
    evidence_fields: tuple[str, ...]
    provenance: tuple[DataProvenance, ...]


class EvidenceProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    field: str
    provenance: DataProvenance


class EventsWeatherResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    agent: Literal["events_weather"] = "events_weather"
    timestamp_utc: datetime
    reasoning_mode: ReasoningMode
    contextual_pressure: PressureAssessment
    weather_assessment: WeatherAssessment
    external_event_assessment: ExternalEventAssessment
    calendar_context: CalendarAssessment
    findings: tuple[DomainFinding, ...]
    possible_operational_contribution: str
    limitations: tuple[str, ...]
    evidence_provenance: tuple[EvidenceProvenance, ...]


class EventsWeatherNarrative(BaseModel):
    """The only fields an optional LLM may refine."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    weather_summary: str
    event_summary: str
    calendar_summary: str
    possible_operational_contribution: str


_CAUSAL_PATTERNS = (
    re.compile(r"\bcaus(?:e|ed|es|ing)\b", re.IGNORECASE),
    re.compile(r"\bis the reason\b", re.IGNORECASE),
    re.compile(r"\bresulted in\b", re.IGNORECASE),
    re.compile(r"\bled to\b", re.IGNORECASE),
)
_UNIT_PATTERN = re.compile(
    r"\b(?:m/s|mph|knots?|meters?|metres?|feet|ft|kilometers? per hour|km/h)\b",
    re.IGNORECASE,
)
_CALENDAR_CONTRIBUTION_PATTERN = re.compile(
    r"\b(?:weekends?|holidays?|calendar|local timing|day of week)\b",
    re.IGNORECASE,
)
_STRATEGY_PATTERN = re.compile(
    r"\b(?:recommend(?:ed|ation)?|strategy|should (?:reroute|deploy|close|delay|divert))\b",
    re.IGNORECASE,
)
_CONGESTION_PREDICTION_PATTERN = re.compile(
    r"\b(?:will become congested|will be congested|predicts? congestion|congestion risk is)\b",
    re.IGNORECASE,
)


def classify_weather(index: float | None) -> PressureLevel:
    """Marsa heuristic bands for a relative 0-1 index, not official thresholds."""
    if index is None:
        return PressureLevel.UNKNOWN
    if index < 0.33:
        return PressureLevel.LOW
    if index < 0.67:
        return PressureLevel.MODERATE
    return PressureLevel.ELEVATED


def classify_context(weather: PressureLevel, event_active: bool, severity: str | None) -> PressureLevel:
    """Combine only domain evidence; forecast context is intentionally absent."""
    if weather is PressureLevel.ELEVATED or (event_active and severity == "severe"):
        return PressureLevel.ELEVATED
    if weather is PressureLevel.MODERATE or event_active:
        return PressureLevel.MODERATE
    if weather is PressureLevel.LOW:
        return PressureLevel.LOW
    return PressureLevel.UNKNOWN


def evidence_confidence(context: EventsWeatherAgentInput) -> float:
    """Evidence-completeness score, not a calibrated probability."""
    score = 0.25  # Valid canonical timestamp and local calendar context.
    if context.weather.weather_inputs_complete:
        score += 0.50
    event = context.external_event
    event_complete = (not event.active) or all(
        value is not None
        for value in (
            event.event_id,
            event.type,
            event.severity,
            event.start_utc,
            event.end_utc,
        )
    )
    if event_complete:
        score += 0.20
    return round(min(score, 0.95), 2)


def _system_prompt() -> str:
    path = Path(__file__).with_name("prompts") / "events_weather.md"
    return path.read_text(encoding="utf-8")


def _narrative_rejection_reason(
    value: EventsWeatherNarrative, *, active_event: bool
) -> str | None:
    text = " ".join(
        [
            value.weather_summary,
            value.event_summary,
            value.calendar_summary,
            value.possible_operational_contribution,
        ]
    )
    if any(pattern.search(text) for pattern in _CAUSAL_PATTERNS):
        return "causal_language"
    if _UNIT_PATTERN.search(text):
        return "unverified_unit"
    if _CALENDAR_CONTRIBUTION_PATTERN.search(value.possible_operational_contribution):
        return "calendar_pressure_attribution"
    if _STRATEGY_PATTERN.search(text):
        return "strategy_recommendation"
    if _CONGESTION_PREDICTION_PATTERN.search(text):
        return "congestion_prediction"
    if active_event and "synthetic" not in value.event_summary.lower():
        return "synthetic_disclosure_missing"
    return None


class EventsWeatherAgent:
    """Investigate contextual evidence without predicting congestion or choosing actions."""

    name = "events_weather"

    def __init__(
        self,
        context_source: EventsWeatherContextSource | None = None,
        llm_provider: StructuredLLMProvider | None = None,
    ) -> None:
        self.context_source = context_source or EventsWeatherContextSource()
        self.llm_provider = llm_provider

    def run(
        self, request: EventsWeatherInvestigationRequest | dict[str, Any]
    ) -> EventsWeatherResult:
        validated = (
            request
            if isinstance(request, EventsWeatherInvestigationRequest)
            else EventsWeatherInvestigationRequest.model_validate(request)
        )
        return self.investigate(validated)

    def investigate(self, request: EventsWeatherInvestigationRequest) -> EventsWeatherResult:
        context = self.context_source.get_context(request.timestamp_utc)
        if context.timestamp_utc != request.timestamp_utc:
            raise ValueError("Context source returned a different authoritative timestamp")
        result = self._deterministic_result(context)
        if self.llm_provider is None:
            logger.info("Gemini provider configured: false; deterministic result returned")
            return result
        fallback_reason = "unknown"
        try:
            logger.info("Gemini provider configured: true")
            logger.info("Gemini request attempted: true")
            raw = self.llm_provider.generate_structured(
                system_prompt=_system_prompt(),
                payload=self._reasoning_payload(request, context, result),
                response_model=EventsWeatherNarrative,
            )
            logger.info("Gemini response received: true")
            try:
                narrative = self._parse_narrative(raw)
            except Exception:
                fallback_reason = "schema_validation_failed"
                logger.info("Gemini response validation: failed; category=%s", fallback_reason)
                raise
            rejection_reason = _narrative_rejection_reason(
                narrative, active_event=context.external_event.active
            )
            if rejection_reason is not None:
                fallback_reason = rejection_reason
                logger.info("Gemini response validation: failed; category=%s", fallback_reason)
                raise ValueError("LLM narrative violated safety constraints")
            logger.info("Gemini response validation: passed")
            logger.info("Deterministic consistency validation: passed")
            return result.model_copy(
                update={
                    "reasoning_mode": ReasoningMode.LLM_ASSISTED,
                    "weather_assessment": result.weather_assessment.model_copy(
                        update={"summary": narrative.weather_summary}
                    ),
                    "external_event_assessment": result.external_event_assessment.model_copy(
                        update={"summary": narrative.event_summary}
                    ),
                    "calendar_context": result.calendar_context.model_copy(
                        update={"summary": narrative.calendar_summary}
                    ),
                    "possible_operational_contribution": (
                        narrative.possible_operational_contribution
                    ),
                }
            )
        except Exception as error:
            if fallback_reason == "unknown":
                fallback_reason = (
                    "provider_error"
                    if not isinstance(error, ValueError)
                    else "response_validation_failed"
                )
            logger.info("Fallback reason: %s", fallback_reason)
            return result.model_copy(
                update={
                    "reasoning_mode": ReasoningMode.DETERMINISTIC_FALLBACK,
                    "limitations": result.limitations
                    + ("Optional LLM narrative was unavailable or invalid; deterministic fallback used.",),
                }
            )

    @staticmethod
    def _parse_narrative(raw: BaseModel | dict[str, Any] | str) -> EventsWeatherNarrative:
        if isinstance(raw, EventsWeatherNarrative):
            return raw
        if isinstance(raw, str):
            return EventsWeatherNarrative.model_validate_json(raw)
        if isinstance(raw, BaseModel):
            return EventsWeatherNarrative.model_validate(raw.model_dump())
        return EventsWeatherNarrative.model_validate(raw)

    @staticmethod
    def _reasoning_payload(
        request: EventsWeatherInvestigationRequest,
        context: EventsWeatherAgentInput,
        result: EventsWeatherResult,
    ) -> dict[str, Any]:
        # Context is already sanitized by the builder and never contains weather_code.
        return {
            "investigation_reason": request.investigation_reason,
            "forecast_context": (
                request.forecast_context.model_dump(mode="json")
                if request.forecast_context
                else None
            ),
            "domain_context": context.model_dump(mode="json"),
            "deterministic_assessment": {
                "contextual_pressure": result.contextual_pressure.model_dump(mode="json"),
                "weather_level": result.weather_assessment.level,
                "event_active": result.external_event_assessment.active,
            },
        }

    @staticmethod
    def _deterministic_result(context: EventsWeatherAgentInput) -> EventsWeatherResult:
        weather_level = (
            classify_weather(context.weather.weather_pressure_index)
            if context.weather.weather_inputs_complete
            else PressureLevel.UNKNOWN
        )
        event = context.external_event
        overall = classify_context(weather_level, event.active, event.severity)
        confidence = evidence_confidence(context)

        if weather_level is PressureLevel.UNKNOWN:
            weather_summary = "Weather evidence is incomplete; no missing condition was inferred."
        elif weather_level is PressureLevel.LOW:
            weather_summary = "No substantial weather pressure is evident in the available observations."
        elif weather_level is PressureLevel.MODERATE:
            weather_summary = "Weather conditions are moderately elevated relative to the historical baseline."
        else:
            weather_summary = "Weather conditions are elevated relative to the historical baseline."

        if event.active:
            event_summary = (
                f"An active synthetic {event.type} event has {event.severity} severity; "
                "it is scenario context, not a verified historical incident."
            )
        else:
            event_summary = "No synthetic external operational event is active for this hour."

        calendar = context.calendar
        labels = []
        if calendar.is_weekend_local:
            labels.append("weekend")
        if calendar.is_public_holiday_local:
            labels.append("US federal observed holiday")
        calendar_summary = (
            f"The local operational period is a {' and '.join(labels)}."
            if labels
            else "The local operational period is neither a weekend nor a US federal observed holiday."
        )

        if overall is PressureLevel.UNKNOWN:
            contribution = "Available evidence is insufficient to assess Events & Weather pressure."
        elif overall is PressureLevel.LOW:
            contribution = "No significant Events & Weather pressure was identified."
        elif event.active and weather_level in {PressureLevel.MODERATE, PressureLevel.ELEVATED}:
            contribution = (
                "Elevated weather context and an active synthetic event may contribute to "
                "operational pressure; causality is not established."
            )
        elif event.active:
            contribution = (
                "The active synthetic event may contribute to operational pressure; causality is "
                "not established."
            )
        else:
            contribution = (
                "Weather context may contribute to operational pressure; causality is not established."
            )

        findings = [
            DomainFinding(
                finding=weather_summary,
                evidence_fields=(
                    "wind_speed_10m",
                    "wave_height",
                    "weather_pressure_index",
                ),
                provenance=(DataProvenance.OBSERVED, DataProvenance.DERIVED),
            )
        ]
        if event.active:
            findings.append(
                DomainFinding(
                    finding=event_summary,
                    evidence_fields=("event_active", "event_type", "event_severity"),
                    provenance=(DataProvenance.SYNTHETIC,),
                )
            )
        if calendar.is_weekend_local or calendar.is_public_holiday_local:
            findings.append(
                DomainFinding(
                    finding=calendar_summary,
                    evidence_fields=("is_weekend_local", "is_public_holiday_local"),
                    provenance=(DataProvenance.DERIVED,),
                )
            )

        limitations = [
            "Wind and wave units are unverified; no unit-specific claim is made.",
            "weather_pressure_index is a Marsa contextual heuristic, not an official threshold.",
            "Contextual evidence does not establish causality or predict congestion.",
            "US federal holiday context does not establish a port closure.",
        ]
        if not context.weather.weather_inputs_complete:
            limitations.append("Weather evidence is incomplete for this authoritative hour.")
        if event.active:
            limitations.append("The active external event is synthetic scenario context.")

        evidence = [
            EvidenceProvenance(field="wind_speed_10m", provenance=DataProvenance.OBSERVED),
            EvidenceProvenance(field="wave_height", provenance=DataProvenance.OBSERVED),
            EvidenceProvenance(
                field="weather_pressure_index", provenance=DataProvenance.DERIVED
            ),
            EvidenceProvenance(field="local_calendar_context", provenance=DataProvenance.DERIVED),
            EvidenceProvenance(field="external_event", provenance=DataProvenance.SYNTHETIC),
        ]

        return EventsWeatherResult(
            timestamp_utc=context.timestamp_utc,
            reasoning_mode=ReasoningMode.DETERMINISTIC,
            contextual_pressure=PressureAssessment(level=overall, confidence=confidence),
            weather_assessment=WeatherAssessment(level=weather_level, summary=weather_summary),
            external_event_assessment=ExternalEventAssessment(
                active=event.active,
                type=event.type,
                severity=event.severity,
                summary=event_summary,
            ),
            calendar_context=CalendarAssessment(
                operational_local_time=context.operational_local_time,
                is_weekend_local=calendar.is_weekend_local,
                is_public_holiday_local=calendar.is_public_holiday_local,
                summary=calendar_summary,
            ),
            findings=tuple(findings),
            possible_operational_contribution=contribution,
            limitations=tuple(limitations),
            evidence_provenance=tuple(evidence),
        )
