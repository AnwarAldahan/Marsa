"""Public request adapters for the three domain agents."""
from __future__ import annotations

from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict, field_validator

from marsa.agents.cargo_agent import CargoForecastContext, CargoInvestigationRequest
from marsa.agents.events_weather_agent import EventsWeatherInvestigationRequest, ForecastContext
from marsa.agents.maritime_agent import MaritimeForecastContext, MaritimeInvestigationRequest


class _AgentApiRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp_utc: datetime
    investigation_reason: str | None = None
    forecast_risk: str | None = None
    horizon_hours: int | None = None

    @field_validator("timestamp_utc")
    @classmethod
    def require_explicit_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp_utc must be timezone-aware and explicitly UTC")
        if value.utcoffset() != timedelta(0):
            raise ValueError("timestamp_utc must use UTC (offset +00:00 or Z)")
        return value

    def _pair(self, context_type):
        if self.forecast_risk is None and self.horizon_hours is None:
            return None
        if self.forecast_risk is None or self.horizon_hours is None:
            raise ValueError("forecast_risk and horizon_hours must be provided together")
        return context_type(risk_level=self.forecast_risk, horizon_hours=self.horizon_hours)


class MaritimeApiRequest(_AgentApiRequest):
    def to_agent_request(self) -> MaritimeInvestigationRequest:
        return MaritimeInvestigationRequest(
            timestamp_utc=self.timestamp_utc,
            investigation_reason=self.investigation_reason,
            forecast_context=self._pair(MaritimeForecastContext),
        )


class CargoApiRequest(_AgentApiRequest):
    def to_agent_request(self) -> CargoInvestigationRequest:
        return CargoInvestigationRequest(
            timestamp_utc=self.timestamp_utc,
            investigation_reason=self.investigation_reason,
            forecast_context=self._pair(CargoForecastContext),
        )


class EventsWeatherApiRequest(_AgentApiRequest):
    def to_agent_request(self) -> EventsWeatherInvestigationRequest:
        return EventsWeatherInvestigationRequest(
            timestamp_utc=self.timestamp_utc,
            investigation_reason=self.investigation_reason,
            forecast_context=self._pair(ForecastContext),
        )
