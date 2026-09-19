"""Typed, deterministic context preparation for the future Events & Weather Agent."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
from functools import cache
from typing import Any, Literal
from zoneinfo import ZoneInfo

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from marsa.provenance.types import DataProvenance

OPERATIONAL_TIMEZONE = "America/Los_Angeles"
_LOCAL_ZONE = ZoneInfo(OPERATIONAL_TIMEZONE)


class WeatherContext(BaseModel):
    """Weather evidence safe for contextual, non-causal reasoning."""

    model_config = ConfigDict(frozen=True)

    wind_speed_10m: float | None = Field(default=None, ge=0.0)
    wave_height: float | None = Field(default=None, ge=0.0)
    weather_pressure_index: float | None = Field(default=None, ge=0.0, le=1.0)
    weather_inputs_complete: bool
    observation_provenance: DataProvenance = DataProvenance.OBSERVED
    pressure_index_provenance: DataProvenance = DataProvenance.DERIVED
    limitations: tuple[str, ...] = (
        "Wind and wave units are unverified.",
        "The pressure index is relative, not an official safety or port metric.",
        "Weather evidence does not establish causality.",
    )


class LocalCalendarContext(BaseModel):
    """DST-aware operational calendar context for Los Angeles/Long Beach."""

    model_config = ConfigDict(frozen=True)

    local_datetime: datetime
    local_date: date
    local_hour: int = Field(ge=0, le=23)
    local_day_of_week: str
    is_weekend_local: bool
    is_public_holiday_local: bool
    holiday_calendar: Literal["US_FEDERAL_OBSERVED_CONTEXT"] = "US_FEDERAL_OBSERVED_CONTEXT"
    provenance: DataProvenance = DataProvenance.DERIVED


class ExternalEventContext(BaseModel):
    """Synthetic operational-event context copied from the canonical hourly row."""

    model_config = ConfigDict(frozen=True)

    active: bool
    event_id: str | None = None
    type: str | None = None
    severity: str | None = None
    start_utc: datetime | None = None
    end_utc: datetime | None = None
    duration_hours: int | None = Field(default=None, ge=0)
    represented_hours: int | None = Field(default=None, ge=0)
    provenance: DataProvenance = DataProvenance.SYNTHETIC

    @field_validator("start_utc", "end_utc")
    @classmethod
    def normalize_optional_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("event timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_active_metadata(self) -> ExternalEventContext:
        metadata = (
            self.event_id,
            self.type,
            self.severity,
            self.start_utc,
            self.end_utc,
            self.duration_hours,
            self.represented_hours,
        )
        if self.active and any(value is None for value in metadata):
            raise ValueError("active events require complete event metadata")
        if not self.active and any(value is not None for value in metadata):
            raise ValueError("inactive events must not carry event metadata")
        return self


class EventsWeatherAgentInput(BaseModel):
    """Structured input contract for future non-causal Agent reasoning."""

    model_config = ConfigDict(frozen=True)

    timestamp_utc: datetime
    operational_timezone: Literal["America/Los_Angeles"] = OPERATIONAL_TIMEZONE
    operational_local_time: datetime
    weather: WeatherContext
    calendar: LocalCalendarContext
    external_event: ExternalEventContext

    @field_validator("timestamp_utc")
    @classmethod
    def normalize_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp_utc must be timezone-aware")
        return value.astimezone(UTC)


@cache
def _observed_federal_holidays(year: int) -> frozenset[date]:
    holidays = USFederalHolidayCalendar().holidays(
        start=pd.Timestamp(year=year, month=1, day=1),
        end=pd.Timestamp(year=year, month=12, day=31),
    )
    return frozenset(timestamp.date() for timestamp in holidays)


def _optional_value(value: Any) -> Any | None:
    return None if value is None or bool(pd.isna(value)) else value


def _stored_utc_hour(
    value: Any,
    *,
    field_name: str = "hour_key",
    require_2025: bool = False,
) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        raise ValueError(f"{field_name} must be timezone-naive in canonical storage")
    if timestamp.minute or timestamp.second or timestamp.microsecond or timestamp.nanosecond:
        raise ValueError(f"{field_name} must be aligned to an exact hour")
    if require_2025 and timestamp.year != 2025:
        raise ValueError(f"{field_name} must belong to the canonical 2025 scope")
    return timestamp.to_pydatetime().replace(tzinfo=UTC)


class EventsWeatherContextBuilder:
    """Convert one canonical 2025 hourly row into a safe Agent input payload."""

    required_fields = frozenset(
        {
            "hour_key",
            "wind_speed_10m",
            "wave_height",
            "weather_pressure_index",
            "event_active",
            "event_id",
            "event_type",
            "event_severity",
            "event_start",
            "event_end",
            "event_duration_hours",
            "event_represented_hours",
        }
    )

    def build(self, row: Mapping[str, Any]) -> EventsWeatherAgentInput:
        missing = sorted(self.required_fields.difference(row.keys()))
        if missing:
            raise ValueError(f"Missing required Events & Weather fields: {missing}")

        timestamp_utc = _stored_utc_hour(row["hour_key"], require_2025=True)
        local_datetime = timestamp_utc.astimezone(_LOCAL_ZONE)
        local_date = local_datetime.date()

        wind = _optional_value(row["wind_speed_10m"])
        wave = _optional_value(row["wave_height"])
        pressure = _optional_value(row["weather_pressure_index"])
        weather_complete = wind is not None and wave is not None and pressure is not None

        event_active = bool(row["event_active"])
        if event_active:
            start = _stored_utc_hour(row["event_start"], field_name="event_start")
            end = _stored_utc_hour(row["event_end"], field_name="event_end")
            external_event = ExternalEventContext(
                active=True,
                event_id=str(row["event_id"]),
                type=str(row["event_type"]),
                severity=str(row["event_severity"]),
                start_utc=start,
                end_utc=end,
                duration_hours=int(row["event_duration_hours"]),
                represented_hours=int(row["event_represented_hours"]),
            )
        else:
            event_values = [
                row[name]
                for name in [
                    "event_id", "event_type", "event_severity", "event_start", "event_end",
                    "event_duration_hours", "event_represented_hours",
                ]
            ]
            if any(_optional_value(value) is not None for value in event_values):
                raise ValueError("Inactive hourly rows must not carry event metadata")
            external_event = ExternalEventContext(active=False)

        return EventsWeatherAgentInput(
            timestamp_utc=timestamp_utc,
            operational_local_time=local_datetime,
            weather=WeatherContext(
                wind_speed_10m=wind,
                wave_height=wave,
                weather_pressure_index=pressure,
                weather_inputs_complete=weather_complete,
            ),
            calendar=LocalCalendarContext(
                local_datetime=local_datetime,
                local_date=local_date,
                local_hour=local_datetime.hour,
                local_day_of_week=local_datetime.strftime("%A"),
                is_weekend_local=local_datetime.weekday() >= 5,
                is_public_holiday_local=(
                    local_date in _observed_federal_holidays(local_date.year)
                ),
            ),
            external_event=external_event,
        )
