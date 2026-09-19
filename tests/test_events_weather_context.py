from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd
import pytest

from marsa.agents.events_weather import EventsWeatherContextBuilder
from marsa.provenance.types import DataProvenance


def row_at(timestamp: str, **updates):
    row = {
        "hour_key": pd.Timestamp(timestamp),
        "wind_speed_10m": 12.0,
        "wave_height": 1.3,
        "weather_code": 61.0,
        "weather_pressure_index": 0.4,
        "event_active": False,
        "event_id": pd.NA,
        "event_type": pd.NA,
        "event_severity": pd.NA,
        "event_start": pd.NaT,
        "event_end": pd.NaT,
        "event_duration_hours": pd.NA,
        "event_represented_hours": pd.NA,
    }
    row.update(updates)
    return row


@pytest.fixture
def builder():
    return EventsWeatherContextBuilder()


def test_pst_conversion(builder):
    result = builder.build(row_at("2025-01-15 08:00:00"))
    assert result.operational_local_time.isoformat() == "2025-01-15T00:00:00-08:00"


def test_pdt_conversion(builder):
    result = builder.build(row_at("2025-07-15 07:00:00"))
    assert result.operational_local_time.isoformat() == "2025-07-15T00:00:00-07:00"


def test_spring_dst_transition_skips_two_am(builder):
    before = builder.build(row_at("2025-03-09 09:00:00"))
    after = builder.build(row_at("2025-03-09 10:00:00"))
    assert before.operational_local_time.isoformat() == "2025-03-09T01:00:00-08:00"
    assert after.operational_local_time.isoformat() == "2025-03-09T03:00:00-07:00"


def test_fall_dst_transition_repeats_one_am_with_distinct_offsets(builder):
    first = builder.build(row_at("2025-11-02 08:00:00"))
    second = builder.build(row_at("2025-11-02 09:00:00"))
    assert first.operational_local_time.isoformat() == "2025-11-02T01:00:00-07:00"
    assert second.operational_local_time.isoformat() == "2025-11-02T01:00:00-08:00"


def test_utc_saturday_can_still_be_local_friday(builder):
    result = builder.build(row_at("2025-01-04 01:00:00"))
    assert result.calendar.local_day_of_week == "Friday"
    assert result.calendar.is_weekend_local is False


def test_utc_holiday_boundary_uses_local_date(builder):
    before_local_midnight = builder.build(row_at("2025-07-04 02:00:00"))
    local_midnight = builder.build(row_at("2025-07-04 07:00:00"))
    assert before_local_midnight.calendar.local_date.isoformat() == "2025-07-03"
    assert before_local_midnight.calendar.is_public_holiday_local is False
    assert local_midnight.calendar.local_date.isoformat() == "2025-07-04"
    assert local_midnight.calendar.is_public_holiday_local is True


def test_weather_null_handling(builder):
    result = builder.build(
        row_at(
            "2025-01-01 00:00:00",
            wind_speed_10m=float("nan"),
            wave_height=float("nan"),
            weather_pressure_index=float("nan"),
        )
    )
    assert result.weather.wind_speed_10m is None
    assert result.weather.wave_height is None
    assert result.weather.weather_pressure_index is None
    assert result.weather.weather_inputs_complete is False


def test_active_event_metadata_and_provenance(builder):
    result = builder.build(
        row_at(
            "2025-08-24 14:00:00",
            event_active=True,
            event_id="EVT_0037",
            event_type="labor_operational_disruption",
            event_severity="severe",
            event_start=pd.Timestamp("2025-08-24 14:00:00"),
            event_end=pd.Timestamp("2025-08-25 15:00:00"),
            event_duration_hours=25,
            event_represented_hours=25,
        )
    )
    assert result.external_event.event_id == "EVT_0037"
    assert result.external_event.provenance is DataProvenance.SYNTHETIC
    assert result.weather.observation_provenance is DataProvenance.OBSERVED
    assert result.weather.pressure_index_provenance is DataProvenance.DERIVED


def test_inactive_event_has_no_metadata(builder):
    result = builder.build(row_at("2025-06-01 12:00:00"))
    assert result.external_event.active is False
    assert result.external_event.event_id is None


def test_utc_json_serialization_uses_z(builder):
    result = builder.build(row_at("2025-08-24 14:00:00"))
    payload = json.loads(result.model_dump_json())
    assert payload["timestamp_utc"] == "2025-08-24T14:00:00Z"
    assert result.timestamp_utc == datetime(2025, 8, 24, 14, tzinfo=UTC)


def test_weather_code_is_excluded_from_reasoning_payload(builder):
    result = builder.build(row_at("2025-08-24 14:00:00", weather_code=95.0))
    payload = result.model_dump(mode="json")
    assert "weather_code" not in payload["weather"]
    assert "weather_code" not in json.dumps(payload)


def test_rejects_timezone_aware_storage_key(builder):
    with pytest.raises(ValueError, match="timezone-naive"):
        builder.build(row_at(pd.Timestamp("2025-01-01", tz="UTC")))
