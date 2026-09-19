"""Build the reproducible Events & Weather Agent state and synthetic event table."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.testing import assert_frame_equal
from pandas.tseries.holiday import USFederalHolidayCalendar

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "data/processed/events_weather_base.parquet"
STATE_PATH = ROOT / "data/processed/events_weather_state.parquet"
EVENTS_PATH = ROOT / "data/processed/synthetic_events.parquet"
SCHEMA_PATH = ROOT / "data/processed/events_weather_schema.json"
ASSUMPTIONS_PATH = ROOT / "docs/EVENTS_WEATHER_ASSUMPTIONS.md"

SEED = 20260917
BASE_COLUMNS = [
    "hour_key",
    "wind_speed_10m",
    "wave_height",
    "weather_code",
    "hour",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "month",
]
EVENT_TYPES = [
    "landside_access_disruption",
    "labor_operational_disruption",
    "logistics_disruption",
    "demand_surge",
]
SEVERITIES = ["minor", "moderate", "severe"]
SEVERITY_WEIGHTS = np.array([0.55, 0.35, 0.10])

# Start probability per eligible hour. Calendar multipliers use only timestamp context.
EVENT_CONFIG = {
    "landside_access_disruption": {"probability": 0.00050, "duration": (4, 14)},
    "labor_operational_disruption": {"probability": 0.00035, "duration": (8, 26)},
    "logistics_disruption": {"probability": 0.00065, "duration": (6, 20)},
    "demand_surge": {"probability": 0.00035, "duration": (12, 36)},
}

WIND_P75 = 10.8
WIND_P95 = 16.9
WAVE_P75 = 1.02
WAVE_P95 = 1.58

FINAL_COLUMNS = BASE_COLUMNS + [
    "is_weekend",
    "is_public_holiday",
    "weather_pressure_index",
    "event_active",
    "event_id",
    "event_type",
    "event_severity",
    "event_start",
    "event_end",
    "event_duration_hours",
    "event_represented_hours",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_calendar(hour_key: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    dates = hour_key.dt.normalize()
    calendar = USFederalHolidayCalendar()
    holiday_series = calendar.holidays(
        start=dates.min(), end=dates.max(), return_name=True
    )
    flags = pd.DataFrame(index=hour_key.index)
    flags["is_weekend"] = hour_key.dt.dayofweek.isin([5, 6])
    flags["is_public_holiday"] = dates.isin(holiday_series.index)
    return flags, holiday_series


def weather_pressure(base: pd.DataFrame) -> pd.Series:
    wind = ((base["wind_speed_10m"] - WIND_P75) / (WIND_P95 - WIND_P75)).clip(0, 1)
    wave = ((base["wave_height"] - WAVE_P75) / (WAVE_P95 - WAVE_P75)).clip(0, 1)
    pressure = pd.concat([wind, wave], axis=1).max(axis=1, skipna=False)
    return pressure.astype("float64")


def generate_events(hour_key: pd.Series, flags: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    records: list[dict[str, Any]] = []
    unavailable_until = pd.Timestamp.min
    normalized_holidays = set(hour_key[flags["is_public_holiday"]].dt.normalize())

    for index, timestamp in hour_key.items():
        if timestamp < unavailable_until:
            continue

        near_holiday = any(
            timestamp.normalize() + pd.Timedelta(days=offset) in normalized_holidays
            for offset in range(-2, 3)
        )
        probabilities = []
        for event_type in EVENT_TYPES:
            probability = EVENT_CONFIG[event_type]["probability"]
            if event_type == "demand_surge" and near_holiday:
                probability *= 2.5
            if event_type == "landside_access_disruption" and flags.at[index, "is_weekend"]:
                probability *= 0.65
            probabilities.append(probability)

        total_probability = sum(probabilities)
        if rng.random() >= total_probability:
            continue

        event_type = str(rng.choice(EVENT_TYPES, p=np.array(probabilities) / total_probability))
        severity = str(rng.choice(SEVERITIES, p=SEVERITY_WEIGHTS))
        low, high = EVENT_CONFIG[event_type]["duration"]
        duration_multiplier = {"minor": 0.8, "moderate": 1.0, "severe": 1.4}[severity]
        duration = max(2, round(int(rng.integers(low, high + 1)) * duration_multiplier))
        event_end = timestamp + pd.Timedelta(hours=duration)
        event_id = f"EVT_{len(records) + 1:04d}"
        records.append(
            {
                "event_id": event_id,
                "event_type": event_type,
                "event_severity": severity,
                "event_start": timestamp,
                "event_end": event_end,
                "event_duration_hours": int(duration),
                "generation_seed": SEED,
            }
        )
        cooldown = int(rng.integers(4, 13))
        unavailable_until = event_end + pd.Timedelta(hours=cooldown)

    return pd.DataFrame.from_records(records)


def add_represented_hours(events: pd.DataFrame, hour_key: pd.Series) -> pd.DataFrame:
    result = events.copy()
    result["event_represented_hours"] = [
        int(((hour_key >= event.event_start) & (hour_key < event.event_end)).sum())
        for event in result.itertuples(index=False)
    ]
    return result


def attach_events(state: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    result = state.copy()
    result["event_active"] = False
    result["event_id"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["event_type"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["event_severity"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["event_start"] = pd.Series(pd.NaT, index=result.index, dtype="datetime64[us]")
    result["event_end"] = pd.Series(pd.NaT, index=result.index, dtype="datetime64[us]")
    result["event_duration_hours"] = pd.Series(pd.NA, index=result.index, dtype="Int32")
    result["event_represented_hours"] = pd.Series(pd.NA, index=result.index, dtype="Int32")

    for event in events.itertuples(index=False):
        mask = (result["hour_key"] >= event.event_start) & (result["hour_key"] < event.event_end)
        result.loc[mask, "event_active"] = True
        result.loc[mask, "event_id"] = event.event_id
        result.loc[mask, "event_type"] = event.event_type
        result.loc[mask, "event_severity"] = event.event_severity
        result.loc[mask, "event_start"] = event.event_start
        result.loc[mask, "event_end"] = event.event_end
        result.loc[mask, "event_duration_hours"] = event.event_duration_hours
        result.loc[mask, "event_represented_hours"] = event.event_represented_hours
    return result


def schema_metadata(state: pd.DataFrame) -> list[dict[str, Any]]:
    descriptions = {
        "is_weekend": "True on Saturday or Sunday in the timestamp calendar.",
        "is_public_holiday": "True on US federal holidays observed under pandas USFederalHolidayCalendar rules.",
        "weather_pressure_index": "Transparent Agent-facing wind/wave pressure indicator; not an official metric.",
        "event_active": "True when the hour falls within a generated synthetic event interval.",
        "event_id": "Unique synthetic event identifier, repeated over active hourly rows.",
        "event_type": "Synthetic event category.",
        "event_severity": "Synthetic severity: minor, moderate, or severe.",
        "event_start": "Inclusive synthetic event start timestamp.",
        "event_end": "Exclusive synthetic event end timestamp.",
        "event_duration_hours": "Elapsed clock-hour duration between event_start and event_end.",
        "event_represented_hours": "Number of authoritative hourly rows present within the event interval.",
    }
    base_descriptions = {
        "hour_key": "Hourly timestamp key from the authoritative base dataset.",
        "wind_speed_10m": "Wind speed at 10 meters, preserved exactly from the base dataset.",
        "wave_height": "Wave height, preserved exactly from the base dataset.",
        "weather_code": "Uninterpreted numeric weather code, preserved exactly from the base dataset.",
        "hour": "Hour-of-day integer preserved exactly from the base dataset.",
        "hour_sin": "Sine encoding of hour-of-day preserved exactly from the base dataset.",
        "hour_cos": "Cosine encoding of hour-of-day preserved exactly from the base dataset.",
        "day_of_week": "Day-of-week integer preserved exactly from the base dataset.",
        "month": "Month integer preserved exactly from the base dataset.",
    }
    descriptions.update(base_descriptions)
    event_fields = {
        "event_active", "event_id", "event_type", "event_severity",
        "event_start", "event_end", "event_duration_hours", "event_represented_hours",
    }
    dependencies = {
        "is_weekend": ["hour_key"],
        "is_public_holiday": ["hour_key"],
        "weather_pressure_index": ["wind_speed_10m", "wave_height"],
    }
    arrow_schema = pa.Schema.from_pandas(state[FINAL_COLUMNS], preserve_index=False)
    entries = []
    for field in arrow_schema:
        name = field.name
        provenance = "SYNTHETIC" if name in event_fields else "DERIVED"
        if name in {"hour_key", "wind_speed_10m", "wave_height", "weather_code"}:
            provenance = "OBSERVED"
        variable_type = "EVENT" if name in event_fields else "CONTEXT"
        if name == "weather_pressure_index":
            variable_type = "DERIVED"
        if name in BASE_COLUMNS:
            variable_type = "STATE"
        notes = ""
        if name == "weather_pressure_index":
            notes = (
                "Agent-facing only; not an official meteorological or port metric. "
                "If used for ML, fit percentile thresholds on the training period only."
            )
        elif name in event_fields:
            notes = "Synthetic MVP feature generated without congestion, vessel, cargo, prediction, or future information."
        elif name == "is_public_holiday":
            notes = "Historical US/LA-LB context only; excludes Saudi holidays, Ramadan, and Eid."
        elif name == "weather_code":
            notes = "Coding system remains unverified and is not mapped to labels."
        entries.append(
            {
                "feature_name": name,
                "description": descriptions[name],
                "dtype": str(field.type),
                "unit": "unitless" if name in {"weather_pressure_index", "hour_sin", "hour_cos"} else None,
                "source": str(BASE_PATH.relative_to(ROOT)) if name in BASE_COLUMNS else "Stage 2 deterministic builder",
                "provenance": provenance,
                "variable_type": variable_type,
                "dependencies": dependencies.get(name, ["hour_key"] if name in event_fields else []),
                "constraints": {"ml_candidate": False} if name == "weather_pressure_index" or name in event_fields else {},
                "missing_value_count": int(state[name].isna().sum()),
                "unique_value_count": int(state[name].nunique(dropna=True)),
                "agent_visibility": ["events_weather_agent"],
                "ml_candidate": not (name == "weather_pressure_index" or name in event_fields),
                "notes": notes,
            }
        )
    return entries


def assumptions_markdown(events: pd.DataFrame, holidays: pd.Series) -> str:
    holiday_lines = "\n".join(
        f"- {timestamp.date().isoformat()}: {name}" for timestamp, name in holidays.items()
    )
    config_lines = "\n".join(
        f"- `{name}`: start probability {values['probability']:.5f} per eligible hour; "
        f"base duration {values['duration'][0]}-{values['duration'][1]} hours."
        for name, values in EVENT_CONFIG.items()
    )
    return f"""# Events & Weather Assumptions

## Scope and provenance

The authoritative timeline is `data/processed/events_weather_base.parquet`. Its 22,534 rows and all nine base columns are preserved exactly; the 25 timeline gaps are not filled. Missing weather values are retained without imputation. Synthetic events are MVP simulation data, not verified historical events at the Ports of Los Angeles or Long Beach.

## Calendar

`is_weekend` is true for Saturday and Sunday. `is_public_holiday` uses `pandas.tseries.holiday.USFederalHolidayCalendar`, a reproducible US federal holiday calendar appropriate as general historical context for Los Angeles/Long Beach. Holidays use their calendar rule's observed-date convention: fixed-date holidays falling on Saturday are observed Friday, and those falling on Sunday are observed Monday. This is not a port closure calendar; an observed federal holiday does not imply either port was closed.

The exact observed holiday dates present in this dataset are:

{holiday_lines}

Saudi holidays, Ramadan, and Eid are intentionally excluded.

## Weather pressure index

This transparent Agent-facing indicator is not an official meteorological or port metric and is not currently an ML candidate.

```text
wind_component = clip((wind_speed_10m - 10.80) / (16.90 - 10.80), 0, 1)
wave_component = clip((wave_height - 1.02) / (1.58 - 1.02), 0, 1)
weather_pressure_index = max(wind_component, wave_component)
```

The thresholds are the full-base p75/p95 values used for this MVP. The index remains null if either input is null. If this feature later becomes an ML input, these percentile thresholds must be fitted on the training period only and then applied unchanged to validation/test periods.

## Synthetic events

Generation uses NumPy `default_rng` with fixed seed `{SEED}`. Events have an inclusive start and exclusive end, cannot overlap, and have a random 4-12 hour cooldown. Severity probabilities are minor 55%, moderate 35%, and severe 10%; duration multipliers are 0.8, 1.0, and 1.4 respectively.

{config_lines}

`demand_surge` start probability is multiplied by 2.5 within two calendar days of an observed holiday. Weekend `landside_access_disruption` probability is multiplied by 0.65. These modifiers use only timestamp-derived calendar context.

Synthetic generation does not read or use congestion labels, `vessels_waiting`, cargo state, ML predictions, future vessel state, future targets, or any future information. Event fields are Agent-facing and have `ml_candidate=false`.

`event_duration_hours` is elapsed clock time from the inclusive start to the exclusive end. `event_represented_hours` is the number of rows actually present in the authoritative timeline within that interval. The represented count can be lower only when an event crosses a preserved source timeline gap; this is expected and is validated explicitly.

## Limitations

- The event rates, durations, severities, and calendar effects are explicit simulation assumptions, not empirical estimates.
- Federal holiday context is not a substitute for a verified port operating calendar.
- Weather units and `weather_code` semantics remain unverified in local project documentation; numeric base values are preserved without reinterpretation.
- Timeline gaps mean an event's clock duration can exceed the number of represented hourly rows when source hours are absent.

Generated events in this build: {len(events)}.
"""


def validate(
    base: pd.DataFrame,
    state: pd.DataFrame,
    events: pd.DataFrame,
    base_hash_before: str,
    repeated_events: pd.DataFrame,
) -> dict[str, Any]:
    if len(base) != 22534 or len(state) != len(base):
        raise AssertionError("Base and final row counts must both equal 22,534")
    if list(state.columns) != FINAL_COLUMNS:
        raise AssertionError("Unexpected final column order")
    if state["hour_key"].duplicated().any():
        raise AssertionError("Duplicate final timestamps")
    base_gap_boundaries = int(base["hour_key"].diff().gt(pd.Timedelta(hours=1)).sum())
    final_gap_boundaries = int(state["hour_key"].diff().gt(pd.Timedelta(hours=1)).sum())
    if base_gap_boundaries != 25 or final_gap_boundaries != base_gap_boundaries:
        raise AssertionError("Final timeline must preserve all 25 source gap boundaries")
    assert_frame_equal(state[BASE_COLUMNS], base[BASE_COLUMNS], check_exact=True, check_dtype=True)
    if sha256(BASE_PATH) != base_hash_before:
        raise AssertionError("Authoritative base parquet changed during build")
    assert_frame_equal(events, repeated_events, check_exact=True, check_dtype=True)
    if not events.empty and (events["event_start"].shift(-1) < events["event_end"]).iloc[:-1].any():
        raise AssertionError("Synthetic event intervals overlap")
    expected_active = pd.Series(False, index=state.index)
    events_crossing_gaps: list[dict[str, Any]] = []
    event_metadata_fields = [
        "event_id", "event_type", "event_severity", "event_start", "event_end",
        "event_duration_hours", "event_represented_hours",
    ]
    for event in events.itertuples(index=False):
        event_mask = (state["hour_key"] >= event.event_start) & (state["hour_key"] < event.event_end)
        expected_active |= event_mask
        if event.event_duration_hours != int((event.event_end - event.event_start) / pd.Timedelta(hours=1)):
            raise AssertionError("Event duration is inconsistent")
        represented = int(event_mask.sum())
        if event.event_represented_hours != represented:
            raise AssertionError(f"Represented-hour count is inconsistent for {event.event_id}")
        event_rows = state.loc[event_mask, event_metadata_fields]
        for field in event_metadata_fields:
            if not event_rows[field].eq(getattr(event, field)).all():
                raise AssertionError(f"Incorrect {field} metadata on represented rows for {event.event_id}")
        expected_timestamps = pd.date_range(
            event.event_start,
            event.event_end - pd.Timedelta(hours=1),
            freq="h",
        )
        missing_timestamps = expected_timestamps.difference(pd.DatetimeIndex(base["hour_key"]))
        if event.event_duration_hours - represented != len(missing_timestamps):
            raise AssertionError(f"Unexplained elapsed/represented mismatch for {event.event_id}")
        if len(missing_timestamps):
            events_crossing_gaps.append(
                {
                    "event_id": event.event_id,
                    "elapsed_hours": event.event_duration_hours,
                    "represented_hours": represented,
                    "missing_source_timestamps": [str(value) for value in missing_timestamps],
                }
            )
    if not expected_active.equals(state["event_active"]):
        raise AssertionError("Hourly event flags do not match event intervals")
    missing_weather = base[["wind_speed_10m", "wave_height"]].isna().any(axis=1)
    if not state.loc[missing_weather, "weather_pressure_index"].isna().all():
        raise AssertionError("Weather pressure must be null when inputs are missing")
    return {
        "base_rows": len(base),
        "final_rows": len(state),
        "source_gap_boundaries": base_gap_boundaries,
        "final_gap_boundaries": final_gap_boundaries,
        "timeline_gaps_filled": 0,
        "duplicate_timestamps": int(state["hour_key"].duplicated().sum()),
        "base_sha256_unchanged": True,
        "base_columns_exactly_equal": True,
        "event_generation_reproducible": True,
        "event_overlaps": 0,
        "events_crossing_source_gaps": events_crossing_gaps,
        "unexplained_event_mapping_mismatches": 0,
    }


def main() -> None:
    base_hash_before = sha256(BASE_PATH)
    base = pd.read_parquet(BASE_PATH)
    if list(base.columns) != BASE_COLUMNS:
        raise RuntimeError(f"Unexpected base columns: {list(base.columns)}")

    flags, holidays = build_calendar(base["hour_key"])
    state = base.copy()
    state["is_weekend"] = flags["is_weekend"].astype(bool)
    state["is_public_holiday"] = flags["is_public_holiday"].astype(bool)
    state["weather_pressure_index"] = weather_pressure(base)
    schedule_columns = [
        "event_id", "event_type", "event_severity", "event_start", "event_end",
        "event_duration_hours", "generation_seed",
    ]
    existing_schedule = pd.read_parquet(EVENTS_PATH)[schedule_columns] if EVENTS_PATH.exists() else None
    events = add_represented_hours(generate_events(base["hour_key"], flags), base["hour_key"])
    repeated_events = add_represented_hours(generate_events(base["hour_key"], flags), base["hour_key"])
    if existing_schedule is not None:
        assert_frame_equal(
            events[schedule_columns], existing_schedule, check_exact=True, check_dtype=False
        )
    state = attach_events(state, events)[FINAL_COLUMNS]

    validation = validate(base, state, events, base_hash_before, repeated_events)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ASSUMPTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    state.to_parquet(STATE_PATH, index=False, compression="zstd")
    events.to_parquet(EVENTS_PATH, index=False, compression="zstd")
    SCHEMA_PATH.write_text(json.dumps(schema_metadata(state), indent=2), encoding="utf-8")
    ASSUMPTIONS_PATH.write_text(assumptions_markdown(events, holidays), encoding="utf-8")

    # Verify persisted artifacts and immutable source one final time.
    persisted_state = pd.read_parquet(STATE_PATH)
    persisted_events = pd.read_parquet(EVENTS_PATH)
    assert_frame_equal(persisted_state, state, check_exact=True, check_dtype=True)
    assert_frame_equal(persisted_events, events, check_exact=True, check_dtype=True)
    if sha256(BASE_PATH) != base_hash_before:
        raise AssertionError("Base parquet changed while outputs were written")

    summary = {
        "validation": validation,
        "columns": FINAL_COLUMNS,
        "event_count": len(events),
        "active_event_percentage": float(state["event_active"].mean() * 100),
        "holiday_dates": int(holidays.size),
        "holiday_rows": int(state["is_public_holiday"].sum()),
        "output_paths": [str(path) for path in [STATE_PATH, EVENTS_PATH, SCHEMA_PATH, ASSUMPTIONS_PATH]],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
