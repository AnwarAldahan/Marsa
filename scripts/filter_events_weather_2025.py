"""Create immutable 2025-only inspection artifacts from finalized Events & Weather outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[1]
SOURCE_STATE = ROOT / "data/processed/events_weather_state.parquet"
SOURCE_EVENTS = ROOT / "data/processed/synthetic_events.parquet"
SOURCE_SCHEMA = ROOT / "data/processed/events_weather_schema.json"

STATE_PARQUET = ROOT / "data/processed/events_weather_state_2025.parquet"
EVENTS_PARQUET = ROOT / "data/processed/synthetic_events_2025.parquet"
SCHEMA_JSON = ROOT / "data/processed/events_weather_schema_2025.json"
STATE_CSV = ROOT / "data/processed/events_weather_state_2025.csv"
EVENTS_CSV = ROOT / "data/processed/synthetic_events_2025.csv"

YEAR_START = pd.Timestamp("2025-01-01 00:00:00")
YEAR_END = pd.Timestamp("2026-01-01 00:00:00")
EXPECTED_HOURS = 8760


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_like_parquet(csv_path: Path, reference: pd.DataFrame) -> pd.DataFrame:
    datetime_columns = [name for name in ["hour_key", "event_start", "event_end"] if name in reference]
    string_columns = [name for name in ["event_id", "event_type", "event_severity"] if name in reference]
    boolean_columns = [
        name for name in ["is_weekend", "is_public_holiday", "event_active"] if name in reference
    ]
    integer_columns = [
        name
        for name in [
            "hour", "day_of_week", "month", "event_duration_hours",
            "event_represented_hours", "event_represented_hours_2025", "generation_seed",
        ]
        if name in reference
    ]
    dtypes: dict[str, str] = {name: "string" for name in string_columns}
    dtypes.update({name: "boolean" for name in boolean_columns})
    for name in integer_columns:
        dtypes[name] = "Int32" if reference[name].isna().any() else "int64"
    return pd.read_csv(
        csv_path,
        dtype=dtypes,
        parse_dates=datetime_columns,
        float_precision="round_trip",
    )[reference.columns]


def build_schema(state_2025: pd.DataFrame, events_2025: pd.DataFrame) -> list[dict[str, Any]]:
    schema = json.loads(SOURCE_SCHEMA.read_text(encoding="utf-8"))
    if [entry["feature_name"] for entry in schema] != list(state_2025.columns):
        raise AssertionError("Finalized source schema does not match finalized state columns")

    for entry in schema:
        name = entry["feature_name"]
        entry["missing_value_count"] = int(state_2025[name].isna().sum())
        entry["unique_value_count"] = int(state_2025[name].nunique(dropna=True))
        entry["source"] = "data/processed/events_weather_state.parquet (2025 row filter only)"
        scope_note = "2025-only scope; retained values are exact copies of the finalized 2023-2025 state."
        entry["notes"] = " ".join(part for part in [entry.get("notes", ""), scope_note] if part)
        if name == "weather_pressure_index":
            entry["notes"] += (
                " Thresholds remain those derived from the full 2023-2025 historical weather "
                "baseline; they were not refitted to 2025."
            )
        if name == "event_represented_hours":
            entry["notes"] += (
                " This preserves the original event-level represented count. All retained events "
                "are wholly inside 2025; the 2025 event registry also exposes "
                "event_represented_hours_2025 explicitly."
            )
        entry["table_scope"] = ["events_weather_state_2025"]
    schema.append(
        {
            "feature_name": "event_represented_hours_2025",
            "description": "Number of authoritative 2025 hourly rows present within the event interval.",
            "dtype": str(events_2025["event_represented_hours_2025"].dtype),
            "unit": "hours represented as rows",
            "source": "Derived by counting finalized 2025 state rows within the original event interval",
            "provenance": "DERIVED",
            "variable_type": "EVENT",
            "dependencies": ["hour_key", "event_start", "event_end"],
            "constraints": {"ml_candidate": False},
            "missing_value_count": int(events_2025["event_represented_hours_2025"].isna().sum()),
            "unique_value_count": int(events_2025["event_represented_hours_2025"].nunique()),
            "agent_visibility": ["events_weather_agent"],
            "ml_candidate": False,
            "table_scope": ["synthetic_events_2025"],
            "notes": (
                "Registry-only 2025 scope count. It does not alter event_start, event_end, "
                "event_duration_hours, or the original event_represented_hours definition."
            ),
        }
    )
    return schema


def validate(
    source_state: pd.DataFrame,
    source_events: pd.DataFrame,
    state_2025: pd.DataFrame,
    events_2025: pd.DataFrame,
) -> dict[str, Any]:
    source_subset = source_state.loc[
        (source_state["hour_key"] >= YEAR_START) & (source_state["hour_key"] < YEAR_END)
    ].copy().reset_index(drop=True)
    assert_frame_equal(state_2025, source_subset, check_exact=True, check_dtype=True)

    if not state_2025["hour_key"].between(YEAR_START, YEAR_END, inclusive="left").all():
        raise AssertionError("2025 state contains an out-of-scope timestamp")
    if state_2025["hour_key"].duplicated().any():
        raise AssertionError("2025 state contains duplicate timestamps")
    if not pd.Index(state_2025["hour_key"]).isin(pd.Index(source_state["hour_key"])).all():
        raise AssertionError("2025 hour_key is not an exact source-timeline subset")

    expected = pd.date_range(YEAR_START, YEAR_END - pd.Timedelta(hours=1), freq="h")
    missing_source_hours = expected.difference(pd.DatetimeIndex(state_2025["hour_key"]))
    if len(state_2025) + len(missing_source_hours) != EXPECTED_HOURS:
        raise AssertionError("2025 timeline accounting does not equal 8,760 hours")

    retained_source_events = source_events.loc[
        (source_events["event_start"] < YEAR_END) & (source_events["event_end"] > YEAR_START)
    ].copy().reset_index(drop=True)
    source_event_columns = list(source_events.columns)
    assert_frame_equal(
        events_2025[source_event_columns], retained_source_events,
        check_exact=True, check_dtype=True,
    )

    crossing_gaps: list[dict[str, Any]] = []
    crossing_boundaries: list[str] = []
    for event in events_2025.itertuples(index=False):
        clipped_start = max(event.event_start, YEAR_START)
        clipped_end = min(event.event_end, YEAR_END)
        represented_mask = (
            (state_2025["hour_key"] >= clipped_start) &
            (state_2025["hour_key"] < clipped_end)
        )
        represented = int(represented_mask.sum())
        if event.event_represented_hours_2025 != represented:
            raise AssertionError(f"Incorrect 2025 represented count for {event.event_id}")
        rows = state_2025.loc[represented_mask]
        if len(rows) and not (
            rows["event_active"].all()
            and rows["event_id"].eq(event.event_id).all()
            and rows["event_type"].eq(event.event_type).all()
            and rows["event_severity"].eq(event.event_severity).all()
            and rows["event_start"].eq(event.event_start).all()
            and rows["event_end"].eq(event.event_end).all()
            and rows["event_duration_hours"].eq(event.event_duration_hours).all()
            and rows["event_represented_hours"].eq(event.event_represented_hours).all()
        ):
            raise AssertionError(f"Hourly event metadata mismatch for {event.event_id}")

        scoped_expected = pd.date_range(
            clipped_start, clipped_end - pd.Timedelta(hours=1), freq="h"
        )
        event_missing = scoped_expected.difference(pd.DatetimeIndex(state_2025["hour_key"]))
        if len(event_missing):
            crossing_gaps.append(
                {
                    "event_id": event.event_id,
                    "missing_source_timestamps": [str(value) for value in event_missing],
                }
            )
        if event.event_start < YEAR_START or event.event_end > YEAR_END:
            crossing_boundaries.append(event.event_id)

    active_event_ids = set(state_2025.loc[state_2025["event_active"], "event_id"].dropna())
    if active_event_ids != set(events_2025["event_id"]):
        raise AssertionError("2025 event registry and active state rows disagree")

    return {
        "hourly_rows": len(state_2025),
        "expected_calendar_hours": EXPECTED_HOURS,
        "missing_source_hours": len(missing_source_hours),
        "first_timestamp": str(state_2025["hour_key"].min()),
        "last_timestamp": str(state_2025["hour_key"].max()),
        "unique_hour_keys": int(state_2025["hour_key"].nunique()),
        "duplicate_timestamps": int(state_2025["hour_key"].duplicated().sum()),
        "retained_events": len(events_2025),
        "event_active_rows": int(state_2025["event_active"].sum()),
        "active_event_percentage": float(state_2025["event_active"].mean() * 100),
        "events_crossing_source_gaps": crossing_gaps,
        "events_crossing_year_boundaries": crossing_boundaries,
        "hour_key_exact_source_subset": True,
        "all_retained_state_values_exact": True,
        "all_retained_event_definitions_exact": True,
    }


def main() -> None:
    source_hashes = {path: sha256(path) for path in [SOURCE_STATE, SOURCE_EVENTS, SOURCE_SCHEMA]}
    source_state = pd.read_parquet(SOURCE_STATE)
    source_events = pd.read_parquet(SOURCE_EVENTS)

    state_2025 = source_state.loc[
        (source_state["hour_key"] >= YEAR_START) & (source_state["hour_key"] < YEAR_END)
    ].copy().reset_index(drop=True)
    events_2025 = source_events.loc[
        (source_events["event_start"] < YEAR_END) & (source_events["event_end"] > YEAR_START)
    ].copy().reset_index(drop=True)
    events_2025["event_represented_hours_2025"] = [
        int(
            (
                (state_2025["hour_key"] >= max(event.event_start, YEAR_START))
                & (state_2025["hour_key"] < min(event.event_end, YEAR_END))
            ).sum()
        )
        for event in events_2025.itertuples(index=False)
    ]

    validation = validate(source_state, source_events, state_2025, events_2025)
    schema_2025 = build_schema(state_2025, events_2025)

    state_2025.to_parquet(STATE_PARQUET, index=False, compression="zstd")
    events_2025.to_parquet(EVENTS_PARQUET, index=False, compression="zstd")
    SCHEMA_JSON.write_text(json.dumps(schema_2025, indent=2), encoding="utf-8")
    state_2025.to_csv(STATE_CSV, index=False, date_format="%Y-%m-%d %H:%M:%S")
    events_2025.to_csv(EVENTS_CSV, index=False, date_format="%Y-%m-%d %H:%M:%S")

    persisted_state = pd.read_parquet(STATE_PARQUET)
    persisted_events = pd.read_parquet(EVENTS_PARQUET)
    assert_frame_equal(persisted_state, state_2025, check_exact=True, check_dtype=True)
    assert_frame_equal(persisted_events, events_2025, check_exact=True, check_dtype=True)
    assert_frame_equal(
        read_csv_like_parquet(STATE_CSV, state_2025), state_2025,
        check_exact=True, check_dtype=False,
    )
    assert_frame_equal(
        read_csv_like_parquet(EVENTS_CSV, events_2025), events_2025,
        check_exact=True, check_dtype=False,
    )
    if any(sha256(path) != value for path, value in source_hashes.items()):
        raise AssertionError("A finalized 2023-2025 source artifact was modified")

    validation["missing_values"] = {
        name: int(value) for name, value in state_2025.isna().sum().items()
    }
    validation["event_counts_by_type"] = {
        name: int(value) for name, value in events_2025["event_type"].value_counts().sort_index().items()
    }
    validation["event_counts_by_severity"] = {
        name: int(value) for name, value in events_2025["event_severity"].value_counts().sort_index().items()
    }
    validation["source_files_unchanged"] = True
    validation["outputs"] = [
        str(path) for path in [STATE_PARQUET, EVENTS_PARQUET, SCHEMA_JSON, STATE_CSV, EVENTS_CSV]
    ]
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
