"""Dataset schema, feature governance, and domain grouping rules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


TIMESTAMP_COLUMN = "hour_key"
STATUS_COLUMN = "operations_status"
TARGET_COLUMN = "cargo_congestion"
POSITIVE_STATUS = {"ELEVATED", "CRITICAL"}
NEGATIVE_STATUS = {"NORMAL", "MODERATE"}
HORIZONS = (1, 3, 6, 12)
CONTEXT_LENGTHS = (24, 48, 72)
PRIMARY_RANDOM_SEED = 42

DEFAULT_DATA_PATH = Path("data/merged_port_dataset_2025.csv")
DEFAULT_OUTPUT_DIR = Path("prediction_model")

TIME_FEATURES = [
    "hour",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "dow_sin",
    "dow_cos",
    "month",
    "is_weekend",
    "is_public_holiday",
]

CARGO_FEATURES = [
    "import_teu",
    "export_teu",
    "container_arrivals",
    "container_departures",
    "containers_in_yard",
    "yard_capacity",
    "average_dwell_time_hours",
    "truck_arrivals",
    "truck_departures",
    "truck_waiting_time_minutes",
    "gate_throughput",
    "yard_occupancy_percent",
    "cargo_flow_ratio",
]

VESSEL_AIS_FEATURES = [
    "vessel_count",
    "waiting_vessel_count",
    "approaching_vessel_count",
    "departing_vessel_count",
    "average_speed",
    "port_throughput",
]

WEATHER_EVENT_FEATURES = [
    "wind_speed_10m",
    "wave_height",
    "weather_code",
    "weather_pressure_index",
    "event_active",
    "event_type",
    "event_severity",
    "event_duration_hours",
    "event_represented_hours",
]

IDENTIFIER_COLUMNS = {"event_id"}
TIMESTAMP_LIKE_COLUMNS = {TIMESTAMP_COLUMN, "event_start", "event_end"}
TARGET_DERIVED_COLUMNS = {STATUS_COLUMN, TARGET_COLUMN}


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    columns: tuple[str, ...]


def available(columns: list[str], requested: list[str]) -> list[str]:
    """Return requested columns that exist in the dataset."""
    present = set(columns)
    return [column for column in requested if column in present]


def infer_domain(column: str) -> str:
    if column in CARGO_FEATURES or column == STATUS_COLUMN:
        return "Cargo / Yard / Gate Operations"
    if column in VESSEL_AIS_FEATURES:
        return "Vessel / AIS State"
    if column in WEATHER_EVENT_FEATURES or column in {"event_id", "event_start", "event_end"}:
        return "Weather / External Conditions / Events"
    if column in TIME_FEATURES or column == TIMESTAMP_COLUMN:
        return "Temporal"
    if column.startswith("target_") or column == TARGET_COLUMN:
        return "Engineered target"
    return "Unknown"


def build_feature_governance(columns: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for column in columns:
        role = "A. usable predictor"
        include = "include"
        reason = "Current or historical information available at prediction time."

        if column == TIMESTAMP_COLUMN:
            role = "C. timestamp / temporal metadata"
            include = "exclude_raw"
            reason = "Raw timestamp is used for sorting, splits, gaps, and engineered time features."
        elif column == STATUS_COLUMN:
            role = "B. target / target-derived"
            include = "exclude_primary"
            reason = (
                "Defines the cargo_congestion label; primary models exclude it to avoid "
                "target-derived leakage."
            )
        elif column == TARGET_COLUMN or column.startswith("target_"):
            role = "B. target / target-derived"
            include = "exclude"
            reason = "Future label or derived target, never an input feature."
        elif column in IDENTIFIER_COLUMNS:
            role = "F. identifier / non-predictive"
            include = "exclude"
            reason = "Identifier has no stable predictive meaning for future operations."
        elif column in TIMESTAMP_LIKE_COLUMNS:
            role = "G. uncertain"
            include = "exclude_primary"
            reason = (
                "Event timing metadata can encode future event duration/end information; "
                "kept out of the primary leakage-safe feature set."
            )
        elif column in {"cargo_flow_ratio", "yard_occupancy_percent"}:
            role = "D. mathematically derived but potentially usable"
            include = "include"
            reason = (
                "Derived from current operational measurements. Correlation with the "
                "future rule-derived target is not automatically leakage when measured at t."
            )
        elif column in {"hour", "hour_sin", "hour_cos", "day_of_week", "dow_sin", "dow_cos", "month"}:
            role = "C. timestamp / temporal metadata"
            include = "include_engineered"
            reason = "Calendar information is known at prediction time."
        elif column in {"is_weekend", "is_public_holiday"}:
            role = "C. timestamp / temporal metadata"
            include = "include"
            reason = "Known calendar/event indicator at prediction time."
        elif column == "event_duration_hours":
            role = "G. uncertain"
            include = "include_with_caution"
            reason = (
                "Included only if the event schedule is assumed known. It is reported "
                "explicitly because it may be unavailable in real-time deployments."
            )

        rows.append(
            {
                "column": column,
                "domain/source": infer_domain(column),
                "role": role,
                "include_or_exclude": include,
                "reason": reason,
            }
        )
    return rows


def primary_feature_columns(columns: list[str]) -> list[str]:
    excluded = TARGET_DERIVED_COLUMNS | IDENTIFIER_COLUMNS | TIMESTAMP_LIKE_COLUMNS
    return [
        column
        for column in columns
        if column not in excluded and not column.startswith("target_")
    ]


def feature_groups(columns: list[str], include_operations_status: bool = False) -> list[FeatureGroup]:
    time_cols = available(columns, TIME_FEATURES)
    cargo = available(columns, CARGO_FEATURES)
    vessel = available(columns, VESSEL_AIS_FEATURES)
    weather = available(columns, WEATHER_EVENT_FEATURES)

    groups = [
        FeatureGroup("CargoOnly", tuple(time_cols + cargo)),
        FeatureGroup("CargoVesselAIS", tuple(time_cols + cargo + vessel)),
        FeatureGroup("CargoVesselWeatherEvents", tuple(time_cols + cargo + vessel + weather)),
        FeatureGroup("FullValidNoStatus", tuple(primary_feature_columns(columns))),
    ]
    if include_operations_status and STATUS_COLUMN in columns:
        groups.append(
            FeatureGroup(
                "DiagnosticFullWithCurrentStatus",
                tuple(primary_feature_columns(columns) + [STATUS_COLUMN]),
            )
        )
    return groups
