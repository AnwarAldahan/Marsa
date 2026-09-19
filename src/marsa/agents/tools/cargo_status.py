"""Exact-hour access to the teammate's frozen synthetic Cargo CSV."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from marsa.common.exceptions import DataNotReadyError
from marsa.common.paths import resolve_project_path


class CargoEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    containers_in_yard: float = Field(ge=0)
    yard_capacity: int = Field(ge=0)
    yard_occupancy_percent: float = Field(ge=0, le=100)
    import_teu: float = Field(ge=0)
    export_teu: float = Field(ge=0)
    container_arrivals_last_1h: float = Field(ge=0)
    container_departures_last_1h: float = Field(ge=0)
    average_dwell_time_hours: float = Field(ge=0)
    truck_waiting_time_minutes: float = Field(ge=0)
    gate_throughput_last_1h: int = Field(ge=0)
    cargo_flow_ratio: float = Field(ge=0)


SOURCE_FIELDS = {
    "containers_in_yard": "containers_in_yard",
    "yard_capacity": "yard_capacity",
    "yard_occupancy_percent": "yard_occupancy_percent",
    "import_teu": "import_teu",
    "export_teu": "export_teu",
    "container_arrivals_last_1h": "container_arrivals",
    "container_departures_last_1h": "container_departures",
    "average_dwell_time_hours": "average_dwell_time_hours",
    "truck_waiting_time_minutes": "truck_waiting_time_minutes",
    "gate_throughput_last_1h": "gate_throughput",
    "cargo_flow_ratio": "cargo_flow_ratio",
}
STATUSES = {"NORMAL", "MODERATE", "ELEVATED", "CRITICAL"}


class CargoStatusSource:
    def __init__(self, dataset_path: str | Path = "data/agent2_cargo_operations_2025.csv") -> None:
        self.dataset_path = resolve_project_path(dataset_path)
        self._rows: dict[datetime, dict[str, object]] | None = None

    def _load_rows(self) -> dict[datetime, dict[str, object]]:
        if self._rows is not None:
            return self._rows
        if not self.dataset_path.is_file():
            raise DataNotReadyError("Cargo source CSV is unavailable")
        try:
            frame = pd.read_csv(self.dataset_path)
            required = {"timestamp", "operations_status", "yard_capacity", *SOURCE_FIELDS.values()}
            if not required.issubset(frame.columns) or len(frame) != 8688 or frame.isna().any().any():
                raise ValueError("Cargo source schema, row count, or completeness is invalid")
            timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
            if (
                timestamps.duplicated().any()
                or not timestamps.is_monotonic_increasing
                or (timestamps.dt.year != 2025).any()
                or (timestamps.dt.floor("h") != timestamps).any()
                or not frame["operations_status"].isin(STATUSES).all()
            ):
                raise ValueError("Cargo source timestamps or statuses are invalid")
            rows = dict(zip(timestamps.dt.to_pydatetime(), frame.to_dict("records"), strict=True))
            for row in rows.values():
                CargoEvidence.model_validate(
                    {public: row[source] for public, source in SOURCE_FIELDS.items()}
                )
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise DataNotReadyError("Cargo source CSV failed validation") from error
        self._rows = rows
        return rows

    def get_status(self, timestamp_utc: datetime) -> tuple[str, CargoEvidence]:
        if timestamp_utc.tzinfo is None:
            raise ValueError("timestamp_utc must be timezone-aware")
        timestamp = timestamp_utc.astimezone(UTC)
        if timestamp.minute or timestamp.second or timestamp.microsecond:
            raise ValueError("timestamp_utc must be aligned to an exact hour")
        if timestamp.year != 2025:
            raise ValueError("timestamp_utc must be within calendar year 2025")
        row = self._load_rows().get(timestamp)
        if row is None:
            raise DataNotReadyError(f"Cargo hour unavailable: {timestamp.isoformat()}")
        evidence = CargoEvidence.model_validate(
            {public: row[source] for public, source in SOURCE_FIELDS.items()}
        )
        return str(row["operations_status"]), evidence
