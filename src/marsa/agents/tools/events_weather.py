"""Controlled read-only access to canonical Events & Weather context."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from marsa.agents.events_weather import EventsWeatherAgentInput, EventsWeatherContextBuilder
from marsa.common.exceptions import DataNotReadyError
from marsa.common.paths import resolve_project_path


class EventsWeatherContextSource:
    """Resolve an exact UTC hour from the authoritative 2025 timeline."""

    def __init__(
        self,
        path: str | Path = "data/processed/events_weather_state_2025.parquet",
        builder: EventsWeatherContextBuilder | None = None,
    ) -> None:
        self.path = resolve_project_path(path)
        self.builder = builder or EventsWeatherContextBuilder()
        self._state: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if self._state is None:
            if not self.path.exists():
                raise DataNotReadyError(f"Events & Weather state not found: {self.path}")
            state = pd.read_parquet(self.path)
            if state["hour_key"].duplicated().any():
                raise DataNotReadyError("Events & Weather state contains duplicate hour_key values")
            self._state = state.set_index("hour_key", drop=False)
        return self._state

    def get_context(self, timestamp_utc: datetime) -> EventsWeatherAgentInput:
        if timestamp_utc.tzinfo is None:
            raise ValueError("timestamp_utc must be timezone-aware")
        canonical = timestamp_utc.astimezone(UTC)
        if canonical.minute or canonical.second or canonical.microsecond:
            raise ValueError("timestamp_utc must be aligned to an exact hour")
        if canonical.year != 2025:
            raise ValueError("timestamp_utc must be within calendar year 2025")
        stored_key = pd.Timestamp(canonical.replace(tzinfo=None))
        state = self._load()
        if stored_key not in state.index:
            raise DataNotReadyError(
                f"Timestamp unavailable in authoritative timeline: {canonical.isoformat()}"
            )
        return self.builder.build(state.loc[stored_key].to_dict())
