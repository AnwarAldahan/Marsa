"""Exact-hour adapter around the unchanged teammate PortOperationsAgent."""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from marsa.common.exceptions import DataNotReadyError
from marsa.common.paths import resolve_project_path


class MaritimeEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    vessels: int = Field(ge=0)
    waiting_vessels: int = Field(ge=0)
    waiting_ratio: float = Field(ge=0)
    approaching_vessels: int = Field(ge=0)
    departing_vessels: int = Field(ge=0)
    average_speed_value: float
    port_throughput_value: int


class MaritimeStatusSource:
    """Own loading and validation, but delegate all status logic to teammate code."""

    def __init__(
        self,
        code_path: str | Path = "data/port_agent.py",
        dataset_path: str | Path = "data/port_operations_agent_dataset.csv",
        core: Any | None = None,
    ) -> None:
        self.code_path = resolve_project_path(code_path)
        self.dataset_path = resolve_project_path(dataset_path)
        self._core = core

    def _load_core(self) -> Any:
        if self._core is None:
            if not self.code_path.is_file() or not self.dataset_path.is_file():
                raise DataNotReadyError("Maritime teammate code or dataset is unavailable")
            spec = importlib.util.spec_from_file_location("marsa_teammate_port_agent", self.code_path)
            if spec is None or spec.loader is None:
                raise DataNotReadyError("Maritime teammate code could not be loaded")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._core = module.PortOperationsAgent(str(self.dataset_path))
            if self._core.df["hour_key"].duplicated().any():
                raise DataNotReadyError("Maritime source contains duplicate hour keys")
        return self._core

    def get_status(self, timestamp_utc: datetime) -> tuple[str, MaritimeEvidence]:
        if timestamp_utc.tzinfo is None:
            raise ValueError("timestamp_utc must be timezone-aware")
        timestamp = timestamp_utc.astimezone(UTC)
        if timestamp.minute or timestamp.second or timestamp.microsecond:
            raise ValueError("timestamp_utc must be aligned to an exact hour")
        if timestamp.year != 2025:
            raise ValueError("timestamp_utc must be within calendar year 2025")

        core = self._load_core()
        # The teammate method floors timestamps; validation above prevents silent substitution.
        raw = core.get_status(timestamp.isoformat())
        if not isinstance(raw, dict) or "error" in raw:
            raise DataNotReadyError(f"Maritime hour unavailable: {timestamp.isoformat()}")
        try:
            returned = datetime.fromisoformat(raw["timestamp"])
            evidence = MaritimeEvidence(
                vessels=raw["vessels"],
                waiting_vessels=raw["waiting_vessels"],
                waiting_ratio=raw["waiting_ratio"],
                approaching_vessels=raw["approaching_vessels"],
                departing_vessels=raw["departing_vessels"],
                average_speed_value=raw["average_speed_knots"],
                port_throughput_value=raw["port_throughput_last_4h"],
            )
            status = raw["port_status"]
        except (KeyError, TypeError, ValueError) as error:
            raise DataNotReadyError("Maritime teammate result is incomplete") from error
        if returned.tzinfo is None or returned.astimezone(UTC) != timestamp:
            raise DataNotReadyError("Maritime teammate returned a different hour")
        if status not in {"NORMAL", "ELEVATED", "CONGESTED"}:
            raise DataNotReadyError("Maritime teammate returned an unknown status")
        return status, evidence
