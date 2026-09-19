"""Provenance categories used across Marsa."""

from __future__ import annotations

from enum import StrEnum


class DataProvenance(StrEnum):
    OBSERVED = "OBSERVED"
    DERIVED = "DERIVED"
    SYNTHETIC = "SYNTHETIC"
    PREDICTED = "PREDICTED"
    SIMULATED = "SIMULATED"
    AGENT_GENERATED = "AGENT_GENERATED"

