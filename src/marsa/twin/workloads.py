"""Reproducible retrospective service-effort proxies, keyed by AIS call_id.

These are not container quantities or an as-of cargo forecast. Availability is
recorded explicitly because a completed call's duration contains hindsight.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from marsa.common.exceptions import DataNotReadyError
from marsa.config import DATA_DIR
from marsa.twin.calls import CALLS_PATH

WORKLOADS_PATH = DATA_DIR / "processed" / "vessel_cargo_workloads.csv"
UNITS = "normalized_service_effort"
METHOD = "recorded_berth_hours_reference_v1"
HASH_NORMALIZATION = "csv_crlf_to_lf_v1"
TRACE_COLUMNS = ["call_id", "vessel_id", "mmsi", "imo", "length_m", "arrival", "berth_start", "berth_end"]


def sha256(path: Path) -> str:
    """Hash CSV content consistently across Git LF/CRLF checkouts.

    Only CRLF line endings are canonicalized; all other bytes remain significant.
    Files are never rewritten by provenance validation.
    """
    content = path.read_bytes()
    if path.suffix.lower() == ".csv":
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def derive_workloads(calls: pd.DataFrame) -> pd.DataFrame:
    """Materialize the validated Twin's reference effort without altering AIS."""
    if not set(TRACE_COLUMNS + ["berth_hours"]) <= set(calls):
        raise ValueError("AIS call table is missing workload source fields")
    if calls.call_id.isna().any() or calls.call_id.duplicated().any():
        raise ValueError("AIS call_id must be present and unique")
    hours = pd.to_numeric(calls.berth_hours, errors="raise")
    if not np.isfinite(hours).all() or (hours <= 0).any():
        raise ValueError("reference berth durations must be finite and positive")
    result = calls[TRACE_COLUMNS].copy()
    for column in ("arrival", "berth_start", "berth_end"):
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
    measured = (result.berth_end - result.berth_start).dt.total_seconds() / 3600
    if (result.berth_start < result.arrival).any() or not np.allclose(measured, hours, atol=1e-6, rtol=0):
        raise ValueError("AIS service timestamps do not agree with reference durations")
    result["estimated_workload"] = hours
    result["units"] = UNITS
    result["method"] = METHOD
    result["provenance"] = "ESTIMATED_RETROSPECTIVE_PROXY"
    result["available_at_utc"] = result.berth_end
    return result.sort_values("call_id").reset_index(drop=True)


def write_workloads(source: Path, output: Path) -> dict:
    if source.resolve() == output.resolve():
        raise ValueError("derived output must not overwrite the AIS source")
    table = derive_workloads(pd.read_csv(source))
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, lineterminator="\n", float_format="%.9g")
    manifest = {"schema_version": 1, "hash_normalization": HASH_NORMALIZATION,
                "source_file": source.name, "source_sha256": sha256(source),
                "dataset_sha256": sha256(output), "records": len(table), "units": UNITS,
                "method": METHOD, "formula": "estimated_workload = recorded berth_hours * 1 unit/hour",
                "temporal_scope": "RETROSPECTIVE; available_at_utc = berth_end; not an as-of forecast",
                "source_fields_reviewed": list(pd.read_csv(source, nrows=0).columns)}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


@lru_cache(maxsize=4)
def load_workloads(path: Path = WORKLOADS_PATH, source: Path = CALLS_PATH) -> tuple[pd.DataFrame, dict]:
    try:
        manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        table = pd.read_csv(path)
    except FileNotFoundError as error:
        raise DataNotReadyError("Vessel cargo workload source unavailable; run scripts/build_vessel_cargo_workloads.py") from error
    if not source.is_file():
        raise DataNotReadyError("AIS source required to validate vessel cargo workloads is unavailable")
    required = set(TRACE_COLUMNS + ["estimated_workload", "units", "method", "provenance", "available_at_utc"])
    if not required <= set(table):
        raise ValueError("vessel cargo workload fields are incomplete")
    if not isinstance(manifest, dict):
        raise ValueError("vessel cargo workload manifest must be an object")
    if (manifest.get("schema_version") != 1 or manifest.get("hash_normalization") != HASH_NORMALIZATION
            or manifest.get("method") != METHOD
            or manifest.get("units") != UNITS or manifest.get("records") != len(table)
            or manifest.get("dataset_sha256") != sha256(path)
            or manifest.get("source_sha256") != sha256(source)):
        raise ValueError("vessel cargo workload provenance is invalid or stale; regenerate derived data")
    # Validate the whole derived artifact against its deterministic source contract.
    expected = derive_workloads(pd.read_csv(source))
    for column in ("arrival", "berth_start", "berth_end", "available_at_utc"):
        table[column] = pd.to_datetime(table[column], utc=True, errors="raise")
    try:
        pd.testing.assert_frame_equal(table.reset_index(drop=True), expected, check_dtype=False, rtol=1e-8)
    except AssertionError as error:
        raise ValueError("vessel cargo workload mapping does not match its AIS source") from error
    return table.set_index("call_id"), manifest


def reference_workload(row: pd.Series, decision_time: pd.Timestamp, *, retrospective: bool = False) -> float:
    """As-of users must explicitly reject unavailable reference-duration inputs."""
    if decision_time.tzinfo is None or decision_time.utcoffset().total_seconds() != 0:
        raise ValueError("workload decision time must be explicit UTC")
    if not retrospective and row.available_at_utc > decision_time:
        raise ValueError("retrospective vessel workload was not available as of the decision time")
    return float(row.estimated_workload)


def attach_workloads(scenario, ts: pd.Timestamp) -> dict:
    table, manifest = load_workloads()
    mapped = {}
    for vessel in scenario.berthed + scenario.vessels:
        if vessel.call_id not in table.index:
            raise ValueError(f"missing vessel cargo workload for call_id {vessel.call_id}")
        row = table.loc[vessel.call_id]
        if int(row.mmsi) != vessel.mmsi:
            raise ValueError(f"vessel cargo identity mismatch for call_id {vessel.call_id}")
        total = reference_workload(row, ts, retrospective=True)
        mapped[vessel.call_id] = {
            "total_workload": total,
            "initial_workload": max(0., total + vessel.berth_h) if vessel.berth_h is not None else total,
            "method": METHOD, "provenance": "ESTIMATED_RETROSPECTIVE_PROXY",
            "available_at_utc": row.available_at_utc.isoformat(), "vessel_id": row.vessel_id,
        }
    scenario.workloads = mapped
    return {"input": "per-call vessel cargo reference workloads", "source": WORKLOADS_PATH.name,
            "source_sha256": manifest["source_sha256"], "dataset_sha256": manifest["dataset_sha256"],
            "records": manifest["records"], "units": UNITS, "provenance": "ESTIMATED_RETROSPECTIVE_PROXY",
            "method": METHOD, "temporal_scope": manifest["temporal_scope"]}
