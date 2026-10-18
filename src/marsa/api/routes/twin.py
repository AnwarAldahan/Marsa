"""Digital Twin endpoints: run one decision the user picked or typed, and list known vessels.

POST /api/twin/simulate runs the twin at one hour twice: with no change (baseline) and with the
chosen actions, under the same stress scenario and added vessels. It returns both, the difference
and the sources. It does not rank or recommend: that is the Strategy agent's job.
GET /api/twin/vessels lists vessels with stays completed before the hour (for adding a known vessel).
GET /api/twin/hours lists real 2025 hours by condition (rain, drizzle, strongest wind or waves, active
event) from the Events & Weather file, so the user can test decisions on hours that really happened.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from marsa.common.exceptions import DataNotReadyError
from marsa.config import DATA_DIR, load_config
from marsa.data.load import parse_exact_utc_hour
from marsa.twin.adapter import build_scenario, scenario_summary
from marsa.twin.calls import ais_hours
from marsa.twin.engine import simulate_candidate
from marsa.twin.estimates import completed_before
from marsa.twin.scoring import score_results

router = APIRouter(prefix="/api/twin", tags=["digital-twin"])
WEATHER_PATH = DATA_DIR / "processed" / "events_weather_state_2025.parquet"
# WMO weather codes (Open-Meteo): 51-55 drizzle, 61-65 rain. Wind and waves have no threshold:
# the hours are listed from the strongest down, so no limit is invented.
CONDITIONS = {"rain": (61, 65), "drizzle": (51, 55)}


# Stress scenario chosen by the user; empty means replay the real AIS schedule.
class StressScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    berths_closed: int = Field(default=0, ge=0)
    arrival_multiplier: float = Field(default=1.0, gt=0.0)
    gate_closed_hours: int = Field(default=0, ge=0)              # yard stress: gate clears nothing
    container_inflow_multiplier: float = Field(default=1.0, gt=0.0)   # yard stress: x arrivals


# A vessel added by the user; its stay comes from its own history, similar lengths, or service_hours.
class AddedVessel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    arrival_in_hours: float = Field(default=0.0, ge=0.0)
    vessel_id: str | None = None
    length_m: float | None = Field(default=None, gt=0.0)
    service_hours: float | None = Field(default=None, gt=0.0)


# One twin action: queue_policy, add_berths or gate_capacity (checked by the twin engine).
class TwinAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    value: str | int | float


# Check that a timestamp is timezone-aware UTC.
def require_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp_utc must be timezone-aware and explicitly UTC")
    if value.utcoffset() != timedelta(0):
        raise ValueError("timestamp_utc must use UTC (offset +00:00 or Z)")
    return value


class SimulateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamp_utc: datetime
    actions: list[TwinAction] = Field(default_factory=list)
    stress: StressScenario | None = None
    added: list[AddedVessel] = Field(default_factory=list)
    twin_mode: Literal["replay", "forecast"] = "replay"   # forecast = only what is known at the hour

    @field_validator("timestamp_utc")
    @classmethod
    def utc_only(cls, value: datetime) -> datetime:
        return require_utc(value)


# Turn the request's added vessels into plain dicts, dropping fields the user left empty.
def added_specs(added: list[AddedVessel]) -> list[dict]:
    return [a.model_dump(exclude_none=True) for a in added]


@router.post("/simulate")
def simulate(request: SimulateRequest) -> dict:
    try:
        cfg = load_config()
        ts = parse_exact_utc_hour(request.timestamp_utc.isoformat())
        stress = request.stress.model_dump() if request.stress else None
        scenario, sources = build_scenario(ts, cfg, stress, added_specs(request.added),
                                           mode=request.twin_mode)
        actions = [a.model_dump() for a in request.actions]
        baseline = simulate_candidate(scenario, {"id": "baseline", "actions": []})
        tried = simulate_candidate(scenario, {"id": "user_decision", "actions": actions})
        compared = {r["candidate_id"]: r for r in score_results([baseline, tried])}
    except DataNotReadyError as error:   # a required data file is missing -> clear 404, not a crash
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {
        "timestamp_utc": ts.isoformat(),
        "simulation_horizon_hours": scenario.horizon_hours,
        "marker_hours": [int(cfg["ml_contract"]["horizon_hours"])],
        "scenario": scenario_summary(scenario),
        "sources": sources,
        "baseline": baseline,
        "result": {**tried, "vs_baseline": compared["user_decision"].get("vs_baseline", {})},
        "provenance": "SIMULATED",
        "human_approval_required": True,
    }


@router.get("/vessels")
def vessels(timestamp_utc: datetime) -> dict:
    try:
        ts = parse_exact_utc_hour(require_utc(timestamp_utc).isoformat())
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    history = completed_before(ts)
    table = (history.groupby("vessel_id")
             .agg(mmsi=("mmsi", "last"), length_m=("length_m", "median"),
                  stays=("call_id", "count"), median_stay_hours=("berth_hours", "median"))
             .reset_index().sort_values("stays", ascending=False))
    table = table.astype(object).where(pd.notna(table), None)
    return {"timestamp_utc": ts.isoformat(), "source": "la_cargo_calls_2023_2025.csv (AIS)",
            "vessels": table.to_dict(orient="records")}

@router.get("/hours")
def hours(condition: str, limit: int = 50) -> dict:
    if condition not in {"rain", "drizzle", "wind", "wave", "event"}:
        raise HTTPException(status_code=422, detail="condition must be rain, drizzle, wind, wave or event")
    w = pd.read_parquet(WEATHER_PATH)
    w["hour_key"] = pd.to_datetime(w["hour_key"], utc=True)
    w = w[w["hour_key"].isin(ais_hours())]          # only hours the twin can run
    if condition in CONDITIONS:
        lo, hi = CONDITIONS[condition]
        w = w[w["weather_code"].between(lo, hi)].sort_values("hour_key")
    elif condition == "wind":
        w = w.sort_values("wind_speed_10m", ascending=False)
    elif condition == "wave":
        w = w.sort_values("wave_height", ascending=False)
    else:
        w = w[w["event_active"]].sort_values("hour_key")
    total = len(w)
    rows = [{"timestamp_utc": r.hour_key.strftime("%Y-%m-%dT%H:00:00Z"),
             "weather_code": None if pd.isna(r.weather_code) else int(r.weather_code),
             "wind_speed_10m": None if pd.isna(r.wind_speed_10m) else float(r.wind_speed_10m),
             "wave_height": None if pd.isna(r.wave_height) else float(r.wave_height),
             "event_type": None if pd.isna(r.event_type) else str(r.event_type),
             "event_severity": None if pd.isna(r.event_severity) else str(r.event_severity)}
            for r in w.head(max(1, limit)).itertuples()]
    return {"condition": condition, "hours_found": total, "source": WEATHER_PATH.name,
            "provenance": "SYNTHETIC" if condition == "event" else "OBSERVED", "hours": rows}
