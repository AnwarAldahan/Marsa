"""Integrated decision-support endpoint."""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from marsa.agents.strategy_advisory import AdvisorySettings, run_advisory_experiment
from marsa.api.dependencies import get_strategy_provider
from marsa.api.routes.twin import AddedVessel, StressScenario, TwinAction, added_specs, require_utc
from marsa.common.exceptions import DataNotReadyError
from marsa.config import load_config
from marsa.pipeline import run
from marsa.twin.sensitivity import SensitivityRequest

router = APIRouter(prefix="/api/decision-support", tags=["decision-support"])


class DecisionSupportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamp_utc: datetime
    stress: StressScenario | None = None
    added: list[AddedVessel] = Field(default_factory=list)
    user_actions: list[TwinAction] | None = None   # the user's own decision, evaluated by the Strategy agent

    @field_validator("timestamp_utc")
    @classmethod
    def require_explicit_utc(cls, value: datetime) -> datetime:
        return require_utc(value)


@router.post("/analyze")
def analyze(request: DecisionSupportRequest) -> dict:
    try:
        stress = request.stress.model_dump() if request.stress else None
        return run(request.timestamp_utc.isoformat(), llm=get_strategy_provider(), stress=stress,
                   added=added_specs(request.added),
                   user_actions=(None if request.user_actions is None
                                 else [a.model_dump() for a in request.user_actions]))
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except DataNotReadyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="Decision-support analysis failed.") from error


class AdvisoryRequest(DecisionSupportRequest):
    experiment: SensitivityRequest | None = None


@router.post("/advisory")
def advisory(request: AdvisoryRequest) -> dict:
    cfg = load_config()
    raw = cfg.get("strategy_decision_support", {})
    if not isinstance(raw, dict) or raw.get("enabled", False) is not True:
        raise HTTPException(status_code=403, detail="strategy_decision_support is disabled; enable it explicitly on the server")
    try:
        AdvisorySettings.model_validate(raw)
        evidence = None
        if request.experiment is not None:
            if request.timestamp_utc != request.experiment.scenario.timestamp_utc:
                raise ValueError("advisory and experiment decision timestamps must match")
            try:
                evidence = run_advisory_experiment(request.experiment, cfg)
            except (OSError, DataNotReadyError) as error:
                raise HTTPException(status_code=503, detail={"code": "SOURCE_UNAVAILABLE",
                                    "message": "Required simulation source is unavailable."}) from error
        stress = request.stress.model_dump() if request.stress else None
        # Deterministic endpoint: no new Gemini calls. Existing /analyze is untouched.
        return run(request.timestamp_utc.isoformat(), cfg=cfg, llm=None, stress=stress,
                   added=added_specs(request.added), advisory_evidence=evidence,
                   user_actions=None if request.user_actions is None else [a.model_dump() for a in request.user_actions])
    except DataNotReadyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
