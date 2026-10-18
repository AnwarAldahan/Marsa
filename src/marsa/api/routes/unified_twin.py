"""Website Hybrid workflow; legacy analyze/simulate contracts remain unchanged."""
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import Field, field_validator

from marsa.api.dependencies import get_strategy_provider
from marsa.common.exceptions import DataNotReadyError
from marsa.config import load_config
from marsa.data.load import parse_exact_utc_hour
from marsa.twin.coupled import Contract, CoupledParameters
from marsa.twin.sensitivity import SensitivityPlan
from marsa.twin.unified import (
    ContextUnavailable,
    UnifiedAction,
    UnifiedSettings,
    unified_analysis,
    unified_sensitivity,
)

router = APIRouter(prefix="/api/decision-support", tags=["unified-hybrid"])


class UnifiedRequest(Contract):
    timestamp_utc: datetime
    context_id: str | None = Field(default=None, min_length=32, max_length=32)
    parameters: CoupledParameters | None = None
    horizon_hours: int | None = Field(default=None, gt=0, le=720, strict=True)
    user_actions: list[UnifiedAction] | None = Field(default=None, max_length=3)

    @field_validator("timestamp_utc")
    @classmethod
    def decision_hour(cls, value):
        return parse_exact_utc_hour(value.isoformat()).to_pydatetime()


class FrozenSensitivityRequest(Contract):
    context_id: str = Field(min_length=32, max_length=32)
    plan: SensitivityPlan | None = None


def enabled_config():
    cfg = load_config()
    for key in ("unified_hybrid", "coupled_normalized", "hybrid_normalized"):
        if not isinstance(cfg.get(key), dict) or cfg[key].get("enabled") is not True:
            raise HTTPException(status_code=403, detail=f"{key} is disabled")
    try:
        settings = UnifiedSettings.model_validate(cfg["unified_hybrid"])
        if settings.parameters is None:
            raise ValueError("explicit unified demo parameters required")
    except ValueError as error:
        raise HTTPException(status_code=503, detail="Invalid unified scenario configuration") from error
    return cfg


@router.post("/unified")
def analyze(request: UnifiedRequest):
    cfg = enabled_config()
    try:
        return unified_analysis(request.timestamp_utc.isoformat(), cfg, context_id=request.context_id,
                                parameters=request.parameters, horizon_hours=request.horizon_hours,
                                user_actions=None if request.user_actions is None else [a.model_dump() for a in request.user_actions],
                                llm=None if request.context_id else get_strategy_provider())
    except ContextUnavailable as error:
        raise HTTPException(status_code=410, detail=str(error)) from error
    except (OSError, DataNotReadyError) as error:
        raise HTTPException(status_code=503, detail={"code": "SOURCE_UNAVAILABLE", "message": "Required unified simulation source is unavailable."}) from error
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Historical evidence unavailable for this hour") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ArithmeticError as error:
        raise HTTPException(status_code=503, detail="Unified simulation evidence validation failed") from error


@router.post("/unified/sensitivity")
def sensitivity(request: FrozenSensitivityRequest):
    cfg = enabled_config()
    if cfg.get("sensitivity_analysis", {}).get("enabled") is not True:
        raise HTTPException(status_code=403, detail="sensitivity_analysis is disabled")
    try:
        return unified_sensitivity(request.context_id, plan=request.plan)
    except ContextUnavailable as error:
        raise HTTPException(status_code=410, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ArithmeticError as error:
        raise HTTPException(status_code=503, detail="Sensitivity conservation validation failed") from error
