"""Independent opt-in endpoint; never runs the Strategy pipeline or changes replay."""
from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from marsa.common.exceptions import DataNotReadyError
from marsa.config import load_config
from marsa.twin.coupled import (
    CoupledParameters,
    CoupledScenario,
    CoupledSettings,
    ScenarioInputs,
    simulate_coupled,
)
from marsa.twin.hybrid import HybridRequest, HybridSettings, simulate_hybrid
from marsa.twin.sensitivity import SensitivityRequest, SensitivitySettings, analyze_sensitivity

router = APIRouter(prefix="/api/twin", tags=["digital-twin"])


class CoupledRequest(ScenarioInputs):
    parameters: CoupledParameters | None = None


@router.post("/simulate-coupled")
def simulate(request: CoupledRequest) -> dict:
    raw = load_config().get("coupled_normalized", {})
    if not isinstance(raw, dict):
        raise HTTPException(status_code=503, detail="Invalid coupled_normalized server configuration")
    if raw.get("enabled", False) is not True:
        raise HTTPException(status_code=403, detail="coupled_normalized is disabled; enable it explicitly in server configuration")
    try:
        settings = CoupledSettings.model_validate(raw)
    except ValidationError as error:
        raise HTTPException(status_code=503, detail="Invalid coupled_normalized server configuration") from error
    try:
        scenario = CoupledScenario.model_validate({
            **request.model_dump(),
            "parameters": request.parameters or settings.parameters,
        })
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return simulate_coupled(scenario)


@router.post("/simulate-hybrid")
def hybrid(request: HybridRequest) -> dict:
    raw = load_config().get("hybrid_normalized", {})
    if not isinstance(raw, dict):
        raise HTTPException(status_code=503, detail="Invalid hybrid_normalized server configuration")
    if raw.get("enabled", False) is not True:
        raise HTTPException(status_code=403, detail="hybrid_normalized is disabled; enable it explicitly in server configuration")
    try:
        settings = HybridSettings.model_validate(raw)
    except ValidationError as error:
        raise HTTPException(status_code=503, detail="Invalid hybrid_normalized server configuration") from error
    try:
        return simulate_hybrid(request, defaults=settings.workload)
    except (OSError, DataNotReadyError) as error:
        raise HTTPException(status_code=503, detail={"code": "SOURCE_UNAVAILABLE",
                            "message": "Required simulation source is unavailable."}) from error
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/analyze-sensitivity")
def sensitivity(request: SensitivityRequest) -> dict:
    cfg = load_config()
    raw = cfg.get("sensitivity_analysis", {})
    if not isinstance(raw, dict):
        raise HTTPException(status_code=503, detail="Invalid sensitivity_analysis configuration")
    if raw.get("enabled", False) is not True:
        raise HTTPException(status_code=403, detail="sensitivity_analysis is disabled; enable it explicitly on the server")
    try:
        settings = SensitivitySettings.model_validate(raw)
        workload = HybridSettings.model_validate(cfg.get("hybrid_normalized", {})).workload
    except ValidationError as error:
        raise HTTPException(status_code=503, detail="Invalid sensitivity or hybrid defaults") from error
    try:
        return analyze_sensitivity(request.scenario, plan=request.plan or settings.plan, defaults=workload)
    except (OSError, DataNotReadyError) as error:
        raise HTTPException(status_code=503, detail={"code": "SOURCE_UNAVAILABLE",
                            "message": "Required simulation source is unavailable."}) from error
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
