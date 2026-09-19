"""HTTP transport for Cargo investigation."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from marsa.agents.cargo_agent import CargoInvestigationAgent, CargoResult
from marsa.api.dependencies import get_cargo_agent
from marsa.api.schemas.agents import CargoApiRequest
from marsa.common.exceptions import DataNotReadyError

router = APIRouter(prefix="/api/agents/cargo", tags=["cargo-agent"])


@router.post("/investigate", response_model=CargoResult)
def investigate_cargo(
    request: CargoApiRequest,
    agent: CargoInvestigationAgent = Depends(get_cargo_agent),
) -> CargoResult:
    try:
        return agent.investigate(request.to_agent_request())
    except DataNotReadyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (ValidationError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="Cargo investigation failed.") from error
