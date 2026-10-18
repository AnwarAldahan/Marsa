"""Active Marsa API with three domain agents and one integrated decision flow."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from marsa.api.routes import cargo, decision_support, events_weather, flow, maritime, twin
from marsa.config import ROOT

app = FastAPI(
    title="Marsa",
    description="Predictive Port Digital Twin decision support with four agents.",
    version="1.0.0",
)

# Allow the local dashboard (dashboard/marsa.html) to call the API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "architecture": "four_agents_no_orchestrator"}


app.include_router(maritime.router)
app.include_router(cargo.router)
app.include_router(events_weather.router)
app.include_router(decision_support.router)
app.include_router(twin.router)
app.include_router(flow.router)


@app.get("/", include_in_schema=False)
def website():
    # the official dashboard (API-driven only); "Marsa updated.html" is kept in the repo but not served
    return FileResponse(ROOT / "dashboard" / "marsa.html", headers={"Cache-Control": "no-store"})
