"""Active Marsa API with three domain agents and one integrated decision flow."""
from fastapi import FastAPI

from marsa.api.routes import cargo, decision_support, events_weather, maritime

app = FastAPI(
    title="Marsa",
    description="Predictive Port Digital Twin decision support with four agents.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "architecture": "four_agents_no_orchestrator"}


app.include_router(maritime.router)
app.include_router(cargo.router)
app.include_router(events_weather.router)
app.include_router(decision_support.router)
