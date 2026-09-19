# Marsa

Marsa is a predictive port Digital Twin with four-agent decision support for the
Port of Los Angeles/Long Beach 2025 MVP dataset.

The active architecture contains exactly four agents:

1. Maritime Agent
2. Cargo Agent
3. Events & Weather Agent
4. Strategy Agent

There is no Orchestrator Agent. ML and the Digital Twin are deterministic software
components, not agents. A human makes the final operational decision.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH="src"
python scripts/run_pipeline.py 2025-01-01T00:00:00Z
python -m uvicorn marsa.api.main:app --host 127.0.0.1 --port 8001
```

Integrated API request:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8001/api/decision-support/analyze `
  -ContentType application/json `
  -Body '{"timestamp_utc":"2025-01-01T00:00:00Z"}'
```

Individual domain endpoints remain available at:

- `POST /api/agents/maritime/investigate`
- `POST /api/agents/cargo/investigate`
- `POST /api/agents/events-weather/investigate`

## Verification

```powershell
python -m pytest -q
python -m ruff check src tests scripts
```

The trained XGBoost C1 artifact is preserved, but current target runtime data does
not provide every approved trained feature under the same semantics. The integrated
runtime therefore reports `ml_unavailable` instead of fabricating a prediction.

See [FINAL_INTEGRATION.md](docs/FINAL_INTEGRATION.md) for architecture, contracts,
provenance, assumptions, and limitations.
