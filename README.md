# Marsa (مرسَى) — Predictive Port Digital Twin · AgentX Hackathon MVP

Challenge: **Predicting cargo congestion and shipment delays in ports** (Ministry of Transport & Logistic Services track).

Marsa forecasts maritime congestion from real AIS data, lets three specialist agents diagnose *where*
the pressure is (sea, yard/gate, context), proposes candidate interventions, tests every candidate in a
digital twin, and returns a ranked, explainable recommendation.

```
merged hourly dataset (v2)
        │
        ├─► ML forecast (LightGBM)  ── waiting vessels in 6/12/24h + top drivers
        │
        ├─► Agent 1 Maritime (AIS)       ─┐
        ├─► Agent 2 Cargo / yard / gate   ├─► Strategy orchestrator ─► candidates
        └─► Agent 3 Context (weather+events)┘          │
                                                       ▼
                                          Digital twin (Monte Carlo simulation)
                                                       │
                                          scoring (documented weights) ─► ranking ─► recommendation
```

## Quick start
## Quick start

```bash
# 1. clone
git clone https://github.com/<your-username>/Marsa.git
cd Marsa

# 2. virtual environment
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# 3. dependencies
pip install -r requirements.txt

# 4. secrets (only if you use an LLM provider) — never commit this file
copy .env.example .env      # then put GEMINI_API_KEY=... inside

# 5. run
python scripts/train_model.py                       # trains the forecasters + baselines
python scripts/run_pipeline.py 2025-07-10T19:00     # full run for one hour -> outputs/run_20250710T19.json
python -m pytest tests                              # twin + feature sanity tests
```

> Note: the placeholder files (`HERE:` in `src/marsa/agents/*` and `src/marsa/model/*`) must be
> implemented before `train_model.py` and `run_pipeline.py` will run.

## Repository layout
| Path | What |
|---|---|
| `config/port_config.yaml` | **Every assumption in one place**, each tagged SOURCE / DERIVED / ASSUMPTION |
| `data/processed/merged_port_dataset_2025_v1.csv` | Pure merge of the 3 agent datasets (AIS, cargo, events+weather). Never edited. |
| `data/processed/merged_port_dataset_2025_v2.csv` | v1 + `pola_berthed_cargo_vessels`, `pola_berthed_tanker_vessels` (derived from raw AIS) |
| `data/processed/event_registry_2025.csv` | Synthetic event registry (agent 3) |
| `src/marsa/data/` | loading, snapshot, feature/target construction (time-based shifts, gap-safe) |
| `src/marsa/model/` | training with strict time split + persistence baselines; forecast with SHAP-style drivers |
| `src/marsa/agents/` | maritime / cargo / context agents (deterministic JSON), strategy orchestrator, LLM hook |
| `src/marsa/twin/` | simulation engine, scenario builder, composite scoring |
| `src/marsa/pipeline.py` | end-to-end run |
| `scripts/berth_occupancy_from_ais.py` | reproduces the berth columns from the raw Kaggle parquet files |
| `docs/` | data dictionary, architecture, limitations |

## Where the team's existing work plugs in
- **Prediction model**: replace/extend `src/marsa/model/train.py`; the pipeline only needs `forecast()` to return the same JSON.
- **Agents**: each agent is one `assess(snapshot, cfg) -> dict`. The team's `PortOperationsAgent` thresholds are already ported into `maritime_agent.py`; the Events & Weather agent (FastAPI + Gemini) can replace `context_agent.py` as long as it returns `twin_multipliers`.
- **LLM narrative**: implement `LLMProvider.complete()` in `agents/llm.py` (Gemini, Claude, …). Numbers never come from the LLM.
- **Digital-twin console (HTML)**: the JS engine's contract (`scenario` + `candidates`) maps 1:1 to `twin/engine.py`; the UI can read `outputs/run_*.json`.

## Results
_(fill after the team model is trained; see models/training_report.json)_
<!--
| horizon | persistence | Marsa (AIS only) | improvement |
|---|---|---|---|
| 6h | 1.30 | 1.27 | +2% |
| 12h | 1.70 | 1.57 | +8% |
| 24h | 1.99 | 1.78 | +10% |

-->
Adding the synthetic cargo columns does **not** improve accuracy (they are generated from AIS), so the
official numbers use AIS-only features. Synthetic data is used where it belongs: in the agents and the twin.

## Data provenance
- **OBSERVED**: AIS (NOAA via Kaggle *LA/LB Maritime Trajectory & AIS Dataset 2023-2025*), Open-Meteo weather.
- **DERIVED**: hourly AIS aggregates, berth occupancy, calendar features.
- **SYNTHETIC**: cargo/yard/gate operations (calibrated to official Port of Los Angeles 2025 monthly TEU and PMSA dwell times), external events.

Port of Los Angeles is used as a **stand-in** because Eastern Province port data was not available.
Everything port-specific lives in `config/port_config.yaml`; see `docs/LIMITATIONS.md` and `docs/NEXT_STEPS.md`.
