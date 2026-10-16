# Marsa (مرسَى) — Predictive Port Digital Twin

**AgentX Hackathon · Logistics & Ports track · Challenge (Ministry of Transport and Logistic Services): *Predicting cargo congestion and shipment delays in ports.***

Marsa forecasts port congestion from real vessel-traffic (AIS) data, lets three specialist AI agents diagnose
*where* the pressure is (sea side, yard/gate, external context), asks a strategy agent to propose interventions,
tests every intervention in a digital twin, and returns a ranked recommendation with evidence.
Nothing is executed automatically — the port operator takes the final decision.

<p align="center"><img src="docs/marsa_flow.png" width="900" alt="Marsa flow"></p>

---

## 1. How Marsa works (one decision cycle)

| Step | Component | What it does | Where |
|---|---|---|---|
| 1 | **Port state** | One hour of the merged dataset is treated as "now" (replay of 2025) | `src/marsa/data/load.py` |
| 2 | **Prediction model** | XGBoost forecasts congestion risk in the next 6 h + risk level | `src/marsa/model/` · `prediction_model/` |
| 3 | **Agent 1 · Maritime** | Vessel queue and waiting ratio from AIS (LA + Long Beach, cargo + tankers) | `src/marsa/agents/maritime_agent.py` |
| 4 | **Agent 2 · Cargo** | Yard occupancy, container flow, dwell, truck/gate pressure | `src/marsa/agents/cargo_agent.py` |
| 5 | **Agent 3 · Context** | Observed weather, synthetic external events, local calendar | `src/marsa/agents/events_weather_agent.py` |
| 6 | **Strategy agent** | Keeps the 3 diagnoses + forecast; proposes actions only if the twin shows a berth shortage | `src/marsa/agents/strategy_agent.py` |
| 7 | **Digital twin** | Replays the real AIS schedule of Port of Los Angeles cargo vessels (arrival, berth and service time of every vessel) over 24 h; optional user stress scenario (berths closed, more arrivals); deterministic | `src/marsa/twin/` |
| 8 | **Strategy agent** | Ranks twin results by total vessel-hours waiting (no weights), writes the recommendation (Arabic, Gemini-assisted) | `src/marsa/agents/strategy_agent.py` |
| 9 | **Dashboard / API** | Shows state, forecast, drivers, recommendation | `dashboard/marsa.html` · `src/marsa/api/` |

Design rules: agents describe and diagnose, they never invent numbers; the LLM (Gemini) only rewrites narrative
and is rejected if it adds numbers or causal claims; every value carries provenance (`OBSERVED`, `DERIVED`,
`SYNTHETIC`, `SIMULATED`); the twin never picks a winner — ranking is total vessel-hours waiting;
the dashboard shows only values returned by the API (no demo or fallback numbers).

---

## 2. Run it — step by step

### Requirements
- Python **3.11 or newer** (`python --version`)
- Git
- (Optional) a Gemini API key from https://aistudio.google.com → *Get API key*

### Step 1 — clone
```bash
git clone https://github.com/AnwarAldahan/Marsa.git
cd Marsa
```

### Step 2 — create and activate a virtual environment
```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```
You should now see `(.venv)` at the start of the prompt.
If PowerShell refuses to run the script: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (once), then retry.

### Step 3 — install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — (optional) enable the LLM narrative
Without this step Marsa runs fully deterministic (same numbers, plainer text).
```powershell
copy .env.example .env        # macOS/Linux: cp .env.example .env
```
Open `.env` and set your key — no quotes, no spaces:
```
GEMINI_API_KEY=your_key_here
```
`.env` is git-ignored and must never be committed. Each should creates their own.

### Step 5 — check everything works
```bash
python -m pytest -q tests
python -m ruff check src tests scripts
```
Expected: all tests pass (≈ 240) and ruff reports no issues.

### Step 6 — run one decision cycle from the command line
```bash
python scripts/run_pipeline.py 2025-07-10T19:00:00Z --save
```
Timestamps must be **UTC with a trailing `Z`** and must exist in the 2025 timeline.
Output: the decision-support JSON in the terminal and `outputs/run_20250710T19.json`.

### Step 7 — start the backend (terminal 1)
```powershell
$env:PYTHONPATH="src"; python -m uvicorn marsa.api.main:app --reload      # Windows
```
```bash
PYTHONPATH=src python -m uvicorn marsa.api.main:app --reload             # macOS / Linux
```
Wait for `Application startup complete`. Swagger UI: http://127.0.0.1:8000/docs

### Step 8 — start the dashboard (terminal 2)
```bash
cd dashboard
python -m http.server 5500 --bind 127.0.0.1
```
Open **http://127.0.0.1:5500/marsa.html**.
Type a UTC hour of 2025 and press **تحليل**. Optional stress scenario: **أرصفة مغلقة** (berths closed) and
**زيادة السفن القادمة %** (the real arrival schedule of the next 24·(1+x) h is compressed into 24 h).
If the API is not running the page says so and shows no numbers.

### Step 9 — stop
`Ctrl+C` in each terminal. After editing Python files or `.env`, restart terminal 1 only;
after editing `marsa.html`, just refresh the browser.

### Good hours to demo
| Hour (UTC) | Stress | Why |
|---|---|---|
| `2025-07-10T19:00:00Z` | none | busy hour: 17 of 23 LA berths used, 1 vessel waiting for its berth time → "no intervention needed" |
| `2025-01-04T22:00:00Z` | none | 20 of 23 berths used, the highest in 2025 |
| `2025-01-03T04:00:00Z` | 6 berths closed, +100 % arrivals | berth shortage; all three options give different results |
| `2025-01-13T18:00:00Z` | none | active synthetic external event |
| `2025-05-14T06:00:00Z` | none | calm |

---

## 3. API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | liveness |
| POST | `/api/agents/maritime/investigate` | `{"timestamp_utc": "2025-07-10T19:00:00Z"}` | Agent 1 assessment |
| POST | `/api/agents/cargo/investigate` | same | Agent 2 assessment |
| POST | `/api/agents/events-weather/investigate` | same | Agent 3 assessment |
| POST | `/api/decision-support/analyze` | same, optional `"stress": {"berths_closed": 6, "arrival_surge_pct": 100}` | full cycle: state, forecast, agents, twin results, recommendation |

Example from PowerShell (backend running on port 8000):
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/decision-support/analyze `
  -ContentType application/json -Body '{"timestamp_utc":"2025-07-10T19:00:00Z"}'
```

Every response carries `reasoning_mode`: `deterministic`, `llm_assisted`, or `deterministic_fallback`
(LLM answer rejected by the guardrails). Errors: 404 = hour not in timeline, 422 = bad timestamp or stress value.

---

## 4. Repository structure

```
Marsa/
├── README.md
├── requirements.txt · pyproject.toml · .env.example · .gitignore · .gitattributes
├── config/port_config.yaml        ★ every port parameter and assumption, tagged SOURCE / DERIVED / ASSUMPTION
├── data/
│   ├── external/                  berth-occupancy columns derived from raw AIS
│   └── processed/
│       ├── merged_port_dataset_2025_v1.csv   pure merge of the three agent datasets (frozen)
│       ├── merged_port_dataset_2025_v2.csv   v1 + berth columns  ← runtime input
│       ├── la_cargo_calls_2023_2025.csv      every LA cargo-vessel berth call from raw AIS  ← twin input
│       ├── event_registry_2025.csv           synthetic events registry
│       └── ml_modeling_table_c1.parquet      AIS-only table (2023–2025) used by the C1 model
├── src/marsa/
│   ├── data/          loader, hourly snapshot, gap-safe features/targets
│   ├── model/         runtime inference contract for the prediction model
│   ├── agents/        maritime · cargo · events_weather · strategy · gemini/llm · remote (HTTP adapter)
│   ├── twin/          calls (AIS state at an hour) · engine (berth simulation) · adapter · scoring
│   ├── api/           FastAPI app, routes, schemas
│   ├── provenance/    provenance types
│   └── pipeline.py    one decision cycle
├── prediction_model/  training code, artifacts and results of the cargo-congestion model (XGBoost 1/3/6/12 h)
├── artifacts/models/  C1 maritime congestion model (XGBoost) + metadata + SHAP explainers
├── reports/           data audit, model evaluation, calibration, thresholds
├── scripts/           run_pipeline.py · build_v2_dataset.py · berth_occupancy_from_ais.py · build_la_cargo_calls.py · events/weather builders
├── dashboard/         marsa.html (dashboard, API-driven only) · README_DASHBOARD.md
├── tests/             agents, API, twin, features, integration, prediction pipeline
└── docs/              DATA_DICTIONARY · ARCHITECTURE · LIMITATIONS · NEXT_STEPS · figures
```

---

## 5. Data

| Layer | Source | Provenance | Notes |
|---|---|---|---|
| Vessel traffic | NOAA AIS via Kaggle *LA/LB Maritime Trajectory & AIS Dataset 2023-2025* (CC BY 4.0) | OBSERVED → hourly DERIVED | 21.7 M pings → 8,688 hours of 2025 |
| Berth occupancy | derived from raw AIS (cargo vessels stationary inside the LA port area) | DERIVED | validated against Marine Exchange daily counts |
| LA cargo calls (twin) | derived from raw AIS 2023-H2 → 2025 (`scripts/build_la_cargo_calls.py`) | OBSERVED → DERIVED | 3,448 berth calls; arrival, wait, berth and service time of each vessel |
| Weather | Open-Meteo (LA/LB), hourly | OBSERVED | wind, wave height, WMO code |
| Cargo / yard / gate | generated, calibrated to official Port of Los Angeles 2025 monthly TEU and PMSA dwell times | SYNTHETIC | port-level, normalised yard scale (10,000 units) |
| External events | generated with a fixed seed, independent of AIS | SYNTHETIC | 15 events, 172 hours |

Port of Los Angeles is a **stand-in**: Eastern Province port data was requested and not made available.
Everything port-specific lives in `config/port_config.yaml` (23 container berths — an approximate public figure, see the source note there).
72 source hours are missing from the AIS timeline and are deliberately **not** filled.
Full column list: `docs/DATA_DICTIONARY.md`.

---

## 6. Prediction models

| Model | Target | Inputs | Test period | Result |
|---|---|---|---|---|
| **C1 maritime congestion** (`artifacts/models/`) | observed AIS label (≥ 17 waiting vessels *or* high density + low speed) in the next 6 h | AIS + weather | Jul–Dec 2025 | ROC-AUC 0.97 · PR-AUC 0.83 (persistence baseline 0.49) |
| **Cargo-congestion proxy** (`prediction_model/`, used at runtime) | rule-based proxy from cargo/yard/gate stress, next 1/3/6/12 h | AIS + synthetic cargo | walk-forward folds | `prediction_model/final_proxy/results/` |

The C1 model has an **observed** target, so its accuracy is the headline number. The runtime proxy model extends
the forecast to the landside; because its target is synthetic, its accuracy is internal consistency, not
real-world accuracy (see `docs/LIMITATIONS.md`).

---

## 7. Digital twin

Sea side of the Port of Los Angeles, cargo vessels only. The starting state at any hour comes from AIS:
which vessels are at a berth (and when each one really left), which are waiting for an LA berth (destination
taken from where the vessel berthed next), and which arrive in the next 24 h, each with its real service time.
Ports know the next day's arrivals from advance arrival notices, so the twin uses the real arrival schedule.

Without a stress scenario the twin reproduces AIS exactly (tested: every vessel berths at its recorded time).
In 2025 LA berths never filled (max 20 of 23) and 7.4 % of cargo calls waited — for reasons the data does not
show (terminal windows), so those waits are kept as recorded. A stress scenario chosen by the user (berths
closed, compressed arrival schedule) makes berths scarce; vessels then also wait for a free berth
(*capacity wait*). Actions: `queue_policy` (FCFS / shortest real service first) and `add_berths`.
The simulation is deterministic (no random sampling) and options are ranked by total vessel-hours waiting.
Yard and gates are not simulated (their data is synthetic).

---

## 8. Tech stack

Python 3.11 · pandas / NumPy / PyArrow · XGBoost · scikit-learn · SHAP-style explainers · Pydantic v2 ·
FastAPI + Uvicorn · Gemini API (optional narrative) · pytest · ruff · HTML/JS dashboard.

---

## 9. Limitations and next steps

`docs/LIMITATIONS.md` and `docs/NEXT_STEPS.md`. In short: stand-in port, container chain only, synthetic landside,
sea-side-only twin of one port treated as one berth pool, approximate berth count.
With Mawani / TOS data for King Abdulaziz Port the same code runs unchanged: only the config and the three
input datasets change.

---

## 10. Team — AgentX 2026 · Team Marsa

| Name | GitHub | Role |
|---|---|---|
| Anwar Aldahan | [@AnwarAldahan](https://github.com/AnwarAldahan) | Data integration, repository, digital twin, dashboard wiring & synthetic operations data |
| Sajedah Alqudaihi | [@Sajedah25 ](https://github.com/Sajedah25) | Maritime agent (AIS) & synthetic operations data & Dashboard design |
| Anfal bamardouf | [@nbamardouf-source](https://github.com/nbamardouf-source) | Cargo agent & synthetic operations data &Prediction model |
| Fatima Alawami | [@Fatima Alawami](https://github.com/FatimaAlawami3) | Events & weather agent, API, LLM integration & synthetic operations data|

