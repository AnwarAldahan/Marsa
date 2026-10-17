# Final Marsa Integration

## Architecture and data flow

Marsa has exactly four agents: Maritime, Cargo, Events & Weather, and Strategy.
There is no active Orchestrator Agent. ML and the Digital Twin are deterministic
components. The final response is decision support and never executes a port action.

The flow is exact-hour validation, authoritative snapshot, ML boundary, three domain
investigations, Strategy synthesis and candidate generation, Digital Twin simulation,
deterministic scoring, decision-support response, and human review.

## Time contract

Public runtime requests must be timezone-aware, explicitly UTC, aligned to an exact
whole hour, and in calendar year 2025. The authoritative timeline has 8,688 hours and
preserves 72 source gaps. Runtime selection never floors, interpolates, nearest-matches,
or substitutes. Invalid timestamp shapes produce 422 behavior at the API; missing
authoritative hours produce 404 behavior.

## Agent responsibilities

The Maritime Agent preserves the teammate `PortOperationsAgent` status calculation.
Its status uses unrounded internal values and is independent of forecast context.

The Cargo Agent reads frozen synthetic/calibrated synthetic state. Stored
`operations_status` is authoritative for its current status and is never recomputed
from rounded output. Cargo values are not observed terminal measurements.

The Events & Weather Agent reports deterministic weather pressure, synthetic event
context, and local calendar context. It does not predict congestion or own simulation
multipliers.

The Strategy Agent preserves the three domain results and optional forecast signal,
including conflicts. It first runs the Digital Twin with no change. Only when that run
shows vessels waiting for a free berth (capacity wait) does it add two candidates:
`shortest_first` (queue policy) and `extra_berth` (one more open berth). Domain statuses,
synthetic events and the ML signal never create a twin action on their own.

## ML contract

The integrated XGBoost model version `layer1-cargo-proxy-v1` (trained on data identical to
`merged_port_dataset_2025_v2.csv`: same 31 source columns, same 284 positive labels) targets the
`operational_cargo_congestion_proxy` at exactly `t + 6h`. It returns an uncalibrated
`congestion_score` plus LOW/MEDIUM/HIGH classification; it does not return a
probability. Runtime construction requires the timestamp and every exact source hour
through `t - 24h`, with no interpolation. Missing history produces a structured
`ml_unavailable` result.

This is a retrospective predictive prototype over the finalized 2025 Marsa dataset.
The target is an operational proxy rather than independently observed congestion
ground truth. Some synthetic Cargo inputs use retrospective full-period/month
construction, and `weather_pressure_index` retains a full-period baseline concern.
The runtime builder itself reads only finalized rows at or before `t`, but the system
must not be represented as fully leakage-free production online inference.

## Digital Twin

Berths of the Port of Los Angeles (cargo vessels only) plus the yard and gate (`src/marsa/twin/`). The starting state at
the requested hour is read from `data/processed/la_cargo_calls_2023_2025.csv` (built from raw AIS by
`scripts/build_la_cargo_calls.py`): vessels at a berth with their real departure time, vessels waiting
for an LA berth with their real berth time, and vessels arriving in the next 24 hours with their real
service times. Non-AIS inputs: the number of container berths (23: sum of the per-terminal berth counts in
the port's official Port 101 brochure; source in `config/port_config.yaml`) and the yard/gate columns of
the Cargo agent's file `data/agent2_cargo_operations_2025.csv` (synthetic, `src/marsa/twin/yard.py`).

With no stress scenario and no action the twin reproduces the recorded AIS schedule exactly (tested).
An optional user stress scenario (`berths_closed`, `arrival_multiplier`: ×1 = real schedule, ×2 = twice the real vessels, no upper limit) makes berths scarce; vessels then
also wait for a free berth. Supported actions are `queue_policy` (fcfs / shortest_service_first),
`add_berths` and `gate_capacity` ("max_recorded" = highest hourly container departures in the Cargo file up
to the chosen hour, or a number). The yard replays the Cargo file; a faster gate lowers it from then on. The
yard does not change vessel waiting (no data on that effect). Added vessels (`src/marsa/twin/estimates.py`)
get a stay from their own earlier stays, else from the 5 vessels closest in length (k chosen by
`scripts/select_similar_k.py`), or a typed duration. The simulation is deterministic (no Monte Carlo). KPIs: total and capacity vessel-hours
waiting, vessels waited, average/max wait, vessels arrived/berthed, waiting at the end, peak berths used,
yard peak/end occupancy, extra containers cleared, plus an hourly timeline and a per-vessel list. Options are listed by total vessel-hours waiting (ties keep
"no change" first); yard measures are reported beside them, not mixed into one score.

Strategy agent candidate rules (`generate_candidates`): maritime ELEVATED/CONGESTED with waiting vessels, or
a twin capacity wait → shortest_first + extra_berth; Cargo ELEVATED/CRITICAL, or a real HIGH model alert
→ gate_max; an active event → every test; weather is context only (no measured effect on stays). Every outcome is marked `SIMULATED`; every input is listed in `digital_twin.sources`.

## Provenance

The shared vocabulary is `OBSERVED`, `DERIVED`, `SYNTHETIC`, `PREDICTED`, `SIMULATED`,
and `AGENT_GENERATED`. Domain agents retain field-level provenance. Candidate strategies
and Strategy narrative are `AGENT_GENERATED`; Digital Twin outcomes are `SIMULATED`.
Unavailable or test-supplied ML context is not labeled as a real `PREDICTED` result.

## Gemini

Gemini is optional and narrative-only. It cannot own evidence, status, probabilities,
simulation parameters/results, scores, provenance, or action parameters. Invalid or
unavailable provider output falls back to deterministic text. The complete pipeline
runs without Gemini.

## API and commands

Start the API:

```powershell
$env:PYTHONPATH="src"
python -m uvicorn marsa.api.main:app --host 127.0.0.1 --port 8001
```

Run one decision:

```powershell
python scripts/run_pipeline.py 2025-01-01T00:00:00Z
```

`POST /api/decision-support/analyze` accepts `timestamp_utc` and an optional
`stress: {"berths_closed": int, "arrival_multiplier": float}` (invalid values → 422). The response separates
timestamp, ML state, domain-agent results, Strategy synthesis/candidates, Digital Twin
scenario/results, deterministic ranking, limitations, and the human-approval flag. It also accepts
`added` vessels and `user_actions` (the user's own decision; the Strategy agent evaluates it next to its
options and returns `decision_support.user_decision_assessment`). `stress` also accepts yard assumptions
`gate_closed_hours` and `container_inflow_multiplier`. `GET /api/twin/hours?condition=rain|drizzle|wind|wave|event`
lists real 2025 hours by condition. `POST /api/twin/simulate` runs one user decision against no change on the same hour,
stress and added vessels; `GET /api/twin/vessels` lists known vessels for the add-vessel form.

## Known limitations

- ML inference remains unavailable until an approved runtime source provides exact feature parity.
- Cargo, yard, gate and event inputs are synthetic; the twin replays the yard from the Cargo file and
  does not link it to vessel waiting.
- The twin treats the LA berths as one pool rather than terminal/operator-level resources.
- In 2025 LA berths never filled; berth-shortage results appear only under a user stress scenario.
- Simulation ranking is not guaranteed real-world superiority.
- The MVP uses Los Angeles/Long Beach as a stand-in port and only calendar year 2025.
