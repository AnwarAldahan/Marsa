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

Sea side of the Port of Los Angeles, cargo vessels only (`src/marsa/twin/`). The starting state at
the requested hour is read from `data/processed/la_cargo_calls_2023_2025.csv` (built from raw AIS by
`scripts/build_la_cargo_calls.py`): vessels at a berth with their real departure time, vessels waiting
for an LA berth with their real berth time, and vessels arriving in the next 24 hours with their real
service times. The only non-AIS input is the number of container berths (23, approximate public figure
with its source in `config/port_config.yaml`).

With no stress scenario and no action the twin reproduces the recorded AIS schedule exactly (tested).
An optional user stress scenario (`berths_closed`, `arrival_surge_pct`) makes berths scarce; vessels then
also wait for a free berth. Supported actions are `queue_policy` (fcfs / shortest_service_first) and
`add_berths`. The simulation is deterministic (no Monte Carlo). KPIs: total and capacity vessel-hours
waiting, vessels waited, average/max wait, vessels arrived/berthed, waiting at the end, peak berths used,
plus an hourly timeline and a per-vessel list. Options are ranked by total vessel-hours waiting; ties keep
"no change" first. Every outcome is marked `SIMULATED`; every input is listed in `digital_twin.sources`.

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
`stress: {"berths_closed": int, "arrival_surge_pct": float}` (out-of-range values → 422). The response separates
timestamp, ML state, domain-agent results, Strategy synthesis/candidates, Digital Twin
scenario/results, deterministic ranking, limitations, and the human-approval flag.

## Known limitations

- ML inference remains unavailable until an approved runtime source provides exact feature parity.
- Cargo, yard, gate and event inputs are synthetic; the twin does not use them.
- The twin treats the LA berths as one pool rather than terminal/operator-level resources.
- In 2025 LA berths never filled; berth-shortage results appear only under a user stress scenario.
- Simulation ranking is not guaranteed real-world superiority.
- The MVP uses Los Angeles/Long Beach as a stand-in port and only calendar year 2025.
