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
including conflicts. It identifies supported pressure areas, creates a small set of
explainable candidates, invokes the Digital Twin, and reports deterministic ranking.
Candidate generation is deterministic and capped at four candidates including the
baseline. Maritime queue actions require both an ELEVATED/CONGESTED status and a
non-empty waiting queue; Cargo gate action generation continues to use the stored
ELEVATED/CRITICAL status without re-thresholding rounded evidence. A combined queue
and gate candidate requires both component candidates plus an active SYNTHETIC event
whose existing configured mapping increases modeled arrivals. The event multiplier
remains a scenario adjustment applied once by the Digital Twin adapter; it is not
copied into candidate actions and is not claimed to cause current pressure.

## ML contract

The preserved XGBoost model targets `c1_congestion_next_6h`: whether C1 queue-pressure
congestion occurs in `(t, t + 6h]`. Output would be a probability plus classification
using the frozen threshold `0.6866225600242615`.

Real online inference is currently unavailable. The target merged runtime table lacks
several approved features under their trained semantics, including vessel-type counts,
median SOG, ship density, average port speed, and exact previous-hour throughput.
The lineage modeling table is not promoted to an online runtime source. Marsa returns
structured `ml_unavailable` state with no probability or classification rather than
renaming or approximating features. Internal tests may inject an explicitly marked
`supplied_test_context`; it is never represented as a real model prediction.

## Digital Twin

The preserved teammate engine is an hour-stepped Monte Carlo queue and yard stock-flow
simulation. Its default horizon is 24 hours. Supported actions are `queue_policy`,
`add_berths`, `crane_boost`, `gate_boost`, `delay_arrivals`, and `prioritise_vessel`.

The deterministic scenario adapter derives the starting state from the authoritative
snapshot and `config/port_config.yaml`. It records assumptions for berth allocation,
LA share of waiting vessels, service distributions, yard scale, gate/discharge rates,
and deterministic weather/event multiplier mappings. These mappings are simulation
assumptions, not domain-agent facts and never come from Gemini.

Simulation output contains mean, p10, and p90 for average/max wait, delayed vessels,
vessels served/arrived, berth utilization, yard peak/end occupancy, and ending queue.
Every outcome is marked `SIMULATED`.

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

`POST /api/decision-support/analyze` accepts `timestamp_utc`. The response separates
timestamp, ML state, domain-agent results, Strategy synthesis/candidates, Digital Twin
scenario/results, deterministic ranking, limitations, and the human-approval flag.

## Known limitations

- ML inference remains unavailable until an approved runtime source provides exact feature parity.
- Cargo, yard, gate, event, capacity, and several simulation inputs are synthetic or assumptions.
- The port is simplified into one container-chain model rather than terminal/operator-level resources.
- Simulation score weights are documented choices, not operational cost estimates.
- Simulation ranking is not guaranteed real-world superiority.
- The MVP uses Los Angeles/Long Beach as a stand-in port and only calendar year 2025.
