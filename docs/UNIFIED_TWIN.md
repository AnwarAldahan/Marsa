# Unified website Digital Twin

The canonical website is root `Marsa updated.html`, served by FastAPI at `/`, `/marsa.html` and `/Marsa%20updated.html` with `Cache-Control: no-store`. No repository directory is exposed. Start from the repository root:

```powershell
python -m uvicorn marsa.api.main:app --app-dir src --host 127.0.0.1 --port 8000
```

## One operational contract

`POST /api/decision-support/unified` accepts `timestamp_utc` (an exact UTC hour in 2025), optional `parameters` (the existing complete `CoupledParameters` schema), optional `horizon_hours`, and optional native `user_actions`.

On initial analysis, existing agents, XGBoost and bounded Gemini investigation run once through `collect_evidence`. Hybrid initialization freezes the timestamp, horizon, eligible AIS cohort, synthetic workloads, resource pool, inventory and parameters. All compared candidates use `simulate_coupled` with that same population; no legacy simulation contributes operational KPIs. Historical agents/forecast remain separately labelled context. The forecast is not a prediction of the counterfactual scenario.

The response supplies `context_id` and `context_signature`. Manual decisions must resend that context and timestamp; parameter/horizon changes on a saved context are rejected. Evidence and initialization are not reloaded. Changed configuration cannot silently alter an existing context. Contexts are JSON snapshots in a bounded process-local cache (32 entries, 30-minute TTL); restart, expiry or a different worker can produce HTTP 410. Reload the selected hour to create a new context. This prototype should run with one worker.

## Explicit demo assumptions

`config/port_config.yaml: unified_hybrid` enables the website contract and configures horizon 24 hours, 23 abstract slots, yard capacity 10,000 normalized units, initial inventory 0, unloading 100 units/hour, gate 100 units/hour, and zero release delay. These are scenario assumptions, not calibrated operational capacities. Existing Hybrid workload configuration supplies 1,000 synthetic units per vessel and a 0.5 remaining fraction for initially berthed calls. Initial inventory uses `inventory_source="explicit"`; no TEU, Cargo-index or gate-transaction conversion is attempted. Validation rejects fewer slots than the initially berthed population.

AIS coverage and terminal membership are heuristic and not verified POLA container-terminal membership. Retrospective initialization is hindsight-assisted and disclosed. No predictive accuracy or operational effectiveness follows from these simulations.

## Native candidates and changed ranking

The website ranks ascending `queue_exposure_vessel_hours`: waiting accumulated **during the simulated horizon**, not the legacy AIS berth-window metric. Baseline wins ties. Completions, unloading, yard occupancy and blocking remain separate outcomes. Queue relief alone is explicitly disclosed and does not establish better cargo progress.

Candidates are baseline; shortest **estimated remaining workload** first; one extra abstract slot; normalized gate rate x1.2; normalized gate rate x3. Factors are configurable. Every variant is validated before execution and must retain identical vessel/workload signatures. Invalid candidates and duplicate runtime variants are excluded with reasons. When base gate rate is zero, multiplicative gate variants are duplicates, not fabricated interventions. A manual absolute `gate_rate` remains available through the API.

Legacy `gate_max` has no justified conversion to normalized units and is excluded. Historical shortest-service ordering is not equivalent to estimated-workload ordering. Crane counts, rescheduling and arbitrary historical service-order overrides have no native engine implementation and are not ranked. Legacy adaptive prioritization remains available through legacy APIs; the unified workflow evaluates its small validated native pool exhaustively. Evaluation order never changes ranking.

The authoritative recommendation selects generated candidates. Manual user decisions are compared and ranked separately, preserving Main's existing operator workflow. Gemini cannot change the winner or invent benefits: only the authoritative verified summary is accepted; provider failures/altered text use deterministic fallback. Native advisory consumes only validated paired simulation ledgers, preserves queue-only/trade-off disclosures, and reports robustness `NOT_EVALUATED` until a separate experiment is supplied. No sensitivity sweep runs implicitly.

## Explicit sensitivity and controlled comparison

`POST /api/decision-support/unified/sensitivity` accepts `context_id` and optional `plan` using the existing `SensitivityPlan` schema. It reuses the frozen scenario, not a freshly initialized AIS population. Existing feature gates also apply.

For the controlled July experiment, initialize `2025-07-10T19:00:00Z`, horizon 24, with explicit parameters: slots 17, yard capacity/inventory 10,000, unloading 1,000/hour, gate 0, release delay 0, eligible initial inventory immediately, units `normalized_units`, provenance `SYNTHETIC_SCENARIO_ASSUMPTION`. Reuse its context and timestamp with `user_actions: [{"type":"gate_rate","value":1000}]`. This changes only clearance in the paired baseline/intervention; do not compare it to a separately initialized run.

The July test produced baseline/intervention queue exposure 75/15 vessel-hours, completions 0/22, unloading 0/13,500 normalized units and blocked unloading 24,000/0. These are synthetic controlled simulation outcomes, not observed port benefits. The original gate-zero generated winner adds a slot and produces queue relief without cargo progress; the advisory explicitly says so.

## Website and compatibility

Main's CSS, navigation, artwork, animations and component layout are retained. Existing operational cards, replay and manual current-port controls consume the unified response. The map aggregates virtual slots and uses actual hourly ledgers; it does not assign physical terminal berths. Null completion times remain censored, not invented. Hand-entered examples remain a labelled independent educational simulation and cannot affect operational recommendations. Unsupported live manual controls are disabled. Stale asynchronous responses cannot overwrite a newer hour.

Legacy `/api/decision-support/analyze`, `/api/twin/simulate`, Hybrid/Coupled, sensitivity and strict supplied-evidence advisory APIs remain available with their existing contracts. `dashboard/marsa.html` remains a legacy reference page. Coupled/Hybrid physics, models and datasets are unchanged. Invalid unified contracts fail visibly without falling back to legacy or illustrative operational results.
