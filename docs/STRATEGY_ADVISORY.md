# Phase 4: additive Strategy decision-support advisory

The deterministic `strategy_advisory` component consumes actual Phase 3 Hybrid
Twin sensitivity outputs. It is an advisory attachment, not a fifth agent or a
new ranking system. The original Strategy candidate generation, Gemini tool
selection, adaptive priorities, forecasts, historical replay, simulations,
ranking and winner are preserved. Coupled/Hybrid physics and sensitivity
calculations are not changed. No frontend or datasets are changed.

## Activation and execution boundary

```yaml
strategy_decision_support:
  enabled: false
```

Disabled: no advisory field is emitted; supplied internal evidence is ignored
without validation or computation. Existing responses remain identical.

Enabled: the pipeline adds only `strategy_advisory`. If no evidence is supplied,
it reports UNAVAILABLE. **Ordinary `/api/decision-support/analyze` requests never
start sensitivity experiments**, even when the feature is enabled.

The complementary `POST /api/decision-support/advisory` route is independently
gated by this server flag (403 when disabled). Its optional `experiment` field
explicitly requests the expensive work. It reads Phase 3 plan/Hybrid workload
defaults without enabling their existing HTTP endpoints. Omit `experiment` to
receive an unavailable advisory alongside the original decision flow.

The endpoint uses deterministic fallback for existing Strategy narrative and
investigation (`llm=None`) and makes no Gemini calls. Existing `/analyze` retains
its original provider behavior. Library pipeline calls retain the caller's LLM
and add no advisory-related calls. Gemini availability therefore never controls
advisory facts or numerical content. No autonomous action is implemented.

## API request and response example

After enabling the server flag:

```json
{
  "timestamp_utc": "2025-03-04T10:00:00Z",
  "experiment": {
    "scenario": {
      "timestamp_utc": "2025-03-04T10:00:00Z",
      "horizon_hours": 24,
      "time_information_mode": "retrospective",
      "cohort_id": "explicit-heuristic-cargo-experiment",
      "resource_pool_id": "abstract-advisory-pool",
      "inventory_source": "cargo_snapshot",
      "parameters": {
        "yard_capacity": 10000,
        "initial_inventory": 0,
        "berth_slots": 13,
        "unloading_units_per_hour": 1000,
        "gate_units_per_hour": 100
      }
    }
  }
}
```

Decision and experiment timestamps must match. The original optional `stress`
and `added` fields still affect legacy replay only; they are not silently applied
to the separately defined advisory experiment. As-of requests use Phase 2's
caller-supplied observation snapshot and explicit initial inventory; they do
not reconstruct availability from the finalized CSV. Legacy replay itself is
still retrospective and is not relabeled as an as-of recommendation.

The response contains the existing decision-support payload unchanged, plus
an additive object with these representative fields:

```json
{
  "strategy_advisory": {
    "mode": "simulation_advisory_v1",
    "status": "AVAILABLE",
    "decision_timestamp": "2025-03-04T10:00:00+00:00",
    "time_information_mode": "retrospective",
    "hindsight_assisted": true,
    "original_winner": "baseline",
    "recommendation_preserved": true,
    "reasoning_mode": "deterministic",
    "units": "normalized_units",
    "operator_review_required": true,
    "autonomous_execution": false
  }
}
```

This is an excerpt, not the complete response schema. Full outputs include:

- Cohort, resource pool, temporal/source provenance and workload assumptions.
- Baseline/intervention KPIs, exact parameter changes and conservation checks.
- Binding constraints, hours and plain-language simulation explanations.
- Original candidate/action mappings, separate advisory experiments.
- Phase 3 robustness, actual sensitivity ranges/context knobs, and tradeoffs.
- Explicit limitations and operator-review requirement.

Library consumers can call `run_advisory_experiment(SensitivityRequest(...), cfg)`
explicitly, then pass its envelope to `pipeline.run(..., advisory_evidence=...)`.
The HTTP interface deliberately accepts requests, not arbitrary result blobs.
The pipeline can also consume a precomputed internal envelope without rerunning
the analysis. No module writes outputs unless the caller explicitly requests
the existing pipeline `save=True` behavior.

## Evidence validation and unavailable states

The internal envelope records the experiment request, decision timestamp and
server-generated sensitivity report. Validation checks:

- Mode, normalized units, simulated provenance, temporal mode/hindsight labels,
  timestamp, cohort/resource identity and population hash/unique call IDs.
- Initial-state counts, base parameters, source inventory and workload settings.
- Complete comparison catalog, paired workload signatures, allowed parameter
  increments and finite numeric outcomes.
- Vessel counts, end/peak occupancy, reported **and recomputed** vessel/yard/
  system residuals, using `1e-9 * max(1, initial work + initial stock)` tolerance.
- Diagnostic methods, per-hour constraint coverage/counts and classification.
- Exact Phase 3 comparison deltas and recomputed robustness consistency.

This verifies internal evidence consistency; it is not independent operational
validation or authentication of third-party simulation reports. No public API
accepts externally supplied reports. Missing evidence returns UNAVAILABLE.
Malformed/incomplete evidence returns INSUFFICIENT_EVIDENCE with no intervention
claims. Conservation failure is explicitly flagged and likewise suppresses
benefit claims. Original recommendations remain untouched on every such path.

## Action compatibility

The current generator's catalog is `baseline`, `shortest_first`, `extra_berth`,
and **`gate_max`**. `gate_extension` is an older/example identifier, not the
current generator's action. No catalog is changed in this phase.

| Original candidate | Advisory experiment | Mapping |
|---|---|---|
| baseline with no actions | baseline | REFERENCE_ONLY; different model, not winner validation |
| extra_berth, exactly add_berths=1 | berth_plus_one | CONDITIONAL_SEMANTIC_MATCH for the one-slot increment |
| shortest_first | None | NOT_EVALUATED; sensitivity did not test queue-policy changes |
| gate_max/max_recorded | None | NOT_EVALUATED; historical maximum is not a normalized multiplier |
| gate_extension | None | NOT_EVALUATED; gate hours cannot be reinterpreted as a multiplier |
| Other/multi-action/different increment | None | NOT_EVALUATED |

Mappings preserve the exact original candidate identifiers and action objects.
Even the one-slot mapping is qualified: effective slots, workloads, cohorts and
resources differ from the legacy model. It supports an abstract experiment,
not confirmation of the original ranking or physical berth feasibility. Gate
+20%, gate x3 and unloading +20% remain separate advisory experiments rather
than newly eligible Strategy candidates. No unsupported candidate gets evidence
merely because a similar-sounding experiment performed well.

## Benefits, uncertainty and tradeoffs

Numerical Phase 3 comparisons/robustness are preserved exactly. The advisory
adds a more explicit interpretation:

- QUEUE_RELIEF_ONLY: lower queue exposure without more completions/unloading;
  admission relief is not demonstrated throughput recovery.
- CARGO_PROGRESS_ONLY: more unloading without a queue/completion benefit.
- SIMULATED_COMPLETION_OR_CARGO_SUPPORTED_IMPROVEMENT: a queue/completion benefit
  supported by completion or cargo progress under this context, subject to
  sensitivity and conservation caveats.
- TRADEOFF: queue and completion outcomes conflict; no winner is declared.
- Otherwise: no demonstrated delay/completion benefit in this context.

Robustness categories describe the tested assumption contexts, not confidence
or probability. A Phase 3 robust queue/completion classification can include
queue-only relief; it does not imply robust throughput gains. Tradeoffs from
all contexts are exposed with exact baseline/intervention KPIs and assumptions,
even when a base comparison is favorable. No weighted score is introduced.

## Two historical examples and reproduction

```powershell
python -B -m scripts.demo_strategy_advisory
python -B -m pytest -p no:cacheprovider
python -B -m ruff check src/marsa/agents/strategy_advisory.py src/marsa/agents/strategy_agent.py src/marsa/pipeline.py src/marsa/api/routes/decision_support.py scripts/demo_strategy_advisory.py tests/test_strategy_advisory.py
```

The demonstration disables LLM investigation, runs legacy and enabled paths,
asserts full response equality after removing the advisory, and prints real
simulator/advisory evidence without creating files.

Both original winners remain **baseline**. This is not fabricated agreement:
A's original generator supplies only baseline; B supplies all four candidates.
The advisory cannot validate baseline's rank because its model/population differs.
B's shortest-first and recorded-maximum gate candidates remain unevaluated by
the advisory; the extra-slot comparison is an explicitly qualified analogue.

A = March 4, 2025, 10:00 UTC (20 calls; stock 7204.03, finalized synthetic Cargo).
B = July 10, 2025, 19:00 UTC (23 calls; controlled stock 10000).
Both H=24h, synthetic work=1000/call, initial remaining fraction=0.5, shared
unloading=1000 units/hour; gate baseline=100, modest=120, large=300.

| Case / experiment | Queue vessel-hours | Completed / unfinished | Unloaded / remaining aboard | Advisory interpretation |
|---|---:|---:|---:|---|
| A baseline | 85 | 0 / 20 | 5195.97 / 8304.03 | Reference only |
| A gate +20% | 85 | 0 / 20 | 5675.97 / 7824.03 | Cargo progress only; sensitive |
| A gate x3 | 30 | 13 / 7 | 9995.97 / 3504.03 | Simulated improvement; sensitive |
| A unloading +20% | 85 | 0 / 20 | 5195.97 / 8304.03 | No base benefit; sensitive |
| A extra slot | 63 | 0 / 20 | 5195.97 / 8304.03 | Queue relief only; robustness inconclusive |
| B baseline | 75 | 0 / 23 | 2400 / 12100 | Reference only |
| B gate +20% | 75 | 0 / 23 | 2880 / 11620 | Cargo progress only; delay benefit ineffective |
| B gate x3 | 75 | 0 / 23 | 7200 / 7300 | Cargo progress only at base; sensitive overall |
| B unloading +20% | 75 | 0 / 23 | 2400 / 12100 | Ineffective |
| B extra slot | 51 | 0 / 23 | 2400 / 12100 | Queue relief only; robust queue/completion classification |

A's baseline binds yard/gate/unloading for 21/24/3 hours; B binds yard/gate for
24/24 hours. All listed base comparisons end at full yard occupancy, with peak
also 100%. Residuals are numerical roundoff below tolerance. In A's low-workload
context, an extra slot lowers queue exposure 6 -> 5 but completions fall 16 -> 14;
the advisory exposes this tradeoff instead of endorsing the original candidate.

## Demonstration readiness

Ready for operator-reviewed demonstrations with assumptions visible, not
automatic operational recommendations. No actual POLA wait reduction, observed
intervention outcome or prediction accuracy is claimed. Workloads are synthetic,
remaining progress unobserved, yard aggregated, identity conflicts possible and
container-terminal membership unverified. As-of availability is caller-asserted;
retrospective experiments use finalized future-arrival/membership information.
Next improve independent cohort/progress evidence and validate decision criteria
before considering any ranking integration.

Local integration note (2026-10-17): the repository demo configuration explicitly enables this
backend feature. Missing sections and the typed settings defaults remain disabled. Latest Main
frontend and AIS replay are preserved; experiments require separate explicit API requests.
