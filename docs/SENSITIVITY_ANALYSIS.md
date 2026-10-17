# Phase 3: sensitivity and bottleneck analysis

This is a standalone, opt-in deterministic experiment layer around the existing
Hybrid Twin. It neither changes simulation mechanics nor feeds results into the
four agents, Gemini tools, Adaptive Candidate Prioritization or existing scoring.
Historical replay, Phase 1/2 routes, models, original datasets and frontend remain
unchanged. Numerical benefits are conditional on synthetic assumptions, not
estimated real-world effects or empirical confidence levels.

## Reproduction and configuration

From the repository root:

```powershell
python -B -m scripts.demo_sensitivity_twin
python -B -m scripts.demo_sensitivity_twin --full
python -B -m pytest -p no:cacheprovider
python -B -m ruff check src/marsa/twin/sensitivity.py src/marsa/api/routes/coupled_twin.py scripts/demo_sensitivity_twin.py tests/test_sensitivity_twin.py
```

The first command prints historical base comparisons, ranges and robustness.
`--full` includes every context, resolved parameter set, KPI, pairing signature,
per-hour diagnostic evidence and metadata. Neither command writes result files,
calls external services or requires API credentials. Explicit Python calls use
`analyze_sensitivity(HybridRequest(...), plan=SensitivityPlan(...))`.

The independent server switch `sensitivity_analysis.enabled` defaults to false.
Enable it explicitly to use `POST /api/twin/analyze-sensitivity`. Existing hybrid
and coupled endpoint switches remain independent. Missing configuration is
disabled (403); invalid server settings return 503; invalid requests return 422.
The HTTP body is:

```json
{
  "scenario": {
    "timestamp_utc": "2025-03-04T10:00:00Z",
    "horizon_hours": 24,
    "time_information_mode": "retrospective",
    "cohort_id": "explicit-heuristic-cargo-experiment",
    "resource_pool_id": "abstract-yard-pool",
    "inventory_source": "cargo_snapshot",
    "parameters": {
      "yard_capacity": 10000,
      "initial_inventory": 0,
      "berth_slots": 13,
      "unloading_units_per_hour": 1000,
      "gate_units_per_hour": 100
    }
  },
  "plan": {
    "modest_increase": 0.2,
    "large_gate_factor": 3,
    "include_interactions": true
  }
}
```

The complete server plan is in `config/port_config.yaml`. Omitted request plans
use it. Omitted scenario workload settings inherit the Hybrid server defaults.
Resources are still supplied explicitly, never inferred from published POLA
berths or cranes. As-of requests retain Phase 2's independently supplied snapshot
requirement and never read finalized AIS/Cargo data. An existing Phase 2
`intervention_gate_units_per_hour` is superseded by the explicitly returned
Phase 3 comparison plan; it does not add a sixth comparison silently.

## Ranges, pairing and experiment size

The baseline uses the exact Hybrid scenario. For each factor, test low and high
against that base, skipping a level equal to base. This gives at most 15 OAT
contexts including base. Two optional named interactions bring the maximum to
17 contexts / 85 runs per historical scenario, not a Cartesian product.
Runtime-identical contexts are deduplicated and reported as aliases, not counted
twice in robustness coverage. B's high-occupancy/low-gate interaction duplicates
its low-gate OAT case because its base yard is already full.

| Factor | Low | Base | High | Interpretation |
|---|---:|---:|---:|---|
| Initial occupancy | min(50%, base) | Actual scenario fill | max(100%, base) | Exploratory stock assumption |
| Yard capacity, base 10000 | 9090.91 | 10000 | 10909.09 | Approved relative storage anchors; other base scales use the same ratios |
| Full workload/call | 0.5x | 1x | 1.5x | Exploratory synthetic work, not measured cargo |
| Initial berthed remaining fraction | min(0.25, base) | Usually 0.5 | max(0.75, base) | Unobserved progress sensitivity |
| Gate capacity | 0.5x | 1x | 1.5x | Normalized units/hour |
| Shared unloading | 0.5x | 1x | 1.5x | Whole-pool normalized units/hour |
| Berth slots | max(initial occupancy, base-1) | Explicit base | base+1 | Synthetic effective slots; not physical POLA berth counts |

For the two historical demonstrations: workload 500/1000/1500; remaining fraction
0.25/0.5/0.75; gate 50/100/150; unloading 500/1000/1500. Occupancy is
50%/72.0403%/100% for A and 50%/100%/100% for B. Slot levels are
13/13/14 and 17/17/18: lowering them below existing occupancy is not allowed.

Capacity sensitivity holds **fill fraction** constant and derives initial stock
as capacity times fill fraction. Thus it is an OAT change in the declared capacity
factor, not a fixed-absolute-stock experiment. Derived sensitivity stocks are
scenario assumptions, not unchanged observed Cargo measurements; the report
explicitly returns this policy. All bounds/units are validated, with no clipping.

Selected interactions only:

- High occupancy plus low gate capacity.
- High workload plus low unloading productivity.

Each context compares baseline, gate +20% (or configurable +10%), gate x3,
unloading +20%, and one extra slot. A zero base rate stays zero under a multiplier;
no hidden additive service is invented. Source population is built once. Every
pair retains identical call IDs, arrival times, initial states and workload
assignments, checked by signatures. No RNG is used. Workload-changing contexts
intentionally differ from each other, but remain paired internally.

## State-based diagnostic definitions

`ledger_binding_constraints_v1` reconstructs opening berth work from admitted
initial work minus cumulative unloading. Eligible opening yard stock is initial
eligible inventory plus earlier released unloading lots minus cumulative gate
clearance. It uses the actual flow ledger and timeline, with a relative numerical
tolerance; it does not inspect intervention names, winners or ranking scores.

| Constraint | Deterministic evidence |
|---|---|
| Yard-capacity blocking | Lost unloading opportunity, or positive berth work with no post-gate headroom |
| Gate-clearance limitation | Eligible stock exceeds actual clearance and configured gate service is saturated; zero capacity can bind |
| Unloading-productivity limitation | Berth work exceeds the budget and actual unloading consumes the budget; zero productivity binds positive arrived demand |
| Independent berth-slot limitation | Full slots with waiting demand and spare unloading/headroom, or zero slots with a queue |
| Insufficient demand | No identified capacity constraint and no queue pressure; at the model's hourly resolution |
| Mixed | Multiple constraints bind during the horizon, possibly at different hours |
| Inconclusive | Queue pressure exists but an independent constraint cannot be attributed, including some slot/budget co-saturation cases |

Full berths caused by yard blocking are reported separately as berth queue
pressure, not automatically diagnosed as an independent slot shortage. Gate
service can constrain yard draining without being the constraint on vessel
completion. Ineligible stock can prevent gate capacity from helping; its blocking
hours are returned separately. Hourly completion/berth reuse affects attribution.
Outputs include binding hours, simultaneous gate/yard hours, pool utilization,
per-hour work/headroom/eligibility evidence, and active constraints.

`bottleneck_changed` compares active-constraint sets over the horizon. The hours
can change even if that set remains identical; inspect `constraint_hours` too.
These are simulation binding diagnostics, not empirically verified causal claims.

## Historical base comparisons

Both experiments retain the Phase 2 2025 cohorts, H=24 and 10000-unit yard.
A: March 4, 10:00 UTC; 20 calls (13 berthed, 0 waiting, 7 future), initial
finalized synthetic inventory 7204.03. B: July 10, 19:00 UTC; 23 calls
(17 berthed, 1 waiting, 5 future), controlled initial stock 10000. Neither source
has verified container-terminal membership. Both are retrospective/hindsight-assisted.

Base assumptions: work 1000/call, fraction 0.5, gate 100 and unloading 1000
normalized units/hour. Interventions: gate 120, gate 300, unloading 1200, or one
extra synthetic slot. Initial slots are 13 (A) / 17 (B).

| Case / intervention | Queue vessel-hours | Completed / unfinished | Unloaded | Remaining aboard | Yard-blocked opportunity |
|---|---:|---:|---:|---:|---:|
| A baseline | 85 | 0 / 20 | 5195.97 | 8304.03 | 18804.03 |
| A gate +20% | 85 | 0 / 20 | 5675.97 | 7824.03 | 18268.06 |
| A gate x3 | 30 | 13 / 7 | 9995.97 | 3504.03 | 12216.12 |
| A unloading +20% | 85 | 0 / 20 | 5195.97 | 8304.03 | 23604.03 |
| A extra slot | 63 | 0 / 20 | 5195.97 | 8304.03 | 18804.03 |
| B baseline | 75 | 0 / 23 | 2400 | 12100 | 21600 |
| B gate +20% | 75 | 0 / 23 | 2880 | 11620 | 21120 |
| B gate x3 | 75 | 0 / 23 | 7200 | 7300 | 16800 |
| B unloading +20% | 75 | 0 / 23 | 2400 | 12100 | 26400 |
| B extra slot | 51 | 0 / 23 | 2400 | 12100 | 21600 |

All ten runs have peak/end yard occupancy of 100%/100%. The maximum absolute
vessel/yard/system conservation residual across **all sensitivity runs** is
7.51e-12 for A and 2.19e-11 for B (rounded upper bounds).
Blocked opportunity can exceed actual cargo: repeated unmet service budgets are
not unique cargo units. Higher productivity can increase that opportunity metric
without changing actual unloading or losing cargo.

Constraint labels: Y=yard, G=gate, U=unloading, B=independent berth slots.

- A baseline, gate +20%, unloading +20%, extra slot: mixed Y/G/U.
- A gate x3: mixed Y/G/U/B; active set changes, with one hour of independent
  slot limitation. Base Y/G/U binding hours are 21/24/3; x3 gives 20/24/3 plus B=1.
- B: mixed Y/G in every base comparison; no active-set change. Both bind all
  24 hours. More gate capacity unloads additional cargo but does not finish the
  initial berthed group by H, so queues do not improve.

## Robustness and benefit definition

No scoring weights are added. A queue/completion benefit requires lower queue
exposure or more completions, with neither metric worsening. Opposing changes
are a tradeoff. Cargo-only progress, clearance and yard-stock changes are reported
separately, never relabeled as demonstrated delay improvement.

| Classification | Rule across tested contexts |
|---|---|
| Robust across tested assumptions | Beneficial in every context, at least two contexts |
| Sensitive to specific parameters | Benefit in some contexts, ineffective/harmful in others, without tradeoffs |
| Ineffective under tested conditions | No queue/completion benefit in any context; outcome counts disclose any harm |
| Inconclusive | Any tradeoff, inadmissible excluded context, or fewer than two valid contexts |

| Intervention | A: 16 contexts / 80 runs | B: 14 contexts / 70 runs |
|---|---|---|
| Gate +20% | Sensitive: 4 beneficial, 12 ineffective | Ineffective: 0 beneficial, 14 ineffective |
| Gate x3 | Sensitive: 14 beneficial, 2 ineffective | Sensitive: 4 beneficial, 10 ineffective |
| Unloading +20% | Sensitive: 1 beneficial, 15 ineffective | Ineffective: 0 beneficial, 14 ineffective |
| Extra slot | Inconclusive: 15 beneficial, 1 tradeoff | Robust queue/completion benefit in 14/14 |

Concrete switches and qualifications:

- A unloading +20% is useful at 50% initial occupancy but ineffective at base
  72.0403% and full occupancy. Yard/gate pressure masks the productivity benefit.
- B gate x3 helps at lower workload or 25% remaining work, but not at base
  workload/progress. A gate +20% similarly helps in selected lower-pressure cases.
- A at 500-unit full workload: an extra slot reduces queue exposure 6 -> 5,
  but completions fall 16 -> 14 because the fixed productivity is shared across
  more occupied slots. The result is a tradeoff, not a fabricated win.
- B's extra slot reduces queue waiting without increasing completions/unloading
  in the base case. Moving a vessel from queue to berth is not throughput recovery.

These are coverage statements for a small exploratory plan, not probabilities
or guarantees. Untested interactions, other horizons, fractional-hour mechanics
and different workload/resource assumptions can change the conclusions.

## Integration recommendation and limitations

Ready for cautious **read-only explanations and operator-reviewed experiments**,
not automatic Strategy recommendations or ranking. First validate identities and
cohorts, remaining-work assumptions and information availability; examine
queue/berth-service tradeoffs and an independent calibration set. Finalized AIS
arrival linkage, synthetic Cargo construction, unknown actual cargo and physical
capacity remain the Phase 2 limitations. No empirical prediction accuracy is
claimed. Existing Strategy behavior is deliberately unchanged.

Local integration note (2026-10-17): the repository demo configuration explicitly enables this
backend feature. Missing sections and the typed settings defaults remain disabled. Latest Main
frontend and AIS replay are preserved; experiments require separate explicit API requests.
