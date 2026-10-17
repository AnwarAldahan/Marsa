# Phase 2: historical-call hybrid initialization

This adapter combines individual AIS-derived timing/identities with synthetic
normalized import workloads in the existing coupled engine. It is an abstract
aggregated yard experiment, not seven terminals or a calibrated POLA forecast.
Historical replay and Strategy recommendations remain unchanged. No datasets,
models, Gemini tools, agents, rankings or frontend files are changed.

## Opt-in route and configuration

`POST /api/twin/simulate-hybrid` has its own server flag, independent of the
Phase 1 endpoint. Both flags remain false by default. Missing hybrid configuration
also means disabled (403). Invalid enabled server settings return 503; invalid
requests, source contracts, missing inventory hours or insufficient slots return
422. No request can turn on a server flag.

```yaml
hybrid_normalized:
  enabled: false
  workload:
    method: constant_per_call_v1
    full_import_units: 1000
    berthed_remaining_fraction: 0.5
    remaining_fraction_sensitivity: [0.25, 0.5, 0.75]
```

The workload numbers are explicit synthetic assumptions, not estimates validated
against measured cargo. The fractions are optional sensitivity anchors, not an
empirically fitted range. No random sampling, TEU conversion or hidden per-vessel
quantity is used. Library calls `build_hybrid()` and `simulate_hybrid()` are
explicit opt-in; they do not require enabling HTTP.

Retrospective request example, after server enablement:

```json
{
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
  },
  "workload": {
    "method": "constant_per_call_v1",
    "full_import_units": 1000,
    "berthed_remaining_fraction": 0.5
  },
  "intervention_gate_units_per_hour": 1000
}
```

Resource parameters are required, synthetic, and are echoed in outputs. In this
example `cargo_snapshot` replaces the placeholder initial inventory with the exact
timestamp's normalized stock, **once**. It does not replace yard capacity or read
historical inflows/gate outflows into the simulation. Over-capacity stock is
rejected, never clipped or silently scaled. Missing hours are not forward-filled.
Use `inventory_source: explicit` to retain the supplied initial inventory,
including zero. Initial stock eligibility/dwell controls remain Phase 1 settings.

Optional `call_ids` limits the source cohort; IDs must be unique and present.
Omitting it selects all qualifying stays in the historical window. Every scenario
names its cohort and resource pool; no regional aggregate vessel counts or
Poisson arrivals are added. No terminal membership is inferred from length.

## Dataset and retrospective selection

The unchanged source has 4,704 records, 4,704 unique call IDs, five UTC timestamp
columns and the original 13-column schema. 4,400 records have no linked waiting
stay, 240 lack IMO, and actual cargo per call is absent. Each load is validated on
a copy: schema, IDs, numeric attributes, timestamp ordering, linked waits and
duration consistency. Invalid records cause an error rather than silent dropping.

For decision t0 and horizon H:

- Initial berthed: `berth_start <= t0 < berth_end`.
- Initial waiting: matched waiting evidence, `arrival <= t0 < berth_start`.
- Future: `t0 < arrival <= t0+H` (retrospective only).
- Completed-before-t0 and arrivals after H are excluded.

These sets are disjoint by call ID. Actual future berth starts are **not** passed
to the coupled engine as appointment or readiness times. Actual berth ends only
select retrospective initial membership; they never set simulated completion.
Identity-overlap checks use both vessel ID and MMSI, including cohort-history
overlaps involving selected calls. Possible duplicate physical identities remain
flagged and retained; no automatic de-duplication is claimed.

Unmatched `arrival=berth_start` is labeled `berth_start_fallback` and waiting
history UNKNOWN, never observed zero. Per-call quality includes this basis and
possible dataset-boundary censoring. All stays carry a gap-continuity limitation:
the processed file cannot establish uninterrupted ping-level observations.
Boundary flags are conservative and do not prove a stay was truncated.

Source timing, waiting linkage and active membership use finalized hindsight.
Outputs say `hindsight_assisted: true`; they do not claim future-arrival accuracy.
The synthetic Cargo snapshot also has retrospective/mixed-scope limitations.

## As-of experiments: a separate information boundary

The final call table has no publication/last-seen feed. Filtering it by a past
timestamp cannot establish active occupancy or attribute availability without
future information. Accordingly **as-of mode never reads or validates that table,
nor the finalized Cargo file**. It requires independently supplied active-call
observations and explicit scenario inventory. No future arrivals are invented.

To use the same route in as-of mode, set `time_information_mode: as_of`,
`inventory_source: explicit`, and provide `as_of_observations`, for example:

```json
[
  {
    "call_id": "snapshot-call-1",
    "vessel_id": "snapshot-vessel-1",
    "mmsi": 123456789,
    "available_at": "2025-03-04T10:00:00Z",
    "state": "berthed",
    "arrival": "2025-03-04T08:00:00Z",
    "arrival_basis": "berth_start_fallback",
    "berth_start": "2025-03-04T08:00:00Z"
  }
]
```

The example is a fabricated schema illustration, not an AIS observation. All
attributes and active membership are caller-asserted at t0; the project has no
independent feed to verify that assertion. `available_at <= t0`; arrivals and
known berth starts must already have happened by available_at. A waiting
observation requires `matched_wait_start` and cannot supply a future berth start.
Final berth end, realized duration, and other extra fields are rejected. Empty
snapshots are allowed; absent snapshots are not. Future arrivals are explicitly
UNSPECIFIED, not predicted to be zero. As-of numerical outcomes consequently
condition on current calls only and may understate later congestion.

Tests change future-only CSV rows, lengths and final durations (even making that
unread table invalid) and prove identical as-of scenario inputs/results. They
also fail if either finalized-source loader is called. This is information
isolation conditional on the supplied snapshot, not proof of a genuine historic
observation feed or predictive accuracy.

## Workload and accounting contract

Method/version `constant_per_call_v1`:

```
estimated original import work = full_import_units
work at t0 = estimated original work * berthed_remaining_fraction  # initially berthed
work at arrival = estimated original work                        # waiting/future
```

All assignments use `call_id`, normalized units and `SYNTHETIC_ESTIMATED`
provenance. They do not depend on length, stay duration, future berth completion,
future actual arrival count, the gate intervention or RNG. An equal-work model
is intentionally simple; it is not evidence that real vessels exchange equal
cargo. Existing stay estimators use finalized attributes and are not repurposed
as an as-of cargo model.

For initial berthed calls, `initial_import_workload` means **remaining at t0**,
not original cargo. The hypothetical original amount and fraction remain in
workload-estimation metadata. Prior assumed unloaded work is not injected into
the yard: initial inventory already supplies the yard state. Consequently the
conservation ledger starts with initial remaining work plus initial yard stock.
Calls remain berthed until their remaining synthetic work is unloaded, even if
the historical berth end lies outside H. Slot assumptions must accommodate
initial occupancy; extra slots are never created automatically.

The Phase 1 transfer mechanism, gate eligibility, shared productivity and H
endpoint convention are reused. Pre-t0 waits and pre-t0 berth occupation are
excluded from within-experiment exposure. Initially berthed vessels are excluded
from new admissions/mean completed queue wait, but included in completions and
occupied berth-hours. Unfinished calls retain work and state at H. Exact-H future
arrivals join waiting without service. See COUPLED_TWIN.md for other denominators.

`simulate_hybrid()` builds once and runs baseline plus an optional intervention
that changes **only** gate capacity. Cohort, timestamps, work assignments,
inventory, resources and queue policy remain paired. No ranking is applied.

## Actual historical demonstrations

Run `python -B -m scripts.demo_hybrid_twin` from the root. It writes no files.
Both examples use H=24h, 10,000-unit yard, 1,000-unit/hour shared unloading,
100-unit/hour baseline gate and 1,000-unit/hour intervention. Work is 1,000/call,
with 50% remaining for initial berthed vessels. Slot counts (13 and 17) are
synthetic choices that accommodate the selected occupancy, not physical POLA
infrastructure. All initial stock is assumed eligible at t0.

| Measure | A: March 4, 10:00 UTC | B: July 10, 19:00 UTC |
|---|---:|---:|
| Eligible calls | 20 | 23 |
| Initial berthed / waiting / future | 13 / 0 / 7 | 17 / 1 / 5 |
| Matched / unmatched waiting histories | 1 / 19 | 5 / 18 |
| Initial inventory | 7,204.03, finalized synthetic Cargo snapshot | 10,000, controlled full-yard assumption |
| Total remaining/arrival workload | 13,500 | 14,500 |
| Completed vessels, baseline -> intervention | 0 -> 19 | 0 -> 22 |
| Queue exposure, vessel-hours | 85 -> 12 | 75 -> 15 |
| Unloaded units | 5,195.97 -> 12,500 | 2,400 -> 13,500 |
| Remaining aboard at H | 8,304.03 -> 1,000 | 12,100 -> 1,000 |
| Yard-blocked service, vessel-hours | 273 -> 0 | 408 -> 0 |
| System residual, absolute | < 4e-12 both runs | < 4e-12 both runs |

Each example has one exact-H arrival that receives no service. The intervention
benefit emerges from the chosen yard/resource assumptions; it is not estimated
real-world operational benefit. A zero-productivity control shows no gate
benefit; sufficient headroom or ineligible stock can also make gate capacity
nonbinding, as covered by the Phase 1 tests.

## Next phase and limitations

Next prioritize a verified timestamped call-status feed, identity reconciliation,
remaining-work sensitivity and cohort validation, then calibrate work/resource
ratios against independent data. Real container-terminal membership, manifests,
physical capacity and progress are still absent. No export loading, terminal
routing, physical crane allocation, model retraining, AI tools or operational
prediction claims are introduced here.

Local integration note (2026-10-17): the repository demo configuration explicitly enables this
backend feature. Missing sections and the typed settings defaults remain disabled. Latest Main
frontend and AIS replay are preserved; experiments require separate explicit API requests.
