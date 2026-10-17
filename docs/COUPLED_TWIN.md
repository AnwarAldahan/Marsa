# Phase 1 coupled normalized Twin

This independent, opt-in import-only engine models one **abstract aggregated
container-handling yard** and an explicit scenario cohort/resource pool. It is
not a physical POLA yard, an agent, or an operational forecast. Historical replay
remains the default `/api/twin/simulate` behavior and the Strategy pipeline remains
unchanged, including ranking, Gemini tools, adaptive priorities and XGBoost.

## Selection and configuration

`config/port_config.yaml` has a separate `coupled_normalized` section, disabled
by default. Only the new `POST /api/twin/simulate-coupled` route reads it. Set
`enabled: true` on the server to opt in; a client cannot bypass this flag.
Missing sections in older configs leave the route disabled (HTTP 403). Invalid
enabled server settings return 503. Invalid scenario inputs return 422.

The standalone Python entry point `simulate_coupled(CoupledScenario(...))` is
explicit opt-in and does not need the HTTP feature flag. It reads no datasets,
environment credentials or configuration files. There is no implicit fallback
between engines and no coupled recommendation/ranking in Phase 1.

Example server configuration (all numeric values are synthetic assumptions):

```yaml
coupled_normalized:
  enabled: false
  mode: coupled_normalized
  parameters:
    yard_capacity: 10000
    initial_inventory: 0
    berth_slots: 1
    unloading_units_per_hour: 100
    gate_units_per_hour: 100
    release_delay_hours: 0
    initial_inventory_eligible_in_hours: 0
    units: normalized_units
    provenance: SYNTHETIC_SCENARIO_ASSUMPTION
  storage_sensitivity: [9090.91, 10000, 10909.09]
```

Example JSON body for the **new route**, after server enablement:

```json
{
  "mode": "coupled_normalized",
  "timestamp_utc": "2025-01-01T00:00:00Z",
  "horizon_hours": 4,
  "cohort_id": "explicit-two-vessel-demo",
  "resource_pool_id": "abstract-demo-yard",
  "parameters": {
    "yard_capacity": 10000,
    "initial_inventory": 10000,
    "berth_slots": 1,
    "unloading_units_per_hour": 100,
    "gate_units_per_hour": 100
  },
  "vessels": [
    {"simulation_id": "A", "arrival_timestamp": "2025-01-01T00:00:00Z", "initial_import_workload": 100},
    {"simulation_id": "B", "arrival_timestamp": "2025-01-01T00:00:00Z", "initial_import_workload": 100}
  ]
}
```

Omitting `parameters` uses the complete server parameter object. Supplying it
replaces that object, with required berth/productivity/gate/initial-stock values;
it is not a partial override. Results echo all resolved parameters, cohort,
resource pool, scope, vessel provenance, timeline and per-interval flow ledger.

## Units and provenance

Inventory, explicit vessel work and gate clearance use **the same normalized
unit**. Productivity is the whole pool's normalized units/hour, not per crane.
The default single berth and 100-unit/hour rates are illustrative synthetic
scenario assumptions. They are not calibrated physical capacities.

The 10,000-unit base is the existing Cargo modeling scale. Storage sensitivities
are rounded `10000 * [3.0, 3.3, 3.6] / 3.3`; these are relative assumptions using
the historical stacking-height envelope, not TEU conversions.

Physical references remain outside runtime calculations:

- [2025 terminal map](https://kentico.portoflosangeles.org/getmedia/b1277c2b-c88c-4c1e-aedb-113706d28543/2025-pola-container-terminals-map): 22 reported berths across seven terminals.
- [Facts & Figures](https://portoflosangeles.org/business/statistics/facts-and-figures): current reported 85 STS cranes.
- [Pier 300 study, December 2011 appendix](https://kentico.portoflosangeles.org/getmedia/f5ff7577-27a6-485f-8763-1525a59e6162/Appendix-C2-Automation-WP): historical engineering assumptions, including 3.0-3.6 stacking and 27 moves/crane-hour.

No Cargo CSV arrivals/departures or annual POLA TEU are fed into this engine.
`call_id` is optional compatibility metadata and must be unique if supplied.
No AIS records are auto-loaded or selected. Historical arrival inputs must be
explicitly labeled as retrospective; the engine does not establish when an
arrival was known. Workloads are always explicitly synthetic, never observed TEU.

## Hourly event convention

Inputs are finite, UTC aware, with arrivals on the grid relative to t0. Phase 2
adds explicit initial waiting/berthed states for pre-t0 arrivals; the standalone
Phase 1 defaults still start vessels as future. H is an integer
1-720 hours. Each interval `[h,h+1)`:

1. Admit arrivals; assign vacant slots FCFS (arrival, then simulation ID).
2. Clear eligible opening yard lots up to the gate rate. Demand is assumed
   sufficient to collect all eligible stock; no physical truck queue is modeled.
3. Allocate the shared productivity pool by max-min fair sharing; redistribute
   unused shares from small workloads. Limit those allocations by yard headroom,
   using the same fair-sharing rule.
4. Transfer actual quantities from vessels to the yard. New cargo becomes eligible
   at `h+1+release_delay_hours`. The initial stock has one explicitly configured
   eligibility time. Eligibility alone never removes inventory.
5. Complete zero-remaining-work vessels at h+1 and release their slots. A new
   vessel can use those slots only at the next opening. No mid-hour reuse or
   fractional completion is inferred.

At H, finish the last interval, admit exact-H arrivals to waiting, and record
state. No berth assignment, clearance or unloading happens at H. Later arrivals
remain future; all work remains represented. State snapshots are taken after
opening admissions/assignment and before gate/unloading. Flows describe intervals.

The optional `shortest_estimated_workload_first` policy requires a separate,
explicit estimate for every vessel; it does not inspect actual remaining work
to infer an unavailable estimate. Workload estimates remain fixed across runs.

## Invariants and KPIs

Per interval: `unloaded <= remaining work`, `unloaded <= pool rate * 1h`,
`unloaded <= headroom after gate`, `gate outflow <= eligible opening stock`,
`closing yard = opening yard + unloaded - cleared`. Vessel work decreases only
by actual transfer. No inventory or work is clipped/discarded. Global residuals:

```
initial vessel work - remaining vessel work - cumulative unloaded
initial yard + cumulative unloaded - cumulative cleared - current yard
initial vessel work + initial yard - remaining vessel work - current yard - cumulative cleared
```

Runtime checks reject conservation, negative-stock or yard-capacity violations
beyond `1e-9 * max(1, initial vessel work + initial yard)`; figures are not rounded
before accounting. Nonfinite totals/budgets are rejected.

- Queue exposure: number waiting after assignment times interval hours, summed.
- Completed queue wait: arrival-to-admission duration, averaged across **all
  admitted vessels including zero waits**, even if vessel service is unfinished.
  Unadmitted arrived vessels have separately reported right-censored waits.
  For Phase 2, only the part of each wait within [t0,H] is counted; initially
  berthed vessels have no new queue admission and are excluded from this average
  and `vessels_berthed`. Their service completions are still counted normally.
- Occupied berth-hours: occupied slots per full interval; utilization divides by
  `berth_slots * H`. Zero slots gives null utilization.
- Yard peak includes t0 and the H endpoint; occupancy percentages divide by the
  configured normalized capacity. Gate-first ordering cannot create a hidden
  within-step peak higher than the interval endpoints.
- Yard-blocked unloading: potential allocation minus accepted transfer, in units
  of **cumulative lost service opportunity**, not unique stranded cargo.
- Blocked service vessel-hours: count of vessels whose potential allocation was
  yard-restricted per interval; yard-blocked hours counts each such interval once.
- Completed vessels require zero remaining work; admissions are separate.
- Unfinished vessels counts arrived waiting/berthed vessels, including exact-H
  arrivals; future vessels are reported separately. Remaining aboard includes
  **all supplied vessels, including future ones**, for total conservation.

## Acceptance demonstration and limitations

Run `python -B -m scripts.demo_coupled_twin` from the root. It writes no files.
Both cases: H=4, yard capacity/initial stock=10000, one berth, two vessels arriving
at t0 with 100 units each, pool productivity=100, immediate eligibility.

| Measure | A: gate 0 | B: gate 100 units/hour |
|---|---:|---:|
| Unloaded | 0 | 200 |
| Completed vessels | 0 | 2 |
| A/B berth start hours | 0 / none | 0 / 1 |
| A/B completion hours | none / none | 1 / 2 |
| Queue exposure, vessel-hours | 4 | 1 |
| Occupied berth-hours | 4 | 2 |
| Cargo cleared | 0 | 400 |
| End yard | 10000 | 9800 |
| System conservation residual | 0 | 0 |

Gate clearance brings no unloading benefit when productivity is zero or cargo
is ineligible. With sufficient yard headroom and a binding productivity limit,
extra gate capacity can drain the yard without advancing vessel completion.
Tests cover these controls as well as capacity, conservation, boundary semantics,
input validation and isolation from the existing agent/replay pipeline.

Phase 2's independent adapter and endpoint are documented in [HYBRID_TWIN.md](HYBRID_TWIN.md).
There is no physical crane allocation, export loading, seven-terminal routing,
empirical workload estimation or predictive-accuracy claim.

Local integration note (2026-10-17): the repository demo configuration explicitly enables this
backend feature. Missing sections and the typed settings defaults remain disabled. Latest Main
frontend and AIS replay are preserved; experiments require separate explicit API requests.
