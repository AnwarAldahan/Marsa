# Methodology note — Agent 2 Port Operations 2025

## Real inputs
- Timestamp index and maritime exogenous signals are derived from the supplied 2025 LA/LB AIS files. Exactly 8,688 available UTC hours are exported; the 72 missing AIS hours are not exported.
- AIS cargo vessel types 70–79 are emphasized. Cargo vessel count, near-port activity, approaching activity, waiting-area behavior, ship density and port throughput are aggregated hourly.
- Cargo influence uses a distributed 2–10 hour lag so maritime relevance does not create instantaneous yard inflow.

## Public calibration anchors supplied in the brief
- 2025 Port of Los Angeles monthly loaded import/export TEU targets. Hourly weights are normalized within each month across available AIS timestamps, so monthly synthetic sums match the targets to numerical precision.
- Truck-bound San Pedro Bay dwell-time examples supplied for April, July, August and December guide synthetic dwell baselines. Other monthly dwell baselines are explicitly synthetic interpolation/assumptions, not claimed observations.

## Synthetic assumptions
- yard_capacity = 10,000 normalized container-equivalent units. This is a modeling scale, NOT physical Port of Los Angeles capacity.
- Truck demand, gate processing capacity, queue dynamics, normalized yard-flow scaling and non-anchor monthly dwell baselines are synthetic assumptions.
- Missing AIS hours are internally simulated only to preserve continuous stock-flow dynamics; they are excluded from the final CSV and official monthly TEU allocation.
- Random seed = 42. Randomness perturbs structured dynamics; it does not define causal relationships or status labels.

## Status logic
operations_status is derived from a transparent multi-signal score using occupancy thresholds (65/75/85%), truck-wait thresholds (30/45/70 min), dwell elevation over monthly baseline, cargo-flow imbalance and six-hour accumulation persistence. CRITICAL requires multiple severe simultaneous/persistent signals; it is never randomly assigned and is not a future congestion prediction.
