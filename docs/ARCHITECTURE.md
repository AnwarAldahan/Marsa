# Architecture and contracts

**Pipeline** (`src/marsa/pipeline.py`): snapshot → forecast → agents → strategy → twin → score → recommendation.

**Forecast JSON**: `{timestamp_utc, target, current_value, horizons: {"6h": {predicted, change_vs_now, top_drivers[]}}}`

**Agent JSON** (all three): `{agent, status/level, findings[{metric, value, observation}], possible_bottleneck, provenance[]}`.
Agents describe; they never recommend. The context agent also returns `twin_multipliers` (crane/gate/arrival).

**Candidate** (strategy → twin): `{id, label, actions:[{type, value}]}` with action types
`queue_policy | add_berths | crane_boost | gate_boost | delay_arrivals | prioritise_vessel`.

**Twin result** (twin → strategy): `{candidate_id, runs, kpis:{avg_wait_hours, max_wait_hours, delayed_vessels, vessels_served,
berth_utilization, yard_peak_occupancy, yard_end_occupancy, queue_end} each {mean,p10,p90}}`.
The twin never picks a winner.

**Ranking**: `twin/scoring.py` — min-max normalised weighted sum, lower is better, paired Monte Carlo seeds so
candidates are compared on identical arrival streams.

**Recommendation**: rule "no intervention" if the best candidate's score gain < 0.05; the LLM (optional) only rewrites the narrative.
