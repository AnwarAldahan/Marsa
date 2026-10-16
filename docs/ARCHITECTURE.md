# Marsa Final Architecture

```text
Exact 2025 UTC hour
  -> authoritative port snapshot
  -> ML cargo-congestion proxy forecast
  -> Maritime Agent
  -> Cargo Agent
  -> Events & Weather Agent
  -> Strategy Agent
  -> candidate strategies
  -> Digital Twin simulations
  -> deterministic scoring
  -> decision support
  -> human decision
```

The four agents are Maritime, Cargo, Events & Weather, and Strategy. There is no
Orchestrator Agent. ML is not an agent. The Digital Twin is not an agent.

Domain agents report evidence and limitations independently. They do not change
their status in response to an ML forecast. The Strategy Agent preserves conflicting
signals, runs the Digital Twin with no change, proposes actions only when that run shows a
berth shortage, and
explains deterministic simulation ranking.

The ML contract is an uncalibrated XGBoost score for the operational
cargo-congestion proxy at `t + 6h`. It is a retrospective predictive prototype,
not independently observed congestion ground truth or production online inference.
The Digital Twin replays the real AIS schedule of Port of Los Angeles cargo vessels over
24 hours (optionally under a user stress scenario) and ranks options by total vessel-hours
waiting. Simulation output is not an ML forecast, and the ML forecast does not feed the twin.

## Runtime surfaces

- `scripts/run_pipeline.py <UTC hour>Z` — one cycle from the command line (deterministic narrative).
- `POST /api/twin/simulate` — one user decision vs no change (optional `stress`, `added`); `GET /api/twin/vessels` — known vessels
- `POST /api/decision-support/analyze` — same cycle over HTTP (optional `stress`, `added`); the strategy narrative is Gemini-assisted
  when `GEMINI_API_KEY` is set (Arabic, guard-railed), otherwise deterministic.
- `dashboard/marsa.html` — renders only the analyze response (sections ١–٤, berth map, twin replay); no demo data.

## Decision-support response (key fields)

```json
{
  "authoritative_timestamp": "2025-07-10T19:00:00+00:00",
  "port_state": { "...one row of merged_port_dataset_2025_v2..." },
  "ml_forecast": { "status": "success", "classification": "HIGH", "congestion_score": 0.96, "horizon_hours": 6 },
  "domain_agents": { "maritime": {...}, "cargo": {...}, "events_weather": {...} },
  "strategy_candidates": [ { "candidate_id": "extra_berth", "actions": [ {"type": "add_berths", "value": 1} ] } ],
  "digital_twin": { "scenario": {...}, "sources": [...], "results": [ { "candidate_id": "...", "rank": 1, "score": 0.0, "kpis": {...}, "timeline": [...], "vessels": [...] } ] },
  "decision_support": { "highest_ranked_candidate": "...", "summary": "...", "reasoning_mode": "llm_assisted", "ranking": [...] },
  "human_approval_required": true
}
```

See `docs/FINAL_INTEGRATION.md` for complete runtime contracts.