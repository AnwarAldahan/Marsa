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
signals, generates a small candidate set from supported Digital Twin actions, and
explains deterministic simulation ranking.

The ML contract is an uncalibrated XGBoost score for the operational
cargo-congestion proxy at `t + 6h`. It is a retrospective predictive prototype,
not independently observed congestion ground truth or production online inference.
The Digital Twin independently simulates candidates over the configured horizon,
currently 24 hours. Simulation output is not an ML forecast.

## Runtime surfaces

- `scripts/run_pipeline.py <UTC hour>Z` — one cycle from the command line (deterministic narrative).
- `POST /api/decision-support/analyze` — same cycle over HTTP; the strategy narrative is Gemini-assisted
  when `GEMINI_API_KEY` is set (Arabic, guard-railed), otherwise deterministic.
- `dashboard/marsa.html` — reads the analyze response and fills sections ١–٤ and the twin cards.

## Decision-support response (key fields)

```json
{
  "authoritative_timestamp": "2025-07-10T19:00:00+00:00",
  "port_state": { "...one row of merged_port_dataset_2025_v2..." },
  "ml_forecast": { "status": "success", "classification": "HIGH", "congestion_score": 0.96, "horizon_hours": 6 },
  "domain_agents": { "maritime": {...}, "cargo": {...}, "events_weather": {...} },
  "strategy_candidates": [ { "candidate_id": "gate_extension", "actions": [ {"type": "gate_boost", "value": 1.3} ] } ],
  "digital_twin": { "scenario": {...}, "results": [ { "candidate_id": "...", "rank": 1, "score": 0.0, "kpis": {...} } ] },
  "decision_support": { "highest_ranked_candidate": "...", "summary": "...", "reasoning_mode": "llm_assisted", "ranking": [...] },
  "human_approval_required": true
}
```

See `docs/FINAL_INTEGRATION.md` for complete runtime contracts.