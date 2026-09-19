# Marsa Final Architecture

```text
Exact 2025 UTC hour
  -> authoritative port snapshot
  -> ML C1 forecast boundary
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

The ML contract is the probability of C1 congestion occurring in `(t, t + 6h]`.
The Digital Twin independently simulates candidates over the configured horizon,
currently 24 hours. Simulation output is not an ML forecast.

See `docs/FINAL_INTEGRATION.md` for complete runtime contracts.
