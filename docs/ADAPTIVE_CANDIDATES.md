# Evidence-driven candidate evaluation priority

This optional Strategy policy uses complete historical diagnostics to change the order in which existing candidates reach the Digital Twin. It preserves the legacy candidate pool, baseline, actions, parameters, feasibility gates, forecast/event rules, physics and scoring formulas. It introduces no new agent, model, data, dependency or Gemini call.

## Enable independently

Both features default to disabled in code when omitted. The current Stage 2 prototype profile in `config/port_config.yaml` explicitly enables them:

```yaml
strategy_investigation:
  enabled: true
  use_gemini: true  # false uses the existing deterministic diagnostic router
  max_rounds: 2
  max_tool_calls: 3
  window_hours: 6

adaptive_candidates:
  enabled: true
```

The new section accepts only a strict boolean `enabled`. Its absence or `false` preserves the exact previous pipeline output and candidate behavior, including when investigation is enabled. It does not change how the existing provider loads `.env`. No additional LLM requests are needed. A completely offline Python run supplies `llm=None`.

The pipeline is now:

`domain assessments → unchanged investigation → legacy eligible pool → optional deterministic priority → Twin in priority order → unchanged scorer in original comparison order → recommendation/human review`

## Policy and evidence validation

The baseline always runs first. Remaining experiments sort by the following ordinal bands, with stable legacy order within each band. The numbers are ordering categories, not confidence percentages.

| Band | Historical support for experiment relevance | Rule |
|---|---|---|
| 0 | Reference | Baseline; always selected |
| 1 | Strongly supported | Queue: both waiting count and ratio increase. Gate: inventory and net flow increase, and truck waiting increases. Combined: all component hypotheses supported |
| 2 | Mixed | Some component signals support worsening and others contradict it, or a component hypothesis is mixed |
| 3 | Unsupported worsening trend / conditional | No component supports worsening; preserve the existing experiment because current status or forecast still makes it eligible |
| 3 | Berth experiment, conditional | Never promote a berth experiment based on regional waiting trends alone; these do not verify POLA berth availability or necessity |

For `extra_berth`, a supported rising regional trend is qualified as mixed and remains conditional. Declining trends stay unsupported as evidence of *worsening*, not as a claim that the action is useless. Improving Cargo never automatically removes a gate experiment. The existing six-hour ML cargo-proxy forecast influences eligibility through its existing rules only; this policy does not interpret scores as probabilities or override current evidence. Weather and event conditions remain unchanged, with unknown weather unresolved.

Required evidence is `maritime_history` for queue/berth candidates, both Cargo diagnostics for gate candidates, and all three for combined candidates. Results must have complete seven-point exact-hour coverage, the matching authoritative timestamp and six-hour window, matching tool/evidence identity, non-null required metrics, appropriate source scope/limitations, and synthetic Cargo provenance. Overlapping endpoint facts must match canonical domain evidence. Required hypothesis states and metric references are checked against the existing deterministic hypothesis calculation. LLM preferences and generated explanations are not policy inputs.

Unavailable/skipped investigation, missing or incomplete required evidence, conflicting endpoints, stale/future timestamps, invalid provenance, tampered hypotheses, duplicates or invalid settings produce `legacy_fallback` with a reason code. The candidate objects and original evaluation order are preserved. Complete diagnostics from deterministic provider fallback can support adaptive priority just like complete diagnostics selected by Gemini. An incomplete unrelated diagnostic does not invalidate a complete domain needed by the actual candidate pool.

The policy reads already acquired evidence only. It does not read additional history, labels, future rows or weather forecasts. Exact-window and future-isolation safeguards remain in the original diagnostic tools.

## Selection, budget and ranking distinctions

- **Selected for simulation:** every candidate in the legacy eligible pool, including baseline.
- **Prioritized for evaluation:** the actual execution order passed to the Twin, based on the bands above.
- **Conditionally retained:** candidates with mixed/unsupported recent worsening evidence, or insufficient evidence during fallback. They still execute.
- **Excluded for verified infeasibility:** none added. The available diagnostics cannot establish new terminal-level feasibility constraints; original eligibility gates remain authoritative.
- **Not selected due to budget:** none removed from the supplied pool. The original generator already caps that pool at four; this policy preserves that cap and does not construct a fifth alternative. The report's available IDs describe this capped pool, not every action in the Twin's catalog. Original rules that withhold a combined alternative when the pool is full remain unchanged.

Priority changes the order of real simulation calls, not just explanatory text. All selected alternatives are still evaluated with the same per-candidate seeds. Before calling the unchanged scoring function, raw results are put back in legacy comparison order. This preserves its stable tie-breaking behavior and prevents priority from silently selecting a different equal-score winner.

The API adds an optional root `candidate_selection` section with mode, available/selected IDs, priorities, historical support, hypothesis/metric references, rationales, conditional IDs, exclusions, budget status and fallback reason. Existing candidate objects are not edited by this policy. `strategy_candidates` follows execution order when adaptive mode is enabled. Original response fields remain available. The original investigation and its annotations remain unchanged; their legacy-preserved policy describes eligibility/actions, while the new section describes execution priority.

## Historical evaluation: modes A / B / C

- **A:** investigation disabled, original generation.
- **B:** deterministic investigation enabled, adaptive feature disabled.
- **C:** deterministic investigation and adaptive priority enabled.

The two additional cases below were found by searching the existing hourly sources for increasing Cargo inventory/delay with declining Maritime count/ratio, then verifying complete exact-hour windows and actual canonical statuses through the pipeline. This retrospective case search tests policy behavior; it is not an unbiased operational-quality evaluation.

For compactness: `B` = baseline, `Q` = shortest_first, `G` = gate_extension, `R` = extra_berth.

| Timestamp UTC | A order | B order | C order | Additional evidence / conditional retention | Winner in A/B/C |
|---|---|---|---|---|---|
| 2025-07-10 19:00 | B,Q,G | B,Q,G | B,Q,G | Inventory −28.01, net flow −28.02, truck wait −3.50 min; Maritime waiting +4 and ratio +0.087432. Q prioritized; G retained conditionally despite draining | G |
| 2025-01-12 02:00 | B,G | B,G | B,G | Inventory/net flow +72.23; truck wait +34.57 min. Strong gate relevance; current ELEVATED Cargo and LOW future risk remain separate signals | G |
| 2025-11-02 08:00 | B,Q | B,Q | B,Q | Waiting +1, ratio +0.053419. Queue relevance supported; unknown weather stays unresolved | B |
| 2025-07-18 10:00 | B,Q,G | B,Q,G | **B,G,Q** | Inventory +22.85, net flow +22.87, truck wait +14.63 min; waiting −1, ratio −0.036051. G runs before conditionally retained Q | G |
| 2025-08-10 07:00 | B,Q,G,R | B,Q,G,R | **B,G,Q,R** | Inventory +123.97, net flow +123.98, truck wait +13.69 min; waiting −1, ratio −0.012443. G prioritized; Q and R conditional. Four-candidate cap reached | R |

Numerical diagnostic excerpts are rounded for display. No membership, action or parameter changed in these five cases; no useful candidate was excluded. In the last case the conditional berth experiment still wins the simulation, illustrating why declining regional pressure is insufficient grounds for removal and why experiment priority must remain separate from ranking.

For all five cases, tests confirm identical domain assessments, ML outputs, scenario inputs, simulated KPIs, scores, ranking and deterministic recommendations across modes. A and B differ only by optional investigation data/annotations. C adds `candidate_selection` and reorders existing candidate records in two cases. Simulated alternatives and final recommendations do not change. Direct spies verify the Twin's invocation order and action dictionaries, rather than relying only on response text.

The demonstrated additional relevance is that worsening Cargo evidence moves its experiment ahead of improving Maritime experiments in two actual historical cases. There is no claimed improvement in operational outcomes, forecast accuracy, winner quality, compute cost or latency. All alternatives still run; this version does not reduce simulation work.

## Files changed for this capability

| File | Purpose |
|---|---|
| `src/marsa/agents/candidate_selection.py` | Independent settings, evidence checks, deterministic priority and optional selection report |
| `src/marsa/agents/strategy_agent.py` | Exposes policy within Strategy; restores legacy comparison order before unchanged scoring |
| `src/marsa/pipeline.py` | Connects selection after investigation and before Twin execution; adds optional response section |
| `config/port_config.yaml` | Independent feature flag; current prototype profile enables it |
| `tests/test_adaptive_candidates.py` | Policy, fallback, provenance, budget, invocation order, tie and API regression tests |
| `README.md` | Links to this feature and explains priority versus ranking |
| `docs/STRATEGY_INVESTIGATION.md` | Notes the independent feature without changing investigation contracts |
| `docs/ADAPTIVE_CANDIDATES.md` | Rules, configuration, historical comparisons and validation |

Original investigation/tool implementation, datasets, models, training, domain agents, frontend, Twin physics and scoring formulas are unchanged. Existing uncommitted work was preserved.

## Validation and limits

Before changes: **319 passed, 1 existing warning in 29.45s**, using `python -m pytest tests -o addopts= -q`.

New targeted tests: **32 passed in 10.60s**. They cover independently disabled behavior, rising/mixed/improving signals, incomplete evidence, unavailable investigation/provider fallback, future timestamps, source scope, endpoint conflicts, hypothesis tampering, budget preservation, baseline retention, event combinations, unchanged actions/parameters, actual evaluation order, stable score ties, API compatibility and all five historical A/B/C comparisons.

Automated tests block real HTTP transport through `tests/conftest.py`, even when a live `.env` is present. `httpx.MockTransport` tests still exercise the Gemini adapter; provider-failure tests are mocked. Legacy comparison tests explicitly configure their feature flags, so enabling the prototype YAML does not change their baseline. This feature makes **no additional live Gemini calls**. The existing investigation's earlier live validation is separate from this task.

Initial adaptive implementation result: **351 passed, 0 failures, 1 existing warning in 28.87s**. All 319 original tests and 32 adaptive tests passed. The original offline regression invocation was:

```python
from unittest.mock import patch
import pytest

with patch("httpx.HTTPTransport.handle_request",
           side_effect=RuntimeError("Live HTTP disabled for regression tests")):
    result = pytest.main(["tests", "-o", "addopts=", "-q"])
```

Pre-push verification after test-isolation hardening: **352 passed, 0 failures, 1 existing warning in 37.53s** with the normal `python -m pytest tests -o addopts= -q` command and the enabled prototype YAML. `tests/conftest.py` now blocks live HTTP automatically; explicit legacy configurations make the comparison tests independent of the user's YAML flags. `python -m ruff check src tests scripts` and `git diff --check` also pass.

The additional regression holds initial planner inputs constant and changes only a labelled counterfactual first-tool result; a conditional mock then selects a different second tool. This verifies result-dependent loop integration, not live Gemini reasoning. Live executions have demonstrated valid batch selection; evidence-dependent live follow-up remains unverified.

The existing warning is `Timestamp.utcnow()` deprecation at `prediction_model/src/final_proxy.py:346`. No unrelated warning fix is included.

Suitable as an opt-in Stage 2 prototype priority policy within these tested limits. It is not a validated operational optimization or a production feasibility system. Strong support refers to descriptive evidence for prioritizing a comparative experiment, not to confidence that its intervention will help. Safe candidate suppression/membership changes would require additional feasibility/causal evidence and separately tested rules.
