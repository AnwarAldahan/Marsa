# Marsa Stage 2 — Dynamic Tool Selection implementation

## Outcome and scope

Implemented optional, bounded Gemini diagnostic selection inside the existing Strategy responsibility. The three domain agents retain ownership of their canonical assessments. Python computes historical diagnostics and evidence-backed hypotheses; the existing Digital Twin and scoring determine simulated ranking. There is no fifth agent, new framework, dependency, database, dataset, model training, or operational execution.

The conservative candidate integration adds **separate diagnostic justification only**. Every existing candidate, action and parameter remains governed by the legacy deterministic rules. Improving recent progression does not remove a candidate or downgrade current status. No decision-quality improvement percentage or production-readiness claim is made.

## Inspection and integration points

Before editing, inspected `pipeline.py`, `strategy_agent.py`, the structured Gemini provider, all three domain agents and source adapters, their Pydantic evidence models, configuration, API routing/dependency factories, and existing tests. No `AGENTS.md` was present in the workspace search. The baseline full suite passed before implementation.

The integrated pipeline collects the existing three assessments, invokes optional Strategy investigation, and passes the returned report into existing candidate generation. Investigation only attaches a new candidate field. Scenario construction, simulation, scoring and recommendation then use their original code paths. Enabled reports are added as a new optional root key. Disabled runs do not add either field.

The existing API already supplies a structured Gemini provider to `pipeline.run`; this provider now supports both investigation selection and the existing independent recommendation narrative. Direct Python/script runs with `llm=None` use the deterministic diagnostic router. Existing domain agents, API request schemas, routes, ML and Twin implementations were not modified.

## Exact files added or changed for this implementation

| File | Change |
|---|---|
| `src/marsa/agents/tools/diagnostics.py` | Three typed, exact-hour, read-only diagnostics and source/coverage contracts |
| `src/marsa/agents/investigation.py` | Strict settings/plans; Gemini planning; deterministic routing; bounded execution; hypotheses; candidate annotations |
| `src/marsa/agents/strategy_agent.py` | Exposes investigation within Strategy and optionally annotates existing candidates |
| `src/marsa/pipeline.py` | Calls optional investigation and adds its report when enabled |
| `config/port_config.yaml` | Disabled-by-default investigation settings |
| `tests/test_diagnostic_tools.py` | Historical calculations, missing values, timestamp validation, provenance and future isolation |
| `tests/test_strategy_investigation.py` | Mocked planning/transport, invalid inputs, budgets, failures, API and historical compatibility |
| `docs/STRATEGY_INVESTIGATION.md` | Configuration, PowerShell usage, contracts, planner/fallback behavior and limitations |
| `README.md` | Describes optional tool selection and links to the guide/report |
| `reports/STAGE2_DYNAMIC_TOOL_SELECTION_IMPLEMENTATION.md` | This implementation and validation report |

Previously created audit reports were preserved. Datasets, artifacts, training code, frontend, domain-status definitions, supported action catalog, geography assumptions, physics and ranking formulas were not edited.

## Tools and historical safeguards

- `cargo_balance`: endpoint inventory delta, six-bin net cargo flow, yard-occupancy change. Inventory/flow retain the Cargo source's normalized modeling units; occupancy changes are percentage points.
- `cargo_delay_profile`: truck waiting-time delta in minutes and six-bin truck/gate flow totals, without interpreting gate throughput as measured queue size or causal capacity evidence.
- `maritime_history`: waiting-count and waiting-ratio deltas, start/end count and ratio context. Count deltas use vessels; ratios remain fractions rather than percentages.

For timestamp `t`, each endpoint window is exactly `[t-6h,t]` at UTC hours. Flow totals use `(t-6h,t]`, six bins rather than seven. Missing clock hours remain missing. Zero is preserved. Missing/invalid values are explicitly reported; incomplete diagnostics cannot establish supported/unsupported hypotheses. The timestamp must be an exact UTC hour within the existing 2025 replay contract.

CSV columns are explicitly allowlisted, with labels excluded. Timestamp filtering precedes numeric conversion and computation; all future observations are excluded from resulting evidence. Because these are CSV files, file scanning can read future rows as strings; this is not an as-of storage system. A mutation test places corrupted future rows first and verifies unchanged diagnostic results for all three tools.

Each result includes source identifier, source fields, scope, provenance, per-metric units, missing timestamps/values and limitations. Cargo remains synthetic/calibrated synthetic with mixed regional/POLA construction. Maritime remains regional LA/LB with unverified POLA-only membership. Prefix filtering cannot remove retrospective source-construction limitations. No additional geographic filtering was introduced.

## Gemini selection, execution and fallback

The deterministic question catalog covers cargo progression/delays and maritime waiting progression. Elevated current Cargo and low future ML risk can coexist; both signals are preserved. Conversely, a HIGH forecast can prompt Cargo diagnostics even when current Cargo is NORMAL. Missing weather context remains unresolved because no weather diagnostic exists in this initial catalog.

Gemini receives whitelisted domain evidence/statuses, forecast metadata including calibration/limitations, open questions, allowed tool names/descriptions, the sole allowed window, remaining budgets and validated prior results/hypotheses. It returns a strict structured plan containing question IDs, tool names and arguments. A second round can adapt to the first round's evidence.

Validation rejects unknown tools, extra fields, file paths, caller-selected timestamps, arbitrary interventions, invalid windows/types, unrelated question/tool mappings, duplicate requests and over-budget batches **before execution**. No free-form Gemini factual text is added to diagnostics or hypotheses. The existing Gemini provider is reused unchanged; only the new plan schema inlines its Pydantic definitions for the existing responseSchema adapter.

The limits are at most two planning/execution rounds and three diagnostic attempts per analysis, including failed/incomplete attempts. Lower budgets are configurable; increasing hard caps is rejected. A tool/window combination cannot be repeated. Stop decisions `NO_TOOL_NEEDED` and `INSUFFICIENT_EVIDENCE` leave unanswered questions visible. Planning stops when no useful unattempted diagnostic remains or budget is exhausted. These limits apply to investigation; the existing optional recommendation-narrative call remains independent.

When diagnostic Gemini is disabled, no provider is supplied, or the provider/plan fails validation, the deterministic router handles relevant Cargo balance/delays first and Maritime history next, subject to the same global budgets. Valid results from earlier rounds are retained after later fallback. Raw provider errors and invalid response contents are not exposed. Tool failures become typed errors; unexpected optional failures produce a skipped report. Overlapping endpoint facts must match canonical domain evidence or the diagnostic fails validation. The original analysis continues.

## Hypotheses and candidate justification

Python evaluates cargo accumulation, high current Cargo pressure with recent draining, worsening truck delays, and rising Maritime waiting pressure. States are `supported`, `mixed`, `unsupported`, or `insufficient`. Supporting/contradicting metric references are included; insufficient evidence references diagnostic coverage. These are descriptive checks, not causal findings.

Candidates receive a separate `diagnostic_justification` containing the policy `legacy_preserved`, relevant hypothesis IDs/states and diagnostic evidence IDs. For supported recent Cargo draining, the gate candidate is explicitly framed as a comparative what-if despite current elevated pressure. Baseline and all supported alternatives remain available under original rules. No selection-changing rule was introduced, so no operational scenario is added or removed by Gemini.

## Historical example

Actual offline run at **2025-07-10T19:00:00Z**, with investigation enabled and no Gemini provider:

```json
{
  "mode": "deterministic_fallback",
  "rounds_used": 2,
  "tool_calls_used": 3,
  "stopping_reason": "no_useful_tool_remains"
}
```

Selected tools: `cargo_balance` and `cargo_delay_profile` in round one, `maritime_history` in round two. All returned complete coverage: seven exact hourly observations. Selected metrics below are rounded for readability:

| Diagnostic finding | Value |
|---|---:|
| Inventory change | −28.01 normalized cargo units |
| Net cargo flow | −28.02 normalized cargo units |
| Yard occupancy change | −0.28 percentage points |
| Truck waiting-time change | −3.50 minutes |
| Waiting-vessel change | +4 vessels |
| Waiting-ratio change | +0.087431694 |

Cargo remains **ELEVATED** and future proxy risk remains **HIGH**. The hypothesis `cargo_high_but_recently_draining` is supported by `cargo_balance.metrics.inventory_change` and `cargo_balance.metrics.net_cargo_flow`. Rising Maritime waiting pressure is supported by the two Maritime deltas. Neither finding overrides the other. The candidates remain baseline, shortest-first and gate-extension; gate-extension remains the simulated highest-ranked option.

## Before/after compatibility

Before edits, full deterministic/no-save/no-LLM response hashes were captured using SHA-256 of `json.dumps(result, sort_keys=True, ensure_ascii=True).encode()`:

| Timestamp UTC | Original hash |
|---|---|
| 2025-07-10T19:00:00Z | `cf8fb1c09938648b244a3dee7782f8cbae5cab89c8a60051eb1c669c43768806` |
| 2025-01-12T02:00:00Z | `d5748eb80830a61bde4cc830cb2fbb3bb3fcb4ed62c4dee71115b157a18d52f3` |
| 2025-11-02T08:00:00Z | `92d3f316e22a4cc3a36ec94b5f98461e4c6b9e4ee7cf4906d01629b53e4c436d` |

Regression tests confirm disabled mode matches these original hashes. For each timestamp, enabled deterministic mode equals the complete original result after removing **only** `strategy_investigation` and each candidate's `diagnostic_justification`. Thus all original domain evidence/statuses, ML outputs, candidate definitions, scenario inputs, simulations, rankings and deterministic recommendations are unchanged in those comparisons.

| Timestamp UTC | Original signals retained | Added context | Original candidates / winner retained |
|---|---|---|---|
| Jul 10 19:00 | Cargo ELEVATED, Maritime ELEVATED, ML HIGH | Cargo draining and delays improving; Maritime waiting rising | baseline, shortest_first, gate_extension / gate_extension |
| Jan 12 02:00 | Cargo ELEVATED, Maritime NORMAL, ML LOW | Inventory/net flow +72.23; truck waiting +34.57 minutes; no claim either current/forecast signal is wrong | baseline, gate_extension / gate_extension |
| Nov 2 08:00 | Cargo NORMAL, Maritime ELEVATED, ML LOW, weather unknown | Waiting +1 vessel; ratio +0.053418803; weather explicitly unresolved | baseline, shortest_first / baseline |

These differences provide additional diagnostic context; they do not demonstrate better operational outcomes or forecast accuracy.

## Validation

Baseline command, before implementation:

```text
python -m pytest tests -o addopts= -q
239 passed, 1 warning in 27.01s
```

Final full suite with the same command: **319 passed, 1 existing warning in 26.25s**. The 80 new cases include rising/draining Cargo, rising/falling Maritime, exact alignment, year-start/in-window gaps, zero/null, invalid counts, source failures, geographic metadata, future isolation, valid adaptive plans, mocked Gemini HTTP transport/schema, invalid tools/arguments/extra fields, repeated requests, budget/config caps, terminal decisions, provider timeouts, endpoint conflicts, mixed/insufficient hypotheses, mutation isolation, full-pipeline Gemini/failure behavior, optional failure containment, API compatibility, and all three requested historical comparisons.

The existing warning is `Timestamp.utcnow()` deprecation at `prediction_model/src/final_proxy.py:346`; it was present in the baseline and was not changed in this task. Ruff checks pass for all changed/new Python files. `git diff --check` passes.

**Gemini validation uses mocked provider responses and `httpx.MockTransport`; no live Gemini calls or credits were used for new-feature validation.** Live provider/schema acceptance, real model selection quality, latency, API quota behavior and operational decision quality remain unevaluated. Existing narrative generation is independently tested and preserved.

## Remaining limitations

- Feature is opt-in and disabled by default. An enabled configuration alone does not supply a provider to direct Python calls; the existing API factory does that.
- No intervention selection changes, new action types, automatic execution or frontend investigation display. This version improves evidence acquisition and justification only.
- Only three diagnostics and a fixed six-hour window; no weather investigation. Missing observations remain unresolved rather than inferred.
- Cargo provenance, regional Maritime membership and retrospective source construction remain limitations; source data and geography assumptions were deliberately preserved.
- Endpoint checks catch direct conflicts with supplied domain evidence, but cannot establish that an externally injected history is operationally authoritative.
- Scanning CSV sources adds I/O; this task does not add indexes/caching or make claims about production throughput.
- Mocked Gemini exercises integration and fallback, not real-service availability or reasoning quality. Optional planning can add up to two provider calls before the existing narrative call.

For enablement and PowerShell commands, see [the configuration guide](../docs/STRATEGY_INVESTIGATION.md).
