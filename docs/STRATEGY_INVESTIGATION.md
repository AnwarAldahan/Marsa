# Optional Strategy investigation

The four-agent pipeline now supports bounded diagnostic acquisition **inside Strategy**:

`current state → ML + three domain assessments → optional investigation → legacy candidates + diagnostic justification → unchanged twin → unchanged ranking → human review`

This version qualifies existing candidates with historical evidence. It does not change candidate membership, identity, titles, original rationale, actions, intervention parameters, simulation inputs, ranking, or domain statuses. Historical direction alone cannot establish whether an intervention is necessary or effective.

The separate `adaptive_candidates.enabled` switch can now prioritize actual candidate evaluation using this evidence.
It does not change investigation behavior. Both flags default to false in code when omitted; the current
repository prototype configuration explicitly enables both. See
[adaptive rules and historical comparisons](ADAPTIVE_CANDIDATES.md).

## Enable

Edit the existing `config/port_config.yaml` section:

```yaml
strategy_investigation:
  enabled: true
  use_gemini: true
  max_rounds: 2
  max_tool_calls: 3
  window_hours: 6
```

Code defaults are disabled; the current repository prototype configuration explicitly enables investigation and adaptive candidate priority. Set both flags to false, or omit both sections, for the complete legacy response without new fields. Settings are strict booleans/integers. Rounds may be 0–2, calls 0–3, and the only supported window is six hours; higher limits and extra configuration keys are rejected, and investigation is skipped without failing the analysis. Zero budgets leave questions unresolved.

The existing API `/api/decision-support/analyze` supplies its existing Gemini provider, configured through `GEMINI_API_KEY` and `MARSA_GEMINI_MODEL` as before. Restart the backend after changing configuration. No new endpoint, request parameter, dependency, or frontend change is required.

From PowerShell in the repository, with dependencies installed:

```powershell
$env:PYTHONPATH = "src"
python -m uvicorn marsa.api.main:app --reload
```

From a second PowerShell terminal:

```powershell
$result = Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/api/decision-support/analyze" `
  -ContentType "application/json" `
  -Body '{"timestamp_utc":"2025-07-10T19:00:00Z"}'
$result.strategy_investigation | ConvertTo-Json -Depth 20
```

Set `use_gemini: false` for deterministic **diagnostic routing**. The existing recommendation narrative has its own legacy provider behavior and is unaffected by that setting. For a completely offline Python execution, call `marsa.pipeline.run(timestamp, cfg=cfg, llm=None)`. The existing `scripts/run_pipeline.py` does not create an LLM provider: enabled investigation uses deterministic routing there. Its `--save` option saves the optional investigation in the full output; terminal output remains the recommendation summary.

## Diagnostic contracts

All three read existing CSV sources, use application-owned paths, and accept only the server's authoritative timestamp and strict `{"window_hours":6}`. Seven exact UTC endpoint observations cover `t-6h` through `t`. Endpoint deltas compare `t-6h` and `t`; flow totals sum the six hourly bins `(t-6h,t]`, avoiding a seven-bin total.

| Tool | Existing source | Computation and units |
|---|---|---|
| `cargo_balance` | `data/agent2_cargo_operations_2025.csv` | Inventory delta and net arrivals minus departures in original normalized cargo units; yard-occupancy delta in percentage points |
| `cargo_delay_profile` | Same Cargo source | Truck waiting-time delta in minutes; six-bin truck arrivals/departures and source gate-throughput totals |
| `maritime_history` | `data/port_operations_agent_dataset.csv` | Waiting-count delta in vessels; waiting-ratio delta as a fraction; start/end total and waiting counts and ratios |

CSV columns are allowlisted; labels are never loaded. CSV scanning reads strings before timestamp filtering; only selected window values undergo numeric conversion and computation. Future observations are excluded from evidence and planner payloads. These are CSV-backed queries, not an as-of database or proof of online data availability at the historical date.

Missing clock hours are reindexed as missing, never replaced with older rows or interpolated. Missing, negative, nonfinite or malformed required numbers yield incomplete evidence; duplicate/misaligned in-window timestamps or unreadable sources yield errors. Maritime counts must be integral and waiting cannot exceed total vessels. Zero is retained; zero waiting with zero total yields a zero ratio, while missing counts remain null. Incomplete diagnostics retain any computable metrics, but cannot support a definitive hypothesis.

Every result contains source identification, geographic scope, provenance, field names, per-metric units, coverage, missing timestamps/values, execution status, and limitations. Cargo is finalized synthetic/calibrated synthetic with mixed regional/POLA construction. Maritime is documented LA/LB regional data with unverified POLA-only membership. Prefix filtering does not remove retrospective source-construction limitations. Waiting ratios depend on the total-vessel denominator and do not establish congestion causes.

## Planner and fallback

Deterministic questions address elevated Cargo progression/delays, elevated Maritime pressure, and current-state versus future cargo-proxy risk differences. Those signals are not treated as mutually exclusive predictions. Unknown weather is explicitly unresolved because this catalog has no weather diagnostic.

Gemini receives whitelisted initial domain statuses/evidence, forecast metadata, questions, tool descriptions, allowed arguments, remaining budgets, and validated previous diagnostic results/hypotheses. It returns a Pydantic-validated plan:

```json
{
  "decision": "CONTINUE",
  "selections": [{
    "question_id": "cargo_progression",
    "selected_tool": "cargo_balance",
    "arguments": {"window_hours": 6}
  }]
}
```

`NO_TOOL_NEEDED` and `INSUFFICIENT_EVIDENCE` require an empty selections list and stop planning while retaining unanswered questions. Every batch is validated before any execution. Unknown tools/questions, unrelated question/tool combinations, arbitrary arguments/paths/timestamps, extra fields, malformed responses, duplicates, and requests exceeding remaining budget are rejected. No generated code, queries, actions, numeric facts, or free-form explanation enter the authoritative evidence.

The existing structured Gemini provider is reused. The plan schema inlines its small Pydantic definitions for the provider's existing Gemini `responseSchema` adapter. This does not change the shared provider or domain/narrative schemas.

Without a provider, with diagnostic Gemini disabled, or after any provider/plan failure, the deterministic router selects Cargo balance/delay first when relevant, then Maritime history. It runs at most two tools per fallback round, within the same global two-round/three-execution limits. A failed or incomplete execution counts toward the budget and cannot be repeated. Gemini may batch all three relevant tools in one round. Valid first-round results are returned for an optional second planning round.

Execution errors are converted to typed diagnostic errors. Diagnostic endpoint facts are checked against overlapping canonical domain evidence, preventing a conflicting injected source from silently qualifying candidates. Unexpected investigation failures produce a skipped report; they do not stop an otherwise valid analysis. Raw provider responses and exception details are not included in reports.

## Response additions

Enabled runs add `strategy_investigation` with mode (`gemini`, `deterministic_fallback`, `skipped`), questions, execution trace, diagnostics, deterministic hypotheses, evidence references, unresolved questions, budgets, stopping reason and limitations. Mode is fallback if any planning round falls back. Candidate records gain a separate `diagnostic_justification`; original fields remain intact. Twin and ranking records remain unchanged.

Python evaluates accumulation/draining, worsening delays and rising Maritime pressure as `supported`, `mixed`, `unsupported`, or `insufficient`. Each hypothesis references diagnostic IDs and supporting/contradicting metric paths, or coverage for insufficient evidence. No LLM-generated factual explanation is stored. A skipped investigation leaves candidate behavior intact and labels its justification as legacy preserved.

This feature adds acquisition and explanation context, not validated improvement in decision quality. No production-readiness or live Gemini compatibility claim is made by the mocked tests. See the implementation report for exact historical comparisons and test results.
