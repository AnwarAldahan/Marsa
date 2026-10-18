# Unified Hybrid website integration validation

Date: 2026-10-17. Branch: `Agent_Ft`. HEAD remains `252355450b3a710a381d997009fff476923eaa0e`. No commit, push, merge or rebase performed.

## Scope and behavior changes

The root Main website now uses `/api/decision-support/unified` for all operational simulation comparisons, recommendations, replay and current-port manual decisions. Evidence collection is shared with the unchanged legacy pipeline. One frozen Hybrid initialization supplies every native candidate. Ranking is now ascending simulated queue exposure in vessel-hours with baseline-first ties; this deliberately replaces legacy berth-window waiting semantics. Completions and unloading are shown separately.

The five implemented candidates are baseline, estimated-workload shortest first, an extra abstract slot, normalized clearance x1.2 and x3. Invalid/duplicate native scenarios are excluded, not approximated. Recorded `gate_max`, historical service-duration ordering, crane changes and rescheduling are not assumed equivalent. Manual decisions retain Main's separate operator-comparison role and do not replace the generated-candidate recommendation.

Demo assumptions are explicit in `unified_hybrid`: 23 abstract slots; capacity 10,000 normalized units; initial inventory 0; shared unloading 100/hour; gate 100/hour; 24 hours. Hybrid workloads reuse the existing synthetic workload contract. No conversion from Cargo indices, TEU or gate transactions is introduced. Initial occupied berths must fit configured slots. Historical agents and the XGBoost forecast remain separate context, not counterfactual predictions.

Main CSS and static element IDs match the pre-change backup exactly. No new panels, buttons, forms, dependencies or physics were introduced. Existing drawings/animations are reused with explicit virtual-slot and synthetic-workload labels. Null completion times remain censored. Typed educational examples remain separate. FastAPI now serves the exact root file at `http://127.0.0.1:8000/`, with no directory browsing and no-store caching.

## Live historical API and browser checks

| UTC timestamp | Eligible cohort | Initially berthed / waiting / future | Winner | Queue exposure | Completed | Unloaded |
|---|---:|---|---|---:|---:|---:|
| 2025-03-04 10:00 | 20 | 13 / 0 / 7 | baseline | 0 | 0 | 2,400 |
| 2025-07-10 19:00 | 23 | 17 / 1 / 5 | baseline | 0 | 0 | 2,400 |
| 2025-11-05 12:00 | 17 | 8 / 0 / 9 | baseline | 0 | 0 | 2,400 |

Queue units are vessel-hours; unloading uses synthetic normalized units. Every generated candidate tied in these default scenarios. Zero completions reflect the explicit low unloading assumption and finite horizon, not observed vessel performance. Initially waiting vessels can be admitted at simulation time zero when abstract slots are available; pre-decision AIS waiting is not the new ranking objective. Calls arriving at the horizon may remain waiting with zero accumulated queue exposure.

All three API runs returned the agents and successful XGBoost inference. Initial measured request times were approximately 5.45–7.23 seconds including evidence/investigation; this is a local observation, not a latency guarantee. July's investigation reported Gemini mode; recommendation narratives used safe deterministic fallback, so live narrative acceptance is not claimed. March/November investigation was skipped when no diagnostic question was selected. Provider acceptance, contradiction rejection, failure and payload mutation are regression-tested.

Real browser checks at the canonical API-served root verified all three timestamps, forecast display (approximately 0.010 / 0.959 / 0.007), historical agent factors, native candidate cards and baseline winner, navigation and hourly replay. July's manual current-port calculation reused the saved context and showed censored unfinished service without invented durations. The independent hand-entered example also ran. Final browser console error/warning list was empty. Desktop was exercised; mobile CSS is preserved, but a separate mobile viewport audit was not performed.

## Controlled full-yard July comparison

Same frozen July cohort and workloads, 24-hour horizon, 17 slots, initial inventory/capacity 10,000 and unloading 1,000/hour. Only normalized gate rate changes from 0 to 1,000/hour:

| Actual engine outcome | Gate 0 baseline | Gate 1,000 intervention |
|---|---:|---:|
| Queue exposure (vessel-hours) | 75 | 15 |
| Completed vessels | 0 | 22 |
| Cargo unloaded (normalized units) | 0 | 13,500 |
| Cargo cleared (normalized units) | 0 | 23,500 |
| Blocked unloading (normalized units) | 24,000 | 0 |
| Occupied berth-hours | 408 | 162 |
| Waiting at horizon | 6 | 1 |

This demonstrates full yard -> blocked unloading -> no completion/release -> continued waiting. Increased clearance restores unloading and completion/release in the unchanged engine. Final conservation residual magnitudes were at most approximately 1.82e-12. These are synthetic simulated benefits, not observed operational improvements. The gate-zero generated pool excludes multiplicative gate duplicates; its extra-slot winner is correctly classified `QUEUE_RELIEF_ONLY`, since it reduces waiting without unloading/completion gains. An explicit frozen sensitivity request completed 16 comparison contexts and retained the original context signature. Robustness is not automatically asserted.

## Regression and safety checks

- Full suite: **596 passed**, exit 0. One existing pandas `Timestamp.utcnow` deprecation warning in `prediction_model/src/final_proxy.py` remains.
- Focused native API/frontend tests: **37 passed**, including legacy/mixed-contract rejection, stale responses, frozen contexts, baseline ties, invalid candidates, initial occupancy, provider failures, cache expiry/bounds, missing sources, feature gates, conservation guards and the blocked-yard causal chain.
- Ruff on all modified/new Python files: passed.
- Node syntax checks for all eight inline scripts: passed.
- `git diff --check`: passed; existing LF/CRLF notices remain for reference files.
- Agent modules, original Twin and Coupled/Hybrid physics hashes match the pre-change backup. Protected data/model/credential hashes match the prior manifest; no secrets were included in changed files.
- Root CSS and static element IDs match the backup. `dashboard/marsa.html` and `port-dt-console (1).html` retain their pre-existing line-ending-only Git flags; this task made no content edits to them.

## Exact content changes

1. `Marsa updated.html`: native operational adapter, actual replay/manual mappings and accurate labels; original design preserved.
2. `config/port_config.yaml`: explicit enabled unified demo profile.
3. `src/marsa/twin/unified.py`: frozen scenario contract, validated candidates, ranking, guarded recommendation/advisory and explicit sensitivity entry.
4. `src/marsa/api/routes/unified_twin.py`: validated unified and frozen sensitivity routes, feature gates and controlled errors.
5. `src/marsa/api/main.py`: router registration and canonical root-file serving.
6. `src/marsa/pipeline.py`: extract unchanged evidence collection for shared use; legacy execution remains intact.
7. `src/marsa/twin/sensitivity.py`: expose existing algorithm on a pre-frozen scenario, avoiding AIS reinitialization.
8. `tests/test_unified_twin.py`: 24 native regression tests.
9. `tests/test_root_api_bindings.py`: 13 frontend regression tests.
10. `README.md`: canonical startup and changed website semantics.
11. `dashboard/README_DASHBOARD.md`: distinguish canonical site from the retained legacy reference.
12. `docs/UNIFIED_TWIN.md`: API/configuration contract, supported mappings and limitations.
13. `reports/UNIFIED_TWIN_VALIDATION.md`: this validation record.

## Remaining limitations

This is an operator-reviewed, retrospective, synthetic normalized-workload demo. AIS terminal membership is heuristic; 23 slots are not verified available operational berths; workloads, initial yard and rates are assumptions. Default scenarios need not exhibit improvements and currently produce no completions within 24 hours. Queue relief and cargo progress are deliberately separate. Legacy adaptive prioritization remains in legacy APIs; the unified small native pool is evaluated exhaustively. The original legacy reference dashboard and APIs remain available with their original meanings, outside the canonical unified site. Frozen contexts use one bounded process-local cache (32 entries, 30-minute TTL); restart, expiry or another worker returns 410 and requires a new analysis. No automatic sensitivity or live operational effectiveness claim is made.
