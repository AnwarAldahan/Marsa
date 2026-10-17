# Latest Main + cargo backend integration

2026-10-17. Target branch: `Agent_Ft`. The approved fast-forward to latest Main
is complete. No new commit was created; nothing was staged or pushed.

## Foundation and recovery

Fetched and independently rechecked remote Main:
`916a264ea431a85e8c2fd62e63214b06c54f664f` (PR #4, plan/objection UI).
Agent_Ft HEAD is now `916a264ea431a85e8c2fd62e63214b06c54f664f`, equal to Main.
Remote origin/Agent_Ft remains `176473a051107e7ca1cd27f14e17cec38bcf0a27`.
The old HEAD was an ancestor of Main: no divergent history needed rebasing.

The initial working tree was clean. Previous uncommitted Coupled/Hybrid work was
absent, so the retained independent extensions were recovered from the verified
`Marsa-remote-main-exact-20261017/before` snapshot, checking each file against its
SHA256 manifest. Previous frontend customizations were not recovered.

Complete current-state backup:
`C:\Users\fatom\.codex\backups\Marsa-latest-main-integration-20261017-172038`.
`workspace.zip` includes all 1,135 original files, including `.git`, ignored files
and credentials; it must remain private. Every archive member was read back and
SHA256-verified against `manifest.json`, and filesystem inventory was rechecked.
Extract into an **empty recovery directory**, not over an active checkout.
`recovery.txt` records HEAD and
`refs/backup/Agent_Ft-Marsa-latest-main-integration-20261017-172038`.
The reversible Main patch and live validation summary are also stored there.

## Main preserved

All latest Main tree content is present. Frontend entry point is
`dashboard/marsa.html`, documented by Main itself. Every Main HTML/CSS/JS file
and the dashboard README is byte-identical to the fetched source. The previous
root-design adapter, Hybrid animation overlay, launcher and frontend tests were
not carried over. No extra panels or controls were added.

Main changes retained include user-decision assessment, plan/objection workflow,
calendar and data-based flow levels, clearer port visualization, removal of the
model-performance section, `/api/flow/norms`, gate-closure/inflow stress,
updated twin APIs, tests and documentation. Dependencies and datasets/models
are unchanged. Main's existing Gemini/adaptive configuration remains as supplied.

Actual local UI: http://127.0.0.1:5504/marsa.html, served from this repository's
`dashboard` directory. Actual integrated API: http://127.0.0.1:8000.

## Digital Twin conflict resolution

Main's Twin still uses finalized AIS arrival/berth/service times for deterministic
historical replay. Yard replay supports gate closure and inflow stress; it does
**not** determine vessel service duration or berth release. Its synthetic yard
stress can exceed 100% to express overflow. This is Main's intended behavior and
was preserved exactly in `engine.py`, `yard.py`, `adapter.py` and `/twin` routes.

The independent normalized engine instead admits unloading only into available
yard capacity, accounts for gate clearance, retains uncompleted vessel workload,
and releases berths only after workload completion. Capacity and conservation
invariants apply there. These are different models/populations/units, not duplicate
implementations of the same contract. Their KPIs are not mixed or ranked together.

Retained backend extensions:

- `src/marsa/twin/{coupled,hybrid,sensitivity}.py`: conservation, workload-based
  completion, explicit AIS cohort initialization, retrospective/as-of boundaries,
  paired experiments, ledger-derived bottlenecks and robustness/trade-offs.
- `src/marsa/agents/strategy_advisory.py`: validated optional evidence envelope;
  unsupported candidate mappings remain NOT_EVALUATED; no ranking replacement.
- `src/marsa/api/routes/coupled_twin.py`: independent Coupled/Hybrid/sensitivity
  routes, strict request validation, feature gates and structured missing-source 503.
- Existing Strategy function adapted with the recovered evidence-backed narrative
  guard. It preserves Main's user-decision assessment and authoritative agent winner;
  invalid/provider-failed prose falls back deterministically.
- Existing decision-support route/pipeline extended only with the optional advisory
  request, hook and endpoint. No experiment is automatically run by ordinary analysis.

The four new configuration sections (`coupled_normalized`, `hybrid_normalized`,
`sensitivity_analysis`, `strategy_decision_support`) are explicitly enabled for this
local demo. Typed/missing-section defaults remain disabled. All numeric workload,
capacity and productivity values are synthetic assumptions, not physical calibration.
Main settings preceding these sections were preserved.

## Changes relative to Main

Only seven existing Main files have content adaptations:

1. `config/port_config.yaml`: append reviewed independent experiment sections.
2. `src/marsa/api/main.py`: register independent experiment router; retain flow router.
3. `src/marsa/api/routes/decision_support.py`: append advisory request/endpoint.
4. `src/marsa/pipeline.py`: optional advisory input/attachment, no rank changes.
5. `src/marsa/agents/strategy_agent.py`: advisory export and safe narrative guard.
6. `tests/test_adaptive_candidates.py` and
7. `tests/test_strategy_investigation.py`: disable the new advisory in legacy-isolation
   fixtures. All original snapshot hashes, physics/ranking assertions and tests remain.

Added files: the five independent modules/routes above; tests
`test_coupled_twin.py`, `test_hybrid_twin.py`, `test_sensitivity_twin.py`,
`test_strategy_advisory.py`, `test_strategy_narrative.py`; four corresponding Phase
1–4 documents and four standalone `scripts/demo_*_twin.py`/advisory demos; this report.
Before ancestry alignment, new Main files appeared untracked against the older
local HEAD, including `flow.py`, `flow_norms.py` and `test_flow_norms.py`. They are
now tracked by the fast-forwarded Main commit and have no content modifications.

## Validation

Full suite: **559 passed**, one existing pandas `Timestamp.utcnow` deprecation
warning, 99.97 seconds. Includes conservation/capacity, retrospective/as-of,
sensitivity pairing, malformed evidence, missing sources, narrative fallback,
Main ranking/user decisions, flow normalization and API contracts.
Ruff passed on every modified/new Python file. Node syntax checks passed for both
scripts in Main dashboard and eight root-reference scripts. `git diff --check` passed.

Actual API requests at all three required UTC timestamps:

| Decision | Main winner | ML | Hybrid eligible (berthed/waiting/future) | Advisory |
|---|---|---|---|---|
| 2025-03-04 10:00 | baseline | success | 20 (13/0/7) | AVAILABLE |
| 2025-07-10 19:00 | baseline | success | 23 (17/1/5) | AVAILABLE |
| 2025-11-05 12:00 | baseline | success | 17 (8/0/9) | AVAILABLE |

Each explicit sensitivity request produced 15 contexts; paired workloads were
validated by the restored tests/evidence validator. Hybrid accounting residuals
were below 1e-6. Deliberately full synthetic-yard baselines unloaded 2,400 units,
blocked 21,600 potential units and completed zero vessels over 24 hours. These are
controlled scenarios, not evidence that real port completions stopped. The queue
exposure was zero; a final-boundary arrival remained waiting. Do not translate
blocking or queue relief into an unsupported completion benefit.

API analysis took 2.6–4.1 seconds; Hybrid 3.1–3.7; sensitivity 3.2–4.6; advisory
4.7–5.2 during concurrent validation. Narrative mode was deterministic_fallback;
fresh live Gemini success is not certified by these calls.

Actual browser checked all three calendar timestamps, Main's recommendations and
Twin replay. July gate-card comparison, horizon scrub and manual gate-closure plus
extra-berth evaluation succeeded. Agent winner remained baseline, with user decision
reported separately. No console errors/warnings were captured. Latest Main's
frontend was served without injected bindings or mock results.

Seventy protected data/model/artifact/environment files matched initial backup
hashes. Pending content contains no actual environment credential values; nothing
is staged. No datasets or models were regenerated.

## Git alignment and verdict

Content is validated for an operator-reviewed demonstration, with synthetic workload,
unverified terminal membership and retrospective hindsight limitations. Main's live
website continues to display Main's replay; the cargo experiments remain explicit APIs.

**Git ancestry alignment complete, without conflicts.** After explicit user approval,
Agent_Ft was fast-forwarded with `git merge --ff-only` to the exact verified Main
commit. No merge commit or other new commit was created. The local branch is five
existing Main commits ahead of origin/Agent_Ft; the remote branch is unchanged.

Before moving HEAD, an additional complete backup was created at
`C:\Users\fatom\.codex\backups\Marsa-ancestry-alignment-20261017-173121`.
Its `workspace-before-alignment.zip` contains 1,182 files including Git metadata and
ignored files. All archive members were read back and SHA256 verified. The manifest
and `RESTORE.txt` document recovery into an empty directory. The seven-file patch
plus 19 added files were recovered onto a clean exact Main tree and verified before
any tracked work was temporarily restored for fast-forwarding.

After fast-forwarding, the patch applied cleanly and all 26 saved delta files matched
their pre-alignment bytes. All 177 unadapted Main files matched Main content; all
70 protected data/model/artifact/environment files matched pre-alignment hashes.
The only subsequent file update is this report of the completed operation.

Final content status: seven adapted tracked files and 19 untracked cargo files;
nothing staged, no unmerged entries, no deleted files. Git also reports the same
four pre-existing frontend line-ending flags (`Marsa updated.html`,
`port-dt-console (1).html`, `dashboard/marsa.html`, `dashboard/README_DASHBOARD.md`).
They have no content diff against Main and their pre-alignment bytes were preserved.
No unrelated functionality or frontend changes were introduced.

Post-alignment checks: **559 passed**, one existing pandas deprecation warning,
87.01 seconds. Ruff passed on all modified/new Python files; dashboard and root
reference JavaScript syntax checks passed; `git diff --check` passed. No code
changes were made after the saved validated delta was restored.
