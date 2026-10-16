# Marsa — Stage 2 comprehensive audit and improvement plan

**Audit date:** 16 October 2026 (Asia/Riyadh).

**Repository:** AnwarAldahan/Marsa. **Audited commit:** `504041e0b3911e8585136a74354f9932b61b7c48`.

**Scope:** repository implementation, tracked data/artifacts, documentation, tests, and focused executable probes. No application code, dependencies, datasets, or model artifacts were changed. This report is the only deliverable added.

## Executive assessment

Marsa has a working, reproducible historical decision workflow, typed domain assessments, real XGBoost inference, and a functional Python simulation. Its strongest asset is the traceable combination of evidence, forecasts, candidate actions, paired simulations, and human review. Preserve that structure.

Its principal Stage 2 weakness is **decision credibility across component boundaries**. The frontend sometimes presents invented forecast points or hides current pressure; its animation uses a different scenario from the backend. The simulation has reproducible service-progress and stock-conservation defects. Model evaluation has a tied-score metric defect, while the final model is replayed over hours inside its training period. Optional narrative guardrails do not fully preserve decision meaning.

The best next investment is a small, trustworthy decision loop with demonstrable invariants and honest evaluation. More agents or a larger model would not fix these issues.

### Evidence labels and priorities

- **Verified defect:** reproduced behavior contradicts a contract, physical relationship, or displayed meaning.
- **Verified limitation:** implementation or evidence is absent/incomplete; not necessarily a programming error.
- **Recommendation/hypothesis:** a proposed improvement whose operational benefit must be evaluated.
- **Unverified:** cannot be established from available repository evidence.
- **Critical:** can materially misrepresent a decision or invalidate its supporting evidence. **High:** materially weakens reliability or usefulness. **Medium/Low:** supporting quality or longer-term value.

Effort estimates are planning judgments: **Small** approximately up to one engineer-day; **Medium** roughly one to three; **Large** four or more, often dependent on external evidence. They are not delivery guarantees. Suggested acceptance tests are future work, except checks explicitly recorded as executed below.

File references use repository-relative paths and one-based line numbers at the audited commit. Function names identify the relevant implementation even if later edits shift lines.

# Part 1 — Current system architecture

## 1.1 Actual decision cycle

`src/marsa/pipeline.py:36`, `run()`, performs the following sequence:

1. Load YAML configuration and `data/processed/merged_port_dataset_2025_v2.csv`.
2. Validate an explicit UTC whole hour in 2025 and select its exact row. Missing source hours are not interpolated or replaced (`src/marsa/data/load.py:19`, `:30`).
3. Run the native +6h cargo-proxy forecast (`src/marsa/model/predict.py:39`). Construct current/lag/rolling features from exact source hours through t−24h; load the fitted preprocessor and XGBoost artifact (`src/marsa/model/congestion_proxy.py:99`, `:158`). Missing history returns structured `ml_unavailable` output.
4. Construct Maritime, Cargo, and Events & Weather agents and call `investigate()` sequentially (`src/marsa/pipeline.py:47`). They independently read their own frozen source files for the same requested hour.
5. Pass their structured results to Strategy synthesis and candidate generation (`src/marsa/agents/strategy_agent.py:66`, `:163`). Their current statuses are preserved even when they disagree with the model.
6. Convert the snapshot and context into a Python `Scenario` (`src/marsa/twin/adapter.py:9`, `src/marsa/twin/state.py:6`). Weather/event multipliers belong here, not in LLM output.
7. Run the candidates with the same seed sequence, normally 30 repetitions each over 24 hours (`src/marsa/agents/strategy_agent.py:194`, `src/marsa/twin/engine.py:184`).
8. Min–max normalize selected KPIs across candidates, apply configured weights, and sort ascending (`src/marsa/twin/scoring.py:9`).
9. Return the ranking, assumptions, limitations, domain evidence, and recommendation; optional Gemini rewrites the Strategy sentence (`src/marsa/agents/strategy_agent.py:233`). `human_approval_required=true`; no port action is executed (`src/marsa/pipeline.py:81`).

The user’s conceptual “Orchestrator & Strategy Agent” is implemented as **coordination in an ordinary pipeline plus Strategy functions**. The repository explicitly records four agents and no separate Orchestrator agent. This is a naming/implementation-boundary difference, not a reason to add a fifth agent. Describe the coordinating responsibility honestly; a separate orchestrator service is unnecessary for this scale.

## 1.2 Components and responsibilities

| Component | What actually happens | Evidence |
|---|---|---|
| Maritime | Dynamically loads the teammate `PortOperationsAgent`; reports current AIS-derived counts and heuristic status. Waiting ratio ≥0.263 is ELEVATED; ≥0.304 plus zero throughput is CONGESTED. | `src/marsa/agents/tools/maritime_status.py:28`; `data/port_agent.py`, `determine_port_status()` |
| Cargo | Reads the frozen 2025 synthetic/calibrated CSV, preserves stored status rather than reconstructing it from rounded values. | `src/marsa/agents/tools/cargo_status.py:47`; `src/marsa/agents/cargo_agent.py:200` |
| Events & Weather | Builds DST-aware LA local calendar; classifies relative weather pressure and synthetic events. “Confidence” explicitly means evidence completeness. | `src/marsa/agents/events_weather.py`, `EventsWeatherContextBuilder.build()`; `src/marsa/agents/events_weather_agent.py:158`, `:180` |
| Strategy | Fixed evidence-triggered candidate templates, conflict-preserving synthesis, twin invocation, deterministic ranking, optional narrative. | `src/marsa/agents/strategy_agent.py:66`, `:163`, `:194`, `:233` |
| ML | Runtime +6h XGBoost operational cargo proxy, 32 source features →448 engineered inputs →604 transformed model features. | `src/marsa/model/congestion_proxy.py:158`; `tests/test_ml_artifact.py:25` |
| Python twin | Queue, service times, aggregate berth capacity, yard stock, gate clearance, stochastic arrivals. | `src/marsa/twin/engine.py:38`, `:128` |
| Backend | FastAPI health, three investigation endpoints, full analysis endpoint. Synchronous route functions; optional synchronous HTTP Gemini provider. | `src/marsa/api/main.py:5`; `src/marsa/api/routes/decision_support.py:31`; `src/marsa/agents/gemini.py:65` |
| Frontend | Single Arabic HTML/CSS/JS page, API adapter added to an earlier demo, separate client-side twin engine and manual console. | `dashboard/marsa.html:765`, `:808`, `:1909` |
| Persistence | Local CSV/Parquet/JSON, optional saved run JSON, browser-local feedback. No integrated database or durable operational decision/outcome store was found. | `src/marsa/pipeline.py`, save branch; `dashboard/marsa.html:1500`, `:1533` |

Agent communication is in-process Pydantic requests/results converted into dictionaries. It is not an active network of agent-to-agent calls. Individual domain APIs are separate entry points. `RemoteAgent.assess()` is a legacy adapter with a different interface; its documented `context` example does not match the current `investigate()` injection contract (`src/marsa/agents/remote.py:1`; `src/marsa/pipeline.py:52`).

## 1.3 Two distinct prediction tracks

| Track | Target and evidence | Current use | Evaluation interpretation |
|---|---|---|---|
| C1 maritime XGBoost | AIS-derived queue-pressure heuristic occurring anywhere in `(t,t+6h]`; 17 AIS/weather features, 2023–2025 modeling table. | Artifact loads as an XGBClassifier with fitted SimpleImputer; not called by the current decision pipeline. | Saved-artifact inference independently reproduces test ROC-AUC 0.97056 and average precision 0.83068 on 4,349 July–December rows. These results use a derived label, not independently adjudicated congestion truth. |
| Final cargo proxy XGBoost | Yard AND delay AND flow stress, persisting for two consecutive exact hours; predict the state at exactly `t+1/3/6/12h`. Primarily synthetic operational signals plus AIS/weather/calendar. | Four saved models; native runtime uses +6h. | June–September walk-forward predictions and episode analyses are saved. Scores are explicitly uncalibrated. Metrics need the correction in A07. |

Evidence: `reports/phase2_ml/model_evaluation.md`; `artifacts/models/marsa_xgboost_c1_metadata.json`; `prediction_model/src/final_proxy.py:109`, `:124`, `:157`; `prediction_model/final_proxy/artifacts/deployment_manifest.json`; `src/marsa/model/predict.py:39`.

The C1 artifact is a real saved model, but it should not supply headline numbers that appear to describe the live cargo-proxy output. The targets, periods, features, and prediction semantics differ. There is no verified active LightGBM training/inference path merely because LightGBM is listed in requirements.

# Part 2 — Implementation inventory

“Working” below means executable within the prototype’s stated scope, not production validation.

| Capability | Status | Evidence and qualification |
|---|---|---|
| Exact historical replay, UTC validation, preservation of source gaps | Working | `src/marsa/data/load.py:19`; integration tests cover invalid/missing hours. |
| Maritime/Cargo/context deterministic investigations | Working | Typed evidence, field provenance, source validation, forecast-invariance tests. |
| Native cargo XGBoost inference | Working | Artifact parity test; exact history requirements. 8,431/8,688 hours have the required continuous 24h source history. |
| C1 maritime prediction in integrated API | Missing | Saved artifact exists; current `forecast()` calls cargo proxy only. |
| Candidate generation and paired Monte Carlo ranking | Working with material limitations | Four-candidate cap, baseline, deterministic seeds; physical/decision defects in Part 3. |
| Autonomous tool selection/planning between agents | Missing | One fixed lookup per domain; Strategy uses fixed templates; no deliberation loop. |
| LLM narrative and deterministic fallback | Partially implemented | Domain wrappers support it, but integrated pipeline supplies its LLM only to Strategy. Semantic guardrails are incomplete. |
| Domain-source outage recovery | Missing | Domain exceptions terminate the integrated run; no typed partial-result contract. |
| Authoritative state shared by all components | Partial | Same timestamps, separate source loaders. Current cargo merge parity is exact; runtime does not enforce common snapshot version. |
| Python twin | Functional, insufficiently validated | Executable stochastic model, not just graphics; service/flow defects reproduced. |
| Shared crane pool, terminal compatibility, action budgets | Missing | Config lists crane counts, but Python runtime does not enforce allocation; aggregate berths are interchangeable. |
| Uncertainty | Partial | Marginal p10/p90 for KPIs; no paired benefit interval or parameter uncertainty. |
| Dashboard API data | Partial | Main figures and result cards load the response; animation/map/demo scripts remain separate. |
| Performance and feedback section | Mocked/local-only | Hardcoded `PERF`; feedback in localStorage, empty server endpoint. |
| Live AIS/TOS/weather ingestion | Missing | Runtime selects frozen 2025 files. Builders are offline and depend on unavailable raw/intermediate artifacts. |
| Full training reproducibility | Broken/incomplete | Final benchmark imports two missing root modules; default data path missing; main training entry point is a stub. |
| Legacy NumPy logistic fit without validation | Defective, outside active XGBoost runtime | Restores initial random weights; A17. |
| Authentication, rate limits, deployment packaging/CI | Missing in tracked implementation | No auth middleware; permissive CORS; no tracked CI workflow, container definition or dependency lock found. |
| Automated coverage | Substantial backend coverage; missing critical invariants/browser coverage | 239 passing tests; only five twin tests, no frontend test suite found. |
| Saudi port transferability and actual operational benefit | Unverified | No Saudi operational dataset, local calibration, or observed intervention outcomes. |
| Official Stage 2 rubric | Unverified | No official judging criteria found in the repository. |

# Part 3 — Technical gap analysis

## Category A: verified defects and critical gaps

### A01 — Frontend presents unsupported forecast information and can suppress current pressure

**Critical; verified defect.** `renderTimeline()` assigns “now” a handpicked 15/45/70 score from Maritime status, fabricates +3h as the midpoint of now and the +6h score, and presents these as percentages (`dashboard/marsa.html:1936`). The backend provides one integrated +6h forecast and explicitly says it is uncalibrated. The page’s hardcoded 35/60 color bands also differ from the artifact’s classification thresholds.

`renderAlert()` decides whether to show any alert from ML classification whenever ML succeeds, ignoring elevated current domain evidence when ML is LOW (`dashboard/marsa.html:1975`). Bulk scoring with the saved +6h artifact found **47 valid-history rows** with LOW score and ELEVATED/CRITICAL stored Cargo status; one is `2025-01-12T02:00:00Z`, score `0.0236212034`, Cargo ELEVATED, yard 79.26%. This count is an implementation check, not an operational prevalence estimate.

**Fix:** display current domain state separately from forecast risk. Use a score such as `0.024` with “uncalibrated model score”; render only real model horizons. Centralize alert/display policy while retaining conflicting signals. Show current severe pressure even with a LOW future score. Do not silently infer severity from arbitrary percentages.

### A02 — Backend results and frontend animation use different scenarios

**Critical; verified integration defect.** The API adapter fills `twCards`, but `TwinUI` closes over `window.PORT` and `TwinEngine.analyze(P)` from the original six-berth demo (`dashboard/marsa.html:765`, `:1554`). Opening the twin calls `buildCards()` on first use and `select(A.best, ...)`, which can replace API cards and animate the demo winner (`:1680`). The backend may be comparing 23 berths and a gate boost while the animation shows unrelated vessel/crane/berth interventions. Manual console simulation is another separate client calculation.

**Fix:** make the Python decision/run response authoritative. Initially, replace misleading animation with timestamped KPI comparison and explicit illustrative-map labeling. Then expose backend per-step traces and animate those exact traces. Keep manual experiments in an explicitly separate mode with their own scenario ID. Verify opening the console cannot overwrite current backend rankings.

### A03 — Crane outages and changing yard pressure do not consistently govern vessel progress

**Critical; verified simulation defect.** Initial berthed vessels get a fixed departure time independent of crane rate (`src/marsa/twin/engine.py:137`). Newly assigned vessels divide service by `max(rate,0.1)`, so zero crane capacity still allows completion (`:157`). Subsequent hourly rate changes affect yard inflow but do not update an already scheduled departure. `Vessel.teu` is not consumed as remaining cargo in the service loop.

**Reproduced:** with zero crane multiplier, a new one-service-hour vessel departs at hour 10 while yard inflow remains zero. An initially berthed vessel with two service hours and one hour already served departs at hour 1 despite the same outage.

**Fix:** track remaining service work or cargo per vessel. Each timestep advances remaining work only by feasible allocated productivity. Zero productivity means zero work; completion and cargo discharge must describe the same work. Apply this to initial and subsequently berthed vessels. Choose one documented unit system before expanding fidelity.

### A04 — Yard clipping breaks cargo conservation

**Critical; verified simulation defect.** `yard = min(capacity,max(0,yard+inflow-outflow))` silently removes excess cargo above capacity and treats planned gate outflow as possible even when inventory is insufficient (`src/marsa/twin/engine.py:160`). Overflow is neither retained on vessels nor recorded as rejected/backlogged work.

**Reproduced:** starting yard stock 95, inflow 10, outflow zero, capacity 100 yields 100, leaving five units unaccounted for. The normal 95% slowdown reduces rate but does not eliminate overflow.

**Fix:** constrain actual discharge by available yard space and retain blocked vessel cargo/work. Constrain actual gate flow by available stock and gate capacity. Report requested versus actual flows and blocked work. Assert conservation at every step, not only bounds.

### A05 — LLM guardrails preserve schema but not all decision meaning

**High; verified defect.** Strategy validates a single free-text summary with regexes, not the selected candidate identity (`src/marsa/agents/strategy_agent.py:233`, `:260`). A probe with baseline ranked first accepted the Arabic sentence “الخيار الأعلى ترتيبًا هو زيادة قدرة التخليص عبر البوابات.” as `llm_assisted`; the numeric winner remained baseline. Thus the narrative can contradict the ranking.

Events & Weather checks some causal/unit/strategy patterns but not numeric grounding or agreement with all deterministic statuses (`src/marsa/agents/events_weather_agent.py:206`). A fabricated weather sentence containing “999” was accepted by `_narrative_rejection_reason()`. The prompt is stronger than its enforcement. Maritime and Cargo have more extensive numeric/citation checks, which are a useful foundation, not proof of exhaustive semantic safety.

**Fix:** render winner, action parameters, evidence values and limitations deterministically. Let LLM text occupy a separate non-authoritative explanation field with cited evidence IDs. Reject contradictions, numbers not grounded in cited fields, unsupported absence claims, markup, and invalid language where required. Test adversarial Arabic and English variants. Prefer constrained templates when proof is hard.

### A06 — Optional generated text reaches an HTML interpretation sink

**High; verified unsafe data flow, browser exploit unverified.** Backend `decision_support.summary` is interpolated into `innerHTML` (`dashboard/marsa.html:1997`). Strategy validation does not prohibit HTML. Some other result text is similarly interpolated into card markup. This creates an injection surface through provider/output text; no browser payload was executed during this audit.

**Fix:** construct DOM elements and insert narrative via `textContent`. Use sanitization only for explicitly supported rich text. Add a browser check showing a harmless markup-shaped string renders literally. Apply a restrictive content-security policy when deploying.

### A07 — Average precision mishandles tied prediction scores

**High; verified evaluation defect.** `prediction_model/src/evaluation.py:56`, `average_precision()`, sorts scores stably and averages precision at each positive row without grouping ties. Equal scores can change AP when row order changes. The displayed “PR_AUC” is intended as average precision, not trapezoidal PR integration; document that convention.

**Reproduced:** labels `[1,0,1,0]` with four constant 0.5 scores produce 0.833333 in project code, versus 0.5 using installed `sklearn.metrics.average_precision_score`. Reordering labels to `[0,1,0,1]` changes the project value to 0.5 even though model discrimination is unchanged.

Recomputed on the **existing saved fold predictions**, without retraining:

| Model/horizon | Saved/project AP | Tie-aware sklearn AP |
|---|---:|---:|
| XGBoost +1h | 0.857775 | 0.857717 |
| XGBoost +3h | 0.624753 | 0.624942 |
| XGBoost +6h | 0.483436 | 0.480949 |
| XGBoost +12h | 0.462201 | 0.462372 |
| Persistence +1h | 0.637173 | 0.602579 |
| Persistence +3h | 0.279171 | 0.233750 |
| Persistence +6h | 0.146823 | 0.109267 |
| Persistence +12h | 0.152063 | 0.111732 |

XGBoost still exceeds persistence here. This is not a wholesale invalidation of learned discrimination. However, revise aggregate and onset metrics, model-selection comparisons, reports, and slide claims after fixing the metric. Do not apply these corrections to C1 numbers without inspecting that separate evaluation implementation.

### A08 — Replay is not a point-in-time backtest of the final deployed model

**High; verified evaluation limitation.** The final artifact is trained through 30 September 2025 (`run_final_proxy_benchmark.py:66`, `:633`; deployment manifest). Runtime accepts any available hour in 2025 and does not select an as-of artifact (`src/marsa/model/predict.py:39`). **6,499 source rows** are at or before the final training cutoff. The common July demo is therefore a retrospective/in-training-period replay for this artifact, not an out-of-sample deployment demonstration.

Additionally, Cargo monthly allocations use finalized monthly totals (`data/agent2_deliverables/methodology_note.md`), weather-pressure thresholds use the full 2023–2025 baseline (`scripts/build_events_weather_state.py:263`), and the proxy source features include that index (`prediction_model/src/final_proxy.py:65`). Looking up only rows ≤t does not remove upstream retrospective construction.

**Fix:** label replay mode and artifact training cutoff in every response/UI. Use a model trained before the demo hour for predictive evaluation; preserve a later untouched evaluation period. Rebuild full-period-derived features using training-only parameters. Keep synthetic proxy scores distinct from observed congestion efficacy. Independent real operations data is still required for field validity.

### A09 — Clean-checkout training/inference entry points are incomplete

**High; verified reproducibility gap.** `run_final_proxy_benchmark.py:49` imports missing `run_prediction_experiments.py` and `run_stage2_benchmark.py`. `prediction_model/src/schema.py:18` defaults to missing `data/merged_port_dataset_2025.csv`. `run_prediction_inference.py:13` uses retired logistic-artifact defaults and a missing `prediction_model/saved_models` directory. `src/marsa/model/train.py:22` raises `NotImplementedError`. Sequence-model wrappers also retain stubs. These are distinct from functioning saved XGBoost inference.

**Fix:** restore the actual training helpers, provide one authoritative CLI for the final artifacts/data, separate historical experiments, and record environment versions, input/artifact hashes, target parameters, and time boundaries. Add an inexpensive CLI/import smoke check; full training belongs in a deliberate reproducibility run, not every test execution.

### A10 — One request can combine inconsistent state sources

**High; verified integration weakness.** `run(df=...)` uses the supplied dataframe for snapshot/forecast but default agents still read their files (`src/marsa/pipeline.py:40`, `:47`). The twin adapter also treats zero inventory as missing (`src/marsa/twin/state.py:25`).

**Reproduced:** changing the supplied snapshot yard to zero at `2025-01-02T08:00:00Z` yielded snapshot yard 0, Cargo evidence 6750.38, twin yard 6000. Current checked-in Cargo versus merged data had **zero differing cells across all 14 source fields**, so this is an override/refresh consistency defect, not evidence the frozen normal files currently disagree.

**Fix:** distinguish zero from missing; validate snapshots; bind all observations to one immutable run/snapshot version. Inject snapshot-backed domain sources when running alternative datasets. Enforce timestamp, source version, and important shared-field parity across domain results.

### A11 — Domain failures terminate the entire decision cycle

**High; verified reliability limitation.** The sequential `investigate()` calls have no domain failure boundary (`src/marsa/pipeline.py:56`). A deliberately injected Cargo exception propagates as `RuntimeError`; the HTTP route reduces unexpected failures to a generic 500 (`src/marsa/api/routes/decision_support.py:40`). This is different from the good narrative-provider fallback and structured missing-history ML behavior.

**Fix:** define typed unavailable/error domain results. Explicitly decide which evidence is required for each candidate. Suppress unsupported candidates, preserve healthy assessments, and return “recommendation unavailable” when critical state is missing. Do not default missing capacity/context to reassuring normal values. Add per-domain timeout/error metadata and a run-level status.

### A12 — KPIs and timestep accounting can mislead comparisons

**High; verified semantics/approximation gap.** `vessels_served` counts vessels assigned a berth, not vessels completing service (`src/marsa/twin/engine.py:170`). A 20-service-hour vessel in a one-hour simulation reports served=1. `vessels_arrived` includes initial berthed/waiting vessels (`:171`). Arrivals up to t+1 receive full-hour discharge and busy-hour credit (`:148`, `:160`) even with fractional arrival times. Remaining waits are right-censored at the horizon, and initially berthed vessels are outside the wait sample.

**Fix:** expose unambiguous counts: initial fleet, new arrivals, berth starts, completions, ending queue, deferred arrivals. Use consistent interval/substep accounting or documented boundary rounding. Report censoring and include ending queue/deferred work in candidate comparison. Do not advertise precise minute-level operational accuracy from one-hour resolution.

### A13 — Candidate-relative normalization can reverse rankings

**High; verified decision-policy limitation.** Scores use min/max values from the current candidate set (`src/marsa/twin/scoring.py:9`). A dominated third option can change which of two unchanged options wins.

**Reproduced** with configured weights and KPI order `[average wait,max wait,delayed vessels,yard peak]`: A=`[0,0,1,1]`, B=`[1,1,0,0]` produce A=0.4/B=0.6. Add dominated C=`[100,100,1,1]`; B=0.006/A=0.4/C=1.0. Adding C changes the preferred option despite no change in A or B. This is a property of the policy, not an arithmetic bug.

**Fix:** prefer fixed operational scales or baseline-referenced utility, feasibility filters, and a minimum meaningful benefit. Show raw KPI deltas, tradeoffs, and “no robust winner” when appropriate. Keep weights explicitly hypothetical until costs/preferences are elicited.

### A14 — Action feasibility, costs and resource pools are not enforced

**High; verified operational limitation.** Supported action names are checked, but numeric feasibility is not enforced in the twin (`src/marsa/twin/engine.py:102`). Additional berths are assumed available, gate boost has no labor/cost bounds, and configured 79 cranes/28 moves per hour are not a shared resource pool (`config/port_config.yaml:11`; no runtime references to `container_cranes`, `crane_moves_per_hour`, `teu_per_move`, `cranes_per_vessel` were found). Only a subset of supported actions is generated by Strategy (`src/marsa/agents/strategy_agent.py:66`).

**Fix:** add a small feasibility layer: approved berth count/compatibility, crane-hours, gate/overtime capacity, implementation lead time and operator restrictions. Validate action ranges and reject impossible scenarios. Do not build a full seven-terminal port model until suitable data is available. Cost can initially be operator-entered, explicitly hypothetical.

### A15 — Environmental context is frozen across the simulation horizon

**High; verified modeling limitation.** `build_scenario()` maps the current hour’s wind/event into constant multipliers (`src/marsa/twin/adapter.py:21`, `:40`). An event ending shortly afterward remains active for the entire simulated horizon; future weather is not ingested. Wind units are declared unverified by domain code but the adapter applies thresholds named km/h (`config/port_config.yaml:31`).

**Fix:** establish a verified unit contract, then use time-varying availability schedules with known event end times. For unknown weather futures, compare explicitly assumed scenarios; never treat observed future weather in a backtest as a known forecast. Keep uncertainty and calendar context separate from claims of causal disruption.

### A16 — UI freshness and error behavior can misrepresent which run is displayed

**High; verified source behavior, visual execution unverified.** An existing one-second timer overwrites `lastUpdate` with a cycling seconds counter (`dashboard/marsa.html:1447`), erasing the replay timestamp inserted at `:2034`. Request errors update a small status message but leave old/demo content visible (`:2038`). `load()` has no request-version/abort control, so out-of-order replies can replace a newer selection. Forecast tooltips attached by earlier demo code retain demo data (`:1307`, `:1333`).

**Fix:** keep selected replay hour, displayed run hour, fetch time and source mode separately visible. Commit results atomically with request IDs; cancel/ignore obsolete responses. Mark retained data stale on failure. Remove stale demo tooltips and the fake update timer. Disable adoption claims in every demo/error state; an old pre-adapter button says the plan will be applied (`:1401`) although no operational execution exists.

### A17 — Legacy logistic fit without a validation set discards training updates

**Medium; verified defect outside active XGBoost runtime.** `NumpyLogisticRegression.fit()` initializes `best_weights`, updates them only when validation is supplied, then restores them unconditionally (`prediction_model/models/logistic_model.py:46`, `:79`, `:91`). A 20-epoch fit without validation returned exactly the initial seeded weights and intercept 0. `test_probability_bounds` checks only bounded outputs, not learning (`tests/test_prediction_pipeline.py:145`).

**Fix:** retain the final fitted parameters when no validation set exists, or retire this baseline if unused. Add a meaningful separable-data learning test before using it in comparisons. Do not infer that the separately reported sklearn C1 logistic model shares this defect.

## Category B: important limitations and engineering improvements

### B01 — Improve coordination through evidence acquisition, not agent count

**High; recommendation.** Domain boundaries and typed facts are good. Current autonomy is limited: fixed sequential calls and rule-based candidate selection, no tool planning or iterative evidence request. The integrated path does not use the domain API dependency factories or their Gemini providers (`src/marsa/pipeline.py:47`; `src/marsa/api/dependencies.py:20`). Choose intentionally whether domain narrative should be enabled; it is not required for intelligence.

Add a bounded Strategy investigation step: identify disagreement or missing evidence; request one supported diagnostic tool, such as waiting-ratio trend, recent stock accumulation, or data completeness; then propose only feasible typed candidates. Domain agents should report diagnostic hypotheses with evidence and alternative explanations, never invented causes. This gives observable useful coordination without unrestricted debate or extra agents. Runtime must validate injected results rather than trusting `.model_dump()` dictionaries from any supplied implementation.

### B02 — Strengthen twin uncertainty and empirical validation

**High; verified evidence gap plus recommendation.** Common random seeds are valuable. Current p10/p90 summarize marginal stochastic KPI outcomes, not confidence that an intervention helps, and not parameter/model error (`src/marsa/twin/engine.py:184`). Thirty runs are a configuration choice, not demonstrated convergence. Twin tests cover only a few broad behaviors (`tests/test_twin.py:13`).

Retain per-seed baseline/candidate differences; report paired benefit distributions, fraction of runs improving, tail degradation and convergence. Sweep uncertain arrival/service/LA-share/gate parameters. Validate baseline behavior on a later dataset slice; label comparison to synthetic cargo as internal consistency rather than real terminal validation. Use held-out observed berth/queue trajectories where semantics permit. No observed causal intervention uplift can be inferred from the existing data.

### B03 — Establish a defensible prediction evaluation and alert policy

**High; recommendation.** Train-only proxy quantiles, exact future matching, gap checks and baselines are strong choices. But model families are evaluated on slightly different valid timestamp sets (for +6h: XGBoost 2,820 versus sequence models 2,822). Final artifact alert thresholds are selected from in-sample training scores (`run_final_proxy_benchmark.py:441`, `:466`), while risk levels are training-score quantiles (`:475`), not actionable incident probabilities. Stored +6h recall is about 0.237; that operating point should be explained rather than obscured by AUC.

After restoring training, use common eligible hours, chronological inner validation for thresholds, a later untouched holdout, episode-block uncertainty and onset/false-alert-budget analysis. Separate “risk band” from “alert triggered”: for +6h, medium=0.103402, high=0.528033, alert=0.808412 in the manifest. Audit episode-warning lead times against true episode starts: current code uses the earliest target timestamp represented within each fold’s candidate rows, which can truncate a boundary-crossing episode (`run_final_proxy_benchmark.py:348`). Calibrate only when enough independent validation episodes and an appropriate target exist. Do not promise calibration solves synthetic ground truth.

### B04 — Add a versioned data contract and portable port profile

**High; recommendation.** Source timestamp handling is generally careful, including DST-aware local agent context. Storage mixes aware CSV UTC keys and naive-but-UTC Parquet keys, handled by adapters. Schema/year/row validation is strongest for Cargo; merged loader lacks duplicate/completeness validation (`src/marsa/data/load.py:13`). Some feature governance files exclude weather pressure while the final proxy model includes it; version the actual model feature contract rather than relying on one universal registry.

Add dataset IDs, checksums, units, transformations, provenance parents and as-of availability. Distinguish expected null event metadata from missing weather. Validate cross-source alignment and shared values before serving runs. Move hardcoded 2025/8,688/LA timezone, US calendar, 23-berth display assumptions and absolute geographic split into an explicit port profile before claiming Saudi portability (`src/marsa/agents/tools/cargo_status.py:52`; `src/marsa/agents/events_weather.py:17`; `dashboard/marsa.html:1959`; `docs/NEXT_STEPS.md`). This requires code changes as well as new datasets/config.

### B05 — Keep the local architecture, improve execution and observability

**Medium; verified engineering gap plus recommendation.** Every full run reloads configuration, merged data, default domain sources, preprocessing and model. Independent calls are sequential; simulation is synchronous CPU work; optional Strategy Gemini can wait for its provider timeout (`src/marsa/pipeline.py:36`; `src/marsa/model/congestion_proxy.py:158`; `src/marsa/agents/gemini.py:40`). Synchronous FastAPI handlers are executed through the framework’s synchronous handling; simply changing to `async def` would not make CPU work nonblocking.

Cache immutable, version-keyed inputs/models; reuse source adapters; measure cold/warm p50/p95 latency before parallelizing. Add bounded concurrency only for independent slow I/O, with timeouts; consider job IDs for long simulations if measurements justify them. Record run ID, data/model version, seed, component latency, error category and narrative fallback. Current context-provider logging deliberately avoids secrets, a good practice to retain. A small SQLite run/outcome ledger is enough for a hackathon; a vector database, message bus and microservices are not prerequisites.

### B06 — Deployment hygiene and documentation should match executable reality

**Medium; verified gap plus recommendation.** Dependencies have lower bounds without a lock; `duckdb` is used in the raw builder but absent from declared requirements; optional sequence training needs torch/transformers. `prediction_model` and data/artifact paths depend on repository layout rather than a self-contained installed wheel (`pyproject.toml:25`; `src/marsa/common/paths.py`). CORS allows every origin, health is liveness only, no auth/rate limiting is implemented (`src/marsa/api/main.py:14`, `:23`). This is acceptable only as a declared local-demo deployment boundary.

Freeze a tested environment; separate runtime/training extras; add startup readiness checks, response schema for the integrated result, explicit configuration validation, and CI for unit/integration/lint/artifact smoke checks. For a shared deployment, restrict origins, add access control/rate limits, keep keys server-side, and enforce request/job budgets. Provider model availability and a real deployment were not tested.

Correct stale documentation: final integration says ML remains unavailable (`docs/FINAL_INTEGRATION.md:113`); prediction README describes a Master Agent; limitations mention older forecast gains/target claims (`docs/LIMITATIONS.md:11`); generic training config and final runtime horizons differ. Mark legacy files/commands explicitly. Fix the 16 verified import lint errors after audit approval; do not spend Stage 2 rebuilding the frontend framework solely for style.

# Part 4 — Digital Twin and Multi-Agent deep dive

## 4.1 What qualifies today, and what remains unproven

The Python twin is **a functional stochastic port simulation initialized from a historical snapshot**. It is more than a chart: candidate actions change queue selection, modeled berth/gate/crane capacity and arrival timing, producing counterfactual KPIs. Its current strength is controlled what-if comparison, not validated operational fidelity. There is no live state synchronization or closed-loop actual-outcome assimilation.

The represented system is one aggregate container chain, not separate LA terminal operations. Vessel sizes/cargo work are not consistently coupled to discharge. Arrival intensity uses a Little’s-law-inspired estimate from current occupancy and assumed service time; waiting vessels are a 40% assumed LA share of regional counts. Initial waiting ages and residual services are sampled, not observed. Those are legitimate prototype assumptions if prominently disclosed; they cannot substantiate terminal-specific recommendations (`src/marsa/twin/state.py:6`; `src/marsa/twin/engine.py:70`, `:77`; `config/port_config.yaml:24`).

The 24h twin horizon and exact +6h model target are independent. The ML score is not converted into a predicted vessel-arrival sequence. Do not imply the twin “plays out” the ML forecast: it uses its own sampled arrivals and assumed service/capacity dynamics.

## 4.2 Minimum credible simulation contract for Stage 2

Before adding realism, require these invariants and publish their check results:

1. Work/cargo cannot disappear; every discharge unit must leave remaining vessel cargo and enter yard inventory or an explicit transfer stream.
2. Zero crane allocation/productivity prevents service progress for all affected vessels.
3. Actual discharge respects yard space; actual clearance respects available inventory and gate capacity.
4. No more vessels occupy berths than capacity; no allocations exceed resource pools.
5. Arrivals, service starts and completion times obey one documented time convention.
6. Initial state, new arrivals, completions, deferred demand and ending queues have distinct KPI definitions.
7. All candidates use the same exogenous arrival/service realization for each paired comparison.
8. Unchanged candidate actions produce identical outputs; lower-is-better score does not hide a safety/feasibility violation.
9. Missing state prevents unsupported conclusions; zero state remains zero.
10. UI comparison and animation identify the same scenario, candidate, timestamp, horizon and model/data versions.

A 30-run p10/p90 range does not prove these invariants. Nor does a deterministic seed prove validity. After correctness, use convergence/sensitivity experiments and baseline holdout comparison before discussing operational effect.

## 4.3 Candidate generation and decision quality

Current candidates are explainable and intentionally small: baseline; shortest service first when Maritime is ELEVATED/CONGESTED with waiting vessels; gate boost 1.3 for elevated Cargo or a narrowly defined preventive alert; additional berth for CONGESTED Maritime; combined queue/gate only under a synthetic arrival-increasing event and available candidate slot (`src/marsa/agents/strategy_agent.py:66`).

The cap can exclude the combined option when extra berth already fills the fourth slot. That is a template policy, not optimization over all interventions. Shortest-service-first requires plausible vessel service estimates, which are sampled here. It may disadvantage large vessels; the score includes max wait but has no explicit fairness constraint. Arrival delay can push work beyond the evaluation horizon; count that deferred work before treating delay as improvement.

A stronger decision contract is: **feasible candidates → raw KPI tradeoffs → paired uncertainty → fixed preference/cost policy → conditional recommendation or abstention**. Show why the runner-up lost and which assumptions could reverse the choice. Return a no-change recommendation when improvement is marginal or constraints are unknown.

## 4.4 What makes the multi-agent system useful

Existing positive design choices:

- Narrow domain responsibilities; forecasts do not overwrite current evidence.
- Pydantic schemas and largely immutable results.
- Stored Cargo status is retained, avoiding rounding-based reclassification.
- Field provenance and explicit synthetic disclosures.
- Optional LLM failure does not invalidate deterministic facts.
- Strategy uses supported actions and evidence references; numerical ranking is outside the LLM.

Current limits: fixed workflow, no agent memory tied to outcomes, no bounded follow-up investigations, no runtime validation envelope for injected results, and no partial domain failure recovery. Conflicts are preserved but rarely resolved through additional evidence; `synthesize()` returns statuses and pressure areas rather than a structured explanation of contradiction.

Keep four roles. Give the coordinating Strategy role a short investigation budget and read-only diagnostic tools. Make its output a typed plan with evidence requirements and rejectable proposals. An LLM may propose a tool call or alternative from a bounded catalog, while deterministic validation decides admissibility and the twin evaluates effects. Measure whether this improves coverage, evidence completeness and useful candidate diversity over the current rules; do not label added prose as proven reasoning improvement.

# Part 5 — Prioritized improvement comparison

The tables cover every finding/recommendation above. Current implementation and consequences are detailed in Part 3 under the matching IDs. “Impact” is a measurable acceptance goal or hypothesis, never guaranteed real-world uplift.

## Category A — Critical gaps

| ID / gap | Current implementation / why it matters | Concrete solution | Priority | Expected impact | Effort | Dependencies | Validation |
|---|---|---|---|---|---|---|---|
| A01 Truthful risk and alert display | Invented midpoint/percentages; LOW forecast can hide current pressure. | Real forecast points, separate current status, shared alert policy. | Critical | Zero invented forecast points; no suppressed elevated current condition. | Small | Response/UI contracts | Fixture above plus all status/forecast combinations. |
| A02 One scenario across results and animation | Demo `PORT`/closure differs from API scenario. | KPI-only authoritative view first; backend trace animation next. | Critical | Every view shares run/candidate identity. | Small→Medium | A12 trace semantics | Open console after API load; compare displayed IDs/KPIs before/after. |
| A03 Service-work correctness | Fixed departures; zero cranes still completes work. | Remaining-work service loop for initial/new vessels. | Critical | Zero-capacity progress invariant; coherent capacity effects. | Medium | Defined work units | Zero/variable-rate and initial-residual-work probes. |
| A04 Cargo conservation | Capacity clipping discards/overstates flows. | Feasible actual flows and explicit blocked work. | Critical | Zero unaccounted cargo within numerical tolerance. | Medium | A03, shared unit definitions | Per-step conservation on saturated/empty yard cases. |
| A05 Grounded narrative | Regex permits wrong winner and invented weather numbers. | Deterministic action facts plus cited constrained explanation. | High | No accepted contradiction in adversarial suite. | Small–Medium | Typed decision facts | Wrong-winner, numeric/status/absence, multilingual tests. |
| A06 Safe text rendering | Provider text interpolated into HTML. | DOM/textContent for narrative and labels. | High | Markup-shaped data is displayed literally. | Small | A02 renderer edits | Browser text injection regression; no script execution needed. |
| A07 Correct evaluation | Custom AP is tie-order-dependent. | Tie-aware standard metric; recompute saved results and selection. | High | Metric agreement and permutation invariance for ties. | Small | Existing saved predictions | Match sklearn; update aggregate/onset reports with hashes. |
| A08 As-of evaluation | September-trained artifact replays January onward; retrospective inputs. | Explicit replay mode/cutoff, historical artifacts, later holdout, causal feature construction. | High | All predictive demo/evaluation rows have correct as-of model/feature provenance. | Medium–Large | A09; data reconstruction | Assert train_end < inference time for backtest; mutate future source rows. |
| A09 Reproducible entry points | Missing imports/data defaults; training stub. | Restore helpers and one final CLI/artifact manifest. | High | Clean checkout reproduces inference and a documented training run. | Medium | Teammate training sources, environment | Import/CLI smoke; compare seeded artifacts/metrics with declared tolerances. |
| A10 Consistent state and zero handling | Separate sources; `0 or fallback`. | Immutable versioned snapshot; null-specific defaults; parity validation. | High | Snapshot/domain/twin shared fields agree. | Small–Medium | Domain source adapters | Override probe, zero-vs-null, wrong timestamp/version tests. |
| A11 Domain failure policy | One domain exception aborts full run. | Typed partial results; evidence-gated candidates and abstention. | High | Healthy facts retained, unsupported decisions withheld. | Medium | Result schema | Fail each source/provider independently; check candidate suppression. |
| A12 KPI/time semantics | Assigned counted as served; full-bin credit; censoring. | Starts/completions/deferred counts, consistent step traces. | High | KPI names match hand-computed scenarios. | Medium | A03–A04 | Tiny fleet manual calculations and horizon-boundary cases. |
| A13 Stable decision objective | Candidate-relative min–max permits reversal. | Fixed scales/baseline utility, feasibility first, minimum effect. | High | Preference between unchanged options unaffected by dominated additions. | Small–Medium | Operator preferences/scales | Reversal fixture; weight sensitivity and no-benefit abstention. |
| A14 Feasible actions | Unlimited assumed resources/capacity; no cost. | Action bounds, resource budgets, compatibility and lead times. | High | Zero infeasible candidates reach ranking. | Medium | Operator assumptions/capacity data | Invalid/overallocated action tests; manual feasibility review. |
| A15 Time-varying context | Current context frozen for 24h; wind unit uncertain. | Unit contract, known event schedules, explicit uncertain weather scenarios. | High | Multipliers change at correct modeled times. | Medium | Verified units/event availability | Event ends at hour two; unknown future weather remains unknown. |
| A16 Honest freshness/races | Demo ticker and stale/out-of-order content. | Run-aware atomic render, cancel/ignore old requests, stale badges. | High | Selected/displayed run never silently differs. | Small | A01–A02 | Slow-old/fast-new response and offline/error browser cases. |
| A17 Legacy logistic fitting | No-validation fit restores initial weights. | Fix final-parameter path or retire baseline. | Medium | Fit learns a separable fixture when used. | Small | Baseline usage decision | Learning test plus validation early-stop regression. |

## Category B — High-impact strengthening

| ID / opportunity | Current implementation / why it matters | Concrete solution | Priority | Expected impact/hypothesis | Effort | Dependencies | Validation |
|---|---|---|---|---|---|---|---|
| B01 Useful coordination | Fixed single-lookups and templates; no follow-up evidence. | Budgeted diagnostic tools, conflict-triggered investigation, typed proposals. | High | Better evidence coverage/candidate relevance than rule baseline. | Medium | A05/A10/A11; tool schemas | Curated conflicting/unknown cases; compare blinded reviewer ratings and invalid proposals. |
| B02 Twin validation and uncertainty | Marginal 30-run spreads; no external validation. | Paired deltas, convergence, parameter sweeps, held-out baseline comparison. | High | Stable or explicitly uncertain recommendations; measured baseline error. | Medium | A03/A04/A12; evaluation data | Seed-count convergence, sensitivity, calibration/holdout plots. |
| B03 Prediction evaluation/policy | Different eligible samples, training-derived operating thresholds. | Common-time evaluation, inner validation, later holdout, episode-aware intervals. | High | Auditable alert burden, recall and lead-time tradeoffs. | Medium | A07–A09 | Frozen split/threshold metadata; common-hour and true-onset checks. |
| B04 Data contracts/port profile | Frozen-source conventions and hardcoded LA/2025 assumptions. | Versioned units/provenance/as-of schema; explicit portability boundary. | High | No silent cross-port/time/schema assumptions. | Medium | Actual source definitions; Saudi data for validation | Duplicate/gap/unit/calendar/port-config contract tests. |
| B05 Reliable execution/ledger | Repeated loads; sparse run logs; no outcome persistence. | Immutable caching, latency instrumentation, run IDs and small durable ledger. | Medium | Measured latency reduction, replayable decisions and failures. | Medium | Version IDs; job policy if needed | Cold/warm benchmarks, concurrency/memory check, replay matching hashes/seeds. |
| B06 Deployment/docs | Unlocked deps, no readiness/CI, broad CORS; stale claims. | Tested environment, runtime/training extras, CI, readiness and scoped deployment settings. | Medium | Reproducible judge setup; accurate executable documentation. | Small–Medium | A09; deployment target | Clean environment smoke, lint, API readiness/auth-origin checks if shared. |

# Part 6 — Practical Stage 2 innovation ideas

These are proposals, not implemented capabilities. Build at most one or two after critical credibility fixes.

| Idea | Operational problem and current evidence | Implementation approach | Priority | Expected impact/hypothesis | Effort/time scope | Dependencies/data | Validation |
|---|---|---|---|---|---|---|---|
| C01 Robust action envelope | A single winning plan may fail under uncertain arrivals, service and gate capacity; current twin gives marginal spreads. | Evaluate the small feasible candidate set over a bounded parameter grid with paired seeds. Show where each plan wins, worst tradeoffs, and a “no robust winner” state. Operator supplies tolerable delay/yard limits. | High | Reveals fragile advice and supports defensible abstention; measure ranking stability and constraint violations. | Medium; short-cycle after correctness | A03/A04/A13/A14, B02; explicit assumption ranges, no new live feed needed | Known easy/ambiguous cases; hold out parameter combinations; check no claimed real-world probability. |
| C02 Evidence challenge and decision receipt | Operators/judges need to ask why a recommendation exists and what would reverse it. Current evidence references are scattered. | Strategy requests one diagnostic tool when signals conflict; save a receipt with snapshot/model versions, cited evidence, rejected candidates, paired deltas and sensitive assumptions. Clicking “why not runner-up?” reveals deterministic tradeoffs; LLM only explains cited facts. | High | Faster inspection of decision basis; fewer unsupported statements. | Medium; feasible short cycle | A05/A10/A11, B01/B05; current history and result schema | Blinded explanation tasks, citation resolution, reproducible receipt, false-premise questions. |
| C03 Operator-constrained intervention workbench | Recommendations ignore staffing, temporary berth availability and implementation timing. Existing manual console is disconnected. | Operator enters gate/overtime limit, berth availability and priority constraints. Backend validates them and reruns identical baseline/scenarios; compare original and constrained feasible set. | Medium | Makes recommendations actionable under actual restrictions; measure feasibility and decision changes. | Medium; short cycle for a few controls | A02/A12/A14; operator inputs; actual roster optional | Reject impossible budgets; compare identical runs; verify controls reach Python engine. |

**Longer term, not a short-cycle commitment:** terminal-level resource synchronization from TOS/AIS; prospective outcome learning; verified Saudi port calibration; online adaptive models. These require data access, validation and operational ownership. A lightweight outcome ledger can precede them; it does not by itself prove causal impact.

## Competition positioning — general considerations, not an official rubric

Strongest advantages: working end-to-end execution; clear separation of evidence, model, simulation and human choice; documented synthetic limitations; deterministic replay; nontrivial temporal modeling; Arabic operator presentation; paired candidate comparison.

Likely judge challenges and honest answers:

| Question | Defensible current answer | Evidence needed for a stronger Stage 2 answer |
|---|---|---|
| What makes it agentic? | Four specialized roles with typed domain investigations and Strategy coordination; currently mainly deterministic workflow and optional narrative. | Recorded bounded tool selection/investigation trace that improves evidence/candidate quality over rules. |
| What makes it a twin? | Executable snapshot-initialized queue/stock-flow what-if simulation. | Correct physical invariants, synchronized traces, calibrated baseline error and assumption sensitivity. |
| How are recommendations better? | Lower weighted simulated KPI score under declared assumptions. | Paired uncertainty, feasibility, stable objective and later operational validation; no observed uplift currently established. |
| What is measured versus invented? | AIS/weather observed or derived; landside/event state synthetic; ML predicted proxy; twin simulated outcomes. | Always-visible field/source badges and a decision receipt. |
| Can it transfer to Dammam? | Architecture is adaptable; current implementation contains LA/2025-specific code and assumptions. | Saudi profile, actual operational data, compatible units/calendars/resources, fresh training and validation. |

Do not present AUC as “accuracy,” a model score as calibrated probability, synthetic outcomes as observed savings, replay as live telemetry, or bounded agent templates as unrestricted autonomy. A compelling demo can be truthful about all five.

# Part 7 — Recommended development roadmap

This sequence is dependency-driven, not a claim about team availability. Do not start by changing the four-role architecture.

## Phase 1 — Immediate fixes: make every displayed claim defensible

1. A01/A16: remove invented timeline points, percentage probability language and fake freshness; retain elevated current pressure; mark offline/stale/replay state.
2. A02/A06: prevent demo animation/cards from overwriting authoritative results; render provider text safely. Temporarily use an honest KPI-only comparison if traces are not ready.
3. A05: make selected option/actions deterministic in narrative. Remove unsupported generative fields until guarded.
4. A07: correct AP, recompute saved metrics/onset selection, and annotate what changed.
5. A10: preserve zero inventory and validate state identity. Fix lint/docs as supporting work, not the headline improvement.

**Exit gate:** one chosen run has one timestamp/source version, no unsupported forecast points, no contradictory winner, and no hidden current critical condition. Corrected saved metrics are traceable.

## Phase 2 — Quick wins: make setup and failure behavior predictable

Restore missing training/import helpers and current CLI defaults (A09). Freeze a tested environment and add a small CI/readiness suite (B06). Add typed degraded results and evidence-gated abstention (A11). Add run IDs/data-model versions and reuse immutable loaded objects (B05). Retire or fix legacy logistic behavior if retained (A17).

**Exit gate:** a teammate can reproduce inference from a clean checkout; domain outage retains healthy facts without manufacturing a recommendation. Environment/artifact metadata is saved.

## Phase 3 — High-impact development: earn the simulation recommendation

Implement service-work and conservation corrections together (A03/A04), then KPI/time traces (A12). Add feasibility bounds (A14), context schedules/units (A15), and stable ranking with minimum benefit (A13). Only then run B02 convergence/sensitivity and connect backend traces to animation.

In parallel once training is restored, implement B03 common-time, later-period evaluation and A08 as-of replay labels/models. Engineering can remove feature leakage; observed operational target validity still needs better data.

**Exit gate:** physical invariants pass; baseline error is measured; each recommendation shows feasibility, raw deltas, paired uncertainty and remaining assumptions. No candidate is called superior solely from a normalized score.

## Phase 4 — Optional differentiation and demo preparation

Choose C01 plus a compact C02 decision receipt, or C03 if operator constraints are available. Add the bounded B01 diagnostic investigation only when it supplies useful new evidence. Avoid additional agent count, a microservice rewrite or a complex 3D port as substitutes for validated decisions.

Prepare three reproducible cases: meaningful pressure with a feasible alternative; conflicting forecast/current evidence; missing data or fragile benefit where Marsa abstains. Store their data/model versions and seed. Keep the same scenario in dashboard, evidence and twin comparison.

# Part 8 — Top five recommended actions

1. **Unify the operator’s view around one truthful decision record (A01/A02/A16).** This fixes visible contradictions quickly and protects the entire demo. Present a real +6h score, independent current statuses, correct timestamp and backend candidate comparison; keep demo animation clearly separate until synchronized.
2. **Repair service progress and cargo conservation before expanding the twin (A03/A04/A12).** These are foundational to whether changing cranes/gates/berths means anything. Acceptance is measurable with tiny hand-checkable examples, without needing a new data feed.
3. **Make evaluation reproducible and temporally honest (A07/A08/A09/B03).** Correcting saved AP is small work; restoring training/as-of holdout is larger but essential. It prevents judges from challenging the wrong model/metric or an in-training replay presented as predictive proof.
4. **Bind recommendations to validated facts and handle missing evidence explicitly (A05/A06/A10/A11).** Fixed decision identity, safe rendering, common snapshot versions and typed degraded results strengthen trust more than more generated explanation.
5. **Rank feasible interventions with paired uncertainty and stable preferences (A13/A14/B02).** This turns a mathematically lowest score into a defensible conditional recommendation and creates the basis for C01 robust action envelopes. Resource limits and cost inputs can start as declared operator assumptions.

The fifth item has more implementation effort than a cosmetic dashboard improvement but directly addresses the project’s differentiator: choosing an operationally admissible intervention with evidence.

# Part 9 — Critical questions and missing evidence

## 9.1 Information the team should supply

| Missing evidence / question | Why it changes the plan | Owner/source to request |
|---|---|---|
| Official Stage 2 rubric, deadline, required deployment/demo format | Determines whether prospective ingestion, autonomy traces or reproducibility receive greater emphasis. No official requirements verified. | Hackathon organizers/team brief |
| Available engineer-days and domain expertise | Effort estimates cannot become a schedule without staffing and deadline. | Team |
| Missing training modules and original environment | Needed to reproduce model comparisons, thresholds and artifacts. | Prediction-model contributor |
| Raw AIS lineage, transformation scripts and units | Cannot independently verify all speed/throughput/wind/wave semantics or original aggregation claims from prepared files. | Data contributors/dataset documentation |
| Independently observed congestion/queue/service labels | Both targets are operational heuristics/proxies; classification performance is not field validation. | Port authority/operator |
| Actual vessel call/service/cargo records and berth compatibility | Required for realistic residual work, waiting ages, priority and terminal allocation. | TOS/terminal planning |
| Gate labor, throughput capacity, overtime cost, implementation lead times | Needed to turn a 30% gate boost into a feasible recommendation and assess costs. | Terminal/gate operations |
| Verified weather units, equipment rules and forecast availability | Avoids applying arbitrary safety/capacity assumptions to unverified units or unknowable future weather. | Data source/equipment/operator |
| Saudi port scope and dataset access | Config-only transfer is not established; local calendar/geography/resources need adaptation. | Mawani/partner |
| Observed intervention outcomes and adoption logs | Required for real savings, prospective learning or causal claims. | Operator trial/outcome ledger |
| Intended hosting/authentication and concurrency | Determines deployment hardening and whether jobs/background workers are necessary. | Team deployment owner |
| Data/model licensing and redistribution permissions | No complete licensing/access review performed; dataset attribution in README is not full proof. | Data owner/project owner |

Do not block engineering corrections on these questions. Missing operational data does limit claims of real-world performance and cost-benefit.

## 9.2 Executed checks and exact results

All commands ran from the repository root with existing installed packages. No packages were installed or upgraded. Source/data/model files were not edited. Existing pytest/ruff caches may have been refreshed by their normal operation.

| Command/check | Result |
|---|---|
| `git rev-parse HEAD` | `504041e0b3911e8585136a74354f9932b61b7c48` |
| `git status --short` before audit | Clean |
| `python -m pytest tests -o addopts= -q` | **239 passed, 1 warning in 45.42s**. Warning: deprecated `Timestamp.utcnow()` at `prediction_model/src/final_proxy.py:346`. This is test duration, not an inference latency benchmark. |
| `python -m ruff check src tests scripts --output-format concise` | **16 errors:** I001 import ordering/E402 import placement in `src/marsa/api/routes/decision_support.py`, `tests/test_final_proxy_pipeline.py`, `tests/test_prediction_pipeline.py`. Four marked automatically fixable. No fixes applied. |
| Read-only CSV/Parquet scan and continuous-history count | Merged v2: 8,688 rows ×42 columns; 0 duplicate hours; 72 missing calendar hours; one wind/wave-missing row; 8,431 exact continuous 24h histories. |
| Cargo source/merged comparison | Zero differing cells across 14 Cargo fields at all shared hours. |
| Saved fold predictions recomputed with sklearn AP | Values in A07; no retraining or artifact changes. |
| C1 saved artifact load and held-out inference | Loaded joblib dictionary containing XGBClassifier, SimpleImputer and 17 feature names. On 4,349 rows from July 2025 onward: ROC-AUC **0.9705621617**, AP **0.8306771248**, matching the report. A sklearn warning noted that the imputer was fitted without feature names. Training was not rerun. |
| Python simulation, narrative and source probes | Values in A03–A05/A10/A12/A13/A17 reproduced directly through existing functions. |
| Bulk +6h scoring and current Cargo comparison | 47 eligible rows with LOW model band and ELEVATED/CRITICAL Cargo status; first examples in A01. |
| File existence check | Both imported root training helpers, generic default dataset, and retired saved-model directory are absent. |

### Reproduce the central behavioral probes

The following is the substantive Python used for the focused checks (the shell invocation was a PowerShell here-string piped to `python -`). It is included in this report for reproducibility; no test/code file was added to the application.

```python
import sys
import numpy as np
sys.path.insert(0, 'src')
from marsa.twin.engine import Scenario, Vessel, run_once
from marsa.twin.state import scenario_from_snapshot
from marsa.twin.scoring import score_results
from marsa.config import load_config
from marsa.agents.strategy_agent import recommend
from marsa.agents.events_weather_agent import (
    EventsWeatherNarrative, _narrative_rejection_reason,
)
from prediction_model.src.evaluation import average_precision
from prediction_model.models.logistic_model import NumpyLogisticRegression
from sklearn.metrics import average_precision_score

def scenario(**overrides):
    values = dict(berths=1, berthed_now=0, waiting_now=0,
        yard_inventory=0, yard_capacity=100,
        discharge_units_per_hour=10, gate_units_per_hour=0,
        arrival_rate_per_hour=0, mean_service_hours=2, horizon_hours=12)
    values.update(overrides)
    return Scenario(**values)

v = Vessel('W', eta=0, service_hours=1, teu=10)
r = run_once(scenario(crane_multiplier=0), [v], 'fcfs')
print(v.depart_time, r.vessels_served, r.yard_end_occupancy)
# 10.0, 1, 0.0: completion despite zero crane work.

v = Vessel('B', eta=-1, service_hours=2, teu=20, at_berth_since=-1)
r = run_once(scenario(crane_multiplier=0, horizon_hours=3), [v], 'fcfs')
print(v.depart_time)  # 1: initial vessel also completes during outage.

v = Vessel('B', eta=0, service_hours=10, teu=100, at_berth_since=0)
r = run_once(scenario(yard_inventory=95, horizon_hours=1,
    yard_slowdown_occupancy=1), [v], 'fcfs')
print(r.yard_end_occupancy * 100)  # 100 rather than accounting for 105.

v = Vessel('W', eta=0, service_hours=20, teu=200)
r = run_once(scenario(horizon_hours=1), [v], 'fcfs')
print(r.vessels_served, v.depart_time)  # 1, 20.0: served means berth-started.

r = scenario_from_snapshot(dict(containers_in_yard=0,
    pola_berthed_cargo_vessels=0, waiting_vessel_count=0), load_config())
print(r.yard_inventory)  # 6000.0: zero treated as absent.

for y in ([1, 0, 1, 0], [0, 1, 0, 1]):
    p = [.5] * 4
    print(average_precision(y, p), average_precision_score(y, p))
# 0.8333333333333333, 0.5; then 0.5, 0.5.

class Provider:
    def generate_structured(self, **kwargs):
        return {'summary': 'الخيار الأعلى ترتيبًا هو زيادة قدرة التخليص عبر البوابات.'}

ranked = [dict(candidate_id='baseline', title='No operational change',
    rank=1, score=0)]
agents = dict(maritime=dict(maritime_status='NORMAL'),
    cargo=dict(cargo_status='NORMAL'),
    events_weather=dict(contextual_pressure=dict(level='low')))
r = recommend(ranked, dict(status='ml_unavailable'), agents, llm=Provider())
print(r['highest_ranked_candidate'], r['reasoning_mode'], r['summary'])
# baseline, llm_assisted, contradictory gate recommendation.

n = EventsWeatherNarrative(weather_summary='Wind is 999 and weather is elevated.',
    event_summary='No synthetic event is active.', calendar_summary='Context available.',
    possible_operational_contribution='Weather may contribute to pressure.')
print(_narrative_rejection_reason(n, active_event=False))  # None.

keys = ['avg_wait_hours', 'max_wait_hours', 'delayed_vessels', 'yard_peak_occupancy']
def result(name, values):
    return dict(candidate_id=name, kpis={k: dict(mean=v) for k,v in zip(keys, values)})
a, b, c = result('A',[0,0,1,1]), result('B',[1,1,0,0]), result('C',[100,100,1,1])
w = load_config()['scoring']['weights']
print([(x['candidate_id'], x['score']) for x in score_results([a,b], w)])
print([(x['candidate_id'], x['score']) for x in score_results([a,b,c], w)])
# A .4, B .6; then B .006, A .4, C 1.

m = NumpyLogisticRegression(epochs=20, learning_rate=.05).fit(
    np.array([[0.],[1.],[2.],[3.]]), np.array([0,0,1,1]))
print(np.allclose(m.weights, np.random.default_rng(42).normal(0,.01,size=1)),
    m.intercept)  # True, 0.0.
```

### Reproduce saved-prediction metric correction

```python
import pandas as pd
from sklearn.metrics import average_precision_score
from prediction_model.src.evaluation import average_precision
p = pd.read_csv('prediction_model/final_proxy/results/fold_predictions.csv')
for model in ['XGBoost', 'CurrentStatePersistence']:
    for h in [1, 3, 6, 12]:
        x = p[(p.model == model) & (p.horizon_hours == h)]
        y, score = x[f'target_{h}h'], x['score']
        print(model, h, len(x), average_precision(y, score),
              average_precision_score(y, score))
```

### Reproduce bulk UI-policy and source-consistency checks

```python
import sys
import pandas as pd
import xgboost as xgb
sys.path.insert(0, 'src')
from marsa.data.load import load_dataset
from marsa.model.congestion_proxy import (
    prepare_source_frame, load_manifest, horizon_spec, FittedPreprocessor, MODEL_DIR,
)
from prediction_model.src.features import build_tabular_lag_features
from marsa.pipeline import run

f = prepare_source_frame(load_dataset())
spec = horizon_spec(load_manifest())
# Dummy labels only satisfy the historical feature builder; no fitting occurs.
f['target_6h'] = 0
f['target_timestamp_6h'] = f.hour_key + pd.Timedelta(hours=6)
x, _ = build_tabular_lag_features(f, spec['source_columns'], 6, 'audit')
pre = FittedPreprocessor.load(MODEL_DIR / spec['preprocessor_path'])
b = xgb.Booster()
b.load_model(str(MODEL_DIR / spec['model_path']))
scores = b.predict(xgb.DMatrix(pre.transform(x[spec['feature_columns']])))
z = f.set_index('hour_key').loc[x.hour_key].copy()
z['score'] = scores
hit = z[(z.score < spec['risk_thresholds']['medium']) &
        z.operations_status.isin(['ELEVATED', 'CRITICAL'])]
print(len(hit), hit[['score','operations_status']].head(2))  # 47

f = load_dataset()
ts = pd.Timestamp('2025-01-02T08:00:00Z')
f.loc[f.hour_key == ts, 'containers_in_yard'] = 0
context = dict(status='supplied_test_context', test_only=True, classification='LOW')
r = run(ts.isoformat(), df=f, forecast_override=context)
print(r['port_state']['containers_in_yard'],
      r['domain_agents']['cargo']['evidence']['containers_in_yard'],
      r['digital_twin']['scenario']['yard_inventory'])  # 0, 6750.38, 6000

class BrokenAgent:
    def investigate(self, request):
        raise RuntimeError('audit domain unavailable')
try:
    run(ts.isoformat(), forecast_override=context, agents={'cargo': BrokenAgent()})
except Exception as error:
    print(type(error).__name__, str(error))  # propagated RuntimeError
```

### Reproduce the C1 artifact evaluation

```python
import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
p = joblib.load('artifacts/models/marsa_xgboost_c1.joblib')
f = pd.read_parquet('data/processed/ml_modeling_table_c1.parquet')
ts = pd.to_datetime(f.timestamp, utc=True)
test = f.loc[ts >= pd.Timestamp('2025-07-01T00:00:00Z')]
score = p['model'].predict_proba(
    p['imputer'].transform(test[p['features']]))[:, 1]
print(len(test), roc_auc_score(test[p['target']], score),
      average_precision_score(test[p['target']], score))
# 4349, 0.9705621616653201, 0.8306771247918209
```

## 9.3 Explicit audit boundaries

No full training run, raw AIS reaggregation, package installation, cloud deployment, real Gemini request, live port feed, browser interaction or operational experiment was performed in this Stage 2 audit. Frontend findings are grounded in source flow; browser execution/visual quality is not claimed. Saved metrics are reproducible from present fold predictions; missing original training helpers prevent a complete independent training reproduction. Dataset collection rights and physical capacity source claims were not independently reverified. Official judging requirements remain unverified.

The existing 239 passing tests are a strength. They establish current contracts, not physical fidelity or prospective value; the reproduced counterexamples explain why the proposed next validation work matters.
