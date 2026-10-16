# Marsa Stage 2 — Agent intelligence using existing repository data

**Date:** 16 October 2026, Asia/Riyadh.

**Source commit:** `504041e0b3911e8585136a74354f9932b61b7c48`.

**Scope:** agent analysis and improvement design only. No agent/application code was modified, no model was trained, no dependency was installed, and no external dataset was collected. The earlier comprehensive audit remains unchanged. This document is a separate, focused recommendation.

## Recommendation first

Choose **Option B: shared historical diagnostic tools, richer domain assessments, and a bounded Strategy investigation loop**. Begin with **Cargo temporal diagnostics** and a shared exact-time evidence contract; then let Strategy request missing or conflicting evidence from the other existing agents. Keep numerical analysis, evidence validation, candidate admissibility and simulation ranking deterministic. Use an LLM selectively to choose a useful follow-up from a small tool catalog and explain cited, verified findings.

The biggest untapped resource is historical information already present in the files but absent from agent tools. Current agents usually read one hour, assign/preserve a status, and describe it. Marsa can become meaningfully more analytical without adding agents or training another large model.

The improvement to demonstrate is: **“Marsa notices a question, requests relevant history, weighs supporting and contradicting evidence, and either proposes a justified simulation test or states why it cannot.”** More fluent text alone is not that improvement.

All new tools, schemas, routes and reasoning loops named below are **proposed**, not implemented. File/line references describe the existing implementation at the audited commit. Existing field names are literal; proposed output names are explicitly introduced as such.

# 1. Existing-data and agent-capability inventory

## 1.1 What data is actually available

Use these source IDs throughout the document. They are shorthand for exact existing files, not new datasets.

| ID | Existing file | Actual scope verified in this analysis | Appropriate role |
|---|---|---|---|
| P | `data/port_operations_agent_dataset.csv` | 8,688 UTC hours, 2025; 7 columns. | Canonical Maritime tool evidence. |
| C | `data/agent2_cargo_operations_2025.csv` | Same 8,688 hours; timestamp plus 14 Cargo fields; synthetic/calibrated synthetic. | Cargo state and temporal diagnostics. |
| M | `data/processed/merged_port_dataset_2025_v2.csv` | 8,688 hours, 42 columns; includes P/C/context and two berth-count columns. | Cross-domain exact-time views and existing ML/twin snapshot. |
| W | `data/processed/events_weather_state_2025.parquet` | Same 8,688 hours; 20 columns; stored timestamps are naive UTC. | Weather and local calendar context, synthetic-event metadata. |
| E | `data/processed/synthetic_events_2025.parquet` and `data/processed/event_registry_2025.csv` | 15 synthetic events; 172 active represented hours in M. | Scenario context and event phase, not empirical incident learning. |
| H | `data/processed/ml_modeling_table_c1.parquet` | 22,534 hours: 5,134 in 2023, 8,712 in 2024, 8,688 in 2025. Range 2023-06-01 through 2025-12-31. | Longer observed/derived maritime/weather reference history and C1 artifact inputs. |
| X | `prediction_model/final_proxy/artifacts/deployment_manifest.json` and `XGBoost_1h/3h/6h/12h/` model/preprocessor pairs | Four fitted XGBoost cargo-proxy models; final training cutoff 30 September 2025. Native integrated path uses +6h. | Existing forecast signal and optional model attribution. |
| Y | `artifacts/models/marsa_xgboost_c1.joblib`, metadata JSON, and `artifacts/explainers/xgboost_c1_*.json` | C1 model/imputer/features/threshold plus saved global/local explanation examples. | Optional distinct maritime forecast and model explanations; not an existing active domain tool. |

### Exact useful fields

- **P:** `hour_key`, `vessel_count`, `waiting_vessel_count`, `approaching_vessel_count`, `departing_vessel_count`, `average_speed`, `port_throughput`.
- **C:** `timestamp`, `import_teu`, `export_teu`, `container_arrivals`, `container_departures`, `containers_in_yard`, `yard_capacity`, `average_dwell_time_hours`, `truck_arrivals`, `truck_departures`, `truck_waiting_time_minutes`, `gate_throughput`, `yard_occupancy_percent`, `cargo_flow_ratio`, `operations_status`.
- **M additional:** `pola_berthed_cargo_vessels`, `pola_berthed_tanker_vessels`; the weather/event fields listed below.
- **W:** `hour_key`, `wind_speed_10m`, `wave_height`, `weather_code`, `hour`, `hour_sin`, `hour_cos`, `day_of_week`, `month`, `is_weekend`, `is_public_holiday`, `weather_pressure_index`, `event_active`, `event_id`, `event_type`, `event_severity`, `event_start`, `event_end`, `event_duration_hours`, `event_represented_hours`.
- **H:** `timestamp`, `vessels_in_area`, `vessels_waiting`, `waiting_ratio`, `cargo_vessels_in_area`, `tanker_vessels_in_area`, `avg_sog`, `median_sog`, `ship_density`, `avg_port_speed`, `port_throughput_lag1`, `wind_speed_10m`, `wave_height`, `weather_code`, `hour_sin`, `hour_cos`, `day_of_week`, `month`, `c1_congestion_next_6h`, `split`.

H does **not** provide three years of Cargo history, berth assignments, actual waiting age per vessel, gate staffing, service-time labels, crane allocation or observed intervention outcomes. The large raw AIS file described in old reports is not a usable checked-in source for this proposal. Aggregate histories cannot justify named-vessel prioritization or terminal-specific causal claims.

## 1.2 Newly verified constraints that matter for reasoning

1. **Do not alias waiting measures across files.** P/M `waiting_vessel_count` differs from H `vessels_waiting` in **3,244 shared 2025 hours**, with a maximum absolute difference of six vessels. At `2025-07-10T19:00:00Z` they are 19 and 18 respectively. Total `vessel_count` and H `vessels_in_area` do match in the checked 2025 rows. The reason for waiting-definition differences is **Unverified**. A cross-model disagreement might reflect definitions, not a surprising port condition.
2. **Avoid counting redundant features as independent evidence.** H `ship_density` equals `vessels_in_area` throughout the table. C `gate_throughput` equals `truck_arrivals` in approximately **92.14%** of rows. Occupancy is a rounded transformation of yard inventory/capacity. Three correlated metrics do not constitute three independent confirmations.
3. **Preserve time gaps.** 72 hours are absent from the 2025 timeline. Exact t−6h and rolling-window coverage must be checked. Never substitute six previous rows for six clock hours.
4. **Synthetic construction is a separate limitation from future-row leakage.** Cargo has documented maritime influence at 2–10h lags and retrospective monthly calibration; weather pressure uses a full-period historical baseline. A tool can correctly read only rows ≤t while still analyzing data whose original construction was retrospective. Label that honestly.
5. **Weather uncertainty is real.** One 2025 weather hour is missing; H has ten missing wind/wave/code rows over all years and 26 missing lagged-throughput entries. Units and some throughput semantics remain unverified in the agent implementation. `weather_code` is deliberately not interpreted.
6. **The artifact can know later training information.** X’s final training cutoff is September 2025. Its January/July replay outputs are retrospective, not point-in-time prediction evidence. C1 metadata specifies training through 2024, validation January–June 2025; its validation-selected threshold and any validation-dependent model selection must not be presented as available before those validation hours. Artifact creation/as-of selection also requires explicit evidence.

Evidence: direct field/row comparisons in this analysis; `data/agent2_deliverables/methodology_note.md`; `scripts/build_events_weather_state.py:263`; `src/marsa/agents/maritime_agent.py:318`; `src/marsa/agents/events_weather.py:21`; X/Y metadata; `src/marsa/data/features.py:27`.

## 1.3 Actual capabilities of each agent

| Role | What currently executes | Data/history it actually receives | Rules/models/LLM | Tools and missing reasoning |
|---|---|---|---|---|
| Maritime | `MaritimeInvestigationAgent.investigate()` calls source once and builds deterministic findings/status. | One P hour: counts, waiting ratio, raw speed and throughput values. Source internally loads the full CSV, but no historical diagnostic API is exposed. Optional forecast context is only risk level/horizon. | Preserves teammate waiting-ratio heuristic: ELEVATED at 0.263; CONGESTED at ≥0.304 and zero throughput. No ML model is invoked by this agent. Optional narrative provider has numeric/citation checks. | `MaritimeStatusSource.get_status()`. Missing pressure duration, direction, denominator effects, recent recovery/acceleration, historical rarity, vessel-type/berth context. |
| Cargo & Logistics | Reads frozen row; preserves stored `operations_status`; describes occupancy, flows, truck waiting and gate throughput. | One C hour via `CargoEvidence`; truck arrival/departure fields exist in C but are not exposed in current public evidence. | Stored synthetic rule-derived status; no predictive model owned by this agent. Optional narrative is tightly constrained and must disclose synthetic state. | `CargoStatusSource.get_status()`. Missing accumulation/recovery, six-hour net flow, dwell anomaly relative to historical baseline, truck-flow imbalance, counterevidence and bottleneck discrimination. |
| Events & Weather | Builds validated weather/event/calendar context; classifies weather and combined pressure; returns evidence-completeness score. | One W hour. Context builder holds event ID/start/end/duration and raw weather; final result drops some phase/raw details. DST-aware LA local calendar is recomputed. | Weather index bands <0.33/<0.67/otherwise; active events elevate context by rules. Confidence 0.95 with complete evidence, 0.45 in missing-weather example. Optional narrative only. | `EventsWeatherContextSource.get_context()`, builder, `classify_weather()`, `classify_context()`, `evidence_confidence()`. Missing trend, freshness, active-event phase, context-sufficiency requests and clear differentiation of forecast versus observed context. |
| Orchestrator & Strategy | Ordinary pipeline calls three domains sequentially; Strategy preserves statuses, makes candidate templates, runs twin and ranks. | The three outputs, cargo-proxy forecast dict, snapshot/config indirectly through scenario. No historical diagnostic tool requests or follow-up rounds. | Deterministic candidate templates and scoring; optional LLM rewrites one summary. No autonomous plan execution or agent debate. | `generate_candidates()`, `synthesize()`, `evaluate()`, `recommend()`. Missing conflict taxonomy, diagnostic question selection, bounded evidence acquisition, hypothesis revision, decision-quality abstention and paired benefit interpretation. |

Source references: `src/marsa/agents/maritime_agent.py:209`, `:258`; `src/marsa/agents/tools/maritime_status.py:55`; `data/port_agent.py`, `determine_port_status()`; `src/marsa/agents/cargo_agent.py:200`, `:243`; `src/marsa/agents/tools/cargo_status.py:15`, `:81`; `src/marsa/agents/events_weather.py:165`; `src/marsa/agents/events_weather_agent.py:158`, `:180`, `:255`, `:354`; `src/marsa/agents/strategy_agent.py:66`, `:163`, `:194`, `:233`; `src/marsa/pipeline.py:36`.

The integrated `pipeline.run(llm=...)` constructs default domain agents **without** a provider and passes `llm` to Strategy recommendation. Individual agent API dependency factories can supply Gemini separately (`src/marsa/api/dependencies.py:20`). Therefore enabling a key does not mean every domain performs LLM reasoning in the integrated cycle.

Existing strengths to retain: typed evidence and provenance, independent domain judgments, exact timestamps, canonical stored status, deterministic fallback, and an explicit human decision boundary. Rule-based arithmetic is not a weakness; the missing capability is selecting and interpreting relevant additional evidence.

# 2. Untapped opportunities and strict temporal rules

## 2.1 Opportunities supported by the data

| Diagnostic question | Existing inputs/history | Appropriate method | What it establishes—and what it does not |
|---|---|---|---|
| Is Maritime pressure building or easing? | P waiting/total/approaching/departing counts, speed/throughput; 1/3/6/12/24h lookbacks. | Exact-hour differences, regression slope over complete short windows, high-status run length, count versus ratio decomposition. | Current progression under dataset definitions; not vessel-specific waiting time or verified cause. |
| Is high yard inventory still accumulating? | C inventory, arrivals/departures, occupancy/capacity; 3/6/12/24h. | Net inflow sum, inventory delta, persistence and reconciliation residual with documented interval convention. | Accumulation versus loaded-but-draining state. Small residuals may reflect rounding; gaps or convention uncertainty must be shown. |
| Which Cargo indicators support or contradict gate-pressure hypothesis? | C truck waiting, truck arrivals/departures, gate throughput, dwell, yard/inflow. | Separate evidence dimensions: stock, delay, gate flow; rolling changes and prior reference quantiles. | A supported operational hypothesis within synthetic state, not real gate capacity or a causal diagnosis. |
| Is an indicator unusual for comparable hours? | P/C/W; H for longer Maritime/weather history. | Prior-only median/IQR/MAD or empirical percentile; optionally condition on local hour after enough support. | Unusual relative to declared reference, not dangerous or operationally abnormal by official standard. |
| Is there a sustained change? | P waiting and C flow/truck wait/inventory. | Short/long trailing-window contrast first; later forward-only CUSUM/Page–Hinkley on baseline-standardized residuals. | An onset/change signal with uncertainty. Offline centered smoothing/change-point segmentation is inappropriate for as-of replay. |
| Do domains show compatible timing? | Exact P/M Maritime history and C flow/delay at the same timestamps. | Compare predefined 2/6/10h lag windows; fit relationships only in earlier training blocks, if needed. | Temporal consistency or contradiction. Cargo’s designed AIS lag means correlation may reproduce its generator. |
| Is weather context changing or missing? | W wind/wave/index; H prior raw weather; timestamps. | Coverage and last-valid age, differences, short-window persistence; train/prefix-only relative baselines for new diagnostics. | Evidence sufficiency and relative conditions; not an actual crane shutdown forecast. |
| What does an active synthetic event imply for a scenario? | W/E type/severity/start/end; config event mappings. | Event phase and declared schedule; expose assumptions to Strategy. | A synthetic scenario adjustment, never evidence that the event caused current congestion. |
| What earlier states resemble this one? | P/C/M; H for distinct Maritime retrieval. | Numeric nearest neighbors on a compact, robust-scaled feature set; prior-only eligible cases, temporal de-duplication. | Comparable historical patterns; no observed action effectiveness or case-based causal recommendation. |
| Which features moved the existing model score? | X fitted preprocessor, exact current feature row and booster; Y separately. | Native XGBoost feature contributions, group into source/domain/window with signed totals. | Model attribution, not physical cause; redundant features and in-training replay limit interpretation. |

No additional data feed is needed for these calculations. Data sufficiency for calculating a statistic is different from sufficiency for validating an operational claim.

## 2.2 Mandatory as-of contract

Every proposed tool should accept an authoritative `timestamp_utc=t` and obey:

1. Observation access is limited to timestamps ≤t. Baseline fitting normally excludes t: use `[t−28d,t)` or a frozen earlier training block, with a minimum sample requirement. The 28-day choice is a proposed configurable starting point, not a validated optimum.
2. Rolling six-hour flow sums refer to bins `(t−6h,t]`: exactly six hours. A six-hour endpoint delta requires both t and t−6h; a contiguous seven-point slope requires all seven points. Return coverage and missing timestamps. Never fill source gaps merely to obtain a trend.
3. Reference quantiles, scaling, clustering, anomaly thresholds, retrieval indices and model selection are fit before the evaluation cutoff. Do not compute a full-year percentile then display it as known in January.
4. Candidate historical case s must be earlier than t. If showing its next-h-hour outcome, require `s+h≤t`; for evaluation, keep all fitting/case selection within the allowed past and embargo near neighbors, initially at least 24h. Missing future target hours remain unavailable.
5. Do not expose H’s `c1_congestion_next_6h`, `split`, final proxy labels, future observations, or saved evaluation outcomes as current agent evidence. Labels are allowed only in controlled evaluation or matured past-case outcomes.
6. Existing X artifacts may be used in retrospective demonstration with training cutoff visible. For prospective-style evaluation, use only as-of-valid artifacts or explicitly decline that claim. A future-invariant feature builder cannot make a later-trained model point-in-time valid.
7. Current W `weather_pressure_index` remains the canonical frozen heuristic. If creating a prefix-fitted weather comparator, give it a new diagnostic name and fit-end metadata; do not silently replace X’s fitted input semantics or break artifact parity.
8. Synthetic event end/duration is available as a **frozen scenario schedule**. Its future end must not become an observed predictor or proof that a real operator knew that duration. Event represented-hours counts can depend on future timeline coverage and should not enter new as-of predictors.
9. Recompute local calendar from UTC with `America/Los_Angeles`; never reinterpret legacy integer day encodings by guess. DST and UTC/local weekday disagreements are already tested in context code.
10. Return both `read_as_of` and `construction_limitations`. “Reads no future rows” does not erase retrospective Cargo/monthly calibration or full-period weather thresholds.

Existing gap-safe functions are useful building blocks: `src/marsa/data/features.py:27`/`:39`, `prediction_model/src/features.py:72`/`:95`, `src/marsa/model/congestion_proxy.py:99`. They are **not** generic agent investigation APIs today.

## 2.3 Shared proposed evidence schema

Keep current result fields/statuses for compatibility. Add typed diagnostic facts and hypotheses; do not rewrite canonical statuses merely because new evidence arrives.

```json
{
  "diagnostic_id": "cargo.net_inflow_6h",
  "timestamp_utc": "<selected UTC hour>",
  "source_file": "data/agent2_cargo_operations_2025.csv",
  "source_fields": ["container_arrivals", "container_departures"],
  "window": {"hours": 6, "interval": "(t-6h,t]", "expected_rows": 6},
  "value": null,
  "coverage": {"observed_rows": null, "missing_timestamps": []},
  "provenance": ["SYNTHETIC", "DERIVED"],
  "max_source_timestamp": "<not later than t>",
  "reference_fit_end": null,
  "semantics_verified": false,
  "limitations": ["Retrospective synthetic operational state"]
}
```

This is a **proposed schema**, with placeholders/nulls, not a current response. A hypothesis should contain an ID, short statement, `supported|mixed|unsupported|insufficient` assessment, supporting and contradicting evidence IDs, a next useful question and explicit operational limitations. Avoid a fabricated “confidence 87%.” Evidence coverage, provenance, model relevance and simulation stability should remain separate dimensions.

# 3. Individual-agent improvements

## 3.1 Maritime: explain progression, not just waiting ratio

**Problem/current handling:** a ratio-based status does not distinguish a growing absolute queue, denominator shrinkage, a transient spike, or recovery. The existing agent describes one-hour values (`src/marsa/agents/maritime_agent.py:258`).

**Proposed tools and exact data:**

- `maritime_history(t, windows=[3,6,12,24])`: P `vessel_count`, `waiting_vessel_count`, approaching/departing counts, speed/throughput. Compute count/ratio changes, high-status duration using the same unrounded teammate rules and exact contiguous timestamps, short-window trend and counterevidence. Treat speed/throughput as named source values until definitions are verified.
- `maritime_berth_context(t)`: M `pola_berthed_cargo_vessels` and configured container berth count, reported as approximate LA cargo-berth context. The P queue spans the LA/LB region and all types; do not divide regional waiting counts by LA container berths as if they were identical scope.
- `maritime_reference(t)`: P history for canonical definitions; H subtype composition (`cargo_vessels_in_area`, `tanker_vessels_in_area`) in a separate namespace. Do not silently extend P waiting history back to 2023 using H’s different definition.

**Changes:** add prefix-history methods beside `MaritimeStatusSource.get_status()` (`src/marsa/agents/tools/maritime_status.py:55`); add `diagnostics`, `progression`, `hypotheses` to the result in `src/marsa/agents/maritime_agent.py:64`. Preserve existing status, evidence and findings. A planner can request a specific historical diagnostic instead of always running every tool.

**LLM needed?** No for metrics/classification. Optional for interpreting conflicting count/ratio/berth evidence through cited facts or selecting a follow-up. It must not infer anchoring, a berth blockage, vessel service duration, or cause from count data.

**Proposed output additions:** `waiting_delta_6h`, `ratio_delta_6h`, `pressure_duration_hours`, `trend_support`, `scope_mismatches`, and a hypothesis such as “regional waiting pressure is rising while LA berth count alone does not establish insufficient capacity.” All include coverage/source IDs.

**Evaluation:** hand-computed rising/falling/denominator-change windows; gaps force unknown; changing future rows leaves facts unchanged; status remains identical to legacy output; review whether a selected follow-up resolves a documented question. Future queue-change accuracy can evaluate a separately declared forecast, not retroactively convert a descriptive trend into prediction.

## 3.2 Cargo: separate stock, accumulation, delay and gate-flow evidence

**Problem/current handling:** high yard occupancy prompts a general Cargo-pressure finding and often a fixed gate-capacity test, even if the yard has begun draining. Current single-hour flow ratio is not six-hour persistence (`src/marsa/agents/cargo_agent.py:243`; `src/marsa/agents/strategy_agent.py:90`).

**Proposed tools:**

1. `cargo_balance(t,6)`: C arrivals/departures/inventory/occupancy/capacity. Return cumulative net flow, stock change, coverage and approximate stock-flow residual. Expose rounding tolerance and interval convention; do not “repair” gaps or call normalized units TEU.
2. `cargo_delay_profile(t,6,24)`: C truck waiting, dwell, truck arrivals/departures and gate throughput. Return short-term changes and separate earlier-reference percentiles; distinguish elevated stock with improving delays from accumulating stock with worsening delays.
3. `cargo_reference(t)`: prior-only medians/quantiles, optionally local-hour conditioned after enough support. Do not normalize dwell against finalized monthly baselines when evaluating as-of behavior; use earlier available rows or a frozen earlier reference.
4. `cargo_buffer_scenario(t)`: optional deterministic stress extrapolation `(capacity−inventory)/positive trailing net inflow`. Return unavailable for nonpositive net flow, incomplete data or unstable trend; call this a constant-flow scenario, not a congestion prediction. Start after the three tools above.

**Changes:** extend `CargoStatusSource` to expose exact-history views including existing truck arrival/departure columns; add a diagnostic builder beside `_deterministic_result()` (`src/marsa/agents/cargo_agent.py:243`) and a structured hypothesis set such as `accumulating_stock`, `loaded_but_draining`, `gate_delay_context`, `dwell_elevated`. Hypotheses can coexist; avoid forcing one bottleneck label.

**LLM needed?** No for balances/trends/reference comparison. Optional to ask whether gate-delay evidence actually supports testing a gate action, or to phrase a qualified multi-signal diagnosis. The next tool and supporting/counterevidence are more valuable than another paragraph.

**Proposed output additions:** `stock_state`, `flow_progression`, `delay_progression`, `supporting_evidence_ids`, `contradicting_evidence_ids`, `synthetic_state=true`, `next_question`. Keep stored `cargo_status` unchanged.

**Evaluation:** exact sum/delta checks, rounding-aware reconciliation, rising stock with falling delay cases, draining stock with still-high occupancy, gap and zero-inventory cases. Candidate quality is measured as evidence-supported testing, not an assertion that a gate boost will work in reality. If evaluating future Cargo labels, call the result synthetic-proxy consistency.

**Why first:** Cargo has the richest coherent domain fields and directly triggers a common action. The July example below already shows useful distinction between “high inventory” and “worsening accumulation,” with no new model or data required.

## 3.3 Events & Weather: provide phase, freshness and bounded contextual support

**Problem/current handling:** context is useful but largely a current level plus event flag. Complete evidence receives nearly the same confidence regardless of rarity or relevance. A synthetic event can be misconstrued as a cause unless its purpose is explicit (`src/marsa/agents/events_weather_agent.py:180`, `:354`).

**Proposed tools:**

- `weather_history(t,6,24)`: W raw wind/wave/index, coverage, age of last complete observation and recent persistence. If current weather is missing, keep it missing; an old value is labeled stale comparison, not substituted current evidence. No weather-code interpretation or guessed units.
- `weather_reference(t)`: H raw weather values before t or a frozen pre-2025 fit; compare rarity in source units. Generate a separately named prefix-relative diagnostic; preserve W’s original full-baseline pressure value for compatibility and disclose its construction.
- `event_phase(t)`: W/E active event ID/type/severity/start/end. Return elapsed schedule time and remaining **synthetic scenario** duration, plus whether it began before the current pressure episode. Event start timing can refute a claim it preceded already-existing pressure; it cannot prove causality in either direction.
- `context_sufficiency(question,t)`: determine whether current weather observations, a synthetic event schedule or mere calendar flags can answer the Strategy question. Return “insufficient” when asked for tomorrow’s weather or verified real incident information.

**Changes:** add history access to `EventsWeatherContextSource` (`src/marsa/agents/tools/events_weather.py:37`); keep `classify_weather()` and `classify_context()` for canonical current labels; expose event phase/raw-value evidence separately from narrative. Add numeric/status consistency checks to `_narrative_rejection_reason()` (`src/marsa/agents/events_weather_agent.py:206`), which currently does not verify all such facts.

**LLM needed?** Not for phase/freshness/classification. Optional to explain why context supports only a scenario or why a question cannot be answered. Calendar facts must not become port closure/operational-pressure claims.

**Proposed output additions:** `weather_freshness`, `observation_coverage`, `relative_baseline_fit_end`, `event_phase`, `scenario_only`, `cannot_establish`. Keep confidence labeled evidence completeness, not decision reliability.

**Evaluation:** missing hour `2025-11-02T08:00:00Z`; event-start/ongoing/end boundaries; DST repeated-hour behavior; event chronology; forbidden claims that no event record proves no real incident; no invented numeric weather or interpretation of unknown codes.

## 3.4 Strategy: ask a useful question before enlarging the candidate set

**Problem/current handling:** `synthesize()` lists independent signals and pressure areas; `generate_candidates()` applies templates. It does not ask why current Cargo and future proxy disagree, whether pressure is recovering, or whether extra evidence would change the candidate set (`src/marsa/agents/strategy_agent.py:66`, `:163`).

**Proposed tools/behavior:**

1. `detect_investigation_needs()`: distinguish current-vs-future horizon difference, domain-scope difference, missing evidence, worsening/recovery ambiguity and unsupported attribution.
2. `request_diagnostic(agent,tool,args)`: select a bounded tool that addresses an unresolved question; obtain typed facts and verify read-as-of/provenance.
3. `update_hypotheses()`: revise only diagnostic hypotheses, keeping canonical statuses and original facts. Record concise evidence-linked changes, not hidden chain-of-thought.
4. `generate_supported_candidates()`: choose from existing supported actions using declared preconditions, evidence support and simulation assumptions. A hypothesis enables a **test**, not an execution instruction or proof of benefit.
5. `compare_simulated_evidence()`: preserve raw KPI tradeoffs, ties, paired uncertainty and important assumption sensitivity; return `no_distinguishable_benefit` or `insufficient_evidence` when warranted.

**Changes:** add an investigation stage between domain collection and scenario/candidate construction in `src/marsa/pipeline.py:65`; extend Strategy with a typed plan/results envelope. Keep four roles. The coordinating responsibility can live in the existing pipeline/Strategy boundary; no new Orchestrator agent or framework is necessary.

**LLM needed?** Optional in the strong scope; useful for choosing among several valid next questions, articulating alternative noncausal hypotheses and explaining disagreements. A deterministic router should be the measured baseline. Any LLM choice must resolve to a valid tool enum and arguments, be checked for relevance/budget, and fall back to the router.

**Proposed output additions:** `investigation_needs`, `tool_requests`, `evidence_added`, `hypothesis_updates`, `unresolved_questions`, `candidate_preconditions`, `decision_qualification`. The selected action identity and all numeric results are deterministic.

**Evaluation:** compare question/tool selection to predeclared case checklists; measure relevant investigations, invalid calls, grounded findings, candidate evidence coverage, unjustified candidate expansion and correct abstention. An LLM planner is retained only if it adds measurable value over the router.

# 4. Cross-agent collaboration and bounded orchestration

## 4.1 Recommended investigation state machine

```text
Canonical snapshot + existing forecast
    -> initial domain assessments
    -> detect unresolved questions and evidence quality
    -> choose relevant diagnostic tools (bounded)
    -> validate returned facts and update diagnostic hypotheses
    -> optionally one further investigation round
    -> supported candidate tests, with baseline
    -> twin evaluation and qualified deterministic decision
```

Suggested initial budget: at most **two follow-up rounds**, **six diagnostic calls in total**, and at most **two calls to one domain**. These are proposed limits to tune, not existing configuration. Basic snapshot facts are not counted as follow-up calls. Keep tool results cached within the run so an identical request is not repeated. Stop when the question is resolved, evidence cannot be obtained from existing sources, no useful tool remains, or the budget is exhausted.

No tool can access arbitrary files, arbitrary Python/SQL, unrestricted future timestamps or external feeds. A validated catalog defines source, allowed windows, returned schema and failure semantics. An invalid/unnecessary tool request gets a typed rejection and deterministic fallback, not another unconstrained attempt.

## 4.2 Concrete routes worth implementing

| Trigger | Actual question | Follow-up | Allowed conclusion/update |
|---|---|---|---|
| Cargo ELEVATED while +6h proxy LOW | Is this conflicting evidence or current pressure with a different future target? | Cargo balance/delay progression; model target/horizon metadata; optional attribution. | Preserve both. Explain horizon/proxy distinction; still test gate only when current evidence supports it. |
| Maritime waiting ratio rises | Are more vessels waiting or did total count fall? | P count/ratio decomposition; approximate M berth context. | Distinguish denominator effect from absolute growth; do not declare a specific berth fault. |
| Yard occupancy high but one-hour net flow negative | Is that recent improvement or persistent drainage? | Cargo six/12-hour balance and truck-wait changes. | Loaded-but-draining or mixed evidence; do not automatically infer new capacity is necessary. |
| Weather UNKNOWN, Maritime elevated | Can context explain pressure? | Current coverage/last-complete age; known scenario event check. | Context evidence insufficient; do not invent calm weather or causal explanation. |
| Active demand surge but normal current domains | Does scenario context justify a precautionary test? | Event phase plus short-domain trends. | Scenario-only sensitivity test or baseline; no forced intervention from synthetic event alone. |
| ML HIGH while current domains low | Is the model driven by stock/calendar/history or a target mismatch? | Native model attribution plus history/reference checks. | Explain model reliance; qualify retrospective artifact; do not manufacture current domain congestion. |

Revising an assessment means updating `progression` or a hypothesis with new evidence, not rewriting the authoritative stored Cargo status or forcing all domains to agree. Retain counterevidence. Unresolved differences can be the correct final output.

## 4.3 Router alternatives

| Approach | Real reasoning value | Complexity and reliability | Evaluation | Recommendation |
|---|---|---|---|---|
| Deterministic tool routing | Explicit conflict/missingness checks request relevant history; transparent and repeatable. | Low–Medium; easy to bound/test; misses novel combinations outside rules. | Tool relevance, question resolution and candidate support on fixed cases. | Start here; strong Option B core. |
| Bounded LLM tool selection | Can choose between plausible follow-ups and articulate alternative hypotheses when combinations are ambiguous. | Medium; needs strict catalog, semantic checks, budget and fallback; may call redundant tools. | Same cases versus router, invalid/unnecessary calls, evidence novelty, grounded output, repeatability. | Optional small addition after facts are reliable. |
| Hybrid router plus one LLM planning choice | Router handles hard rules and common cases; LLM proposes a follow-up only when multiple admissible questions remain. | Medium; limits novelty to a small safe space. | Incremental quality benefit over router, including cases where LLM must say “no useful tool.” | Best route toward Option C. |
| Unrestricted chat/debate between agents | More conversation, not necessarily new evidence; difficult to attribute value. | High; circular claims, repeated calls and ungrounded consensus. | Expensive to evaluate; no local evidence justifies it. | Do not build for this stage. |

No external agent framework is needed for these patterns. Existing Pydantic models, provider protocol, Python functions and pipeline are sufficient.

# 5. Additional AI/ML capabilities: what is justified

## 5.1 Ranked analytical/model candidates

| Technique | Exact inputs, objective and labels | Training/evaluation and simpler baseline | Circularity/leakage limits | Decision value / verdict |
|---|---|---|---|---|
| Native XGBoost attribution | X’s exact feature row and 604 transformed feature names; explain log-odds contributions. No new labels/training. Y explanations separately. | Check contribution sum plus bias equals raw model margin; compare model-specific evidence to generic status prose. | Model explanation is not cause. Current stock and occupancy share information. Later-trained replay stays retrospective. | **High priority.** Explains an existing forecast rather than adding a score. |
| Historical numeric case retrieval | Canonical P waiting/total ratio and trends; C occupancy, 6h net inflow, truck wait, dwell; optionally M approximate berth count. Objective: relevant prior states; no diagnosis labels required. | Fit scaling on earlier block/prefix; retrieve prior cases only, exclude near-adjacent duplicates. Compare to most recent case and same-local-hour baseline. Score reviewer relevance and matured-outcome summary coverage. | Do not retrieve on future outcomes/status label then claim discovery; do not infer action effects. Keep observed-only and synthetic-assisted distances separate. | **High/Medium after temporal tools.** Useful evidence/examples; numeric distances need no embeddings. |
| Robust anomaly profile | P ratio/count change and C flow/truck wait/dwell with prior median/MAD/IQR. Objective: unusual state, no labels. | Forward reference fit, min-support rules, fixed alert-budget comparison. Baseline: percentile/threshold. Optional IsolationForest only if it helps joint unusualness. | Anomalous ≠ congestion. Synthetic generator patterns can dominate. Avoid duplicate occupancy/stock/density votes. | **Medium.** Start deterministic; a trained detector is optional. |
| Forward change detection | P waiting count/ratio; C net inflow/truck wait. Objective: sustained shift, no official changepoint labels. | Trailing short-vs-long contrast baseline; then forward-only CUSUM/Page–Hinkley, thresholds tuned on earlier blocks. Score warning burden and overlap with explicitly defined future episodes. | Do not center windows or fit on whole timeline. Defined episodes remain proxy/rule events, not ground truth incidents. | **Medium.** Useful only if earlier/more selective than simple slopes. |
| Operational-state clustering | Compact normalized P/C feature set, not every correlated column. Objective: descriptive regimes; no labels. | Fit on earlier block; choose k/stability there. Compare cluster summaries against transparent stock/flow/delay quadrant rules. | “Cluster 2” is not a new operational truth; synthetic class structure may be rediscovered. | **Low/Medium.** Only if regimes materially improve retrieval or tool routing. |
| Lightweight future trend forecast | P future waiting count; C future inventory/truck wait; features current plus lag/trailing summaries. Exact labels are available after their horizon matures. | Walk-forward with common eligible times; persistence and trailing linear extrapolation first; ridge/gradient-boosted model optional using installed sklearn. Evaluate MAE, bias, episodes and conditional decision relevance. | C targets are synthetic; overlapping samples need temporal separation. Missing future hours dropped. Models must fit before query. | **Medium/Low.** Do not duplicate cargo risk without a distinct decision question. |
| C1 Maritime sidecar | Y’s 17 feature contract from H; target congestion anywhere next six hours, unlike cargo proxy at exactly +6h. Existing trained model, no new training. | Preserve original scaler/imputer/model and threshold timing. Compare C1 versus simple P trends separately, respecting definition differences. | H waiting differs from P; C1 threshold validation cutoff; C1/weather missing values and target definitions. | **Medium.** Potentially useful, but not a five-minute alias of current evidence or an independent vote. |
| Supervised root-cause/action-effect model | Would need verified bottleneck causes or actual intervention/outcome labels. Existing files lack them. | No defensible training/evaluation target from existing data. | Learning stored status/event rules would be circular; simulation-generated “best action” labels reproduce simulator assumptions. | **Insufficient data. Do not build.** |

There is enough data for temporal analytics, retrieval, model attribution and modest exploratory unsupervised/time-series experiments. There is not enough independent operational evidence to establish causal diagnosis or optimal interventions. Only 15 synthetic events are especially inadequate for learning real event effects.

## 5.2 Existing explanation capability was tested, not assumed

Using X’s +6h booster and fitted preprocessor at `2025-07-10T19:00:00Z`, `Booster.predict(..., pred_contribs=True)` returned **605 contributions including bias** for **604 named transformed features**. The contribution sum was `3.15844536`, versus raw margin `3.15844488` (floating-point agreement). No SHAP package installation was needed.

The five largest absolute individual contributions in that probe were:

| Transformed feature | Signed contribution in log-odds |
|---|---:|
| `containers_in_yard__current` | +0.789351 |
| `hour__lag_1h` | +0.395194 |
| `yard_occupancy_percent__current` | +0.325972 |
| `hour__current` | +0.208699 |
| `container_departures__roll_mean_12h` | +0.189520 |

This suggests a useful explanation question—how much of this model score is associated with stock, flow history and calendar variables—not an assertion that hour-of-day causes congestion. Group signed contributions by source and family while preserving correlated-feature limitations. Do not mix Y’s stored C1 SHAP examples into an explanation of X’s cargo score.

Source integration points: `src/marsa/model/congestion_proxy.py:28`, `:99`, `:158`; `prediction_model/src/final_proxy.py:262`; Y explainer JSON. The current runtime does not expose local contributions. The feasibility probe is analysis only.

# 6. Strategy and decision reasoning changes that matter

## 6.1 Keep model risk, current pressure and simulated benefit distinct

The +6h XGBoost score affects forecast context and a narrow preventive gate-test rule. **It does not generate the twin’s arrival process.** `src/marsa/pipeline.py:67` calls `build_scenario(snap,cfg,events_weather)`; `src/marsa/twin/state.py:16` derives arrival intensity from current berthed count and assumed service time. The model score is absent from that scenario function.

Recommended Strategy output should separate:

- **Current evidence:** canonical statuses and temporal diagnostics.
- **Predicted proxy state:** target, horizon, artifact cutoff, uncalibrated score and availability.
- **Hypotheses:** supported/mixed/insufficient explanations of current patterns, with counterevidence.
- **Candidate test rationale:** why this action is worth simulating, and what required real feasibility data is unavailable.
- **Simulated result:** raw KPI differences under explicit assumptions; ties and uncertainty.
- **Decision qualification:** no distinct winner, conditional simulation preference, or insufficient evidence.

Do not multiply three domain confidence values or average statuses into a supposed congestion probability. Their evidence is correlated, their scopes differ, and current context completeness is not predictive reliability.

## 6.2 Better candidate selection without expanding the architecture

Keep the existing action catalog and baseline, but distinguish candidate rationale:

- Rising waiting counts with supported queue evidence → queue-policy comparison. Do not invent real vessel service estimates; the twin samples them.
- High Cargo stock with sustained accumulation and worsening truck delay → stronger evidence to test the existing gate-capacity alternative.
- High stock but falling inventory/delays → keep gate as a conditional experiment if the canonical elevated state supports it; explicitly say evidence is mixed, rather than asserting it is necessary.
- Current MODERATE Cargo plus the existing HIGH active proxy alert → investigate forecast attribution/progression before the preventive gate test, preserving existing strict target/status checks.
- Synthetic event without current operational pressure → scenario sensitivity or monitoring, not compulsory intervention. The present baseline-only behavior can be correct.
- Missing context → retain unknown and test labeled assumption ranges only if the twin supports them; never fabricate measured values.

Use a small candidate budget to preserve comparisons, but choose among supported alternatives by evidence/question coverage rather than insertion order alone. New arbitrary crane/berth/gate magnitudes should not be generated by an LLM. Parameter grids, if used, must be bounded and explicitly hypothetical because current data lacks staffing/cost/feasibility labels.

## 6.3 Ties, uncertainty and abstention are reasoning features

At the missing-weather timestamp below, baseline and shortest-first both score 0.0; existing stable sorting places baseline first. Strategy should say **“no difference established under these simulations”**, not imply a confidently best operational policy.

Use proposed qualifiers such as:

```json
{
  "decision_state": "no_distinguishable_benefit",
  "evidence_quality": {
    "history_complete": true,
    "weather_current": "unavailable",
    "cargo_origin": "synthetic",
    "prediction_mode": "retrospective_finalized_data"
  },
  "conditional_preference": null,
  "unresolved_questions": ["Effect of unavailable current weather"],
  "human_review_required": true
}
```

This is a proposed structure, not an implemented output. Paired run differences and parameter sensitivity would strengthen the qualifier. Their numerical summaries should stay deterministic. The earlier audit’s service-progress/conservation/ranking issues remain relevant when interpreting twin results: agent improvements cannot validate a defective simulator, and simulated uplift is not observed operational uplift. Fixes to those foundations are prerequisites for claiming better decision outcomes, even though this plan focuses on agents.

# 7. Ranked opportunity comparison

Source IDs resolve to the exact paths in Section 1.1. Sufficiency means sufficiency for the stated analytical task; synthetic/causal limitations remain. Ranking balances expected intelligence/decision value and effort, not model novelty.

| Rank / improvement | Agent | Current limitation | Existing data | Technical approach / proposed implementation | LLM required? | Expected value | Complexity | Data sufficiency | Evaluation | Priority |
|---|---|---|---|---|---|---|---|---|---|---|
| 1. Exact-time evidence contract and source-semantic checks | All | One-row facts; no diagnostic coverage/as-of envelope; P/H definitions differ. | P/C/W/M timestamps, shared fields; H waiting names and metadata. | Typed diagnostic facts; exact windows; prefix references; separate P/H namespaces. | No | Prevents misleading cross-agent deductions; enables every next tool. | Low–Medium | Sufficient for engineering; some definitions unverified | Future perturbation, gaps, duplicate fields, P/H mismatch and source/version tests. | Critical |
| 2. Cargo stock/flow/delay progression | Cargo | Elevated stock/status is not persistence or worsening. | C inventory/capacity, arrivals/departures, truck wait/flows, dwell, throughput. | `cargo_balance`, `cargo_delay_profile`, prior reference tools; keep canonical status. | No; optional explanation | Distinguishes loaded-and-draining from accumulating/worsening; stronger test rationales. | Low–Medium | Sufficient within synthetic state | Arithmetic, recovery/counterevidence and candidate-rationale cases. | High |
| 3. Conflict-triggered diagnostic routing | Strategy + domains | Fixed sequence and status list; no follow-up. | Existing outputs + P/C/W/M histories and X metadata. | Deterministic needs detector, tool catalog, two-round budget, hypothesis updates. | Optional; router baseline first | Observable evidence acquisition and useful collaboration. | Medium | Sufficient for bounded diagnostics | Relevant calls, resolution/abstention, supported candidate coverage versus current flow. | High |
| 4. Maritime pressure trajectory and scope-aware berth context | Maritime | One-hour ratio misses denominator effects/duration; berth data unused by wrapper. | P counts/ratio components, M cargo berth count, config capacity; H subtype context separately. | Exact deltas/slopes/status runs, count/ratio decomposition, geographic/type caveats. | No | More precise progression; avoids assuming all elevated Maritime states need capacity. | Low–Medium | Sufficient for aggregate diagnostics; insufficient vessel-level decisions | Rising/falling/denominator/gap fixtures, invariant canonical status. | High |
| 5. Existing model attribution plus target-aware disagreement explanation | Strategy/Maritime/Cargo | Risk label not explained; different target/horizon conflated. | X model/preprocessor/feature names; current exact model row; Y separately. | Native contributions; source/window grouping, target/training-cutoff metadata. | No calculations; optional interpretation | Explains high/low score despite current evidence and exposes calendar/synthetic reliance. | Low–Medium | Sufficient for model behavior; not causal validity | Additivity, feature contract, wrong-model/cause-claim rejection. | High |
| 6. Weather/event phase and sufficiency tools | Events & Weather | Level and completeness score hide freshness, schedule phase and missingness implications. | W wind/wave/index/time/event metadata; E schedule; H earlier weather references. | Freshness/history, scenario event phase, prefix-relative comparator, explicit unknown. | No; optional qualified synthesis | Prevents context overreach; gives Strategy an answer to a specific question. | Low–Medium | Sufficient for observed context/synthetic schedules; insufficient future weather | Missing hour, DST, event boundary, stale observation and numeric grounding cases. | High |
| 7. Hypothesis ledger and decision qualification | Strategy | No explicit support/counterevidence or tie/uncertainty assessment. | Existing domain facts, new diagnostics and twin results. | Structured hypothesis updates, admissibility checks, tie/no-benefit/insufficient states. | Optional wording only | Defensible, auditable decisions instead of forced agreement. | Medium | Sufficient for logical checks; operational confidence limited | Contradiction retention, no-benefit cases, missing-evidence abstention. | High |
| 8. Prior similar-state retrieval | Domains/Strategy | No historical analogues; only current facts. | P/C canonical histories, compact numeric features; H separately for Maritime/weather. | Prefix-fitted robust scaling, nearest neighbors, embargo, episode de-duplication, matured outcomes. | No; optional cited comparison | Concrete analogues and alternative patterns without training a new predictor. | Medium | Limited early in 2025 for Cargo; longer P-equivalent waiting history not established | Retrieval relevance, temporal eligibility, neighbor diversity, simpler-baseline comparison. | Medium |
| 9. Robust anomaly profile / forward change detection | Maritime/Cargo | Changes may remain inside coarse bands; no rarity reference. | P waiting/count deltas; C net flow, truck wait, dwell; H compatible raw weather. | Prior median/MAD/IQR; short/long contrast; optional CUSUM or IsolationForest after baseline. | No | Flags questions for investigation; not another authoritative congestion label. | Low→Medium | Sufficient descriptive history, weak anomaly ground truth | Alert burden, change sensitivity, proxy episode utility, no future fitting. | Medium |
| 10. Independent C1 sidecar context | Maritime/Strategy | Real saved Maritime model is not a current tool. | H exact 17 model features, Y model/imputer/threshold metadata. | Separate inference tool and target-aware comparison; never alias waiting fields. | No | Additional maritime forecast viewpoint if it changes useful investigations. | Medium | Sufficient artifact inputs; definitions/cutoff must be reconciled | Feature parity, as-of-valid period, contribution beyond P trend baseline. | Medium |
| 11. Lightweight future count/stock forecast | Maritime/Cargo | Existing forecast is binary cargo proxy, not quantitative progression. | P waiting, C inventory/truck wait, exact future labels and past features. | Persistence/linear extrapolation first; optional small chronological fitted model. | No | Quantitative planning context if materially better than simpler forecasts. | Medium | Limited independent operational validation; Cargo synthetic | Walk-forward MAE/bias and decision relevance, common times, embargo. | Low–Medium |
| 12. Descriptive state clustering | Domains | No state regimes; may help retrieval only. | Compact nonduplicate P/C features with past fit/scaling. | Small k-means/regime summaries only if quadrant baseline inadequate. | No | Potentially helps routing/retrieval; value unproven. | Medium | Limited to descriptive patterns | Cluster stability and downstream gain, not attractive labels. | Low |
| 13. Learned causes / best-action predictor | All | No independent cause/action-outcome labels. | No adequate existing target. | Do not implement. | No | Avoids false intelligence and circular proof. | High | Insufficient | Cannot validate claimed operational efficacy with current sources. | Low / reject |

# 8. Three development scopes

Effort is an indicative planning estimate for an engineer familiar with this repository, not a promised schedule. Parallel staffing, review and simulator corrections can change elapsed time.

## Option A — Minimal upgrade

**Features:** shared exact-time fact envelope; Cargo six-hour flow/stock/delay changes; Maritime count/ratio changes; context freshness; explicit current-versus-future disagreement and unknowns. Deterministic findings with supporting/counterevidence IDs; canonical statuses remain unchanged.

**Implementation:** extend existing source adapters with prefix-history access; implement reusable exact-window/delta functions; add optional diagnostic fields to result models; add a few deterministic routes for conflicting signals. No free-form agent loop and no new ML training.

**Complexity/effort:** Low–Medium, roughly 2–4 engineer-days including meaningful tests. **Dependencies:** current pandas/NumPy/Pydantic, frozen data and source identity decisions. **Risks:** misleading interval semantics, mixing P/H waiting, overinterpreting synthetic flow, treating history slope as prediction.

**Required tests:** exact arithmetic; gap and zero handling; future mutation invariance; canonical-status compatibility; known conflict/missing-weather cases. **Demonstrable value:** each agent explains whether pressure is rising, recovering or unknown instead of only reporting its band.

## Option B — Strong Stage 2 upgrade — recommended

**Features:** Option A plus bounded Strategy investigation, per-domain diagnostic catalogs, evidence-linked hypothesis updates, native XGBoost explanation, event phase/sufficiency, decision ties/abstention; optionally compact numeric case retrieval if core loop is finished.

**Implementation:** deterministic conflict router and validated tool requests; two follow-up rounds/six calls initially; shared prefix evidence contract; structured final diagnostic/decision record. Allow one constrained LLM planning choice or evidence synthesis only after a no-LLM baseline works. Current Gemini `generate_structured()` can return a proposed plan schema; application code validates and executes it. This is not provider-native autonomous tool execution today.

**Complexity/effort:** Medium, roughly 5–8 engineer-days for the focused core; retrieval adds work and can be deferred. **Dependencies:** Option A, X artifact/preprocessor parity, typed tool registry; known twin corrections before claiming outcome superiority. **Risks:** noisy unnecessary investigations, misleading model attributions, excessive candidate changes, flawed simulation feedback.

**Required tests:** Option A plus route relevance, budget/cycle limits, tool argument rejection, support/counterevidence consistency, no forced consensus, wrong-winner narrative rejection, model contribution checks, deterministic/LLM fallback equivalence for facts. Compare against current flow and Option A, not just against a blank baseline.

**Demonstrable value:** a real conflicting case triggers a specific history request; the agent revises a diagnosis with both confirming and contrary facts, then explains why an action is only a simulation test or why no distinct winner is established. This is a clear core-AI improvement achievable without a framework rewrite.

## Option C — Advanced agentic upgrade

**Features:** Option B plus hybrid LLM planning across multiple admissible questions; prior-case retrieval; selected anomaly/change signals; a stronger hypothesis ledger; paired simulation-sensitivity questions and richer abstention. Add a small quantitative forecast only after it beats persistence for a specific planning question.

**Implementation:** planner proposes a typed question/tool plan, deterministic validator limits scope, tools return facts, planner may select one further useful investigation. Track evidence dependencies and reject repeated questions that add no facts. Optionally rank proposed tools by unresolved question coverage, not opaque “agent confidence.” No additional agents required.

**Complexity/effort:** High relative to this project, roughly 9–15+ engineer-days with evaluation; exact scope must be cut to team capacity. **Dependencies:** robust Option B tools, retrieval eligibility/scaling, adequate early-reference periods, optional validated Gemini access, reliable simulation for sensitivity claims. **Risks:** harder attribution of gains, circular synthetic relationships, hallucinated causality, LLM variability, duplicated models and overfitting evaluation examples.

**Required tests:** all Option B checks plus adversarial tool plans, temporal retrieval leakage, planner-vs-router ablation, duplicate-case exclusion, missing-history behavior, paraphrase robustness and optional model walk-forward evaluation.

**Demonstrable value:** an evidence-grounded dynamic investigation whose trace shows useful tool choices and principled stopping. Keep it only if evaluation shows a gain over the simpler router. Option C is not automatically more intelligent than a well-designed Option B.

# 9. Real before/after examples from the 2025 repository data

## How to read these examples

“Current” values below are actual deterministic `pipeline.run(timestamp)` outputs obtained in this analysis. “Additional audited diagnostics” were calculated in read-only scripts from existing rows at or before the selected hour. **Proposed output structures and interpretations are hypothetical enhanced-agent behavior**, not an implemented improvement or measured operational gain. Every listed six-hour flow window had six available rows; t−6h also existed.

All X scores are uncalibrated. January/May/July examples precede X’s September final training cutoff, so their scores are retrospective replay signals. November occurs after that cutoff but still uses finalized synthetic/derived inputs with the limitations above. No future-row outcome was used in the proposed diagnostic facts.

## Case 1 — High pressure with mixed progression: 2025-07-10T19:00:00Z

**Current actual output excerpt:**

```json
{
  "maritime": {"status": "ELEVATED", "vessels": 57, "waiting_vessels": 19, "waiting_ratio": 0.333},
  "cargo": {"status": "ELEVATED", "yard_occupancy_percent": 94.81, "truck_waiting_time_minutes": 36.89, "cargo_flow_ratio": 0.91},
  "events_weather": {"pressure": "low", "weather": "low", "event_active": false, "evidence_completeness": 0.95},
  "ml": {"classification": "HIGH", "congestion_score": 0.9592401385, "alert_active": true},
  "strategy_candidates": ["baseline", "shortest_first", "gate_extension"],
  "highest_ranked_candidate": "gate_extension"
}
```

M also reports 16 approximate LA cargo berths occupied. Maritime throughput source value is 1, which is why the high ratio is ELEVATED rather than CONGESTED. Current agents do not explain the last six hours.

**Additional audited diagnostics:** waiting count **+4**; waiting-ratio change **+0.087432** using unrounded counts; yard occupancy **−0.28 percentage points**; inventory **−28.01 normalized units**; six-hour net flow **−28.02**; truck waiting **−3.50 minutes**. Minor stock/flow difference is compatible with rounding; exact underlying synthetic generation convention remains a qualification.

**Hypothetical enhanced output:**

```json
{
  "canonical_statuses": {"maritime": "ELEVATED", "cargo": "ELEVATED"},
  "diagnostics": {
    "waiting_delta_6h": 4,
    "yard_delta_percentage_points_6h": -0.28,
    "net_inflow_6h": -28.02,
    "truck_wait_delta_minutes_6h": -3.5,
    "flow_window_coverage": "6/6"
  },
  "hypotheses": [
    {"id": "regional_waiting_pressure_rising", "assessment": "supported"},
    {"id": "cargo_state_loaded_but_recently_draining", "assessment": "supported"},
    {"id": "new_gate_capacity_is_necessary", "assessment": "insufficient"}
  ],
  "next_investigation": "Explain the high cargo-proxy score and compare gate/queue tests against baseline",
  "limitations": ["Cargo indicators are synthetic", "Regional queue and LA cargo berth scope differ", "Forecast uses a later-trained retrospective artifact"]
}
```

**Strategy consequence:** keep current supported candidates, but explain mixed progression and request existing model attribution before attributing forecast risk to worsening yard accumulation. A gate action remains a simulation candidate, not a conclusion proven by occupancy alone. The existing simulation winner is not claimed to change.

## Case 2 — Current Cargo pressure with LOW future proxy: 2025-01-12T02:00:00Z

**Current actual output excerpt:**

```json
{
  "maritime": {"status": "NORMAL", "vessels": 47, "waiting_vessels": 8, "waiting_ratio": 0.17},
  "cargo": {"status": "ELEVATED", "yard_occupancy_percent": 79.26, "truck_waiting_time_minutes": 56.62, "gate_throughput_last_1h": 45, "cargo_flow_ratio": 1.27},
  "events_weather": {"pressure": "low", "event_active": false, "evidence_completeness": 0.95},
  "ml": {"classification": "LOW", "congestion_score": 0.0236212034, "alert_active": false},
  "strategy_candidates": ["baseline", "gate_extension"],
  "highest_ranked_candidate": "gate_extension"
}
```

**Additional audited diagnostics:** waiting count unchanged over six hours; yard occupancy **+0.73 pp**; inventory and cumulative net flow both **+72.23 units**; truck waiting **+34.57 minutes**. Current truck arrivals/departures are 53/44. Only 266 earlier Cargo rows lie in the proposed 28-day reference window—do not pretend January has a full month of history.

**Hypothetical enhanced output:**

```json
{
  "investigation_need": "Current Cargo pressure versus different-horizon future proxy",
  "requested_tools": ["cargo_balance_6h", "cargo_delay_profile_6h"],
  "new_evidence": {
    "net_inflow_6h": 72.23,
    "truck_wait_delta_minutes_6h": 34.57,
    "waiting_vessel_delta_6h": 0,
    "prior_reference_rows": 266
  },
  "hypothesis_update": {
    "cargo_accumulation_and_gate_delay_context": "supported",
    "new_maritime_queue_surge": "unsupported_by_this_window"
  },
  "forecast_interpretation": "LOW refers to a separate future operational proxy; it does not negate current synthetic Cargo evidence",
  "candidate_test": "gate_extension",
  "causal_conclusion": null
}
```

**Strategy consequence:** an evidence-backed reason for keeping gate versus baseline and no reason to add a Maritime queue intervention. Do not call the model “wrong” from a current status alone; the target/horizon differ and the January artifact is retrospective. Event absence is absence of a synthetic scheduled event, not proof of no actual disruption.

## Case 3 — Missing weather and a simulation tie: 2025-11-02T08:00:00Z

**Current actual output excerpt:**

```json
{
  "maritime": {"status": "ELEVATED", "vessels": 36, "waiting_vessels": 13, "waiting_ratio": 0.361},
  "cargo": {"status": "NORMAL", "yard_occupancy_percent": 71.95, "truck_waiting_time_minutes": 20.8},
  "events_weather": {"pressure": "unknown", "weather": "unknown", "event_active": false, "evidence_completeness": 0.45},
  "ml": {"classification": "LOW", "congestion_score": 0.0067418884, "alert_active": false},
  "ranking": [
    {"candidate_id": "baseline", "score": 0.0},
    {"candidate_id": "shortest_first", "score": 0.0}
  ],
  "highest_ranked_candidate": "baseline"
}
```

**Additional audited diagnostics:** waiting **+1** over six hours; occupancy **+0.52 pp**; six-hour net flow **+51.95 units**; truck waiting **+0.48 minutes**. Current wind/wave/index are missing; the last earlier complete weather observation is one hour old. The local time is `2025-11-02T01:00:00-07:00`, before the DST repeated hour. Source coverage must use UTC.

**Hypothetical enhanced output:**

```json
{
  "context_diagnostic": {
    "current_weather": "unavailable",
    "last_complete_weather_age_hours": 1,
    "current_weather_was_imputed": false,
    "event_record": "no_active_synthetic_event"
  },
  "hypothesis_update": {"weather_explains_current_waiting": "insufficient"},
  "decision_state": "no_distinguishable_benefit_in_current_simulation",
  "conditional_preference": null,
  "unresolved": ["Missing current weather", "No independent operational efficacy evidence"],
  "human_review_required": true
}
```

**Strategy consequence:** preserve Maritime pressure and unknown context; do not invent a weather explanation or gate intervention. Explain the tie rather than interpreting deterministic list order as confidence in baseline. The low forecast can remain available because its fitted preprocessor handles missing numeric values; that is not evidence weather was observed.

## Case 4 — Synthetic event without current severe pressure: 2025-01-13T18:00:00Z

**Current actual output excerpt:**

```json
{
  "maritime": {"status": "NORMAL", "vessels": 38, "waiting_vessels": 6, "waiting_ratio": 0.158},
  "cargo": {"status": "MODERATE", "yard_occupancy_percent": 79.27, "truck_waiting_time_minutes": 18.01},
  "events_weather": {"pressure": "moderate", "weather": "low", "event_type": "demand_surge", "event_severity": "moderate", "event_provenance": "SYNTHETIC"},
  "ml": {"classification": "MEDIUM", "congestion_score": 0.1541504115, "alert_active": false},
  "strategy_candidates": ["baseline"],
  "highest_ranked_candidate": "baseline"
}
```

**Additional audited diagnostics:** event `EVT_0025` starts exactly now, with 20 remaining hours in its **synthetic schedule**; waiting fell by one over six hours; yard occupancy **−0.75 pp**; inventory/net flow **−75 units**; truck waiting **−2.34 minutes**. Existing config maps moderate demand surge to arrival multiplier 1.25; that is an assumption, not observed extra arrivals.

**Hypothetical enhanced output:**

```json
{
  "event_phase": {"id": "EVT_0025", "phase": "start", "elapsed_hours": 0, "remaining_scenario_hours": 20},
  "observed_or_derived_progression": {"waiting_delta_6h": -1},
  "synthetic_cargo_progression": {"net_inflow_6h": -75, "truck_wait_delta_minutes_6h": -2.34},
  "hypothesis_update": {"current_pressure_is_explained_by_this_event": "not_established"},
  "next_step": "Retain baseline; optionally compare explicitly assumed arrival-surge sensitivity",
  "event_is_verified_historical_incident": false
}
```

**Strategy consequence:** avoid a forced intervention merely because the context agent has an event. A useful additional investigation can conclude baseline/monitoring is appropriate. A sensitivity scenario would be explicitly hypothetical and is proposed functionality, not a newly observed outcome.

## Case 5 — Elevated weather without elevated Cargo: 2025-05-14T06:00:00Z

This timestamp is described as “calm” in the README demo list, but actual replay returned Maritime **ELEVATED**, Cargo **NORMAL**, Events & Weather **elevated**, weather **elevated**, and LOW proxy score **0.0071369787**. Waiting=13/45; yard=58.85%; truck waiting=11.81 minutes; weather index=0.7142857, with raw wind value 4.1 and wave value 1.42 (units unverified). Candidates were baseline/shortest-first, tied at score 0.0, with baseline first.

Six-hour waiting changed **−1**, inventory **+92.82 units**, net flow **+92.83**, occupancy **+0.93 pp**, and truck waiting **−3.47 minutes**. A proposed context diagnostic would explain that the **relative weather index** is elevated and identify its wave component, without asserting official unsafe conditions or observed crane loss. Strategy would preserve the low-stock/recovering-delay counterevidence and avoid inferring a Cargo intervention from weather alone. This is a useful test of context discipline and a reminder to select demos by actual outputs, not labels in documentation.

# 10. Evaluation plan and direct answers

## 10.1 Evaluate evidence and decisions separately

Use existing timestamps and chronological evaluation blocks, not newly collected data. Keep development examples above separate from a broader later-period test. Any internally constructed adversarial unit fixture is a test of logic, not a new observed dataset.

1. **Freeze baselines.** B0=current agents/Strategy. B1=deterministic richer facts with current candidate rules. B2=deterministic bounded routing. B3=optional hybrid/LLM planning. This separates the value of new facts from the value of LLM planning.
2. **Define a case matrix before comparing variants.** Current-versus-future mismatch; Maritime denominator effect; high-but-draining Cargo; accumulating Cargo/delay; no pressure; active synthetic event; missing weather/history; simulation tie. Include all qualifying test-block hours or a declared deterministic sample, not handpicked winners.
3. **Arithmetic and temporal correctness:** exact window coverage, correct units/pp versus ratio distinction, prefix-only fit/scaling, matured retrieval outcomes, future-row permutation/mutation invariance, canonical-status invariance and source namespace checks. Record pass/fail counts.
4. **Grounding:** every numerical/status/action claim resolves to fact IDs; synthetic/derived labels retained; counterevidence preserved; wrong-winner/causal/unsupported absence claims rejected. Human reviewers evaluate facts and useful explanations blind to variant when feasible. Review labels are judgments about explanation quality, not new official port truth.
5. **Investigation quality:** proportion of calls addressing a declared unresolved question, repeated/unnecessary/invalid calls, useful evidence added, budget compliance, justified stopping and abstention. Report tool-call cost/latency only as a constraint on useful reasoning, not a production project.
6. **Decision support:** supported candidate coverage, unsupported candidate rate, conditional rather than unwarranted necessity claims, explicit ties/no-benefit and assumptions. More candidates or more calls is not the goal. Model/diagnostic explanations should help a reviewer understand why a specific simulation test was selected.
7. **Optional models:** walk-forward training/reference fit, common eligible timestamps, exact future labels, validation-only tuning before test, persistence/linear/percentile baselines and episode-aware uncertainty. Correct the custom tied-score AP issue identified in the earlier audit before reporting new classifier AP. Do not use the legacy NumPy logistic fit-without-validation path without fixing its known fitting defect; installed sklearn provides a simpler trustworthy baseline if needed.
8. **Outcome limits:** assess forecast/proxy outcomes only within their definitions. Simulation benefit is evaluated after simulator correctness, with paired uncertainty, not as proof of real-world uplift. Actual causal effectiveness and root-cause accuracy cannot be established from current source labels.

Success thresholds should be agreed before running the comparison, based on baseline counts and effort. No percentage gains are invented here. A credible result might be fewer unsupported claims and more correctly qualified investigations even if predictive AUC does not change.

## 10.2 Direct answers to the eight questions

**1. What are the five highest-impact improvements using existing data?**

- A shared prefix-only evidence contract with source-semantic checks.
- Cargo stock/flow/delay progression and explicit counterevidence.
- Conflict-triggered bounded diagnostic routing in Strategy.
- Maritime trajectory/count-ratio decomposition with scope-aware berth context.
- Existing-model attribution and target-aware disagreement explanations.

Weather freshness/event phase and a hypothesis ledger should be incorporated as small supporting tools, not new independent agents.

**2. Which agent first, and why?** Cargo first, after the shared time/evidence helper. It has rich consistent 2025 state, directly supports an existing common action, and shows immediate analytical value from stock/flow/delay progression. Improve Strategy next so it can request those tools; then Maritime progression and Weather/context sufficiency.

**3. Which improvements genuinely benefit from LLM reasoning?** Optional selection among multiple relevant follow-up questions, generation of qualified alternative hypotheses from cited facts, and concise cross-domain explanation. The gain must exceed deterministic routing. An LLM should neither invent action parameters nor own fact validation, scores or the winner.

**4. Which should remain deterministic?** Timestamp/window selection, gap checks, arithmetic, slopes/reference fits, anomaly statistics, retrieval distance/eligibility, model inference/attribution, provenance, event phase, canonical statuses, tool-argument/budget validation, candidate feasibility, twin execution, ranking/tie handling and numerical confidence summaries.

**5. Is there enough data for more ML?** Enough for limited exploratory anomaly/regime/short-horizon experiments and retrieval. Not enough independent labels for causal bottleneck diagnosis, verified event impacts or optimal-action learning. Start with analytical baselines and existing XGBoost explanations. Add a model only for a declared question it answers better.

**6. How to become more agentic without complexity?** Keep four roles. Let Strategy detect a knowledge gap, choose a bounded diagnostic tool, receive validated facts, update an evidence-backed hypothesis and stop or abstain. This is useful agency because tool selection changes available evidence; a longer fixed prompt is not.

**7. How should new agents be evaluated?** Chronological, prefix-safe ablations B0→B1→B2→B3; arithmetic/grounding first, relevant investigations and candidate justification next; optional predictive metrics on exact targets last. Keep synthetic-data/internal-consistency claims distinct from operational efficacy.

**8. What should be avoided?** More agents, unrestricted debate, vector databases/embeddings for numeric hourly case lookup, additional transformers without a distinct target, training on stored statuses and calling it root-cause understanding, fitting full-year anomaly references, treating engineered AIS→Cargo lag as causal discovery, interpreting C1 and Cargo models as independent votes, inventing weather forecasts, and learning “optimal actions” from an unvalidated simulator. Do not spend this scope on UI or infrastructure.

## 10.3 Verification record and unresolved evidence

The following checks were executed read-only against the unchanged source/artifacts:

- Enumerated P/C/M/W/E/H columns, row counts and ranges; counted H rows by year.
- Compared P/M and H waiting/total measures: 3,244 waiting mismatches, exact total-count match over shared 2025 rows; verified H ship-density redundancy and C gate/truck-arrival redundancy.
- Ran `marsa.pipeline.run()` without saving and without an LLM for all five timestamps in Section 9; recorded actual domain states, model score, candidates and ranking.
- Computed six-hour diagnostics using only exact rows at/before each selected timestamp. Event metadata from naive-UTC storage was explicitly localized as UTC for phase calculations; no source values changed.
- Loaded existing X booster/preprocessor and computed native local contributions, checking sum versus margin.
- Replaced strictly later rows’ arrivals/inventory/waiting/wind values with extreme numbers in an in-memory copy; `build_feature_row()` at the July query remained identical. This verifies that specific runtime feature builder’s future-row invariance, not upstream synthetic point-in-time validity or earlier artifact availability.
- Confirmed `git status` initially contained only the earlier audit report as untracked; no implementation edits were present. No extra model training or package installation was performed. The prior comprehensive audit’s 239-test result remains relevant because source is unchanged; no new agent functionality is claimed tested.

Still **Unverified**: exact origin of the P/H waiting-definition differences; some raw throughput/speed/weather semantics; whether event durations would be known by a real operator; real terminal causes/action outcomes; independent operational value of each proposed tool; official Stage 2 evaluation requirements; deadline/team capacity; live provider usefulness in the proposed plan loop. None prevents deterministic history tools, but each limits claims and should be an explicit question rather than an invented answer.
