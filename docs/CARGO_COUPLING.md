# Cargo unloading in Main's existing Twin

The existing `simulate_candidate`, Strategy evaluation, waiting-based ranking and APIs are retained. No standalone Coupled/Hybrid engine or unified API is required. `twin.cargo_coupling: true` enables connected progression; false preserves exact Main historical replay, including its original response schema and snapshot tests.

## Explicit normalized assumptions

This is a retrospective counterfactual, not measured vessel cargo or TEU. One nominal berth-service hour is one normalized workload unit. Each occupied abstract berth supplies one unit/hour before yard blocking. An added berth therefore supplies another nominal service channel; no physical crane productivity is inferred. Existing AIS service duration supplies reference effort; initially berthed calls use the remaining recorded duration as their reference remaining effort. Real future departure times never force release when the yard blocks progress. Added vessels retain Main's earlier-history stay estimate or explicit user service input.

`cargo_buffer_hours: 24` is an explicit synthetic one-day buffer assumption. Yard capacity is original configured berths x buffer hours (552 units with 23 slots), fixed across candidate comparisons. It does not increase with an extra-berth intervention or a different forecast horizon. Neither 23 slots nor this capacity is a verified available terminal resource.

Opening inventory = normalized capacity x synthetic Cargo occupancy fraction at the selected hour. Base gate rate = normalized capacity x synthetic Cargo departures/capacity ratio at that hour. This deliberately assumes those dimensionless ratios proxy the abstract AIS cohort's yard state; the source populations are not verified identical. It is a modelling assumption, not a physical unit conversion. `gate_max` uses the highest historical clearance available through that hour divided by the selected hour's source capacity, preserving its action ID and API shape while explicitly labelling its normalized proxy meaning. Raw numeric gate actions are expressed on Main's source Cargo scale and mapped by the same declared ratio, not silently interpreted as normalized units. No future Cargo flows are used in connected mode.

Vessel unloading is the only simulated yard inflow. The Cargo arrivals column is not added again. Existing container-inflow stress scales reference vessel workloads, rather than double-counting background cargo. Gate closure stress stops clearance for its requested opening hours. These connected-mode meanings differ from historical CSV replay and are not observed operating conditions.

The event loop admits eligible waiting vessels, progresses work, integrates gate clearance and storage boundaries, and releases completed berths before subsequent admissions. At a full yard, available clearance limits unloading; no clearance blocks it. Clearance and unloading can occur continuously within an interval, with actual fractional completion/readiness events. Cargo is immediately gate-eligible in this minimal model: there is no fitted dwell-time mechanism. Hourly inventory, unloading, clearance, blocking, completion and waiting ledgers come from that same loop. Conservation checks reject inconsistent stock/work balances. Unfinished completion times remain null. Arrivals exactly at the horizon receive no service.

## Workflow and website

`build_scenario` runs once per analysis. Each candidate uses the same vessel objects as read-only inputs, the same workload references, opening stock and capacity; `apply_actions` copies the scenario. The engine returns fresh state and does not mutate its input. Ranking remains total vessel-hours waiting during the horizon, stable baseline-first ties; completion and cargo progress do not silently change the objective. Main's historical readiness floor is retained, so a faster gate need not improve waiting if recorded readiness rather than berth availability is the limiting condition.

The root Main page retains its CSS, layout and SVG artwork. Its existing API adapter now reads scalar/zero KPIs correctly, renders the API candidate results and plays the actual connected ledger. Existing current-port manual choices call the same analysis endpoint and verify identical scenario metadata before display; the hand-entered example remains separate. Unsupported illustrative crane/custom ordering is not presented as an implemented live intervention. The reference dashboard uses actual completion times instead of assuming recorded service duration still forces departure.

Start from this clean worktree:

```powershell
python -m uvicorn marsa.api.main:app --app-dir src --host 127.0.0.1 --port 8010
```

Open `http://127.0.0.1:8010/`. The root page now uses its server's origin for API calls; no directory browsing is exposed. Existing separate static hosting can set `window.MARSA_API_BASE` explicitly. APIs, agents, XGBoost, investigation and adaptive selection remain the Main workflow. The old Agent_Ft server on another port is separate and is not this clean worktree.

## Validation

A two-vessel controlled fixture with one slot, full one-unit yard and zero clearance blocks all unloading/completion and produces four vessel-hours waiting. Increasing only clearance to one unit/hour completes both vessels, releases the first berth at hour 1, admits the second vessel then, and reduces waiting to one vessel-hour. Partial clearance also produces partial-speed unloading. Tests check initial berthed blocking, conservation, no input mutation, deterministic comparison, horizon censoring and independence from later Cargo CSV flows.

July 10, 2025 at 19:00 UTC: baseline unloads approximately 322.94 units, completes six vessels, reaches 100% yard occupancy and loses 40.71 effort units to blocking. `gate_max` unloads 363 units with no yard blocking, also completes six, and ends at 98.14% yard occupancy. Both accumulate 24 vessel-hours waiting; baseline correctly remains the waiting-ranked winner. These are assumption-dependent simulated outcomes, not demonstrated operational effectiveness. March/November also run successfully; no benefit is manufactured when the yard is not binding.

Recorded service duration may already contain historical congestion; using it as nominal effort can double-count historical delays. This is an explicit reference-effort counterfactual, not a calibrated estimate of congestion-free service. No physical throughput accuracy, container-terminal membership, cargo quantities or causal predictive validation is claimed.

## Final website verification

Only the existing root HTML bindings, frontend regression tests and this documentation were changed during final verification; engine physics and backend behavior were not altered.

- Removed invented +3h interpolation and arbitrary current-risk percentages. The existing forecast chart now displays only the actual +6h proxy score, explicitly uncalibrated, with unavailable values left unavailable.
- Replaced embedded snapshot values for Cargo departures/ship-departure counts with actual API evidence. Historical LA/LB/ Cargo context is labelled separately from normalized simulation outcomes. Ship counts are not presented as verified available operational berths.
- Restored factor expansion, corrected notifications to current agent/model statuses, and preserved opening Twin controls on low-risk hours and after adopting a prior recommendation.
- Existing live candidate selection and replay consume only the analysis response; current-port manual calculations reuse scenario metadata and authoritative comparison units. Old/stale requests cannot overwrite newer analyses. Hand-entered educational examples remain separate and labelled.
- Feedback remains browser-local because Main has no feedback endpoint. Native feedback records use the same vessel-hour objective as Strategy, and actual manual results replace any earlier illustrative metadata.
- Corrected the precision claim to 53.4% (63 true-positive hourly alerts out of 118); the five-hour warning median is qualified to early-detected episodes. The existing 14-week chart was checked against saved +6h target timestamps and labels and is unchanged numerically. Its labels now identify a historical synthetic Cargo proxy, not independently observed real congestion or intervention effectiveness.

Focused checks cover JavaScript syntax, scalar/zero KPIs, actual completion-based occupancy, unchanged Main CSS/static component IDs, forecast availability, saved-evaluation parity, stale response/error rejection, causal/conservation tests, Strategy/investigation/adaptive behavior and exact historical-replay compatibility. Desktop browser checks cover March/July/November, navigation, candidates, playback, adoption followed by a new analysis, notifications/factor expansion, manual source separation and a controlled missing-hour response. Live Gemini was not verified in this credential-free worktree; provider/fallback behavior is regression-tested. No separate mobile viewport audit or full-suite run is claimed.

Final focused run: **162 passed**. After the final text-only context-label correction, all ten frontend checks were rerun and passed. Ruff and `git diff --check` passed. Live `/analyze` and `/twin/simulate` returned identical candidate KPIs; replay compatibility was also verified through the API with coupling disabled. The browser showed July's gate scenario unloading 363 normalized units with six completions and no blocking, while the authoritative waiting-ranked winner remained baseline. No commit, push or merge was performed.
