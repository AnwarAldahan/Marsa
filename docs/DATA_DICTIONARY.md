# Data dictionary — merged_port_dataset_2025_v2.csv (8,688 hourly rows, UTC, 42 columns)

72 calendar hours of 2025 are absent from the source AIS timeline and are **not** filled.
One hour (2025-11-02 08:00) has missing weather and is left null.

| Column | Source file | Provenance | Model use | Notes |
|---|---|---|---|---|
| hour_key | AIS | OBSERVED | key only | UTC, `+00:00` |
| vessel_count, waiting_vessel_count, approaching_vessel_count, departing_vessel_count, average_speed, port_throughput | AIS (agent 1) | DERIVED from AIS | ✅ | all vessel types, LA+LB region; `port_throughput` = Waiting→Moored transitions in last 4h |
| import_teu, export_teu | cargo (agent 2) | SYNTHETIC, calibrated to official POLA monthly TEU | ⛔ import_teu (duplicate scale of container_arrivals) | real-port scale |
| container_arrivals, container_departures, containers_in_yard, yard_capacity | cargo | SYNTHETIC | ✅ / ⛔ yard_capacity (constant) | normalised yard scale (capacity 10,000) |
| average_dwell_time_hours | cargo | SYNTHETIC, calibrated to PMSA | ✅ | |
| truck_arrivals, truck_departures, truck_waiting_time_minutes, gate_throughput | cargo | SYNTHETIC | ✅ | gate_throughput == truck_arrivals in 92% of hours |
| yard_occupancy_percent, cargo_flow_ratio | cargo | DERIVED (synthetic) | ⛔ | computed from other columns |
| operations_status | cargo | DERIVED rule label | ⛔ | same-hour label → leakage risk |
| wind_speed_10m, wave_height, weather_code | events/weather (agent 3) | OBSERVED (Open-Meteo, LA/LB) | ✅ (weather_code with caution) | wind unit unverified (assumed km/h) |
| hour, hour_sin, hour_cos, day_of_week, month, is_weekend, is_public_holiday | events/weather | DERIVED **on UTC** | ✅ used on UTC by the final XGBoost model (not converted to Los Angeles time) | day_of_week: Sunday=1 |
| weather_pressure_index | events/weather | DERIVED | ⛔ | thresholds fitted on 2023-25 incl. test period; agent-facing only |
| event_active, event_id, event_type, event_severity, event_start, event_end, event_duration_hours, event_represented_hours | events/weather | SYNTHETIC | ⛔ | independent of AIS by design; used by context agent and twin as scenarios; `event_end` is future information |
| **pola_berthed_cargo_vessels** | raw AIS → `scripts/berth_occupancy_from_ais.py` | DERIVED | ✅ | cargo vessels (type 70-79), sog<0.5 kn, inside LA area (lon<-118.245, lat>33.72), stays >7 days excluded. Mean 10.2, max 20 (23 container berths). Validated: LA+LB total 27.5 vs Marine Exchange 27-31/day. |
| **pola_berthed_tanker_vessels** | same | DERIVED | ✅ | tankers (type 80-89), same filters; not compared to container berths |

Target: `waiting_vessel_count` at t+h (wall-clock shift, rows without the future hour are dropped).

# Twin input — la_cargo_calls_2023_2025.csv (4,704 rows, one per stay at an LA berth)

Built by `scripts/build_la_cargo_calls.py` from the raw Kaggle AIS files (2023-H2 to 2025-H2). All values are from AIS pings.

| Column | Meaning |
|---|---|
| call_id | row id |
| vessel_id | IMO number when the vessel reports one, else "MMSI" + mmsi (links visits of a vessel that changed MMSI) |
| mmsi | vessel id (ITU MMSI) |
| imo | most reported IMO value of the vessel in the raw file (empty if never reported) |
| length_m | median of the vessel's reported lengths in the raw file (m) |
| arrival | first hour in the waiting area if the vessel waited, else first hour at the LA berth (UTC) |
| waited | True when the vessel was in the waiting area and reached an LA berth within 6 h after leaving it |
| wait_start, wait_end | waiting-area stay (UTC) |
| berth_start, berth_end | stay at an LA berth: cargo vessel (type 70-79), sog < 0.5 kn inside the LA box (UTC) |
| wait_hours | berth_start − arrival |
| berth_hours | berth_end − berth_start (real service time) |

Rules: a stay continues across gaps ≤ 6 h counted in hours present in the AIS file (data outages do not end stays: on 1 and 13 July 2025, whole days missing from the AIS file, 9 and 11 vessels are at an LA berth on both sides of the gap);
every stay is kept as recorded, whatever its length (no minimum duration).

`data/processed/similar_k_selection.json` (from `scripts/select_similar_k.py`): for k in 5, 10, 20, 50, 100,
200, the error of estimating each 2025 stay from the k vessels closest in length, using only earlier
stays and never the vessel's own. k = 5 had the lowest median absolute error (10.4 h) and is stored in
`config/port_config.yaml → twin.similar_vessels_k`.
2025: 1,772 LA cargo stays (468 of them 2 h or shorter), 100 came after a wait (median wait 26.5 h);
hourly occupancy matches `pola_berthed_cargo_vessels` (r = 0.96).

`data/processed/vessel_cargo_workloads.csv` (from `scripts/build_vessel_cargo_workloads.py`): one row per
`call_id`; `estimated_workload` = recorded berth hours × 1 normalized unit/hour (a labelled proxy for the cargo
coupling, not containers). Details and manifest: docs/CARGO_COUPLING.md.

`data/processed/twin_forecast_backtest.json` (from `scripts/backtest_twin_forecast.py`): forecast-mode errors
on sampled 2025 hours (vessels at berth and waiting at +6/12/24 h, total waiting) against the recorded AIS and a
no-change baseline.
