# Phase 1 Real Data Audit + Target Design

## Scope

This report audits only the observed HarborMind AIS dataset and derived hourly state tables. No synthetic terminal, landside, traffic, or counterfactual data was generated.

## Raw Dataset

- Raw file: `C:\Users\fatom\OneDrive\Desktop\AgentX\archive (8)\ais_2023_2025_clean.parquet`

- File size: `548.45 MB`

- Parquet rows: `21678617`

- Parquet row groups: `177`

- Columns: `53`


### Schema

| column | type |
| --- | --- |
| hour_key | timestamp[us] |
| mmsi | int32 |
| visit_id | int64 |
| base_date_time | timestamp[us, tz=UTC] |
| longitude | double |
| latitude | double |
| sog | double |
| cog | double |
| heading | int32 |
| vessel_name | large_string |
| imo | large_string |
| call_sign | large_string |
| vessel_type | int32 |
| status | int32 |
| length | int32 |
| width | int32 |
| draft | double |
| transceiver | large_string |
| draft_imputed_flag | int32 |
| length_imputed_flag | int32 |
| width_imputed_flag | int32 |
| sog_imputed_flag | int32 |
| cog_imputed_flag | int32 |
| vessel_area | int32 |
| dimension_ratio | double |
| draft_to_length_ratio | double |
| hour | int32 |
| day_of_week | int32 |
| month | int32 |
| hour_sin | double |
| hour_cos | double |
| day_of_week_sin | double |
| day_of_week_cos | double |
| month_sin | double |
| month_cos | double |
| is_weekend | int32 |
| is_night_shift | int32 |
| is_gate_hours | int32 |
| distance_to_port | double |
| is_in_waiting_area | bool |
| heading_error | double |
| acceleration | double |
| time_in_zone_hours | double |
| ship_density | int64 |
| avg_port_speed | double |
| port_throughput | int64 |
| delay_minutes | double |
| wind_speed_10m | double |
| wind_gusts_10m | double |
| precipitation | double |
| weather_code | int64 |
| wave_height | double |
| swell_wave_height | double |

## Dataset-Level Audit

| row_count | min_base_time | max_base_time | min_hour_key | max_hour_key | unique_mmsi | unique_visit_id | duplicate_mmsi_time | duplicate_observation_subset | observed_hours |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 21678617 | 2023-06-01 03:00:01+03:00 | 2026-01-01 02:59:48+03:00 | 2023-06-01 00:00:00 | 2025-12-31 23:00:00 | 3376 | 50 | 0 | 0 | 22534 |

### Hour Continuity

| expected_hours | observed_hours | missing_hours |
| --- | --- | --- |
| 22680 | 22534 | 146 |

### Missing Values

| column_name | missing_pct |
| --- | --- |
| delay_minutes | 16.2249 |
| heading | 3.635 |
| status | 1.275 |
| is_in_waiting_area | 1.1457 |
| imo | 0.909 |
| acceleration | 0.4341 |
| precipitation | 0.0376 |
| swell_wave_height | 0.0376 |
| wave_height | 0.0376 |
| weather_code | 0.0376 |
| wind_gusts_10m | 0.0376 |
| wind_speed_10m | 0.0376 |
| call_sign | 0.0018 |
| sog | 0.0005 |
| avg_port_speed | 0.0 |
| base_date_time | 0.0 |
| cog | 0.0 |
| cog_imputed_flag | 0.0 |
| day_of_week | 0.0 |
| day_of_week_cos | 0.0 |
| day_of_week_sin | 0.0 |
| dimension_ratio | 0.0 |
| distance_to_port | 0.0 |
| draft | 0.0 |
| draft_imputed_flag | 0.0 |
| draft_to_length_ratio | 0.0 |
| heading_error | 0.0 |
| hour | 0.0 |
| hour_cos | 0.0 |
| hour_key | 0.0 |
| hour_sin | 0.0 |
| is_gate_hours | 0.0 |
| is_night_shift | 0.0 |
| is_weekend | 0.0 |
| latitude | 0.0 |
| length | 0.0 |
| length_imputed_flag | 0.0 |
| longitude | 0.0 |
| mmsi | 0.0 |
| month | 0.0 |
| month_cos | 0.0 |
| month_sin | 0.0 |
| port_throughput | 0.0 |
| ship_density | 0.0 |
| sog_imputed_flag | 0.0 |
| time_in_zone_hours | 0.0 |
| transceiver | 0.0 |
| vessel_area | 0.0 |
| vessel_name | 0.0 |
| vessel_type | 0.0 |
| visit_id | 0.0 |
| width | 0.0 |
| width_imputed_flag | 0.0 |

## Required Field Audits


### visit_id

| visit_ids | ids_with_multiple_mmsi | ids_with_multiple_delay_values | rows_per_visit_quantiles | delay_range_quantiles | max_rows_per_visit_id | max_delay_range |
| --- | --- | --- | --- | --- | --- | --- |
| 50 | 40 | 35 | [11656, 522723, 2435607, 11139798] | [1301.1333333333334, 29614.19956733187, 36511.38138367236, 41450.916666666664] | 11139798 | 41450.916666666664 |

Conclusion: `visit_id` is not safe to treat as a vessel port-call identifier. It has only 50 unique values, many contain multiple MMSIs, and several span extremely long periods. It should be excluded from primary model features and not synthetically repaired.


### delay_minutes

| row_count | null_count | zero_count | positive_count | min_delay | mean_delay | stddev_delay | quantiles | max_delay |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 21678617 | 3517325 | 1030697 | 17130595 | 0.0 | 2912.665646322845 | 6088.789992508953 | [0.0, 3.904058830678632, 149.66364433568452, 184.15651781498008, 231.52122542420307, 2323.74262217583, 9688.39386515995, 16713.390271141092, 30555.07028025879] | 41450.916666666664 |

Conclusion: `delay_minutes` is outcome-like and highly skewed. It must not be used as a predictor. It can support secondary target analysis only after documenting how it was produced.


### Sampling Frequency

| first_observations | intervals | interval_minutes_quantiles | avg_interval_minutes |
| --- | --- | --- | --- |
| 3376 | 21675241 | [1.0166666666666666, 1.1466389149919345, 1.1666664652198382, 2.928188515539336, 3.0, 3.0, 3.061912413961412, 5.657411432633269, 7.86107249839466] | 41.54773376234768 |

### Port-Call Reconstruction Feasibility

| distance_to_port_quantiles | min_distance_to_port | max_distance_to_port | status_codes | vessel_type_codes |
| --- | --- | --- | --- | --- |
| [0.7114499073633986, 1.417527204236957, 1.9346937126439137, 4.158823666472125, 5.780032016409853, 9.718854959704654, 14.01395282346798, 20.473775589694096, 26.344773593005876] | 0.008117657791665582 | 34.2939346351198 | 13 | 18 |
| gaps_ge_6h | gaps_ge_12h | gaps_ge_24h | gap_hour_quantiles | max_gap_hours |
| --- | --- | --- | --- | --- |
| 10440 | 9565 | 8501 | [0, 0, 1, 26] | 20008 |
| entries_to_waiting | exits_from_waiting |
| --- | --- |
| 9178 | 9136 |

Conclusion: reconstructing a `derived_port_call_id` is plausible but should not be done with arbitrary rules. A defensible reconstruction would need calibrated distance/geofence thresholds, waiting-area transitions, status behavior, and large time-gap session breaks. Until validated, keep `visit_id` excluded and use hourly port-state modeling.


### time_in_zone_hours

| min_value | mean_value | quantiles | max_value | decreases_within_6h_gap |
| --- | --- | --- | --- | --- |
| 0.0 | 294.2623932880353 | [1.1557667967451162, 38.86613978842953, 108.60887218128006, 243.86124731501687, 462.4093220513698, 714.6524084720078, 5556.582841121238] | 14464.881944405328 | 29 |

### Hourly System Fields

| hour_keys | hour_density_pairs | hour_throughput_pairs | hour_speed_pairs | hour_wind_pairs | hour_wave_pairs |
| --- | --- | --- | --- | --- | --- |
| 22534 | 22534 | 22534 | 22534 | 22534 | 22534 |

Conclusion: `ship_density`, `port_throughput`, `avg_port_speed`, wind, and wave fields appear to have one distinct value per observed hour. They are derived/system hourly fields in the raw file, not independent per-ping measurements.


### is_in_waiting_area

| row_count | null_count | true_count | false_count |
| --- | --- | --- | --- |
| 21678617 | 248370 | 5124563 | 16305684 |

### Missing Documentation Fields

| field | status |
| --- | --- |
| avg_waiting_time_last_24h | not present in actual Parquet |
| avg_queue_speed | not present in actual Parquet |
| vessel_size_category | not present in actual Parquet |

## Exploratory Hourly Port State

- Written to: `C:\Users\fatom\OneDrive\Desktop\AgentX\data\derived\hourly_port_state.parquet`

- Vessel-hour snapshot written to: `C:\Users\fatom\OneDrive\Desktop\AgentX\data\derived\vessel_hourly_state.parquet`

| hours | min_timestamp | max_timestamp | vessels_in_area_q | vessels_waiting_q | ship_density_q | throughput_q | avg_port_speed_q |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 22534 | 2023-06-01 00:00:00 | 2025-12-31 23:00:00 | [47, 51, 54, 56, 60] | [12, 14, 16, 17, 20] | [47, 51, 54, 56, 60] | [0, 1, 1, 2, 3] | [0.1227729450625132, 0.2917375217112536, 0.5965407691367169, 0.986095170177673, 1.4498512701313997] |

## Target Candidate Design

All thresholds below are computed from the training period only: `timestamp < 2025-01-01`.

| threshold | value |
| --- | --- |
| waiting_p90 | 17.0 |
| density_p90 | 54.0 |
| speed_p25 | 0.6062121429891382 |
| throughput_p25 | 0.0 |
| avg_delay_p75 | 3765.325949559405 |
| avg_delay_p90 | 4380.954738917115 |
| time_in_zone_p90 | 409.84467867433403 |

Candidate 1, recommended MVP target:

`c1_congestion_next_6h = 1` when a queue-pressure state occurs in `(t, t + 6h]`, where queue pressure is `vessels_waiting >= train P90 OR (ship_density >= train P90 AND avg_port_speed <= train P25)`.


Candidate 2, secondary label only:

`c2_high_delay_next_6h = 1` when observed future hourly average delay exceeds the train P90. This is useful for analysis but riskier because `delay_minutes` is outcome-like.


Candidate 3, broader pressure target:

`c3_composite_congestion_next_6h = 1` when at least two pressure indicators are active: high waiting vessels, high density, low speed, low throughput, or high time-in-zone.


### Candidate Positive Rates

| temporal_split | hours | c1_positive_rate | c2_positive_rate | c3_positive_rate |
| --- | --- | --- | --- | --- |
| test | 4349 | 0.0968 | 0.0333 | 0.648 |
| train | 13846 | 0.1992 | 0.1756 | 0.7292 |
| validation | 4339 | 0.1911 | 0.1897 | 0.9899 |

### Congestion Persistence

| episodes | duration_hours_quantiles | max_duration_hours |
| --- | --- | --- |
| 497 | [2, 4, 9, 17, 52] | 74 |

## Leakage and Feature Safety Table

| field | provenance | recommendation | reason |
| --- | --- | --- | --- |
| mmsi | OBSERVED | exclude from model | Identifier; can cause memorization. |
| visit_id | OBSERVED | exclude | Not reliable as port-call ID; only 50 IDs and spans multiple MMSIs. |
| base_date_time/hour_key | OBSERVED | derive time features only | Timestamp itself should not be raw model feature except ordered split/index. |
| latitude/longitude | OBSERVED | aggregate/optional | Use only via derived hourly vessel location summaries. |
| sog/cog/heading/status | OBSERVED | aggregate | Safe if calculated from current or past pings only. |
| vessel_type/length/width/draft | OBSERVED | aggregate | Safe as current vessel composition; keep imputation flags. |
| distance_to_port/is_in_waiting_area/heading_error | DERIVED | aggregate | Safe if derived from current AIS position only. |
| acceleration | DERIVED | audit/use cautiously | Likely uses prior/current speed; safe only if no future smoothing. |
| time_in_zone_hours | DERIVED | audit/use cautiously | May leak if computed from full future zone duration. |
| ship_density | DERIVED | use cautiously | Hourly system field; safe only if computed from current/past hour. |
| port_throughput | DERIVED | use cautiously | May be post-hour or completion based; audit before predictor use. |
| avg_port_speed | DERIVED | use | Hourly current-state summary; useful low-speed congestion proxy. |
| avg_waiting_time_last_24h | N/A | not available | Mentioned in docs but absent in actual Parquet. |
| delay_minutes | OBSERVED | label only | Outcome-like observed field; do not use as predictor. |
| weather fields | OBSERVED | use | Present in dataset; no synthetic weather needed. |
| synthetic terminal/landside fields | SYNTHETIC | not in Phase 1 model | Future digital twin only, joined by timestamp. |

## Updated Provenance Architecture

| category | definition |
| --- | --- |
| OBSERVED | Measured or directly recorded in the raw HarborMind AIS Parquet. |
| DERIVED | Calculated from observed AIS/weather fields at or before the timestamp. |
| PREDICTED | Produced by a trained ML model. |
| SYNTHETIC | Generated operational state, explicitly not observed. |
| SIMULATED | Counterfactual what-if state produced by the digital twin. |
| AGENT_GENERATED | LLM/agent interpretation or recommendation. |

Future synthetic tables should join to `data/derived/hourly_port_state.parquet` by `timestamp` and must preserve field-level provenance metadata. The primary ML model should use real/derived features only; synthetic-assisted models, if any, must be labeled experimental.


## Phase 1 Recommendation

Use Candidate 1 as the MVP target because it is based on current/future observable port-state pressure rather than directly on `delay_minutes`. Use Candidate 2 only as a secondary diagnostic. Candidate 3 is useful for sensitivity testing but less interpretable.


Before Phase 2, review whether `port_throughput` is acceptable as a predictor. If its construction uses completed future movements inside the hour, either lag it by one hour or exclude it from the primary feature set.
