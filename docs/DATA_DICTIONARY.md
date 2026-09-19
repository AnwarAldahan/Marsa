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
| hour, hour_sin, hour_cos, day_of_week, month, is_weekend, is_public_holiday | events/weather | DERIVED **on UTC** | ⛔ replaced by `local_*` features (America/Los_Angeles) | day_of_week: Sunday=1 |
| weather_pressure_index | events/weather | DERIVED | ⛔ | thresholds fitted on 2023-25 incl. test period; agent-facing only |
| event_active, event_id, event_type, event_severity, event_start, event_end, event_duration_hours, event_represented_hours | events/weather | SYNTHETIC | ⛔ | independent of AIS by design; used by context agent and twin as scenarios; `event_end` is future information |
| **pola_berthed_cargo_vessels** | raw AIS → `scripts/berth_occupancy_from_ais.py` | DERIVED | ✅ | cargo vessels (type 70-79), sog<0.5 kn, inside LA area (lon<-118.245, lat>33.72), stays >7 days excluded. Mean 10.2, max 20 (23 container berths). Validated: LA+LB total 27.5 vs Marine Exchange 27-31/day. |
| **pola_berthed_tanker_vessels** | same | DERIVED | ✅ | tankers (type 80-89), same filters; not compared to container berths |

Target: `waiting_vessel_count` at t+h (wall-clock shift, rows without the future hour are dropped).
