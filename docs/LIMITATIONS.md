# Known limitations (state these openly in the presentation)

1. **Stand-in port.** Real AIS from Los Angeles/Long Beach; the Eastern Province data was requested and not provided.
2. **Container chain only.** Yard/truck/gate data are containers; tankers appear on the sea side only.
3. **Synthetic landside.** Cargo/yard/gate values are generated (calibrated to official monthly TEU and PMSA dwell) — never presented as measurements.
4. **Port modelled as one berth pool.** 23 berths (an approximate public figure: the sum of the 7 container-terminal pages; the official Facts & Figures card gives no berth count); terminals/operators are not separated (a vessel cannot use another operator's berth in reality).
5. **LA/LB boundary** in AIS is an estimate from the stationary-vessel map (-118.245), not an official polygon.
6. **Laid-up rule** (berth-occupancy column in v2 only): stationary stays >7 days are excluded (effect ≈ 1 vessel on the mean). The twin's call table does not use this rule.
7. **Twin covers the sea side only.** Vessels, arrival/berth/service times come from AIS; yard and gates are not simulated because their data is synthetic. Without a stress scenario the twin replays AIS; in 2025 LA berths never filled, so berth-shortage results only appear under a user-chosen stress scenario. Observed waits (7.4 % of 2025 LA cargo calls) are kept as recorded; their cause (terminal windows) is not in the data.
8. **Future arrivals inside the 24 h horizon are the real ones** (a port knows them from arrival notices). Vessel destination (LA vs Long Beach) for waiting vessels is taken from where each vessel berthed next.
9. **Forecast gains are modest** (2–10% over persistence); the target is real, so the number is honest.
10. Synthetic events are independent of AIS by design (to avoid leakage) and therefore not learnable; they are used as scenarios only.
11. **Waiting area is shared by LA and Long Beach.** The maritime agent's counts cover both ports and tankers; the twin counts only cargo vessels that berthed in LA.
12. **Time features in the final ML model are on UTC** (`is_weekend`, `is_public_holiday`), not Los Angeles local time; the shift is 7–8 h at the weekend boundary.
