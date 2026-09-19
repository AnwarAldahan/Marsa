# Known limitations (state these openly in the presentation)

1. **Stand-in port.** Real AIS from Los Angeles/Long Beach; the Eastern Province data was requested and not provided.
2. **Container chain only.** Yard/truck/gate data are containers; tankers appear on the sea side only.
3. **Synthetic landside.** Cargo/yard/gate values are generated (calibrated to official monthly TEU and PMSA dwell) — never presented as measurements.
4. **Port modelled as one terminal.** 23 berths and one yard; the 7 terminals/operators are not separated (a vessel cannot use another operator's berth in reality).
5. **LA/LB boundary** in AIS is an estimate from the stationary-vessel map (-118.245), not an official polygon.
6. **Laid-up rule**: stationary stays >7 days are excluded (effect ≈ 1 vessel on the mean).
7. **Twin is a simplified hour-stepped queue + stock-flow model**, calibrated to dataset scales (Little's law), not to a real terminal. Service-time distribution, gate capacity, crane speed and event impacts are assumptions in `config/port_config.yaml`.
8. **Score weights** (0.4 avg wait, 0.2 max wait, 0.25 delayed vessels, 0.15 yard peak) are a documented choice, not derived from cost data.
9. **Forecast gains are modest** (2–10% over persistence); the target is real, so the number is honest.
10. Synthetic events are independent of AIS by design (to avoid leakage) and therefore not learnable; they are used as scenarios only.
