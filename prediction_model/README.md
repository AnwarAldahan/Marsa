# Cargo Congestion Prediction Model

This module is the **Prediction Layer** of the Predictive Port Digital Twin.

Its purpose is to provide early warning of future **cargo-side operational congestion** so that Agent 1 (Master Decision Agent) can use the prediction together with information from the other agents to decide whether intervention is needed.

---

## 1. What Does the Model Predict?

The model forecasts cargo congestion at four future horizons:

- +1 hour
- +3 hours
- +6 hours
- +12 hours

The **+6 hour horizon is the primary horizon** used by the system because it provides useful advance warning while maintaining reasonable predictive performance.

For example, if the latest available port observation is at 10:00:

| Model | Prediction Target |
|---|---|
| +1h | Cargo congestion state at 11:00 |
| +3h | Cargo congestion state at 13:00 |
| +6h | Cargo congestion state at 16:00 |
| +12h | Cargo congestion state at 22:00 |

The prediction is therefore forward-looking. The model does not simply classify the current port state.

---

## 2. Dataset

The model was developed using the hourly 2025 merged port dataset:

`data/merged_port_dataset_2025.csv`

The dataset combines:

- AIS-derived vessel activity for the Los Angeles / Long Beach port region
- Synthetic cargo, yard, and gate operational variables
- Weather and event information
- Time-based features

The final dataset contains **8,688 hourly observations**.

---

## 3. Target Definition

The prediction target is an **operational cargo congestion proxy**.

For every timestamp `t`:

```text
cargo_congestion_proxy(t) = 0 or 1
```

where:

- `0` = no operational cargo congestion
- `1` = operational cargo congestion

The target is constructed directly from cargo-side operational indicators rather than using `operations_status` as the prediction target.

### Why?

`operations_status` is useful as an operational status variable in the synthetic dataset, but the prediction layer needs a target specifically representing **cargo congestion**.

Therefore, a reproducible cargo-specific proxy was defined using three groups of indicators.

### A. Yard Stress

Yard stress is present when either:

- `yard_occupancy_percent` is high, or
- `containers_in_yard` is high.

### B. Delay Stress

Delay stress is present when either:

- `average_dwell_time_hours` is high, or
- `truck_waiting_time_minutes` is high.

### C. Flow Stress

Cargo flow imbalance is calculated as:

```text
cargo_flow_imbalance =
container_arrivals - container_departures
```

Flow stress represents situations where incoming cargo pressure exceeds the system's ability to clear cargo efficiently, including gate throughput conditions.

### Thresholds

The operational thresholds are derived from the **training data only** using quantiles.

This is important because future/test data must not be used to define the target thresholds during model development.

The proxy primarily uses:

- 80th percentile thresholds for yard stress
- 80th percentile thresholds for delay stress
- flow/gate thresholds based on training-data quantiles

### Persistence Requirement

A single abnormal hour is not automatically considered congestion.

The composite congestion condition must persist for **2 consecutive hourly observations ending at time `t`** before:

```text
cargo_congestion_proxy = 1
```

This reduces the chance of treating a short operational spike as a congestion event.

The final proxy contains:

- **284 congestion observations**
- **8,404 non-congestion observations**
- approximately **3.27% congestion prevalence**
- **74 congestion episodes**

---

## 4. Forecasting Targets

After constructing the hourly congestion target, future labels are created for each prediction horizon.

Conceptually:

```text
X(t) → congestion(t + 1h)
X(t) → congestion(t + 3h)
X(t) → congestion(t + 6h)
X(t) → congestion(t + 12h)
```

This means the model uses information available at time `t` to estimate future congestion.

---

## 5. Models Evaluated

Four machine-learning / deep-learning approaches were evaluated:

1. XGBoost
2. GRU
3. PatchTSMixer
4. PatchTST

A **Persistence baseline** was also included.

Persistence assumes that the future congestion state will remain the same as the current state. It is useful as a baseline because congestion can persist across consecutive hours, but it does not represent a true learned early-warning model.

---

## 6. Evaluation Strategy

Because this is a time-series forecasting problem, the data was evaluated chronologically rather than using a random train/test split.

This prevents future observations from leaking into earlier training periods.

Evaluation considered:

- PR-AUC
- ROC-AUC
- Precision
- Recall
- F1-score
- Brier score
- onset-only prediction performance
- congestion episode detection
- early-warning lead time
- false-alert rate

PR-AUC is especially important because congestion events are relatively rare compared with normal observations.

---

## 7. Final Model Selection

**XGBoost was selected as the final model.**

Among the learned models, XGBoost achieved the strongest overall PR-AUC across all four forecasting horizons.

### Primary +6h Model

For the primary +6 hour horizon, XGBoost achieved:

| Metric | Result |
|---|---:|
| PR-AUC | 0.483 |
| ROC-AUC | 0.910 |
| Brier Score | 0.086 |
| Precision | 0.534 |
| Recall | 0.237 |
| F1 | 0.328 |

For congestion-onset evaluation:

- Onset-only PR-AUC: **0.480**
- Detected congestion episodes: **22 / 60**
- Median early-warning lead time: approximately **5 hours**
- False-alert rate: approximately **2.1%**

The model was selected based primarily on its ability to distinguish rare congestion events while maintaining a relatively low false-alert rate.

---

## 8. Model Output

The prediction layer generates a result for every forecast horizon.

Example:

```json
{
  "prediction_timestamp": "2025-12-31T23:00:00+00:00",
  "model_version": "layer1-cargo-proxy-v1",
  "target_definition": "operational_cargo_congestion_proxy",
  "predictions": [
    {
      "horizon_hours": 1,
      "target_time": "2026-01-01T00:00:00+00:00",
      "congestion_score": 0.0053,
      "risk_level": "LOW",
      "probability_is_calibrated": false
    },
    {
      "horizon_hours": 6,
      "target_time": "2026-01-01T05:00:00+00:00",
      "congestion_score": 0.0066,
      "risk_level": "LOW",
      "probability_is_calibrated": false
    }
  ]
}
```

### Important: Score vs. Probability

`congestion_score` is a model risk score between 0 and 1.

The current model has **not undergone a separate probability-calibration stage**. Therefore, a score such as:

```text
0.81
```

should currently be interpreted as a **high congestion score**, not as a formally calibrated **81% probability**.

For this reason:

```json
"probability_is_calibrated": false
```

is explicitly included in the output.

---

## 9. Integration with Agent 1

The Prediction Model answers:

> **What congestion risk is expected, and when?**

Agent 1 can then combine this forecast with contextual information from the other agents to determine:

> **Why is congestion expected, and what action should be taken?**

The intended flow is:

```text
Port Data
    ↓
Prediction Model
    ↓
Future Congestion Score
(+1h / +3h / +6h / +12h)
    ↓
Agent 1 — Master Decision Agent
    ↓
Other Agents / Operational Context
    ↓
Recommended Action
    ↓
Digital Twin
    ↓
Scenario Evaluation
```

The **+6h prediction is the primary signal** intended for the main decision workflow, while the other horizons provide short- and longer-term context.

---

## 10. Deployment Artifacts

Final model artifacts are located under:

```text
prediction_model/final_proxy/artifacts/
```

Separate XGBoost models are stored for each forecasting horizon:

```text
XGBoost_1h/
XGBoost_3h/
XGBoost_6h/
XGBoost_12h/
```

Each horizon includes the corresponding model and preprocessing information.

The deployment manifest is:

```text
prediction_model/final_proxy/artifacts/deployment_manifest.json
```

Example prediction output:

```text
prediction_model/final_proxy/results/latest_final_proxy_prediction.json
```

Detailed evaluation results are available in:

```text
prediction_model/final_proxy/results/aggregate_results.csv
prediction_model/final_proxy/results/onset_only_results.csv
prediction_model/final_proxy/results/episode_early_warning_results.csv
```

---

## 11. Current Limitations

The current system should be interpreted as a **prototype congestion forecasting layer**.

Important limitations:

- Cargo/yard/gate operational variables are synthetic.
- The congestion target is an operational proxy rather than an official Port of Los Angeles congestion label.
- Congestion events are relatively rare in the dataset.
- Model scores are not yet probability-calibrated.
- Additional real operational data would be required before production deployment.

These limitations do not prevent the model from being used in the project prototype, but they should be considered when interpreting predictions.

---

## Summary

The final Prediction Layer:

```text
Target:
Operational Cargo Congestion Proxy

Forecast Horizons:
+1h / +3h / +6h / +12h

Primary Horizon:
+6h

Final Model:
XGBoost

Output:
Congestion Score + Risk Level + Target Time

Consumer:
Agent 1 — Master Decision Agent
```

The prediction layer provides the early-warning signal, while the multi-agent layer interprets the operational context and the Digital Twin evaluates possible interventions.