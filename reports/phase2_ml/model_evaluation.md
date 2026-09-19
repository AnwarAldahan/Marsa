# Phase 2 ML Model Evaluation

## Dataset Summary

- Dataset: `data/processed/ml_modeling_table_c1.parquet`
- Rows: `22534`
- Timestamp range: `2023-06-01 00:00:00` to `2025-12-31 23:00:00`
- Duplicate timestamps: `0`
- Missing values: `{"port_throughput_lag1": 26, "wind_speed_10m": 10, "wave_height": 10, "weather_code": 10}`
- Split counts: `{"test": 4349, "train": 13846, "validation": 4339}`
- Target prevalence: `{"test": 0.09680386295700161, "train": 0.19919110212335692, "validation": 0.19105784743028348}`

## Target Definition

Predict `c1_congestion_next_6h`: whether C1 congestion occurs in `(t, t + 6h]`.

```text
vessels_waiting >= 17
OR
(ship_density >= 54 AND avg_port_speed <= 0.6062)
```

## Feature List

vessels_in_area, vessels_waiting, waiting_ratio, cargo_vessels_in_area, tanker_vessels_in_area, avg_sog, median_sog, ship_density, avg_port_speed, port_throughput_lag1, wind_speed_10m, wave_height, weather_code, hour_sin, hour_cos, day_of_week, month

## Excluded Features

`visit_id`, `delay_minutes`, `time_in_zone_hours`, same-hour `port_throughput`, target columns, and all synthetic terminal/landside variables.

## Missing-Value Handling

`SimpleImputer(strategy="median")` fitted on training rows only.

## Temporal Split

Train: 2023-06 through 2024-12. Validation: 2025-01 through 2025-06. Test: 2025-07 through 2025-12.

## Baseline Performance

### Validation

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Balanced Accuracy | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| majority_train_prevalence | 0.5 | 0.1911 | 0.0 | 0.0 | 0.0 | 0.5 | 0.1546 |
| current_congestion_persistence | 0.7301 | 0.5192 | 0.8814 | 0.4753 | 0.6176 | 0.7301 | 0.1125 |
| simple_queue_risk | 0.8707 | 0.6664 | 0.2081 | 0.9976 | 0.3444 | 0.5505 | 0.333 |

### Test

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Balanced Accuracy | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| majority_train_prevalence | 0.5 | 0.0968 | 0.0 | 0.0 | 0.0 | 0.5 | 0.0979 |
| current_congestion_persistence | 0.7427 | 0.4879 | 0.8922 | 0.4917 | 0.634 | 0.7427 | 0.055 |
| simple_queue_risk | 0.9293 | 0.6402 | 0.1069 | 1.0 | 0.1931 | 0.5521 | 0.3839 |

## XGBoost Performance

Validation:

```json
{
  "positive_prevalence": 0.19105784743028348,
  "roc_auc": 0.9195309627155226,
  "pr_auc": 0.7764883122614976,
  "precision": 0.675990675990676,
  "recall": 0.6996381182147166,
  "f1": 0.6876111440426793,
  "balanced_accuracy": 0.8102179195062187,
  "brier_score": 0.10512934625148773,
  "confusion_matrix": [
    [
      3232,
      278
    ],
    [
      249,
      580
    ]
  ],
  "threshold": 0.6866225600242615
}
```

Test:

```json
{
  "positive_prevalence": 0.09680386295700161,
  "roc_auc": 0.9705621616653201,
  "pr_auc": 0.8306771247918209,
  "precision": 0.6967032967032967,
  "recall": 0.7529691211401425,
  "f1": 0.723744292237443,
  "balanced_accuracy": 0.8589183691240427,
  "brier_score": 0.048899389803409576,
  "confusion_matrix": [
    [
      3790,
      138
    ],
    [
      104,
      317
    ]
  ],
  "threshold": 0.6866225600242615
}
```

## Optional Comparison Model

Logistic regression validation:

```json
{
  "positive_prevalence": 0.19105784743028348,
  "roc_auc": 0.9222490282803913,
  "pr_auc": 0.780616540191001,
  "precision": 0.6220703125,
  "recall": 0.7683956574185766,
  "f1": 0.6875337290879655,
  "balanced_accuracy": 0.8290696235810832,
  "brier_score": 0.11116564865205125,
  "confusion_matrix": [
    [
      3123,
      387
    ],
    [
      192,
      637
    ]
  ],
  "threshold": 0.6157525879220486
}
```

Logistic regression test:

```json
{
  "positive_prevalence": 0.09680386295700161,
  "roc_auc": 0.9655013521293014,
  "pr_auc": 0.7874157243635915,
  "precision": 0.6146616541353384,
  "recall": 0.7767220902612827,
  "f1": 0.6862539349422875,
  "balanced_accuracy": 0.8622663404463236,
  "brier_score": 0.0609970871724332,
  "confusion_matrix": [
    [
      3723,
      205
    ],
    [
      94,
      327
    ]
  ],
  "threshold": 0.6157525879220486
}
```

## Validation Threshold Analysis

| Threshold | Precision | Recall | F1 |
| --- | --- | --- | --- |
| 0.6866 | 0.676 | 0.6996 | 0.6876 |
| 0.6849 | 0.6752 | 0.6996 | 0.6872 |
| 0.6829 | 0.674 | 0.7008 | 0.6872 |
| 0.6805 | 0.6728 | 0.7021 | 0.6871 |
| 0.6886 | 0.6756 | 0.6984 | 0.6868 |
| 0.6836 | 0.6744 | 0.6996 | 0.6868 |
| 0.7069 | 0.6872 | 0.6864 | 0.6868 |
| 0.6815 | 0.6732 | 0.7008 | 0.6868 |
| 0.6801 | 0.6721 | 0.7021 | 0.6867 |
| 0.6786 | 0.6709 | 0.7033 | 0.6867 |

Frozen threshold: `0.686623`

## Calibration Results

- Validation Brier: `0.105129`
- Test Brier: `0.048899`

Reliability rows:

```json
[
  {
    "mean_predicted_probability": 0.0008352535418515768,
    "fraction_positive": 0.0
  },
  {
    "mean_predicted_probability": 0.0016615931362170598,
    "fraction_positive": 0.0
  },
  {
    "mean_predicted_probability": 0.002924295443634706,
    "fraction_positive": 0.0
  },
  {
    "mean_predicted_probability": 0.005072676775784328,
    "fraction_positive": 0.0022988505747126436
  },
  {
    "mean_predicted_probability": 0.009103651852186383,
    "fraction_positive": 0.0022988505747126436
  },
  {
    "mean_predicted_probability": 0.016883285518752813,
    "fraction_positive": 0.0
  },
  {
    "mean_predicted_probability": 0.041578927619018775,
    "fraction_positive": 0.009195402298850575
  },
  {
    "mean_predicted_probability": 0.13102961881407377,
    "fraction_positive": 0.05057471264367816
  },
  {
    "mean_predicted_probability": 0.4477818372948416,
    "fraction_positive": 0.19080459770114944
  },
  {
    "mean_predicted_probability": 0.9005220506383085,
    "fraction_positive": 0.7126436781609196
  }
]
```

Plots:

- `reports/phase2_ml/calibration_validation.png`
- `reports/phase2_ml/calibration_test.png`

## Distribution-Shift Analysis

| Feature | Train Mean | Test Mean | Difference | Std Difference |
| --- | --- | --- | --- | --- |
| vessels_waiting | 12.4877 | 11.2014 | -1.2863 | -0.4007 |
| ship_density | 46.8545 | 44.3182 | -2.5363 | -0.4103 |
| avg_port_speed | 1.066 | 1.0695 | 0.0035 | 0.0055 |
| vessels_in_area | 46.8545 | 44.3182 | -2.5363 | -0.4103 |
| waiting_ratio | 0.2656 | 0.252 | -0.0136 | -0.2319 |

The lower test prevalence is preserved as real temporal distribution shift; the test set was not altered.

## Feature Ablation

| Variant | Validation PR-AUC | Validation F1 | Test PR-AUC | Test F1 |
| --- | --- | --- | --- | --- |
| all_approved_features | 0.7765 | 0.6876 | 0.8307 | 0.7237 |
| without_port_throughput_lag1 | 0.7748 | 0.6819 | 0.8274 | 0.7146 |
| core_maritime_features | 0.7762 | 0.6903 | 0.8231 | 0.7206 |

## SHAP Global Analysis

Top SHAP features by mean absolute SHAP:

| Feature | Mean Abs SHAP |
| --- | --- |
| vessels_waiting | 1.66931 |
| vessels_in_area | 1.094621 |
| tanker_vessels_in_area | 0.550528 |
| day_of_week | 0.226047 |
| hour_cos | 0.206962 |
| waiting_ratio | 0.171858 |
| avg_sog | 0.153899 |
| cargo_vessels_in_area | 0.128164 |
| ship_density | 0.126569 |
| wave_height | 0.126116 |
| avg_port_speed | 0.107113 |
| wind_speed_10m | 0.069165 |
| month | 0.04074 |
| hour_sin | 0.03386 |
| weather_code | 0.033223 |

Plots:

- `reports/phase2_ml/shap_summary_bar.png`
- `reports/phase2_ml/shap_summary_beeswarm.png`

## SHAP Local Examples

Stored at `artifacts/explainers/xgboost_c1_local_examples.json`.

SHAP explains features contributing to the model's predicted risk. It does not prove causal reasons for congestion.

## Limitations

- Test prevalence is much lower than train/validation prevalence.
- C1 is a configurable MVP target candidate.
- `port_throughput_lag1` is included, while same-hour throughput remains excluded.
- The model is trained only on observed/derived features, not synthetic operational variables.

## Recommended Next Engineering Step

Integrate the saved model artifact into `PredictionService`, expose probability/thresholded risk via FastAPI, and keep the synthetic operational layer out of primary ML training.
