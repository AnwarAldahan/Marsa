# Final Layer 1 Cargo-Congestion Proxy Report

The final cargo congestion label is an operational proxy constructed from cargo/yard/gate stress indicators because independently observed congestion ground-truth labels were unavailable.

These proxy thresholds are not official Port of Los Angeles or industry-standard congestion thresholds. They are training-only robust distribution thresholds for this project dataset. The previous Stage 2 `operations_status` benchmark remains preserved as historical evidence and is not treated as final ground truth.

## Proxy Definition
For each training fold, thresholds are computed from historical training data only:

- yard stress: `yard_occupancy_percent >= q80` OR `containers_in_yard >= q80`
- delay stress: `average_dwell_time_hours >= q80` OR `truck_waiting_time_minutes >= q80`
- flow stress: `cargo_flow_imbalance >= max(0, q75)` OR (`gate_throughput <= q25` AND (`container_arrivals >= q75` OR `cargo_flow_imbalance > 0`))
- raw composite stress: yard stress AND delay stress AND flow stress
- final proxy state: raw composite stress persists for 2 consecutive hourly observations ending at t

## Label Summary
| training_window_end | samples | positive_labels | negative_labels | positive_prevalence | congestion_episodes |
| --- | --- | --- | --- | --- | --- |
| 2025-09-30T23:59:59+00:00 | 8688 | 284 | 8404 | 0.03268876611418048 | 74 |

## Threshold Metadata
| training_window_end | yard_occupancy_q80 | containers_in_yard_q80 | average_dwell_time_q80 | truck_waiting_time_q80 | cargo_flow_imbalance_q75 | container_arrivals_q75 | gate_throughput_q25 | persistence_hours | fold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-05-31T23:59:59+00:00 | 77.79 | 7779.344 | 70.16 | 19.924000000000003 | 21.339999999999996 | 155.18 | 83.0 | 2 | wf_2025_06 |
| 2025-06-30T23:59:59+00:00 | 78.55 | 7854.74 | 70.5 | 20.69 | 21.854999999999997 | 157.68 | 84.0 | 2 | wf_2025_07 |
| 2025-07-31T23:59:59+00:00 | 81.03200000000001 | 8103.314 | 72.13 | 23.6 | 21.555 | 165.075 | 85.5 | 2 | wf_2025_08 |
| 2025-08-31T23:59:59+00:00 | 81.58 | 8157.82 | 72.06 | 24.12800000000001 | 21.38000000000001 | 167.08 | 85.0 | 2 | wf_2025_09 |

## Aggregate Walk-Forward Results
| Model | Horizon | PR_AUC | Precision | Recall | F1 | ROC_AUC | Brier_Score | Positive_Count | Negative_Count | Distinct_Positive_Episodes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CurrentStatePersistence | +1h | 0.6371729552898203 | 0.7601476014760148 | 0.762962962962963 | 0.7615526802218114 | 0.8687861689814815 | 0.04558303886925795 | 270 | 2560 | 65 |
| CurrentStatePersistence | +3h | 0.2791705142024471 | 0.42066420664206644 | 0.42696629213483145 | 0.4237918215613383 | 0.6828071007372086 | 0.10969568294409059 | 267 | 2559 | 64 |
| CurrentStatePersistence | +6h | 0.14682291823451374 | 0.17712177121771217 | 0.18045112781954886 | 0.17877094972067037 | 0.5465685552958355 | 0.15638297872340426 | 266 | 2554 | 63 |
| CurrentStatePersistence | +12h | 0.15206289185387908 | 0.18518518518518517 | 0.18796992481203006 | 0.1865671641791045 | 0.55071194903072 | 0.15527065527065528 | 266 | 2542 | 63 |
| GRU | +1h | 0.6439466057901195 | 0.6933333333333334 | 0.3837638376383764 | 0.49406175771971494 | 0.9302848143670815 | 0.07296326694412134 | 271 | 2561 | 65 |
| GRU | +3h | 0.5134201832583756 | 0.664179104477612 | 0.332089552238806 | 0.4427860696517413 | 0.8547355993470149 | 0.09326826884571503 | 268 | 2560 | 64 |
| GRU | +6h | 0.4242798828874501 | 0.6171875 | 0.29699248120300753 | 0.4010152284263959 | 0.8155188440585031 | 0.09622049631560167 | 266 | 2556 | 63 |
| GRU | +12h | 0.27734934735576317 | 0.3931034482758621 | 0.21428571428571427 | 0.2773722627737226 | 0.7631844942545042 | 0.0967739955787604 | 266 | 2544 | 63 |
| PatchTSMixer | +1h | 0.25489147866143474 | 0.2926829268292683 | 0.44280442804428044 | 0.3524229074889867 | 0.8233176904201686 | 0.15786933584357726 | 271 | 2561 | 65 |
| PatchTSMixer | +3h | 0.2517180405775892 | 0.2975206611570248 | 0.40298507462686567 | 0.3423137876386688 | 0.8070633162313433 | 0.16197032759631014 | 268 | 2560 | 64 |
| PatchTSMixer | +6h | 0.24195289469246695 | 0.26737967914438504 | 0.37593984962406013 | 0.3125 | 0.8123588019344135 | 0.15507172640302735 | 266 | 2556 | 63 |
| PatchTSMixer | +12h | 0.22394654707168643 | 0.2533039647577093 | 0.4323308270676692 | 0.3194444444444445 | 0.7896584030831797 | 0.18707762936668165 | 266 | 2544 | 63 |
| PatchTST | +1h | 0.2639979343429279 | 0.3104325699745547 | 0.45018450184501846 | 0.3674698795180723 | 0.8257080735586739 | 0.14633005414402864 | 271 | 2561 | 65 |
| PatchTST | +3h | 0.2295832691176857 | 0.28083989501312334 | 0.39925373134328357 | 0.32973805855161786 | 0.7797705806902985 | 0.14714151855012578 | 268 | 2560 | 64 |
| PatchTST | +6h | 0.1686772703863819 | 0.18565400843881857 | 0.16541353383458646 | 0.17495029821073557 | 0.740735347759069 | 0.1421796715034443 | 266 | 2556 | 63 |
| PatchTST | +12h | 0.2951790725777536 | 0.275 | 0.37218045112781956 | 0.31629392971246006 | 0.7796274294226131 | 0.1650219278662893 | 266 | 2544 | 63 |
| XGBoost | +1h | 0.8577750217242768 | 0.7956521739130434 | 0.6777777777777778 | 0.732 | 0.9865907118055556 | 0.036668091900324815 | 270 | 2560 | 65 |
| XGBoost | +3h | 0.6247533335539351 | 0.6890243902439024 | 0.4232209737827715 | 0.5243619489559165 | 0.9425900801020998 | 0.07728149766517005 | 267 | 2559 | 64 |
| XGBoost | +6h | 0.48343563918659804 | 0.5338983050847458 | 0.23684210526315788 | 0.328125 | 0.9104809792688455 | 0.08634253213115389 | 266 | 2554 | 63 |
| XGBoost | +12h | 0.46220100155867216 | 0.6608695652173913 | 0.2857142857142857 | 0.3989501312335958 | 0.8943715504339133 | 0.09688752838912287 | 266 | 2542 | 63 |

## Onset-Only Results
| Model | Horizon | PR_AUC | Precision | Recall | F1 | Positive_Count | Negative_Count | Distinct_Positive_Episodes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CurrentStatePersistence | +1h | 0.029882242879780908 | 0.0 | 0.0 | 0.0 | 64 | 2495 | 64 |
| CurrentStatePersistence | +3h | 0.07224511953520073 | 0.0 | 0.0 | 0.0 | 153 | 2402 | 62 |
| CurrentStatePersistence | +6h | 0.10196495979968685 | 0.0 | 0.0 | 0.0 | 218 | 2331 | 60 |
| CurrentStatePersistence | +12h | 0.10095839435075182 | 0.0 | 0.0 | 0.0 | 216 | 2322 | 53 |
| GRU | +1h | 0.2376115056376208 | 0.3111111111111111 | 0.2153846153846154 | 0.2545454545454546 | 65 | 2496 | 65 |
| GRU | +3h | 0.4072405676236904 | 0.5806451612903226 | 0.23376623376623376 | 0.33333333333333337 | 154 | 2403 | 62 |
| GRU | +6h | 0.43640297206002443 | 0.625 | 0.2981651376146789 | 0.40372670807453415 | 218 | 2333 | 60 |
| GRU | +12h | 0.27069950512121743 | 0.3838383838383838 | 0.17592592592592593 | 0.2412698412698413 | 216 | 2324 | 53 |
| PatchTSMixer | +1h | 0.07190831347383939 | 0.08873720136518772 | 0.4 | 0.1452513966480447 | 65 | 2496 | 65 |
| PatchTSMixer | +3h | 0.17809131126856814 | 0.2230769230769231 | 0.37662337662337664 | 0.28019323671497587 | 154 | 2403 | 62 |
| PatchTSMixer | +6h | 0.2507260582756273 | 0.29152542372881357 | 0.3944954128440367 | 0.3352826510721248 | 218 | 2333 | 60 |
| PatchTSMixer | +12h | 0.24258812981639472 | 0.2658959537572254 | 0.42592592592592593 | 0.3274021352313167 | 216 | 2324 | 53 |
| PatchTST | +1h | 0.07226054137085246 | 0.08178438661710037 | 0.3384615384615385 | 0.1317365269461078 | 65 | 2496 | 65 |
| PatchTST | +3h | 0.1464860315635145 | 0.19148936170212766 | 0.35064935064935066 | 0.24770642201834867 | 154 | 2403 | 62 |
| PatchTST | +6h | 0.16623033327103484 | 0.18719211822660098 | 0.1743119266055046 | 0.18052256532066507 | 218 | 2333 | 60 |
| PatchTST | +12h | 0.26439787980634966 | 0.25949367088607594 | 0.37962962962962965 | 0.30827067669172936 | 216 | 2324 | 53 |
| XGBoost | +1h | 0.7134135849892486 | 0.6818181818181818 | 0.46875 | 0.5555555555555556 | 64 | 2495 | 64 |
| XGBoost | +3h | 0.5709364835898308 | 0.625 | 0.35947712418300654 | 0.45643153526970953 | 153 | 2402 | 62 |
| XGBoost | +6h | 0.47980657829059814 | 0.5242718446601942 | 0.24770642201834864 | 0.3364485981308411 | 218 | 2331 | 60 |
| XGBoost | +12h | 0.4691661713147504 | 0.6494845360824743 | 0.2916666666666667 | 0.40255591054313106 | 216 | 2322 | 53 |

## Episode Early-Warning Results
| Model | Horizon | Congestion_Episodes | Episodes_Detected_Before_Onset | Detection_Rate | Median_Warning_Lead_Time_Hours | Max_Warning_Lead_Time_Hours | False_Alerts | Onset_Negatives | False_Alert_Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CurrentStatePersistence | +12h | 53 | 0 | 0.0 | 0.0 | 0.0 | 0 | 2322 | 0.0 |
| CurrentStatePersistence | +1h | 64 | 0 | 0.0 | 0.0 | 0.0 | 0 | 2495 | 0.0 |
| CurrentStatePersistence | +3h | 62 | 0 | 0.0 | 0.0 | 0.0 | 0 | 2402 | 0.0 |
| CurrentStatePersistence | +6h | 60 | 0 | 0.0 | 0.0 | 0.0 | 0 | 2331 | 0.0 |
| GRU | +12h | 53 | 15 | 0.2830188679245283 | 12.0 | 12.0 | 61 | 2324 | 0.026247848537005163 |
| GRU | +1h | 65 | 14 | 0.2153846153846154 | 1.0 | 1.0 | 31 | 2496 | 0.012419871794871794 |
| GRU | +3h | 62 | 20 | 0.3225806451612903 | 2.0 | 3.0 | 26 | 2403 | 0.010819808572617561 |
| GRU | +6h | 60 | 22 | 0.36666666666666664 | 5.0 | 6.0 | 39 | 2333 | 0.016716673810544362 |
| PatchTSMixer | +12h | 53 | 23 | 0.4339622641509434 | 12.0 | 12.0 | 254 | 2324 | 0.10929432013769363 |
| PatchTSMixer | +1h | 65 | 26 | 0.4 | 1.0 | 1.0 | 267 | 2496 | 0.10697115384615384 |
| PatchTSMixer | +3h | 62 | 25 | 0.4032258064516129 | 3.0 | 3.0 | 202 | 2403 | 0.08406158967956721 |
| PatchTSMixer | +6h | 60 | 26 | 0.43333333333333335 | 6.0 | 6.0 | 209 | 2333 | 0.08958422631804544 |
| PatchTST | +12h | 53 | 20 | 0.37735849056603776 | 12.0 | 12.0 | 234 | 2324 | 0.10068846815834767 |
| PatchTST | +1h | 65 | 22 | 0.3384615384615385 | 1.0 | 1.0 | 247 | 2496 | 0.09895833333333333 |
| PatchTST | +3h | 62 | 26 | 0.41935483870967744 | 3.0 | 3.0 | 228 | 2403 | 0.09488139825218476 |
| PatchTST | +6h | 60 | 13 | 0.21666666666666667 | 6.0 | 6.0 | 165 | 2333 | 0.07072438919845692 |
| XGBoost | +12h | 53 | 22 | 0.41509433962264153 | 11.0 | 12.0 | 34 | 2322 | 0.014642549526270457 |
| XGBoost | +1h | 64 | 30 | 0.46875 | 1.0 | 1.0 | 14 | 2495 | 0.0056112224448897794 |
| XGBoost | +3h | 62 | 28 | 0.45161290322580644 | 3.0 | 3.0 | 33 | 2402 | 0.013738551207327226 |
| XGBoost | +6h | 60 | 22 | 0.36666666666666664 | 5.0 | 6.0 | 49 | 2331 | 0.021021021021021023 |

## Selected Learned Models
| Horizon | Selected_Model | Selection_Basis | Onset_PR_AUC | Onset_F1 | Detection_Rate | False_Alert_Rate | Aggregate_PR_AUC | Aggregate_F1 | Aggregate_Brier_Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| +1h | XGBoost | onset_PR_AUC_then_episode_detection_then_false_alert_rate | 0.7134135849892486 | 0.5555555555555556 | 0.46875 | 0.0056112224448897794 | 0.8577750217242768 | 0.732 | 0.036668091900324815 |
| +3h | XGBoost | onset_PR_AUC_then_episode_detection_then_false_alert_rate | 0.5709364835898308 | 0.45643153526970953 | 0.45161290322580644 | 0.013738551207327226 | 0.6247533335539351 | 0.5243619489559165 | 0.07728149766517005 |
| +6h | XGBoost | onset_PR_AUC_then_episode_detection_then_false_alert_rate | 0.47980657829059814 | 0.3364485981308411 | 0.36666666666666664 | 0.021021021021021023 | 0.48343563918659804 | 0.328125 | 0.08634253213115389 |
| +12h | XGBoost | onset_PR_AUC_then_episode_detection_then_false_alert_rate | 0.4691661713147504 | 0.40255591054313106 | 0.41509433962264153 | 0.014642549526270457 | 0.46220100155867216 | 0.3989501312335958 | 0.09688752838912287 |

## Calibration
Scores are not calibrated probabilities. `probability_is_calibrated=false` in the inference payload. Calibration should wait for independent, non-clustered positive episodes and target validation.

## Inference Example
```json
{
  "generated_at": "2026-09-19T11:12:50.099883+00:00",
  "prediction_timestamp": "2025-12-31T23:00:00+00:00",
  "model_version": "layer1-cargo-proxy-v1",
  "target_definition": "operational_cargo_congestion_proxy",
  "predictions": [
    {
      "horizon_hours": 1,
      "target_time": "2026-01-01T00:00:00+00:00",
      "congestion_score": 0.005269472952932119,
      "risk_level": "LOW",
      "probability_is_calibrated": false
    },
    {
      "horizon_hours": 3,
      "target_time": "2026-01-01T02:00:00+00:00",
      "congestion_score": 0.006097192410379648,
      "risk_level": "LOW",
      "probability_is_calibrated": false
    },
    {
      "horizon_hours": 6,
      "target_time": "2026-01-01T05:00:00+00:00",
      "congestion_score": 0.006618573796004057,
      "risk_level": "LOW",
      "probability_is_calibrated": false
    },
    {
      "horizon_hours": 12,
      "target_time": "2026-01-01T11:00:00+00:00",
      "congestion_score": 0.007775178644806147,
      "risk_level": "LOW",
      "probability_is_calibrated": false
    }
  ]
}
```

## Artifact Manifest
`prediction_model/final_proxy/artifacts/deployment_manifest.json`

