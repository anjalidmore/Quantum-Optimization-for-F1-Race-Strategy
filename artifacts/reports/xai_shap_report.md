# Task 8 - SHAP Report

_Generated 2026-09-05 01:38 UTC._

SHAP distributes the gap between a prediction and the average prediction
among the input features, using Shapley values from cooperative game theory.
The attributions below are computed on the untouched chronological test set.

## Explainer choice

| Model | Explainer | Exact? | Why |
|---|---|---|---|
| Classical (random forest) | `TreeExplainer` | **Yes** | Walks the ensemble directly; exact Shapley values in polynomial time. |
| Deep network (Keras) | `KernelExplainer` | No - sampled | Model-agnostic, needs only a `predict` function. `DeepExplainer`'s Keras 3 support targets the TensorFlow backend; this project runs Keras on PyTorch, so `KernelExplainer` is the correct choice. |

> **Trade-off:** the network's values carry sampling noise the forest's do not.
> Compare them by **rank and sign**, not by magnitude.

## target_laptime

**Task:** regression  |  **Test rows explained:** 180  |  **Dataset:** `data/processed/fastf1_laps_clean.csv`

### Global ranking - deep network

_Approximate Shapley values sampled with nsamples=200 over a k-means background of 25 points. Ranks and signs are meaningful; magnitudes carry sampling noise and are not directly comparable to TreeExplainer's exact values._

| Rank | Feature | Mean \|SHAP\| |
|---:|---|---:|
| 1 | `tracktemp_dev_x_tyrelife` | 1.611257 |
| 2 | `tyre_life` | 1.021669 |
| 3 | `stint_number` | 0.605191 |
| 4 | `gap_expanding` | 0.295765 |
| 5 | `form_vs_baseline` | 0.243807 |
| 6 | `field_median_lag1` | 0.215048 |
| 7 | `gap_roll3_std` | 0.196183 |
| 8 | `tyrelife_x_soft` | 0.177677 |
| 9 | `gap_roll3_mean` | 0.098494 |
| 10 | `track_status` | 0.092138 |
| 11 | `wind_speed` | 0.088798 |
| 12 | `is_fresh_tyre` | 0.077573 |
| 13 | `team_red_bull_racing` | 0.057143 |
| 14 | `team_aston_martin` | 0.038370 |
| 15 | `driver_zho` | 0.028628 |
| 16 | `driver_alo` | 0.020355 |
| 17 | `team_mercedes` | 0.019932 |
| 18 | `driver_mag` | 0.019594 |
| 19 | `compound_soft` | 0.019262 |
| 20 | `driver_per` | 0.018212 |
| 21 | `driver_ver` | 0.015504 |
| 22 | `team_mclaren` | 0.015108 |
| 23 | `team_alphatauri` | 0.012302 |
| 24 | `field_pace_trend` | 0.011728 |
| 25 | `humidity` | 0.009209 |
| 26 | `driver_bot` | 0.008454 |
| 27 | `driver_gas` | 0.007889 |
| 28 | `team_alpine` | 0.007207 |
| 29 | `driver_nor` | 0.007093 |
| 30 | `driver_tsu` | 0.006072 |
| 31 | `compound_medium` | 0.005833 |
| 32 | `team_haas_f1_team` | 0.005276 |
| 33 | `driver_sai` | 0.004716 |
| 34 | `driver_hul` | 0.003521 |
| 35 | `tyrelife_x_medium` | 0.003455 |
| 36 | `driver_str` | 0.003028 |
| 37 | `driver_dev` | 0.002579 |
| 38 | `driver_sar` | 0.001726 |
| 39 | `driver_rus` | 0.000860 |
| 40 | `team_ferrari` | 0.000708 |
| 41 | `team_williams` | 0.000618 |
| 42 | `driver_oco` | 0.000000 |
| 43 | `driver_pia` | 0.000000 |
| 44 | `driver_lec` | 0.000000 |
| 45 | `driver_ham` | 0.000000 |

### Global ranking - decision_tree

_Exact Shapley values; TreeExplainer enumerates the ensemble rather than sampling._

| Rank | Feature | Mean \|SHAP\| |
|---:|---|---:|
| 1 | `tracktemp_dev_x_tyrelife` | 0.682427 |
| 2 | `stint_number` | 0.669129 |
| 3 | `gap_expanding` | 0.414635 |
| 4 | `tyre_life` | 0.222912 |
| 5 | `track_status` | 0.188641 |
| 6 | `gap_roll3_mean` | 0.127319 |
| 7 | `team_red_bull_racing` | 0.069264 |
| 8 | `is_fresh_tyre` | 0.053144 |
| 9 | `field_median_lag1` | 0.051855 |
| 10 | `driver_oco` | 0.033057 |
| 11 | `form_vs_baseline` | 0.025248 |
| 12 | `gap_roll3_std` | 0.010864 |
| 13 | `tyrelife_x_soft` | 0.004318 |
| 14 | `compound_soft` | 0.001351 |
| 15 | `field_pace_trend` | 0.001190 |
| 16 | `wind_speed` | 0.000000 |
| 17 | `humidity` | 0.000000 |
| 18 | `tyrelife_x_medium` | 0.000000 |
| 19 | `driver_dev` | 0.000000 |
| 20 | `team_aston_martin` | 0.000000 |
| 21 | `driver_per` | 0.000000 |
| 22 | `driver_rus` | 0.000000 |
| 23 | `team_mclaren` | 0.000000 |
| 24 | `driver_gas` | 0.000000 |
| 25 | `driver_zho` | 0.000000 |
| 26 | `driver_alo` | 0.000000 |
| 27 | `driver_bot` | 0.000000 |
| 28 | `driver_ver` | 0.000000 |
| 29 | `team_mercedes` | 0.000000 |
| 30 | `driver_str` | 0.000000 |
| 31 | `team_alpine` | 0.000000 |
| 32 | `team_williams` | 0.000000 |
| 33 | `driver_nor` | 0.000000 |
| 34 | `driver_pia` | 0.000000 |
| 35 | `team_alphatauri` | 0.000000 |
| 36 | `driver_sar` | 0.000000 |
| 37 | `driver_lec` | 0.000000 |
| 38 | `team_haas_f1_team` | 0.000000 |
| 39 | `driver_mag` | 0.000000 |
| 40 | `driver_ham` | 0.000000 |
| 41 | `driver_hul` | 0.000000 |
| 42 | `driver_sai` | 0.000000 |
| 43 | `driver_tsu` | 0.000000 |
| 44 | `team_ferrari` | 0.000000 |
| 45 | `compound_medium` | 0.000000 |

### Representative predictions explained

#### Fastest Predicted Lap (test row 89, lap 51)

Deep network prediction: **95.9445**  |  decision_tree: **97.1719**

| Feature | Value | SHAP | Effect |
|---|---:|---:|---|
| `stint_number` | 4.5 | -1.179995 | decreases the prediction |
| `gap_roll3_std` | 4.429 | -0.789670 | decreases the prediction |
| `form_vs_baseline` | -2.849 | -0.657067 | decreases the prediction |
| `tracktemp_dev_x_tyrelife` | -8.074 | -0.542360 | decreases the prediction |
| `gap_expanding` | 0.5021 | +0.328134 | increases the prediction |
| `tyre_life` | 8 | -0.324631 | decreases the prediction |

![waterfall](xai_target_laptime_fastest_predicted_lap_shap_waterfall.png)

#### Median Predicted Lap (test row 92, lap 54)

Deep network prediction: **97.7566**  |  decision_tree: **98.9938**

| Feature | Value | SHAP | Effect |
|---|---:|---:|---|
| `stint_number` | 4.5 | -0.840246 | decreases the prediction |
| `form_vs_baseline` | -1.665 | -0.839618 | decreases the prediction |
| `tracktemp_dev_x_tyrelife` | -12.2 | -0.456342 | decreases the prediction |
| `gap_expanding` | 0.3869 | +0.267688 | increases the prediction |
| `field_median_lag1` | 97.78 | -0.245548 | decreases the prediction |
| `is_fresh_tyre` | 0 | +0.240069 | increases the prediction |

![waterfall](xai_target_laptime_median_predicted_lap_shap_waterfall.png)

#### Slowest Predicted Lap (test row 178, lap 55)

Deep network prediction: **102.2913**  |  decision_tree: **97.1719**

| Feature | Value | SHAP | Effect |
|---|---:|---:|---|
| `form_vs_baseline` | 3.268 | +2.377829 | increases the prediction |
| `driver_zho` | 1 | +0.765858 | increases the prediction |
| `stint_number` | 4 | -0.668103 | decreases the prediction |
| `gap_roll3_std` | 1.851 | -0.607362 | decreases the prediction |
| `tyre_life` | 4 | +0.557475 | increases the prediction |
| `gap_expanding` | 0.919 | +0.430109 | increases the prediction |

![waterfall](xai_target_laptime_slowest_predicted_lap_shap_waterfall.png)

![summary](xai_target_laptime_shap_summary.png)

---

## target_pit_next_lap

**Task:** classification  |  **Test rows explained:** 180  |  **Dataset:** `data/processed/fastf1_laps_clean.csv`

### Global ranking - deep network

_Approximate Shapley values sampled with nsamples=200 over a k-means background of 25 points. Ranks and signs are meaningful; magnitudes carry sampling noise and are not directly comparable to TreeExplainer's exact values._

| Rank | Feature | Mean \|SHAP\| |
|---:|---|---:|
| 1 | `tracktemp_dev_x_tyrelife` | 0.123178 |
| 2 | `tyre_life` | 0.113410 |
| 3 | `tyrelife_x_soft` | 0.044555 |
| 4 | `compound_soft` | 0.022829 |
| 5 | `gap_roll3_mean` | 0.016200 |
| 6 | `field_median_lag1` | 0.014686 |
| 7 | `form_vs_baseline` | 0.006367 |
| 8 | `field_pace_trend` | 0.005147 |

### Global ranking - random_forest

_Exact Shapley values; TreeExplainer enumerates the ensemble rather than sampling._

| Rank | Feature | Mean \|SHAP\| |
|---:|---|---:|
| 1 | `tyre_life` | 0.113556 |
| 2 | `tracktemp_dev_x_tyrelife` | 0.096343 |
| 3 | `form_vs_baseline` | 0.075203 |
| 4 | `field_median_lag1` | 0.063239 |
| 5 | `tyrelife_x_soft` | 0.053029 |
| 6 | `gap_roll3_mean` | 0.048826 |
| 7 | `field_pace_trend` | 0.042995 |
| 8 | `compound_soft` | 0.004071 |

### Representative predictions explained

#### Lowest Pit Probability (test row 179, lap 56)

Deep network prediction: **0.0000**  |  random_forest: **0.1022**

| Feature | Value | SHAP | Effect |
|---|---:|---:|---|
| `form_vs_baseline` | 5.111 | -0.018549 | decreases the prediction |
| `tyre_life` | 5 | -0.017075 | decreases the prediction |
| `gap_roll3_mean` | 3.723 | -0.008885 | decreases the prediction |
| `compound_soft` | 1 | -0.007701 | decreases the prediction |
| `tyrelife_x_soft` | 5 | +0.005519 | increases the prediction |
| `tracktemp_dev_x_tyrelife` | -6.046 | -0.004183 | decreases the prediction |

![waterfall](xai_target_pit_next_lap_lowest_pit_probability_shap_waterfall.png)

#### Closest To Decision Boundary (test row 34, lap 48)

Deep network prediction: **0.2040**  |  random_forest: **0.3379**

| Feature | Value | SHAP | Effect |
|---|---:|---:|---|
| `tyre_life` | 21 | +0.236009 | increases the prediction |
| `tracktemp_dev_x_tyrelife` | -19.1 | -0.151469 | decreases the prediction |
| `gap_roll3_mean` | 1.411 | +0.057514 | increases the prediction |
| `tyrelife_x_soft` | 0 | -0.036969 | decreases the prediction |
| `compound_soft` | 0 | +0.030424 | increases the prediction |
| `field_pace_trend` | -0.157 | +0.016188 | increases the prediction |

![waterfall](xai_target_pit_next_lap_closest_to_decision_boundary_shap_waterfall.png)

![summary](xai_target_pit_next_lap_shap_summary.png)

---

