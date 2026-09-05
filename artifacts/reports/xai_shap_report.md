# Task 8 - SHAP Report

_Generated 2026-09-05 06:50 UTC._

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

### Global ranking - svr

_Approximate Shapley values sampled with nsamples=60 over a k-means background of 25 points. Ranks and signs are meaningful; magnitudes carry sampling noise and are not directly comparable to TreeExplainer's exact values._

| Rank | Feature | Mean \|SHAP\| |
|---:|---|---:|
| 1 | `stint_number` | 0.468331 |
| 2 | `tracktemp_dev_x_tyrelife` | 0.458424 |
| 3 | `tyre_life` | 0.347920 |
| 4 | `field_median_lag1` | 0.278125 |
| 5 | `form_vs_baseline` | 0.166605 |
| 6 | `gap_roll3_mean` | 0.098847 |
| 7 | `gap_roll3_std` | 0.073139 |
| 8 | `tyrelife_x_soft` | 0.058210 |
| 9 | `gap_expanding` | 0.052076 |
| 10 | `track_status` | 0.046786 |
| 11 | `team_red_bull_racing` | 0.046773 |
| 12 | `wind_speed` | 0.041363 |
| 13 | `is_fresh_tyre` | 0.040481 |
| 14 | `field_pace_trend` | 0.030945 |
| 15 | `team_alphatauri` | 0.023876 |
| 16 | `team_haas_f1_team` | 0.023702 |
| 17 | `compound_soft` | 0.022405 |
| 18 | `team_mclaren` | 0.022025 |
| 19 | `team_aston_martin` | 0.021817 |
| 20 | `driver_nor` | 0.021579 |
| 21 | `driver_ver` | 0.015798 |
| 22 | `driver_mag` | 0.014947 |
| 23 | `tyrelife_x_medium` | 0.013768 |
| 24 | `humidity` | 0.013494 |
| 25 | `team_mercedes` | 0.012128 |
| 26 | `team_williams` | 0.011916 |
| 27 | `compound_medium` | 0.010625 |
| 28 | `driver_sar` | 0.010444 |
| 29 | `driver_zho` | 0.008667 |
| 30 | `driver_alo` | 0.007601 |
| 31 | `driver_dev` | 0.007519 |
| 32 | `driver_bot` | 0.007208 |
| 33 | `driver_hul` | 0.003461 |
| 34 | `driver_per` | 0.002562 |
| 35 | `driver_rus` | 0.002181 |
| 36 | `driver_ham` | 0.002156 |
| 37 | `team_alpine` | 0.001917 |
| 38 | `team_ferrari` | 0.001530 |
| 39 | `driver_tsu` | 0.001294 |
| 40 | `driver_gas` | 0.001227 |
| 41 | `driver_sai` | 0.000483 |
| 42 | `driver_str` | 0.000180 |
| 43 | `driver_oco` | 0.000000 |
| 44 | `driver_pia` | 0.000000 |
| 45 | `driver_lec` | 0.000000 |

### Representative predictions explained

#### Fastest Predicted Lap (test row 89, lap 51)

Deep network prediction: **95.9445**  |  svr: **97.7846**

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

Deep network prediction: **97.7566**  |  svr: **98.0624**

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

Deep network prediction: **102.2913**  |  svr: **102.2228**

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
| 1 | `tyre_life` | 0.114313 |
| 2 | `tracktemp_dev_x_tyrelife` | 0.113231 |
| 3 | `tyrelife_x_soft` | 0.031832 |
| 4 | `compound_soft` | 0.020711 |
| 5 | `gap_roll3_mean` | 0.016185 |
| 6 | `field_median_lag1` | 0.006576 |
| 7 | `form_vs_baseline` | 0.004810 |
| 8 | `field_pace_trend` | 0.003543 |

### Global ranking - random_forest

_Exact Shapley values; TreeExplainer enumerates the ensemble rather than sampling._

| Rank | Feature | Mean \|SHAP\| |
|---:|---|---:|
| 1 | `tyre_life` | 0.113444 |
| 2 | `tracktemp_dev_x_tyrelife` | 0.111478 |
| 3 | `form_vs_baseline` | 0.078155 |
| 4 | `field_median_lag1` | 0.060573 |
| 5 | `tyrelife_x_soft` | 0.052117 |
| 6 | `gap_roll3_mean` | 0.045295 |
| 7 | `field_pace_trend` | 0.043422 |
| 8 | `compound_soft` | 0.003690 |

### Representative predictions explained

#### Lowest Pit Probability (test row 43, lap 47)

Deep network prediction: **0.0001**  |  random_forest: **0.0235**

| Feature | Value | SHAP | Effect |
|---|---:|---:|---|
| `tyre_life` | 7 | -0.021578 | decreases the prediction |
| `gap_roll3_mean` | -1.405 | -0.015844 | decreases the prediction |
| `compound_soft` | 1 | -0.011191 | decreases the prediction |
| `tyrelife_x_soft` | 7 | +0.009316 | increases the prediction |
| `form_vs_baseline` | -1.693 | -0.006696 | decreases the prediction |
| `tracktemp_dev_x_tyrelife` | -6.365 | -0.005286 | decreases the prediction |

![waterfall](xai_target_pit_next_lap_lowest_pit_probability_shap_waterfall.png)

#### Closest To Decision Boundary (test row 34, lap 48)

Deep network prediction: **0.2113**  |  random_forest: **0.3436**

| Feature | Value | SHAP | Effect |
|---|---:|---:|---|
| `tyre_life` | 21 | +0.229481 | increases the prediction |
| `tracktemp_dev_x_tyrelife` | -19.1 | -0.144088 | decreases the prediction |
| `gap_roll3_mean` | 1.411 | +0.042296 | increases the prediction |
| `tyrelife_x_soft` | 0 | -0.026945 | decreases the prediction |
| `compound_soft` | 0 | +0.024934 | increases the prediction |
| `field_median_lag1` | 97.43 | +0.015858 | increases the prediction |

![waterfall](xai_target_pit_next_lap_closest_to_decision_boundary_shap_waterfall.png)

![summary](xai_target_pit_next_lap_shap_summary.png)

---

