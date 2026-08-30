# Task 6 — Lap-Time Regression Report

Generated: 2026-08-30T16:15:40.144570+00:00

## Objective

Predict `target_laptime` (seconds) from information available before the lap starts.

## Features (Task 5 contract)

`field_median_lag1`, `stint_number`, `gap_expanding`, `track_status`, `tracktemp_dev_x_tyrelife`, `form_vs_baseline`, `field_pace_trend`, `gap_roll3_std`, `wind_speed`, `tyrelife_x_soft`, `tyre_life`, `gap_roll3_mean`, `team_red_bull_racing`, `driver_dev`, `driver_oco`, `team_aston_martin`, `driver_per`, `driver_rus`, `compound_soft`, `humidity`, `team_mclaren`, `driver_gas`, `driver_zho`, `driver_alo`, `driver_bot`, `driver_ver`, `team_mercedes`, `driver_str`, `team_alpine`, `is_fresh_tyre`, `team_williams`, `driver_nor`, `driver_pia`, `team_alphatauri`, `driver_sar`, `driver_lec`, `tyrelife_x_medium`, `team_haas_f1_team`, `driver_mag`, `driver_ham`, `driver_hul`, `driver_sai`, `driver_tsu`, `team_ferrari`, `compound_medium`

## Validation strategy

Expanding-window lap-forward CV (4 folds) over the development
set, with a chronologically later holdout test set never touched during
model selection.

- Development laps: 4–46 (815 rows)
- Test laps: 47–57 (180 rows)

| Fold | Train laps | Val laps | Train rows | Val rows |
|---|---|---|---:|---:|
| 1 | 4–12 | 13–21 | 180 | 172 |
| 2 | 4–21 | 22–30 | 352 | 171 |
| 3 | 4–30 | 31–38 | 523 | 152 |
| 4 | 4–38 | 39–46 | 675 | 140 |

## Model comparison

Primary ranking metric: **CV MAE** (lower is better).

| Model | CV MAE | CV RMSE | CV R² | Test MAE | Test RMSE | Test R² | Selected |
|---|---:|---:|---:|---:|---:|---:|:---:|
| decision_tree | 1.1898 | 1.6321 | 0.3381 | 0.8673 | 1.3308 | -0.1669 | ✅ |
| svr | 1.3813 | 1.6942 | 0.2553 | 0.7815 | 1.0290 | 0.3023 |  |
| random_forest | 1.3817 | 1.7624 | 0.2141 | 0.9253 | 1.2671 | -0.0579 |  |
| xgboost | 1.4384 | 1.8325 | 0.1351 | 1.0858 | 1.4412 | -0.3686 |  |
| linear_regression | 2.0181 | 2.7964 | -1.2807 | 0.7901 | 1.1414 | 0.1415 |  |

## Best model

**decision_tree**

## Limitations

48 pit event(s) observed across 21 distinct laps (the busiest lap accounts for 12% of all pits) - a reasonably realistic spread of strategic pit timing. This remains a single session, though, and results should not be generalised beyond it.
