# Task 5 — Feature Importance Report

_Generated 2026-09-27 17:43 UTC._

Three criteria are averaged by rank, so a feature only finishes high when more than one agrees: mutual information, random-forest importance, and the absolute value of an L1 (Lasso/logistic) coefficient. `stability` is the share of cross-validation folds in whose top 12 the feature appeared — a feature strong on the full data but unstable across folds is fitting one stretch of the race.

## Ranking for `target_laptime` (the selection target)

| Rank | Feature | Mutual info | Tree importance | L1 coef | Avg rank | Stability | Selected |
|---:|---|---:|---:|---:|---:|---:|:---:|
| 1 | `field_median_lag1` | 0.5212 | 0.1472 | 0.4833 | 2.33 | 1.00 | yes |
| 2 | `stint_number` | 0.3841 | 0.0858 | 0.5276 | 3.33 | 1.00 | yes |
| 3 | `gap_expanding` | 0.4674 | 0.0575 | 0.4440 | 5.00 | 1.00 | yes |
| 4 | `track_status` | 0.0626 | 0.1291 | 0.4886 | 6.33 | 0.50 | yes |
| 5 | `tracktemp_dev_x_tyrelife` | 0.5982 | 0.0863 | 0.0224 | 6.33 | 1.00 | yes |
| 6 | `form_vs_baseline` | 0.2307 | 0.0405 | 0.4443 | 6.33 | 1.00 | yes |
| 7 | `field_pace_trend` | 0.2382 | 0.0173 | 0.2100 | 7.00 | 1.00 | yes |
| 8 | `gap_roll3_std` | 0.1904 | 0.0312 | 0.1030 | 8.67 | 1.00 | yes |
| 9 | `wind_speed` | 0.1954 | 0.0030 | 0.1650 | 10.00 | 0.75 | yes |
| 10 | `tyrelife_x_soft` | 0.1109 | 0.0099 | 0.0824 | 10.67 | 1.00 | yes |
| 11 | `tyre_life` | 0.1685 | 0.2753 | 0.0000 | 14.00 | 1.00 | yes |
| 12 | `gap_roll3_mean` | 0.2314 | 0.0787 | 0.0000 | 14.33 | 1.00 | yes |
| 13 | `team_red_bull_racing` | 0.0375 | 0.0021 | 0.0716 | 14.67 | 0.75 | yes |
| 14 | `driver_dev` | 0.0287 | 0.0018 | 0.0184 | 18.00 | 0.00 | yes |
| 15 | `driver_oco` | 0.0073 | 0.0032 | 0.0528 | 18.33 | 0.00 | yes |
| 16 | `team_aston_martin` | 0.0072 | 0.0021 | 0.0366 | 20.33 | 0.00 | yes |
| 17 | `driver_per` | 0.0142 | 0.0035 | 0.0000 | 22.33 | 0.00 | yes |
| 18 | `driver_rus` | 0.0170 | 0.0021 | 0.0000 | 23.67 | 0.00 | yes |
| 19 | `compound_soft` | 0.0756 | 0.0009 | 0.0000 | 23.67 | 0.00 | yes |
| 20 | `humidity` | 0.0861 | 0.0008 | 0.0000 | 24.00 | 0.00 | yes |
| 21 | `team_mclaren` | 0.0000 | 0.0015 | 0.0885 | 24.67 | 0.00 | yes |
| 22 | `driver_gas` | 0.0212 | 0.0012 | 0.0000 | 25.00 | 0.00 | yes |
| 23 | `driver_zho` | 0.0141 | 0.0019 | 0.0000 | 25.33 | 0.00 | yes |
| 24 | `driver_alo` | 0.0123 | 0.0006 | 0.0134 | 25.33 | 0.00 | yes |
| 25 | `driver_bot` | 0.0378 | 0.0007 | 0.0000 | 25.67 | 0.00 | yes |
| 26 | `driver_ver` | 0.0337 | 0.0008 | 0.0000 | 26.00 | 0.00 | yes |
| 27 | `team_mercedes` | 0.0141 | 0.0012 | 0.0000 | 26.67 | 0.00 | yes |
| 28 | `driver_str` | 0.0017 | 0.0023 | 0.0000 | 27.67 | 0.00 | yes |
| 29 | `team_alpine` | 0.0098 | 0.0010 | 0.0000 | 28.33 | 0.00 | yes |
| 30 | `is_fresh_tyre` | 0.0000 | 0.0037 | 0.0000 | 28.33 | 0.00 | yes |
| 31 | `team_williams` | 0.0166 | 0.0004 | 0.0000 | 29.67 | 0.00 | yes |
| 32 | `driver_nor` | 0.0004 | 0.0016 | 0.0000 | 30.33 | 0.00 | yes |
| 33 | `driver_pia` | 0.0216 | 0.0001 | 0.0000 | 31.67 | 0.00 | yes |
| 34 | `team_alphatauri` | 0.0057 | 0.0007 | 0.0000 | 32.33 | 0.00 | yes |
| 35 | `driver_sar` | 0.0128 | 0.0003 | 0.0000 | 32.67 | 0.00 | yes |
| 36 | `driver_lec` | 0.0070 | 0.0004 | 0.0000 | 33.00 | 0.00 | yes |
| 37 | `tyrelife_x_medium` | 0.0000 | 0.0008 | 0.0000 | 33.67 | 0.00 | yes |
| 38 | `team_haas_f1_team` | 0.0024 | 0.0004 | 0.0000 | 33.67 | 0.00 | yes |
| 39 | `driver_mag` | 0.0068 | 0.0003 | 0.0000 | 34.00 | 0.00 | yes |
| 40 | `driver_ham` | 0.0068 | 0.0004 | 0.0000 | 34.00 | 0.00 | yes |
| 41 | `driver_hul` | 0.0000 | 0.0005 | 0.0000 | 35.67 | 0.00 | yes |
| 42 | `driver_sai` | 0.0016 | 0.0002 | 0.0000 | 37.00 | 0.00 | yes |
| 43 | `driver_tsu` | 0.0000 | 0.0003 | 0.0000 | 37.67 | 0.00 | yes |
| 44 | `team_ferrari` | 0.0000 | 0.0002 | 0.0000 | 38.33 | 0.00 | yes |
| 45 | `compound_medium` | 0.0000 | 0.0001 | 0.0000 | 39.00 | 0.00 | yes |

## How many features to keep

Cross-validated error for the top-K features, K chosen as the smallest set within 2% of the best MAE — the simplest model the data cannot distinguish from the best one.

| K | CV MAE (s) | CV R² |
|---:|---:|---:|
| 4 | 1.6486 | -0.4631 |
| 6 | 1.2579 | 0.2217 |
| 8 | 1.2189 | 0.2648 |
| 10 | 1.2323 | 0.2576 |
| 12 | 1.0457 | 0.4077 |
| 15 | 1.0338 | 0.4237 |
| 20 | 1.0288 | 0.4377 |
| 45 | 0.9891 | 0.4491 |

**Selected K* = 45.** Selected set: CV MAE 0.9891 s (R² 0.4491); all 45 post-funnel features: 1.0001 s (R² 0.4444).

## Pit-decision target

Ranked the same way against `target_pit_next_lap`, with K fixed at 8: at roughly 5% positives a sweep would be choosing between numbers that are mostly noise. CV ROC-AUC by K: K=4: 0.879, K=6: 0.868, K=8: 0.897, K=10: 0.902.

Selected (8): `tyre_life`, `tracktemp_dev_x_tyrelife`, `form_vs_baseline`, `field_median_lag1`, `field_pace_trend`, `tyrelife_x_soft`, `gap_roll3_mean`, `compound_soft`.

