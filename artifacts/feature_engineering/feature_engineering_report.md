# Task 5 — Feature Engineering Report

_Generated 2026-09-27 17:43 UTC._

**Input:** `data/processed/fastf1_laps_clean.csv` (Task 4's cleaned laps).  
**Output:** `data/processed/f1_features_selected.csv` (995 rows) and `data/processed/feature_metadata.json`.  
**Data source:** real_fastf1 — Bahrain 2023 (R).

## 1. Engineered features

50 candidate features were built in six blocks: tyre and stint dynamics, fuel and race progress, the driver's own recent pace in gap-space, field-level pace, environment, and reference-encoded categoricals. Two rules govern all of them: anything derived from history uses `shift(1)` so no feature can see the lap it predicts, and no feature is an exact linear combination of others.

The fuel-burn coefficient was estimated from the data at -0.0613 s per lap (negative means the car speeds up as fuel burns off), and used to build the fuel-corrected target.

## 2. Leakage exclusions

Excluded before any feature was built: `Sector1Time`, `Sector2Time`, `Sector3Time`, `SpeedFL`, `SpeedST`, `IsPersonalBest`.

Reason: Sector times sum exactly to LapTime; speed traps and IsPersonalBest are only knowable during or after the lap being predicted.

## 3. The selection funnel

| Stage | Removes | Dropped here |
|---|---|---|
| 1. Near-zero variance | constant or near-constant columns | 1: `is_rainfall` |
| 2. Correlation pruning (\|r\| > 0.95) | one of each redundant pair | 3: `tyre_life_sq`, `air_temp`, `track_air_delta` |
| 3. Variance inflation (VIF > 10) | multi-way collinearity | 1: `race_progress` |
| 4. Importance and stability | weak or unstable predictors | see the importance report |

Correlated pairs, and which of the two survived (mutual information decided):

| Kept | Dropped | \|r\| |
|---|---|---:|
| `tyre_life` | `tyre_life_sq` | 0.9561 |
| `race_progress` | `air_temp` | 0.9803 |
| `race_progress` | `track_air_delta` | 0.9718 |

Removed for multicollinearity, worst first:

| Feature | VIF at removal |
|---|---:|
| `race_progress` | 24.9 |

## 4. What was exported

- 45 features for `target_laptime`
- 8 features for `target_pit_next_lap`
- 45 columns exported (the union), plus 5 identifier columns and 3 targets
- 60 warm-up rows dropped; first usable lap is 4

## 5. The contract for later tasks

- **Scaling:** NOT applied here. Fit the scaler inside each CV fold to avoid leakage.
- **Validation:** Expanding-window lap-forward split; keep whole laps in one fold. Do not use random K-fold - this is a time-ordered panel.
- **Needs scaling:** 14 numeric features; 31 binary indicators do not.

## 6. Validation scores at selection time

- Regression CV MAE 0.9891 s, R² 0.4491
- Classification CV ROC-AUC 0.8969

> 48 pit event(s) observed across 21 distinct laps (the busiest lap accounts for 12% of all pits) - a reasonably realistic spread of strategic pit timing. This remains a single session, though, and results should not be generalised beyond it.

Full step-by-step walkthrough with intermediate tables: [`docs/notebooks/task5_feature_engineering.ipynb`](../../docs/notebooks/task5_feature_engineering.ipynb).

