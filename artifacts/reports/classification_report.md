# Task 6 — Pit-Decision Classification Report

Generated: 2026-09-05T06:38:02.674166+00:00

## Objective

Predict `target_pit_next_lap` (should the driver pit at the end of this lap?)
from information available before the lap starts.

## Dataset: Real FastF1 session — Bahrain 2023 (R), 1055 laps, 20 drivers

48 pit event(s) observed across 21 distinct laps (the busiest lap accounts for 12% of all pits) - a reasonably realistic spread of strategic pit timing. This remains a single session, though, and results should not be generalised beyond it.

This is a single real Grand Prix session. Results reflect genuine race strategy variation but should not be generalised beyond this one race.

Pit events are rare in every session. As a direct consequence:

- Some CV folds can have zero pit events in their training split (a classifier
  cannot be fit on a single class there) or zero pit events in their
  validation split (ROC-AUC/PR-AUC are mathematically undefined there) — see
  the per-fold table below for this run.
- The final chronological holdout test set contains pit events, so test-set ROC-AUC/PR-AUC are defined for every trained model below.

## Features (Task 5 contract)

`tyre_life`, `tracktemp_dev_x_tyrelife`, `form_vs_baseline`, `field_median_lag1`, `field_pace_trend`, `tyrelife_x_soft`, `gap_roll3_mean`, `compound_soft`

## Validation strategy

Expanding-window lap-forward CV (4 folds).

- Development laps: 4–46 (815 rows)
- Test laps: 47–57 (180 rows)

| Fold | Train laps | Val laps | Train rows | Val rows |
|---|---|---|---:|---:|
| 1 | 4–12 | 13–21 | 180 | 172 |
| 2 | 4–21 | 22–30 | 352 | 171 |
| 3 | 4–30 | 31–38 | 523 | 152 |
| 4 | 4–38 | 39–46 | 675 | 140 |

## Model comparison

Primary ranking metrics: **CV ROC-AUC**, then **CV PR-AUC**, then **CV F1**.
Accuracy is not used to rank models — with 4.8% positives, "always predict
no-pit" already scores 95.2% accuracy while being useless.

| Model | CV ROC-AUC | CV PR-AUC | CV F1 | Test ROC-AUC | Test PR-AUC | Test F1 | Selected |
|---|---:|---:|---:|---:|---:|---:|:---:|
| random_forest | 0.8482 | 0.3863 | 0.1967 | 0.9832 | 0.2500 | 0.0833 | ✅ |
| svm | 0.7843 | 0.3321 | 0.1429 | 0.7989 | 0.0270 | 0.0000 |  |
| xgboost | 0.8125 | 0.3083 | 0.2307 | 0.9274 | 0.0714 | 0.0606 |  |
| logistic_regression | 0.8298 | 0.2964 | 0.2887 | 0.9832 | 0.2500 | 0.0213 |  |
| decision_tree | 0.6711 | 0.1797 | 0.1870 | 0.9693 | 0.0833 | 0.0278 |  |

*undefined = ROC-AUC/PR-AUC could not be computed because the split's ground
truth contained only one class.

## Best model

**random_forest**
