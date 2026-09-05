# Task 8 - Explainability Dashboard

_Generated 2026-09-05 01:38 UTC._

One page per target bringing together **global importance**, **per-prediction SHAP**,
the **trust score**, and the **plain-English recommendation** a race engineer would
actually read. This is the view that answers "why should I believe this?".

## target_laptime

**Task:** regression  |  **Deep network vs `decision_tree`**  |  **Dataset:** `data/processed/fastf1_laps_clean.csv`

### Global feature importance (permutation, model-agnostic)

| Feature | Deep network | Classical | Rank gap |
|---|---:|---:|---:|
| `tyre_life` | 0.883371 | 0.030828 | 2 |
| `tracktemp_dev_x_tyrelife` | 0.746878 | 0.000000 | 5 |
| `form_vs_baseline` | 0.196040 | 0.000000 | 5 |
| `gap_expanding` | 0.194300 | 0.054911 | 3 |
| `tyrelife_x_soft` | 0.077161 | 0.000000 | 7 |
| `stint_number` | 0.076144 | -0.020126 | 39 |
| `gap_roll3_mean` | 0.063166 | 0.000000 | 6 |
| `compound_soft` | 0.049385 | 0.000000 | 11 |
| `is_fresh_tyre` | 0.048781 | 0.018284 | 5 |
| `team_red_bull_racing` | 0.043837 | 0.036679 | 8 |

![importance](xai_target_laptime_importance_comparison.png)

### Per-prediction explanations

#### Fastest Predicted Lap - test row 89, lap 51

> **Expected lap time 95.944s - 3.360s faster than an average lap. Stint number (4.5) is a major factor making this lap faster; consistency of the recent pace gap (4.429) is a moderate factor making this lap faster; current form against this driver's baseline (-2.849) is a moderate factor making this lap faster. Trust in this prediction: LOW.**

| Trust | 0.430 (LOW) |
|---|---|
| Confidence | 0.400 |
| Model agreement | 0.400 |
| Explanation stability | 0.500 |

_Treat as a prompt to look at the evidence, not as a recommendation. The models disagree, or the explanations do._

**Top factors (SHAP):** `stint_number` (-1.1800), `gap_roll3_std` (-0.7897), `form_vs_baseline` (-0.6571)

**Counterfactual:** No value of tyre age between 1 and 22 flips this recommendation with the rest of the race state unchanged - the call is not sensitive to tyre age alone.

#### Median Predicted Lap - test row 92, lap 54

> **Expected lap time 97.757s - 1.548s faster than an average lap. Stint number (4.5) is a major factor making this lap faster; current form against this driver's baseline (-1.665) is a major factor making this lap faster; track temperature acting on tyre age (-12.2) is a moderate factor making this lap faster. Trust in this prediction: LOW.**

| Trust | 0.427 (LOW) |
|---|---|
| Confidence | 0.396 |
| Model agreement | 0.396 |
| Explanation stability | 0.500 |

_Treat as a prompt to look at the evidence, not as a recommendation. The models disagree, or the explanations do._

**Top factors (SHAP):** `stint_number` (-0.8402), `form_vs_baseline` (-0.8396), `tracktemp_dev_x_tyrelife` (-0.4563)

**Counterfactual:** The recommendation flips when tyre age reaches 19.4 (currently 11, a increase of 8.404), holding everything else fixed.

#### Slowest Predicted Lap - test row 178, lap 55

> **Expected lap time 102.291s - 2.987s slower than an average lap. Current form against this driver's baseline (3.268) is a major factor making this lap slower; driver identity (zho) (1) is a moderate factor making this lap slower; stint number (4) is a moderate factor making this lap faster. Trust in this prediction: DO NOT ACT.**

| Trust | 0.150 (DO NOT ACT) |
|---|---|
| Confidence | 0.000 |
| Model agreement | 0.000 |
| Explanation stability | 0.500 |

_The prediction sits on the decision boundary and/or the two model families contradict each other. This carries no more information than a coin flip._

**Top factors (SHAP):** `form_vs_baseline` (+2.3778), `driver_zho` (+0.7659), `stint_number` (-0.6681)

**Counterfactual:** No value of tyre age between 1 and 22 flips this recommendation with the rest of the race state unchanged - the call is not sensitive to tyre age alone.

---

## target_pit_next_lap

**Task:** classification  |  **Deep network vs `random_forest`**  |  **Dataset:** `data/processed/fastf1_laps_clean.csv`

### Global feature importance (permutation, model-agnostic)

| Feature | Deep network | Classical | Rank gap |
|---|---:|---:|---:|
| `tyre_life` | 0.363687 | -0.006704 | 7 |
| `tyrelife_x_soft` | 0.249162 | 0.049721 | 2 |
| `gap_roll3_mean` | 0.192179 | 0.162011 | 1 |
| `tracktemp_dev_x_tyrelife` | 0.145251 | 0.012849 | 1 |
| `form_vs_baseline` | 0.034078 | 0.108939 | 2 |
| `field_median_lag1` | -0.000000 | 0.226257 | 5 |
| `field_pace_trend` | -0.006704 | 0.001117 | 1 |
| `compound_soft` | -0.029050 | -0.005028 | 1 |

![importance](xai_target_pit_next_lap_importance_comparison.png)

### Per-prediction explanations

#### Lowest Pit Probability - test row 179, lap 56

> **Recommend STAYING OUT - model confidence <1% (0.00%). Current form against this driver's baseline (5.111) is a major factor pushing against stopping; tyre age (5) is a major factor pushing against stopping; recent pace gap to the field (3.723) is a moderate factor pushing against stopping. Trust in this recommendation: HIGH.**

| Trust | 0.819 (HIGH) |
|---|---|
| Confidence | 1.000 |
| Model agreement | 0.898 |
| Explanation stability | 0.500 |

_Both model families agree, the prediction is far from the boundary, and SHAP and LIME tell the same story. Safe to act on._

**Top factors (SHAP):** `form_vs_baseline` (-0.0185), `tyre_life` (-0.0171), `gap_roll3_mean` (-0.0089)

**Counterfactual:** No value of tyre age between 1 and 22 flips this recommendation with the rest of the race state unchanged - the call is not sensitive to tyre age alone.

#### Closest To Decision Boundary - test row 34, lap 48

> **Recommend STAYING OUT - model confidence 20%. Tyre age (21) is a major factor pushing towards a stop; track temperature acting on tyre age (-19.1) is a major factor pushing against stopping; recent pace gap to the field (1.411) is a minor factor pushing towards a stop. Trust in this recommendation: MODERATE.**

| Trust | 0.557 (MODERATE) |
|---|---|
| Confidence | 0.592 |
| Model agreement | 0.866 |
| Explanation stability | 0.200 |

_Usable as one input among several. Read the SHAP factors before acting; one of the three components is weak._

**Top factors (SHAP):** `tyre_life` (+0.2360), `tracktemp_dev_x_tyrelife` (-0.1515), `gap_roll3_mean` (+0.0575)

**Counterfactual:** No value of tyre age between 1 and 22 flips this recommendation with the rest of the race state unchanged - the call is not sensitive to tyre age alone.

---

