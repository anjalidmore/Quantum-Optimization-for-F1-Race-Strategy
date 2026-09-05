# Task 8 - LIME Report

_Generated 2026-09-05 01:38 UTC._

## How this differs from the SHAP report

LIME does **not** compute Shapley values. It perturbs the row being explained,
asks the real model what it predicts for each perturbation, and fits a weighted
**linear surrogate** to those answers in the neighbourhood of that row. The
reported weights are that surrogate's coefficients.

| | SHAP | LIME |
|---|---|---|
| Question answered | How is credit for this prediction fairly divided? | What simple model behaves like the real one *around here*? |
| Basis | Shapley values (game theory) | Local weighted linear regression |
| Guarantee | Attributions sum to (prediction - base value) | None; quality is reported as the surrogate's R2 |
| Output units | Same units as the model output | Surrogate coefficients on discretised conditions |

`local_r2` is the diagnostic that matters: a low value means a straight line is a
poor stand-in for the model near this row, so the LIME explanation should be
discounted regardless of how confident it looks.

## target_laptime

### Fastest Predicted Lap (test row 89, lap 51)

Local surrogate R2: **0.177** over 2000 perturbations.

| Condition | Weight | Effect |
|---|---:|---|
| `track_status <= 1.00` | -2.135495 | decreases the prediction |
| `stint_number > 3.00` | -0.941320 | decreases the prediction |
| `form_vs_baseline <= -0.47` | -0.758151 | decreases the prediction |
| `field_median_lag1 <= 98.01` | -0.687338 | decreases the prediction |
| `gap_roll3_std > 1.11` | -0.633516 | decreases the prediction |
| `gap_roll3_mean > 0.71` | -0.424692 | decreases the prediction |

**SHAP top-3:** `stint_number`, `gap_roll3_std`, `form_vs_baseline`
**LIME top-3:** `track_status`, `stint_number`, `form_vs_baseline`
**Agreement (Jaccard):** 0.500

![lime](xai_target_laptime_fastest_predicted_lap_lime.png)

### Median Predicted Lap (test row 92, lap 54)

Local surrogate R2: **0.205** over 2000 perturbations.

| Condition | Weight | Effect |
|---|---:|---|
| `track_status <= 1.00` | -2.378100 | decreases the prediction |
| `stint_number > 3.00` | -0.839599 | decreases the prediction |
| `form_vs_baseline <= -0.47` | -0.797397 | decreases the prediction |
| `field_pace_trend > 0.12` | +0.724793 | increases the prediction |
| `field_median_lag1 <= 98.01` | -0.632515 | decreases the prediction |
| `team_red_bull_racing <= 0.00` | +0.614181 | increases the prediction |

**SHAP top-3:** `stint_number`, `form_vs_baseline`, `tracktemp_dev_x_tyrelife`
**LIME top-3:** `track_status`, `stint_number`, `form_vs_baseline`
**Agreement (Jaccard):** 0.500

![lime](xai_target_laptime_median_predicted_lap_lime.png)

### Slowest Predicted Lap (test row 178, lap 55)

Local surrogate R2: **0.306** over 2000 perturbations.

| Condition | Weight | Effect |
|---|---:|---|
| `track_status <= 1.00` | -2.426426 | decreases the prediction |
| `form_vs_baseline > 0.36` | +1.458877 | increases the prediction |
| `stint_number > 3.00` | -0.992117 | decreases the prediction |
| `gap_roll3_std > 1.11` | -0.680784 | decreases the prediction |
| `field_median_lag1 <= 98.01` | -0.610111 | decreases the prediction |
| `gap_expanding > 0.56` | +0.522464 | increases the prediction |

**SHAP top-3:** `form_vs_baseline`, `driver_zho`, `stint_number`
**LIME top-3:** `track_status`, `form_vs_baseline`, `stint_number`
**Agreement (Jaccard):** 0.500

![lime](xai_target_laptime_slowest_predicted_lap_lime.png)

---

## target_pit_next_lap

### Lowest Pit Probability (test row 179, lap 56)

Local surrogate R2: **0.131** over 2000 perturbations.

| Condition | Weight | Effect |
|---|---:|---|
| `tyre_life <= 5.00` | -0.062288 | decreases the prediction |
| `gap_roll3_mean > 0.71` | +0.046581 | increases the prediction |
| `0.00 < compound_soft <= 1.00` | -0.031306 | decreases the prediction |
| `form_vs_baseline > 0.36` | -0.020471 | decreases the prediction |
| `tracktemp_dev_x_tyrelife <= -2.15` | -0.019711 | decreases the prediction |
| `field_median_lag1 <= 98.01` | -0.016083 | decreases the prediction |

**SHAP top-3:** `form_vs_baseline`, `tyre_life`, `gap_roll3_mean`
**LIME top-3:** `tyre_life`, `gap_roll3_mean`, `compound_soft`
**Agreement (Jaccard):** 0.500

![lime](xai_target_pit_next_lap_lowest_pit_probability_lime.png)

### Closest To Decision Boundary (test row 34, lap 48)

Local surrogate R2: **0.319** over 2000 perturbations.

| Condition | Weight | Effect |
|---|---:|---|
| `tyre_life > 12.00` | +0.098646 | increases the prediction |
| `tyrelife_x_soft <= 0.00` | -0.067732 | decreases the prediction |
| `field_pace_trend <= -0.12` | +0.061148 | increases the prediction |
| `gap_roll3_mean > 0.71` | +0.042690 | increases the prediction |
| `compound_soft <= 0.00` | +0.033812 | increases the prediction |
| `tracktemp_dev_x_tyrelife <= -2.15` | -0.021965 | decreases the prediction |

**SHAP top-3:** `tyre_life`, `tracktemp_dev_x_tyrelife`, `gap_roll3_mean`
**LIME top-3:** `tyre_life`, `tyrelife_x_soft`, `field_pace_trend`
**Agreement (Jaccard):** 0.200

![lime](xai_target_pit_next_lap_closest_to_decision_boundary_lime.png)

---

