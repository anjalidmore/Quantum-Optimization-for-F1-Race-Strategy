# Task 8 - Counterfactual Analysis Report

_Generated 2026-09-05 01:38 UTC._

*What would have to change about the race state for the recommendation to flip?*

Two methods, for two different questions:

* **Single-feature bisection scan** - hold the entire race state fixed, move one
  feature, find the exact value at which the model's output crosses the decision
  threshold. This is the question a race engineer actually asks (*how many more
  laps on these tyres?*), because it yields an actionable instruction.
* **DiCE (random search)** - find complete alternative race states the model would
  classify the other way. Useful when several different routes to a different call
  exist; less actionable, because changing six things at once is not an instruction.

Every number below is the result of a real search against the real model. Where no
counterfactual exists inside the feature's observed range, that is reported as
*not reachable* together with the range searched.

## target_laptime

### Fastest Predicted Lap (test row 89, lap 51)

**Scanned feature:** `tyre_life`  |  **Current value:** 8  |  **Current prediction:** 95.9445  |  **Threshold:** 98.825

**Searched range:** [1, 22] in 60 steps

**Result: not reachable.** No crossing of 98.82499694824219 exists anywhere in the observed range of tyre_life ([1.000, 22.000]) with the rest of the race state fixed. The recommendation is not flippable by this feature alone.

> No value of tyre age between 1 and 22 flips this recommendation with the rest of the race state unchanged - the call is not sensitive to tyre age alone.

![cf](xai_target_laptime_fastest_predicted_lap_counterfactual.png)

### Median Predicted Lap (test row 92, lap 54)

**Scanned feature:** `tyre_life`  |  **Current value:** 11  |  **Current prediction:** 97.7566  |  **Threshold:** 98.825

**Searched range:** [1, 22] in 60 steps

**Result: reachable.** The recommendation flips at `tyre_life` = **19.4** - a increase of **8.404** from the current value.

> The recommendation flips when tyre age reaches 19.4 (currently 11, a increase of 8.404), holding everything else fixed.

![cf](xai_target_laptime_median_predicted_lap_counterfactual.png)

### Slowest Predicted Lap (test row 178, lap 55)

**Scanned feature:** `tyre_life`  |  **Current value:** 4  |  **Current prediction:** 102.2913  |  **Threshold:** 98.825

**Searched range:** [1, 22] in 60 steps

**Result: not reachable.** No crossing of 98.82499694824219 exists anywhere in the observed range of tyre_life ([1.000, 22.000]) with the rest of the race state fixed. The recommendation is not flippable by this feature alone.

> No value of tyre age between 1 and 22 flips this recommendation with the rest of the race state unchanged - the call is not sensitive to tyre age alone.

![cf](xai_target_laptime_slowest_predicted_lap_counterfactual.png)

---

## target_pit_next_lap

### Lowest Pit Probability (test row 179, lap 56)

**Scanned feature:** `tyre_life`  |  **Current value:** 5  |  **Current prediction:** 0.0000  |  **Threshold:** 0.5

**Searched range:** [1, 22] in 60 steps

**Result: not reachable.** No crossing of 0.5 exists anywhere in the observed range of tyre_life ([1.000, 22.000]) with the rest of the race state fixed. The recommendation is not flippable by this feature alone.

> No value of tyre age between 1 and 22 flips this recommendation with the rest of the race state unchanged - the call is not sensitive to tyre age alone.

![cf](xai_target_pit_next_lap_lowest_pit_probability_counterfactual.png)

### Closest To Decision Boundary (test row 34, lap 48)

**Scanned feature:** `tyre_life`  |  **Current value:** 21  |  **Current prediction:** 0.2040  |  **Threshold:** 0.5

**Searched range:** [1, 22] in 60 steps

**Result: not reachable.** No crossing of 0.5 exists anywhere in the observed range of tyre_life ([1.000, 22.000]) with the rest of the race state fixed. The recommendation is not flippable by this feature alone.

> No value of tyre age between 1 and 22 flips this recommendation with the rest of the race state unchanged - the call is not sensitive to tyre age alone.

![cf](xai_target_pit_next_lap_closest_to_decision_boundary_counterfactual.png)

### DiCE - diverse whole-row counterfactuals

_Each row above is a complete alternative race state the model would classify the other way. Values are in standardised units; a delta of +1.0 means one standard deviation of that feature as observed in training._

| # | Features changed | Changes |
|---:|---:|---|
| 1 | 2 | `tracktemp_dev_x_tyrelife` -19.095 -> -1.363 (+17.733); `form_vs_baseline` +0.690 -> +0.318 (-0.372) |
| 2 | 2 | `tracktemp_dev_x_tyrelife` -19.095 -> -1.076 (+18.019); `form_vs_baseline` +0.690 -> +0.482 (-0.208) |
| 3 | 2 | `tracktemp_dev_x_tyrelife` -19.095 -> -13.007 (+6.088); `field_median_lag1` +97.432 -> +97.633 (+0.201) |

---

