# State-Space Formulation

_Generated 2026-08-30 11:23 UTC._

## Problem as search

Race-strategy selection is modelled as a **shortest-path search** whose solution
is the sequence of per-lap decisions (run / pit-and-fit-compound) that completes
the race distance in **minimum estimated total time**.

## Formal components

| Component | Definition |
|-----------|------------|
| **State** | `(lap, compound, tyre_age, stops_made, fuel_kg, compounds_used)` |
| **Initial state** | lap 0, start compound, age 0, 0 stops, full fuel |
| **Actions** | `RUN` (one lap on current tyres); `PIT(compound)` (fit a new compound, run the out-lap) |
| **Transition** | `RUN`: lap+1, age+1, fuel−burn. `PIT`: lap+1, new compound at age 1, stops+1, fuel−burn |
| **Step cost** | estimated lap time (compound, age, fuel, track-temp) + pit-loss if pitting |
| **Goal test** | `lap == total_laps` **and** (dry race ⇒ ≥ 2 dry compounds used) |

## Cost model

Lap time is a transparent, deterministic function:

```
lap_time = base_pace(compound)
         + deg(compound) · tyre_age · temp_factor      (with a 'cliff' past cliff_age)
         + fuel_kg · fuel_time_penalty
```

Softer compounds have lower `base_pace` (faster) but higher `deg` (wear faster),
creating the pace-versus-durability trade-off that makes pit strategy non-trivial.

## Heuristic (for informed search)

```
h(state) = (total_laps − lap) × fastest_possible_lap
```

This is **admissible**: no lap can be completed faster than the fastest fresh-tyre
pace at zero fuel, and the heuristic ignores the (non-negative) mandatory pit
loss a dry race may still require — so it never overestimates the true remaining
cost. Admissibility guarantees A\* returns a cost-optimal solution.

## Goals (per the task specification)

- **Maximum points / optimal finish position** — approximated here by minimising
  total race time, the dominant determinant of finishing position in a
  time-trial sense.
- **Minimum race time** — the direct search objective.
