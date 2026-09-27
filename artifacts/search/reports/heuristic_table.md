# Heuristic Table

_Generated 2026-09-27 17:42 UTC._

## Definition

```
h(state) = (total_laps - state.lap) x fastest_possible_lap
```

| Component | Value | Where it comes from |
|---|---:|---|
| `total_laps` | 24 | the problem instance |
| `fastest_possible_lap` | 90.0000 s | `RaceProblem.fastest_possible_lap()`: fresh-tyre pace at zero fuel, the fastest lap physically available |
| `pit_loss` (ignored by h) | 22.0 s | ignoring a non-negative cost keeps h a lower bound |
| `h(initial state)` | 2160.00 s | 24 laps x 90.00 s |

## Why it is admissible

Two costs are deliberately left out, and both are non-negative:

1. **Tyre degradation.** `fastest_possible_lap` is fresh-tyre pace; a real lap on worn tyres is slower.
2. **Fuel weight and pit loss.** Carrying fuel costs time, and a dry race may still require a stop.

Leaving out non-negative costs can only make h too small, never too large, so `h(n) <= true remaining cost` everywhere. That is exactly the condition A* needs to return a cost-optimal solution, and it is why A* and uniform-cost search agree below.

## Admissibility checked on the optimal path

`g` is the cost paid to reach the state, `h` the estimate of what remains, and `actual remaining` the true figure taken from the optimal solution itself. Admissibility requires `h <= actual remaining` on every row.

| Lap | Compound | Tyre age | g (s) | h (s) | f = g + h | Actual remaining (s) | h <= actual |
|---:|---|---:|---:|---:|---:|---:|:---:|
| 0 | SOFT | 0 | 0.00 | 2160.00 | 2160.00 | 2262.42 | yes |
| 3 | SOFT | 3 | 279.72 | 1890.00 | 2169.72 | 1982.70 | yes |
| 6 | SOFT | 6 | 559.79 | 1620.00 | 2179.79 | 1702.63 | yes |
| 9 | SOFT | 9 | 840.23 | 1350.00 | 2190.23 | 1422.20 | yes |
| 12 | SOFT | 12 | 1121.01 | 1080.00 | 2201.01 | 1141.41 | yes |
| 15 | MEDIUM | 1 | 1423.60 | 810.00 | 2233.60 | 838.82 | yes |
| 18 | MEDIUM | 4 | 1703.19 | 540.00 | 2243.19 | 559.23 | yes |
| 21 | MEDIUM | 7 | 1982.80 | 270.00 | 2252.80 | 279.62 | yes |
| 24 | MEDIUM | 10 | 2262.42 | 0.00 | 2262.42 | 0.00 | yes |

Checked on all 25 states of the optimal path: **0 violations**. Admissibility holds, so A*'s solution is cost-optimal.

Optimal cost: **2262.4206 s**. A* expanded 1005 nodes to find it.

Uniform-cost search, which uses no heuristic, reached 2262.4206 s while expanding 1589 nodes. The two costs agree, which is the empirical check that the heuristic did not cost optimality — it only saved 584 node expansions.

