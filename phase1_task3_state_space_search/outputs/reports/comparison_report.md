# Search Algorithm Comparison Report

_Generated 2026-08-03 04:27 UTC._

## Problem instance

- **Race distance:** 24 laps
- **Start compound:** SOFT
- **Pit loss:** 22.0 s
- **Track temperature:** 38 °C
- **Conditions:** DRY
- **Allowed compounds:** SOFT, MEDIUM, HARD
- **Max pit stops:** 2

## Metrics

| Algorithm | Found | Cost (s) | Gap | Optimal | Expanded | Generated | Max frontier | Time (ms) | Peak mem (KB) | Depth | Stops |
|-----------|-------|----------|-----|---------|----------|-----------|--------------|-----------|---------------|-------|-------|
| **BFS** | ✅ | 2270.66 | +0.4% |  | 1455 | 1590 | 135 | 60.58 | 887.8 | 24 | 1 |
| **DFS** | ✅ | 2299.93 | +1.7% |  | 24 | 29 | 5 | 0.62 | 11.6 | 24 | 2 |
| **UCS** | ✅ | 2262.42 | 0.0% | ★ | 1589 | 1723 | 135 | 66.31 | 757.5 | 24 | 1 |
| **Greedy** | ✅ | 2270.66 | +0.4% |  | 26 | 72 | 47 | 1.94 | 38.4 | 24 | 1 |
| **A*** | ✅ | 2262.42 | 0.0% | ★ | 1005 | 1139 | 134 | 48.10 | 488.9 | 24 | 1 |

## Headline findings

- **Optimal algorithm(s):** UCS, A*
- **Fastest wall-clock:** DFS
- **Fewest nodes expanded:** DFS
- **Leanest frontier (memory):** DFS

## Interpretation

- **Uniform-Cost Search (UCS)** is provably optimal for non-negative step costs and provides the ground-truth minimum race time.
- **A\*** matches UCS's optimal cost while expanding fewer nodes, because the admissible heuristic (laps-remaining × fastest-possible lap) focuses the search toward the goal.
- **BFS** finds the *shallowest* goal, which need not be the cheapest when step costs vary (they do here), so it is generally sub-optimal.
- **DFS** uses the least memory but returns the first goal it reaches, typically far from optimal.
- **Greedy** best-first is fast (few expansions) but ignores accumulated cost, so it is not cost-optimal.

## Optimal strategy

The optimal strategy found by A\* is:

- **Stint 1:** laps 0–14 on **SOFT** (14 laps)
- **Stint 2:** laps 14–24 on **MEDIUM** (10 laps)

Total estimated race time: **2262.4 s** (1 pit stop(s)).
