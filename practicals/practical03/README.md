# Phase 1 · Task 3 — State-Space Search

Part of **Quantum Optimization for Formula 1 Race Strategy**.

> **Scope of this module:** classical state-space search. No machine learning,
> no quantum computing. Race strategy is cast as a shortest-path search and
> solved with five algorithms whose behaviour is compared head-to-head.

---

## 1. What this module does

It formulates F1 race-strategy selection as a **search problem** — when to pit
and which tyre to fit, to complete the race in minimum estimated time — and
solves it with **BFS, DFS, Uniform-Cost Search, Greedy Best-First, and A\***.
Each run is instrumented so the algorithms can be compared on execution time,
nodes expanded, memory, solution cost, and optimality.

| Deliverable | Artifact |
|-------------|----------|
| State-space formulation | `outputs/reports/state_space_formulation.md` |
| Algorithm comparison report | `outputs/reports/comparison_report.md` |
| Metrics comparison chart | `outputs/diagrams/metrics_comparison.png` |
| Optimal-strategy stint plot | `outputs/diagrams/optimal_strategy_path.png` |
| Search-tree diagram | `outputs/diagrams/search_tree.png` |

---

## 2. The formulation

| Component | Definition |
|-----------|------------|
| **State** | `(lap, compound, tyre_age, stops_made, fuel_kg, compounds_used)` |
| **Actions** | `RUN` (one lap on current tyres) · `PIT(compound)` (fit a new compound, run out-lap) |
| **Step cost** | estimated lap time (compound, age, fuel, track-temp) + pit-loss if pitting |
| **Goal** | reach `total_laps` and (dry race) use ≥ 2 dry compounds — a real F1 rule |
| **Heuristic** | `laps_remaining × fastest_possible_lap` — admissible ⇒ A\* is optimal |

The cost model captures the pace-versus-durability trade-off: softer compounds
are faster when fresh but degrade quicker (with an accelerated "cliff" past a
compound-specific age), the car speeds up as fuel burns off, and a hotter track
amplifies degradation. This makes pit strategy a genuine optimisation, not a
lookup.

See `outputs/reports/state_space_formulation.md` for the full formal statement.

---

## 3. The five algorithms

| Algorithm | Frontier | Optimal? | Notes |
|-----------|----------|----------|-------|
| **BFS** | FIFO queue | No | Finds the *shallowest* goal; not cheapest when step costs vary |
| **DFS** | LIFO stack (depth-limited) | No | Very memory-light; returns the first goal reached |
| **UCS** | min-priority on `g(n)` | **Yes** | Dijkstra; ground-truth optimum for non-negative costs |
| **Greedy** | min-priority on `h(n)` | No | Fast (few expansions); ignores accumulated cost |
| **A\*** | min-priority on `g(n)+h(n)` | **Yes** | Optimal *and* efficient with an admissible heuristic |

All are implemented as graph search (explored set / best-`g` map) sharing one
instrumented `Node` and `SearchResult` type for a fair comparison.

---

## 4. Representative result

For a 24-lap dry race starting on softs (38 °C, one to two stops allowed):

| Algorithm | Cost (s) | Optimal | Expanded | Frontier | Time |
|-----------|----------|---------|----------|----------|------|
| BFS | 2270.7 | | 1455 | 135 | 74 ms |
| DFS | 2299.9 | | 24 | 5 | 0.9 ms |
| UCS | **2262.4** | ★ | 1589 | 135 | 76 ms |
| Greedy | 2270.7 | | 26 | 47 | 2 ms |
| A\* | **2262.4** | ★ | 1005 | 134 | 48 ms |

The textbook story emerges cleanly: **UCS and A\* find the identical optimal
cost**; **A\* expands ~37% fewer nodes than UCS** thanks to the heuristic; **DFS
and Greedy are far cheaper to run but sub-optimal**. (Exact numbers vary with
the problem instance and machine.)

---

## 5. Installation & execution

Requires **Python 3.10+**.

```bash
cd practical03
pip install -r requirements.txt
python practical03.py                 # defaults to a 24-lap race
python practical03.py --laps 30       # larger instance
```

The driver runs all algorithms, builds the comparison, renders the three
diagrams, writes both reports, and **asserts the optimality invariant**
(`A* cost == UCS cost`), exiting non-zero if it is ever violated — so it doubles
as a CI correctness gate.

> **Note on race length.** Uninformed search (BFS/UCS) explores a state space
> that grows with lap count, so the default is a modest race length. A\* scales
> considerably further; increase `--laps` to see the heuristic's advantage widen.

---

## 6. Tests

```bash
pytest -q
```

18 tests verify the transition model, the two-dry-compound goal rule, heuristic
admissibility, that **UCS and A\* agree on the optimum**, that A\* expands no
more nodes than UCS, that no algorithm beats the true optimum, and that reported
costs equal recomputed path costs.

---

## 7. Folder structure

```
practical03/
├── practical03.py               # driver / CI correctness gate
├── requirements.txt
├── pytest.ini
├── README.md
├── src/f1search/
│   ├── __init__.py
│   ├── problem.py              # RaceState / RaceProblem, cost model, heuristic
│   ├── algorithms.py           # BFS, DFS, UCS, Greedy, A* (instrumented)
│   ├── comparison.py           # metrics aggregation & optimality
│   ├── visualize.py            # charts, stint plot, search tree
│   └── reports.py              # Markdown generators
├── tests/
│   └── test_search.py
└── outputs/                    # generated on run
    ├── reports/*.md
    └── diagrams/*.png
```

---

## 8. Relationship to other tasks

* **Reuses Task 1 & Task 2 vocabulary.** The state variables (lap, compound,
  tyre age, fuel, track temperature) are the same domain concepts formalised in
  the knowledge representation and reasoned over by the expert system.
* **Feeds Phase 3.** This classical search establishes the *optimal-strategy
  baseline* against which the quantum-optimisation approaches (QAOA / hybrid)
  will be benchmarked for solution quality and cost.

## 9. Next task

**Task 4 — Data Engineering & EDA** builds the data pipeline (loading, cleaning,
feature statistics, correlation, visual analytics) over the historical dataset
and FastF1, completing the classical Phase 1 foundation.
