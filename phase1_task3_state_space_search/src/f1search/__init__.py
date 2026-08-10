"""
f1search — F1 Race Strategy as State-Space Search (Phase 1, Task 3)
==================================================================

Classical state-space search over race-strategy states:

* ``problem``     — RaceState / RaceProblem formulation, cost model, heuristic.
* ``algorithms``  — BFS, DFS, UCS, Greedy best-first, A* (instrumented).
* ``comparison``  — metrics aggregation and optimality assessment.
* ``visualize``   — metric charts, strategy stint plot, search-tree diagram.
* ``reports``     — Markdown comparison report and formulation docs.

No machine learning, no quantum computing — classical search only.
"""
from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["problem", "algorithms", "comparison", "visualize", "reports"]
