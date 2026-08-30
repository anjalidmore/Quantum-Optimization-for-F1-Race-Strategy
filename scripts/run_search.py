#!/usr/bin/env python3
"""
run_search.py
=============

End-to-end driver for Phase 1 / Task 3 (State-Space Search).

It:

1. Defines a realistic race-strategy problem instance.
2. Runs all five search algorithms (BFS, DFS, UCS, Greedy, A*) with full
   instrumentation.
3. Builds the comparison table and optimality assessment.
4. Renders the metric charts, the optimal-strategy stint plot and a search-tree
   diagram.
5. Writes the comparison report and state-space formulation documentation.

Exits non-zero if the key optimality invariant fails (A* cost == UCS cost), so
it doubles as a CI correctness gate.

Usage
-----
    python run_search.py [--laps N] [--output-dir OUTPUTS]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from app.intelligence.search import reports as rep  # noqa: E402
from app.intelligence.search import visualize as viz  # noqa: E402
from app.intelligence.search.algorithms import run_all  # noqa: E402
from app.intelligence.search.comparison import compare, summarise  # noqa: E402
from app.intelligence.search.problem import Compound, RaceProblem  # noqa: E402
from app.core.paths import SEARCH_ARTIFACTS_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("run_search")


def main(laps: int, output_dir: Path) -> int:
    output_dir = output_dir.resolve()
    reports_dir = output_dir / "reports"
    diagrams_dir = output_dir / "diagrams"
    for d in (reports_dir, diagrams_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- 1. Problem instance ----------------------------------------------
    problem = RaceProblem(
        total_laps=laps,
        start_compound=Compound.SOFT,
        pit_loss=22.0,
        track_temp=38.0,
        max_stops=2,
    )
    log.info("Problem: %d laps, start=%s, pit_loss=%.1fs, track=%.0f°C, "
             "compounds=%s", problem.total_laps, problem.start_compound.value,
             problem.pit_loss, problem.track_temp,
             [c.value for c in problem.allowed_compounds])

    # --- 2. Run all algorithms --------------------------------------------
    log.info("Running all search algorithms ...")
    results = run_all(problem)
    for name, r in results.items():
        log.info("  %-7s found=%s cost=%s expanded=%d frontier=%d time=%.2fms",
                 name, r.found,
                 f"{r.solution_cost:.1f}" if r.found else "n/a",
                 r.nodes_expanded, r.max_frontier_size, r.elapsed_ms)

    # --- 3. Comparison -----------------------------------------------------
    rows = compare(results)
    summary = summarise(rows)
    log.info("Optimal: %s | fastest: %s | fewest nodes: %s | leanest: %s",
             summary["optimal_algorithms"], summary["fastest"],
             summary["fewest_nodes_expanded"], summary["leanest_frontier"])

    # --- 4. Visualisations -------------------------------------------------
    log.info("Rendering visualisations ...")
    viz.render_metrics(rows, diagrams_dir / "metrics_comparison.png")
    astar = results["A*"]
    if astar.found:
        viz.render_strategy_path(problem, astar.solution,
                                 diagrams_dir / "optimal_strategy_path.png",
                                 title="A* optimal strategy")
    viz.render_search_tree(problem, astar, diagrams_dir / "search_tree.png",
                           max_depth=4)

    # --- 5. Reports --------------------------------------------------------
    log.info("Generating reports ...")
    written = rep.generate_all(reports_dir, problem, rows, results)
    for name, path in written.items():
        log.info("  wrote %s -> %s", name, path.relative_to(output_dir))

    # --- Correctness gate: A* must match UCS optimal cost -----------------
    ucs, astar_r = results["UCS"], results["A*"]
    if ucs.found and astar_r.found:
        if abs(ucs.solution_cost - astar_r.solution_cost) > 1e-6:
            log.error("Optimality invariant FAILED: A* (%.4f) != UCS (%.4f)",
                      astar_r.solution_cost, ucs.solution_cost)
            return 1
        log.info("Optimality invariant holds: A* cost == UCS cost == %.2f s",
                 ucs.solution_cost)
    log.info("All Task-3 deliverables generated successfully in %s", output_dir)
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run F1 state-space search comparison.")
    p.add_argument("--laps", type=int, default=24,
                   help="Race distance in laps (keep modest so uninformed "
                        "search stays tractable).")
    p.add_argument("--output-dir", type=Path, default=SEARCH_ARTIFACTS_DIR)
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    raise SystemExit(main(a.laps, a.output_dir))
