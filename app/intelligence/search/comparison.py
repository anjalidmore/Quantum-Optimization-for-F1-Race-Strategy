"""
f1search.comparison
====================

Aggregates the results of running every search algorithm on a problem and
derives the comparison metrics requested for Task 3: execution time, nodes
expanded, memory (peak frontier & traced memory), solution cost, and optimality.

Optimality is assessed *relative to Uniform-Cost Search*, which is provably
optimal for non-negative step costs and therefore serves as the ground-truth
minimum cost. An algorithm is marked optimal when its solution cost is within a
small tolerance of the UCS cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .algorithms import SearchResult


@dataclass
class ComparisonRow:
    """One algorithm's metrics in the comparison table."""

    algorithm: str
    found: bool
    solution_cost: float
    cost_gap_pct: float           # % above the optimal (UCS) cost
    is_optimal: bool
    nodes_expanded: int
    nodes_generated: int
    max_frontier_size: int
    elapsed_ms: float
    peak_memory_kb: float
    solution_depth: int
    n_pit_stops: int


def _count_pits(result: SearchResult) -> int:
    from .problem import ActionType
    return sum(1 for a in result.solution_actions if a.type is ActionType.PIT)


def compare(results: Dict[str, SearchResult], tolerance: float = 1e-6) -> List[ComparisonRow]:
    """
    Build comparison rows from a mapping of algorithm name -> result.

    The optimal cost baseline is taken from UCS if present, else the minimum
    solution cost among the algorithms that found a goal.
    """
    found_costs = [r.solution_cost for r in results.values() if r.found]
    if "UCS" in results and results["UCS"].found:
        optimal_cost = results["UCS"].solution_cost
    elif found_costs:
        optimal_cost = min(found_costs)
    else:
        optimal_cost = float("inf")

    rows: List[ComparisonRow] = []
    for name, r in results.items():
        if r.found and optimal_cost not in (0.0, float("inf")):
            gap = (r.solution_cost - optimal_cost) / optimal_cost * 100.0
        else:
            gap = float("inf") if not r.found else 0.0
        rows.append(ComparisonRow(
            algorithm=name,
            found=r.found,
            solution_cost=r.solution_cost,
            cost_gap_pct=gap,
            is_optimal=r.found and abs(r.solution_cost - optimal_cost) <= max(tolerance, optimal_cost * 1e-6),
            nodes_expanded=r.nodes_expanded,
            nodes_generated=r.nodes_generated,
            max_frontier_size=r.max_frontier_size,
            elapsed_ms=r.elapsed_ms,
            peak_memory_kb=r.peak_memory_kb,
            solution_depth=r.solution_depth,
            n_pit_stops=_count_pits(r),
        ))
    # Present in a stable, meaningful order.
    order = {"BFS": 0, "DFS": 1, "UCS": 2, "Greedy": 3, "A*": 4}
    rows.sort(key=lambda row: order.get(row.algorithm, 99))
    return rows


def summarise(rows: List[ComparisonRow]) -> Dict[str, object]:
    """Return headline findings across the comparison."""
    solved = [r for r in rows if r.found]
    optimal = [r for r in solved if r.is_optimal]
    fastest = min(solved, key=lambda r: r.elapsed_ms) if solved else None
    fewest_nodes = min(solved, key=lambda r: r.nodes_expanded) if solved else None
    leanest = min(solved, key=lambda r: r.max_frontier_size) if solved else None
    return {
        "n_algorithms": len(rows),
        "n_found": len(solved),
        "optimal_algorithms": [r.algorithm for r in optimal],
        "fastest": fastest.algorithm if fastest else None,
        "fewest_nodes_expanded": fewest_nodes.algorithm if fewest_nodes else None,
        "leanest_frontier": leanest.algorithm if leanest else None,
    }
