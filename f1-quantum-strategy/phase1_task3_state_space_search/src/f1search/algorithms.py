"""
f1search.algorithms
====================

Uninformed and informed state-space search algorithms, instrumented for the
Task-3 comparison (execution time, nodes expanded, peak frontier size / memory,
solution cost, and optimality).

All five requested algorithms share one :class:`SearchResult` return type and a
common :class:`Node` structure so their behaviour can be compared apples-to-
apples on the same :class:`~f1search.problem.RaceProblem`.

Algorithms
----------
* ``breadth_first_search``      — FIFO frontier; optimal only if step costs are
  uniform (they are not here), so it is included for comparison, not optimality.
* ``depth_first_search``        — LIFO frontier; memory-light, not optimal.
* ``uniform_cost_search``       — priority = path cost g(n); optimal for
  non-negative costs (Dijkstra).
* ``greedy_best_first_search``  — priority = heuristic h(n); fast, not optimal.
* ``astar_search``              — priority = g(n) + h(n); optimal with an
  admissible heuristic.

Each expands graph-search style (with an explored set / best-g map) and returns
the solution path plus metrics.
"""

from __future__ import annotations

import heapq
import itertools
import time
import tracemalloc
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .problem import Action, RaceProblem, RaceState


@dataclass
class Node:
    """A search-tree node."""

    state: RaceState
    parent: Optional["Node"] = None
    action: Optional[Action] = None
    path_cost: float = 0.0  # g(n): cumulative cost from the root
    depth: int = 0

    def path(self) -> List["Node"]:
        """Return the list of nodes from the root to this node."""
        node, seq = self, []
        while node is not None:
            seq.append(node)
            node = node.parent
        return list(reversed(seq))


@dataclass
class SearchResult:
    """The outcome and metrics of a single search run."""

    algorithm: str
    found: bool
    solution: List[Node] = field(default_factory=list)
    solution_cost: float = float("inf")
    nodes_expanded: int = 0
    nodes_generated: int = 0
    max_frontier_size: int = 0
    elapsed_ms: float = 0.0
    peak_memory_kb: float = 0.0

    @property
    def solution_actions(self) -> List[Action]:
        return [n.action for n in self.solution if n.action is not None]

    @property
    def solution_depth(self) -> int:
        return max(0, len(self.solution) - 1)


# --------------------------------------------------------------------------- #
# Instrumentation helper
# --------------------------------------------------------------------------- #

def _instrument(fn: Callable[[], SearchResult]) -> SearchResult:
    """Run ``fn`` while measuring wall-clock time and peak memory."""
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    result.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result.peak_memory_kb = peak / 1024.0
    return result


# --------------------------------------------------------------------------- #
# Uninformed search
# --------------------------------------------------------------------------- #

def breadth_first_search(problem: RaceProblem) -> SearchResult:
    """FIFO graph search. Finds the shallowest goal, not the cheapest."""

    def _run() -> SearchResult:
        res = SearchResult(algorithm="BFS", found=False)
        start = Node(problem.initial_state())
        if problem.is_goal(start.state):
            res.found = True
            res.solution = start.path()
            res.solution_cost = 0.0
            return res
        frontier: deque[Node] = deque([start])
        explored = {start.state.key()}
        res.nodes_generated = 1
        while frontier:
            res.max_frontier_size = max(res.max_frontier_size, len(frontier))
            node = frontier.popleft()
            res.nodes_expanded += 1
            for action in problem.actions(node.state):
                child_state = problem.result(node.state, action)
                if child_state.key() in explored:
                    continue
                cost = node.path_cost + problem.step_cost(node.state, action)
                child = Node(child_state, node, action, cost, node.depth + 1)
                res.nodes_generated += 1
                if problem.is_goal(child_state):
                    res.found = True
                    res.solution = child.path()
                    res.solution_cost = cost
                    return res
                explored.add(child_state.key())
                frontier.append(child)
        return res

    return _instrument(_run)


def depth_first_search(problem: RaceProblem, depth_limit: Optional[int] = None) -> SearchResult:
    """LIFO graph search with an optional depth limit (defaults to a bound that
    guarantees completeness: total_laps + max_stops)."""

    limit = depth_limit if depth_limit is not None else problem.total_laps + problem.max_stops + 1

    def _run() -> SearchResult:
        res = SearchResult(algorithm="DFS", found=False)
        start = Node(problem.initial_state())
        frontier: List[Node] = [start]
        # best-depth map allows revisiting a state via a shallower route.
        best_depth: Dict[Tuple, int] = {start.state.key(): 0}
        res.nodes_generated = 1
        while frontier:
            res.max_frontier_size = max(res.max_frontier_size, len(frontier))
            node = frontier.pop()
            if problem.is_goal(node.state):
                res.found = True
                res.solution = node.path()
                res.solution_cost = node.path_cost
                return res
            res.nodes_expanded += 1
            if node.depth >= limit:
                continue
            for action in problem.actions(node.state):
                child_state = problem.result(node.state, action)
                d = node.depth + 1
                key = child_state.key()
                if key in best_depth and best_depth[key] <= d:
                    continue
                cost = node.path_cost + problem.step_cost(node.state, action)
                best_depth[key] = d
                frontier.append(Node(child_state, node, action, cost, d))
                res.nodes_generated += 1
        return res

    return _instrument(_run)


# --------------------------------------------------------------------------- #
# Priority-queue-based search (shared core)
# --------------------------------------------------------------------------- #

def _best_first(
    problem: RaceProblem,
    algorithm: str,
    priority: Callable[[Node], float],
) -> SearchResult:
    """
    Generic best-first graph search parameterised by a priority function.

    * UCS      : priority = g(n)
    * Greedy   : priority = h(n)
    * A*       : priority = g(n) + h(n)
    """

    def _run() -> SearchResult:
        res = SearchResult(algorithm=algorithm, found=False)
        start = Node(problem.initial_state())
        counter = itertools.count()  # tie-breaker for stable, deterministic order
        frontier: List[Tuple[float, int, Node]] = [(priority(start), next(counter), start)]
        # best_g maps a state key to the cheapest g(n) seen so far.
        best_g: Dict[Tuple, float] = {start.state.key(): 0.0}
        res.nodes_generated = 1
        while frontier:
            res.max_frontier_size = max(res.max_frontier_size, len(frontier))
            _, _, node = heapq.heappop(frontier)
            # Stale entry check: skip if we've since found a cheaper route.
            if node.path_cost > best_g.get(node.state.key(), float("inf")):
                continue
            if problem.is_goal(node.state):
                res.found = True
                res.solution = node.path()
                res.solution_cost = node.path_cost
                return res
            res.nodes_expanded += 1
            for action in problem.actions(node.state):
                child_state = problem.result(node.state, action)
                g = node.path_cost + problem.step_cost(node.state, action)
                key = child_state.key()
                if g < best_g.get(key, float("inf")):
                    best_g[key] = g
                    child = Node(child_state, node, action, g, node.depth + 1)
                    heapq.heappush(frontier, (priority(child), next(counter), child))
                    res.nodes_generated += 1
        return res

    return _instrument(_run)


def uniform_cost_search(problem: RaceProblem) -> SearchResult:
    """Dijkstra-style search. Optimal for non-negative step costs."""
    return _best_first(problem, "UCS", priority=lambda n: n.path_cost)


def greedy_best_first_search(problem: RaceProblem) -> SearchResult:
    """Greedy search on h(n). Fast but not cost-optimal."""
    return _best_first(problem, "Greedy", priority=lambda n: problem.heuristic(n.state))


def astar_search(problem: RaceProblem) -> SearchResult:
    """A* search on f(n) = g(n) + h(n). Optimal with an admissible heuristic."""
    return _best_first(
        problem, "A*",
        priority=lambda n: n.path_cost + problem.heuristic(n.state),
    )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

ALGORITHMS: Dict[str, Callable[[RaceProblem], SearchResult]] = {
    "BFS": breadth_first_search,
    "DFS": depth_first_search,
    "UCS": uniform_cost_search,
    "Greedy": greedy_best_first_search,
    "A*": astar_search,
}


def run_all(problem: RaceProblem) -> Dict[str, SearchResult]:
    """Run every registered algorithm on ``problem`` and return their results."""
    return {name: fn(problem) for name, fn in ALGORITHMS.items()}
