"""
Unit tests for the f1search state-space search package (Phase 1, Task 3).

Run with:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from app.intelligence.search.algorithms import (
    astar_search,
    breadth_first_search,
    depth_first_search,
    greedy_best_first_search,
    run_all,
    uniform_cost_search,
)
from app.intelligence.search.comparison import compare, summarise
from app.intelligence.search.problem import (
    Action,
    ActionType,
    Compound,
    RaceProblem,
    RaceState,
)


def small_problem(laps: int = 16) -> RaceProblem:
    return RaceProblem(total_laps=laps, start_compound=Compound.SOFT,
                       pit_loss=22.0, track_temp=35.0, max_stops=2)


# --------------------------------------------------------------------------- #
# Problem formulation
# --------------------------------------------------------------------------- #

def test_initial_state():
    p = small_problem()
    s = p.initial_state()
    assert s.lap == 0
    assert s.compound is Compound.SOFT
    assert s.stops_made == 0
    assert Compound.SOFT in s.compounds_used


def test_actions_include_run_and_pits():
    p = small_problem()
    acts = p.actions(p.initial_state())
    assert any(a.type is ActionType.RUN for a in acts)
    # PIT options exclude the current compound.
    pit_compounds = {a.compound for a in acts if a.type is ActionType.PIT}
    assert Compound.SOFT not in pit_compounds
    assert Compound.MEDIUM in pit_compounds


def test_run_transition_advances_lap_and_age():
    p = small_problem()
    s0 = p.initial_state()
    s1 = p.result(s0, Action(ActionType.RUN))
    assert s1.lap == 1
    assert s1.tyre_age == 1
    assert s1.fuel_kg < s0.fuel_kg


def test_pit_transition_changes_compound_and_counts_stop():
    p = small_problem()
    s0 = p.initial_state()
    s1 = p.result(s0, Action(ActionType.PIT, Compound.HARD))
    assert s1.compound is Compound.HARD
    assert s1.stops_made == 1
    assert s1.tyre_age == 1
    assert Compound.HARD in s1.compounds_used


def test_goal_requires_two_dry_compounds():
    p = small_problem(laps=2)
    # A state that reaches the flag on a single compound is NOT a goal (dry).
    one = RaceState(lap=2, compound=Compound.SOFT, tyre_age=2, stops_made=0,
                    fuel_kg=90, compounds_used=frozenset({Compound.SOFT}))
    assert not p.is_goal(one)
    two = RaceState(lap=2, compound=Compound.MEDIUM, tyre_age=1, stops_made=1,
                    fuel_kg=90,
                    compounds_used=frozenset({Compound.SOFT, Compound.MEDIUM}))
    assert p.is_goal(two)


def test_wet_race_relaxes_compound_rule():
    p = RaceProblem(total_laps=2, start_compound=Compound.INTERMEDIATE,
                    is_wet=True, max_stops=1)
    s = RaceState(lap=2, compound=Compound.INTERMEDIATE, tyre_age=2, stops_made=0,
                  fuel_kg=90, compounds_used=frozenset({Compound.INTERMEDIATE}))
    assert p.is_goal(s)


def test_lap_time_increases_with_age():
    p = small_problem()
    fresh = p.lap_time(Compound.SOFT, 1, 50)
    worn = p.lap_time(Compound.SOFT, 15, 50)
    assert worn > fresh


def test_heuristic_is_admissible_lower_bound():
    """h(state) must never exceed the true optimal remaining cost (UCS)."""
    p = small_problem(laps=14)
    start = p.initial_state()
    ucs = uniform_cost_search(p)
    assert ucs.found
    # Heuristic from the start must not overestimate the true optimal cost.
    assert p.heuristic(start) <= ucs.solution_cost + 1e-6


# --------------------------------------------------------------------------- #
# Algorithms
# --------------------------------------------------------------------------- #

def test_all_algorithms_find_a_goal():
    p = small_problem()
    for name, r in run_all(p).items():
        assert r.found, f"{name} failed to find a goal"


def test_ucs_and_astar_are_optimal_and_agree():
    p = small_problem(laps=18)
    ucs = uniform_cost_search(p)
    astar = astar_search(p)
    assert ucs.found and astar.found
    assert abs(ucs.solution_cost - astar.solution_cost) < 1e-6


def test_astar_expands_no_more_than_ucs():
    """With an admissible heuristic, A* should not expand more nodes than UCS."""
    p = small_problem(laps=18)
    ucs = uniform_cost_search(p)
    astar = astar_search(p)
    assert astar.nodes_expanded <= ucs.nodes_expanded


def test_bfs_and_greedy_not_cheaper_than_optimal():
    """No algorithm can beat the true optimum; sub-optimal ones are >= UCS."""
    p = small_problem(laps=18)
    optimal = uniform_cost_search(p).solution_cost
    for r in (breadth_first_search(p), depth_first_search(p),
              greedy_best_first_search(p)):
        assert r.solution_cost >= optimal - 1e-6


def test_dfs_is_memory_light():
    p = small_problem(laps=20)
    dfs = depth_first_search(p)
    ucs = uniform_cost_search(p)
    assert dfs.max_frontier_size < ucs.max_frontier_size


def test_solution_reaches_goal_state():
    p = small_problem()
    r = astar_search(p)
    final = r.solution[-1].state
    assert p.is_goal(final)


def test_solution_path_cost_matches_recomputation():
    """The reported cost must equal the sum of step costs along the path."""
    p = small_problem()
    r = astar_search(p)
    total = 0.0
    nodes = r.solution
    for i in range(1, len(nodes)):
        total += p.step_cost(nodes[i - 1].state, nodes[i].action)
    assert abs(total - r.solution_cost) < 1e-6


def test_metrics_are_populated():
    p = small_problem()
    r = astar_search(p)
    assert r.nodes_expanded > 0
    assert r.nodes_generated >= r.nodes_expanded
    assert r.elapsed_ms >= 0.0
    assert r.peak_memory_kb > 0.0


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #

def test_comparison_marks_ucs_astar_optimal():
    p = small_problem(laps=18)
    rows = compare(run_all(p))
    opt = {r.algorithm for r in rows if r.is_optimal}
    assert "UCS" in opt and "A*" in opt


def test_summary_fields_present():
    p = small_problem()
    s = summarise(compare(run_all(p)))
    assert set(s) >= {"n_algorithms", "n_found", "optimal_algorithms",
                      "fastest", "fewest_nodes_expanded", "leanest_frontier"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
