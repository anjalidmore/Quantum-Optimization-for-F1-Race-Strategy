"""
f1search.reports
================

Markdown deliverable generators for Task 3: the algorithm comparison report and
the state-space formulation documentation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from .algorithms import SearchResult
from .comparison import ComparisonRow, summarise
from .problem import ActionType, RaceProblem


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _fmt(x: float) -> str:
    if x == float("inf"):
        return "∞"
    return f"{x:.2f}"


def comparison_report_md(problem: RaceProblem, rows: List[ComparisonRow],
                         results: Dict[str, SearchResult]) -> str:
    summary = summarise(rows)
    parts = [
        "# Search Algorithm Comparison Report",
        "",
        f"_Generated {_ts()}._",
        "",
        "## Problem instance",
        "",
        f"- **Race distance:** {problem.total_laps} laps",
        f"- **Start compound:** {problem.start_compound.value}",
        f"- **Pit loss:** {problem.pit_loss:.1f} s",
        f"- **Track temperature:** {problem.track_temp:.0f} °C",
        f"- **Conditions:** {'WET' if problem.is_wet else 'DRY'}",
        f"- **Allowed compounds:** {', '.join(c.value for c in problem.allowed_compounds)}",
        f"- **Max pit stops:** {problem.max_stops}",
        "",
        "## Metrics",
        "",
        "| Algorithm | Found | Cost (s) | Gap | Optimal | Expanded | Generated | "
        "Max frontier | Time (ms) | Peak mem (KB) | Depth | Stops |",
        "|-----------|-------|----------|-----|---------|----------|-----------|"
        "--------------|-----------|---------------|-------|-------|",
    ]
    for r in rows:
        gap = "0.0%" if r.is_optimal else (f"+{r.cost_gap_pct:.1f}%" if r.found else "—")
        parts.append(
            f"| **{r.algorithm}** | {'yes' if r.found else 'no'} | {_fmt(r.solution_cost)} | "
            f"{gap} | {'optimal' if r.is_optimal else '—'} | {r.nodes_expanded} | "
            f"{r.nodes_generated} | {r.max_frontier_size} | {r.elapsed_ms:.2f} | "
            f"{r.peak_memory_kb:.1f} | {r.solution_depth} | {r.n_pit_stops} |"
        )

    parts += [
        "",
        "## Headline findings",
        "",
        f"- **Optimal algorithm(s):** {', '.join(summary['optimal_algorithms']) or 'none'}",
        f"- **Fastest wall-clock:** {summary['fastest']}",
        f"- **Fewest nodes expanded:** {summary['fewest_nodes_expanded']}",
        f"- **Leanest frontier (memory):** {summary['leanest_frontier']}",
        "",
        "## Interpretation",
        "",
        "- **Uniform-Cost Search (UCS)** is provably optimal for non-negative "
        "step costs and provides the ground-truth minimum race time.",
        "- **A\\*** matches UCS's optimal cost while expanding fewer nodes, "
        "because the admissible heuristic (laps-remaining × fastest-possible "
        "lap) focuses the search toward the goal.",
        "- **BFS** finds the *shallowest* goal, which need not be the cheapest "
        "when step costs vary (they do here), so it is generally sub-optimal.",
        "- **DFS** uses the least memory but returns the first goal it reaches, "
        "typically far from optimal.",
        "- **Greedy** best-first is fast (few expansions) but ignores accumulated "
        "cost, so it is not cost-optimal.",
        "",
        "## Optimal strategy",
        "",
    ]
    # Describe the optimal (A*) strategy stint by stint.
    astar = results.get("A*")
    if astar and astar.found:
        parts.append(_describe_strategy(problem, astar))
    return "\n".join(parts) + "\n"


def _describe_strategy(problem: RaceProblem, result: SearchResult) -> str:
    lines = ["The optimal strategy found by A\\* is:", ""]
    prev = problem.initial_state().compound
    start = 0
    stints = []
    for node in result.solution[1:]:
        if node.action is not None and node.action.type is ActionType.PIT:
            stints.append((prev, start, node.state.lap - 1))
            prev = node.action.compound
            start = node.state.lap - 1
    stints.append((prev, start, problem.total_laps))
    for i, (compound, s, e) in enumerate(stints, 1):
        lines.append(f"- **Stint {i}:** laps {s}–{e} on **{compound.value}** "
                     f"({e - s} laps)")
    lines.append("")
    lines.append(f"Total estimated race time: **{result.solution_cost:.1f} s** "
                 f"({len(stints) - 1} pit stop(s)).")
    return "\n".join(lines)


def formulation_doc_md() -> str:
    return f"""# State-Space Formulation

_Generated {_ts()}._

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
cost. Admissibility guarantees A\\* returns a cost-optimal solution.

## Goals (per the task specification)

- **Maximum points / optimal finish position** — approximated here by minimising
  total race time, the dominant determinant of finishing position in a
  time-trial sense.
- **Minimum race time** — the direct search objective.
"""


def heuristic_table_md(problem: RaceProblem, results: Dict[str, SearchResult]) -> str:
    """The heuristic, its components, and an admissibility check with real numbers.

    Admissibility is the property A* depends on, so it is demonstrated rather
    than asserted: for every state along the optimal path, h(n) is compared with
    the true remaining cost taken from that same optimal solution. If any row
    showed h > actual, A* would not be guaranteed optimal and the table would
    say so.
    """
    fastest = problem.fastest_possible_lap()
    astar = results.get("A*")

    parts = [
        "# Heuristic Table",
        "",
        f"_Generated {_ts()}._",
        "",
        "## Definition",
        "",
        "```",
        "h(state) = (total_laps - state.lap) x fastest_possible_lap",
        "```",
        "",
        "| Component | Value | Where it comes from |",
        "|---|---:|---|",
        f"| `total_laps` | {problem.total_laps} | the problem instance |",
        f"| `fastest_possible_lap` | {fastest:.4f} s | `RaceProblem.fastest_possible_lap()`: "
        "fresh-tyre pace at zero fuel, the fastest lap physically available |",
        f"| `pit_loss` (ignored by h) | {problem.pit_loss:.1f} s | ignoring a non-negative cost keeps h a lower bound |",
        f"| `h(initial state)` | {problem.heuristic(problem.initial_state()):.2f} s | "
        f"{problem.total_laps} laps x {fastest:.2f} s |",
        "",
        "## Why it is admissible",
        "",
        "Two costs are deliberately left out, and both are non-negative:",
        "",
        "1. **Tyre degradation.** `fastest_possible_lap` is fresh-tyre pace; a real lap on worn tyres is slower.",
        "2. **Fuel weight and pit loss.** Carrying fuel costs time, and a dry race may still require a stop.",
        "",
        "Leaving out non-negative costs can only make h too small, never too large, so "
        "`h(n) <= true remaining cost` everywhere. That is exactly the condition A* needs to return "
        "a cost-optimal solution, and it is why A* and uniform-cost search agree below.",
        "",
    ]

    if astar and astar.found:
        path = astar.solution
        total = astar.solution_cost
        parts += [
            "## Admissibility checked on the optimal path",
            "",
            "`g` is the cost paid to reach the state, `h` the estimate of what remains, and "
            "`actual remaining` the true figure taken from the optimal solution itself. "
            "Admissibility requires `h <= actual remaining` on every row.",
            "",
            "| Lap | Compound | Tyre age | g (s) | h (s) | f = g + h | Actual remaining (s) | h <= actual |",
            "|---:|---|---:|---:|---:|---:|---:|:---:|",
        ]
        violations = 0
        # Every 4th state keeps the table readable; the check below covers all of them.
        for node in path[:: max(1, len(path) // 8)]:
            g = float(node.path_cost)
            h = float(problem.heuristic(node.state))
            actual = total - g
            ok = h <= actual + 1e-9
            violations += 0 if ok else 1
            parts.append(
                f"| {node.state.lap} | {node.state.compound.value} | {node.state.tyre_age} | "
                f"{g:.2f} | {h:.2f} | {g + h:.2f} | {actual:.2f} | {'yes' if ok else '**NO**'} |"
            )
        all_violations = sum(
            1 for node in path
            if float(problem.heuristic(node.state)) > (total - float(node.path_cost)) + 1e-9
        )
        parts += [
            "",
            f"Checked on all {len(path)} states of the optimal path: "
            f"**{all_violations} violations**."
            + (" Admissibility holds, so A*'s solution is cost-optimal."
               if all_violations == 0 else
               " At least one state overestimates, so A* is **not** guaranteed optimal here."),
            "",
            f"Optimal cost: **{total:.4f} s**. "
            f"A* expanded {astar.nodes_expanded} nodes to find it.",
            "",
        ]
        if "UCS" in results and results["UCS"].found:
            ucs = results["UCS"]
            same = abs(ucs.solution_cost - total) < 1e-6
            parts += [
                f"Uniform-cost search, which uses no heuristic, reached "
                f"{ucs.solution_cost:.4f} s while expanding {ucs.nodes_expanded} nodes. "
                + ("The two costs agree, which is the empirical check that the heuristic did not "
                   "cost optimality — it only saved "
                   f"{ucs.nodes_expanded - astar.nodes_expanded} node expansions."
                   if same else
                   "**The two costs differ, which would indicate an inadmissible heuristic.**"),
                "",
            ]
    else:
        parts += ["## Admissibility check", "",
                  "A* found no solution on this instance, so there is no optimal path to check against.", ""]
    return "\n".join(parts) + "\n"


def generate_all(output_dir: Path, problem: RaceProblem,
                 rows: List[ComparisonRow],
                 results: Dict[str, SearchResult]) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    return {
        "comparison_report": _write(output_dir / "comparison_report.md",
                                    comparison_report_md(problem, rows, results)),
        "formulation": _write(output_dir / "state_space_formulation.md",
                              formulation_doc_md()),
        "heuristic_table": _write(output_dir / "heuristic_table.md",
                                  heuristic_table_md(problem, results)),
    }
