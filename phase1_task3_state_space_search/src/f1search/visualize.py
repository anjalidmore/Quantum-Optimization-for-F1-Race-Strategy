"""
f1search.visualize
===================

Visual analytics for Task 3:

* **Metrics comparison** — grouped bar charts of nodes expanded, execution time,
  peak frontier size and solution cost across the five algorithms.
* **Strategy path** — the winning (A*/UCS) strategy rendered as a stint diagram:
  laps on the x-axis, coloured by compound, with pit stops marked.
* **Search tree** — a Graphviz/NetworkX rendering of the partial search tree
  explored by A* (depth-limited for legibility), showing how the optimal path
  is discovered.

Matplotlib runs on the headless ``Agg`` backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

from .algorithms import Node, SearchResult
from .comparison import ComparisonRow
from .problem import ActionType, Compound, RaceProblem

# Consistent compound colours (broadly matching F1 tyre marking colours).
_COMPOUND_COLOURS: Dict[Compound, str] = {
    Compound.SOFT: "#E10600",
    Compound.MEDIUM: "#F5C518",
    Compound.HARD: "#EBEBEB",
    Compound.INTERMEDIATE: "#43B02A",
    Compound.WET: "#0067AD",
}


def render_metrics(rows: List[ComparisonRow], path: str | Path) -> Path:
    """Render a 2×2 grid of metric comparisons across algorithms."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    names = [r.algorithm for r in rows]
    metrics = [
        ("Nodes expanded", [r.nodes_expanded for r in rows], "#4363D8"),
        ("Execution time (ms)", [r.elapsed_ms for r in rows], "#E6194B"),
        ("Peak frontier size", [r.max_frontier_size for r in rows], "#3CB44B"),
        ("Solution cost (s)", [r.solution_cost if r.found else 0 for r in rows], "#F58231"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (title, values, colour) in zip(axes.flat, metrics):
        bars = ax.bar(names, values, color=colour, alpha=0.85, edgecolor="black")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        # Annotate solution-cost bars with optimality.
        if title.startswith("Solution cost"):
            for bar, r in zip(bars, rows):
                if r.found:
                    tag = "opt" if r.is_optimal else f"+{r.cost_gap_pct:.1f}%"
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height(), tag, ha="center", va="bottom",
                            fontsize=8)
    fig.suptitle("Search Algorithm Comparison — F1 Race Strategy",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def render_strategy_path(problem: RaceProblem, solution: List[Node],
                         path: str | Path, title: str = "Optimal strategy") -> Path:
    """Render the solution as a horizontal stint bar coloured by compound."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Reconstruct stints from the solution node sequence.
    stints = []  # (compound, start_lap, end_lap)
    prev_compound = problem.initial_state().compound
    stint_start = 0
    for node in solution[1:]:  # skip root
        if node.action is not None and node.action.type is ActionType.PIT:
            stints.append((prev_compound, stint_start, node.state.lap - 1))
            prev_compound = node.action.compound
            stint_start = node.state.lap - 1
    stints.append((prev_compound, stint_start, problem.total_laps))

    fig, ax = plt.subplots(figsize=(13, 2.6))
    for compound, start, end in stints:
        ax.barh(0, end - start, left=start, height=0.6,
                color=_COMPOUND_COLOURS.get(compound, "#888"),
                edgecolor="black")
        ax.text((start + end) / 2, 0, compound.value, ha="center", va="center",
                fontsize=9, fontweight="bold",
                color="black" if compound in (Compound.MEDIUM, Compound.HARD) else "white")
    # Mark pit stops.
    for _, start, _ in stints[1:]:
        ax.axvline(start, color="black", linestyle="--", alpha=0.6)
        ax.text(start, 0.42, "PIT", ha="center", fontsize=8, color="black")

    ax.set_xlim(0, problem.total_laps)
    ax.set_ylim(-0.5, 0.6)
    ax.set_yticks([])
    ax.set_xlabel("Lap")
    ax.set_title(f"{title} — {len(stints) - 1} stop(s), "
                 f"{sum(1 for _ in stints)} stint(s)", fontsize=12,
                 fontweight="bold")
    legend = [mpatches.Patch(color=c, label=k.value)
              for k, c in _COMPOUND_COLOURS.items()
              if any(s[0] == k for s in stints)]
    ax.legend(handles=legend, loc="upper right", ncol=len(legend), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def render_search_tree(problem: RaceProblem, result: SearchResult,
                       path: str | Path, max_depth: int = 4) -> Path:
    """
    Render a depth-limited slice of the search tree explored to reach the goal.

    To keep the figure legible we re-expand from the root breadth-first up to
    ``max_depth`` and highlight the nodes that lie on the solution path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    g = nx.DiGraph()
    solution_keys = {n.state.key() for n in result.solution}

    # Build a shallow tree for visualisation.
    from collections import deque
    root = problem.initial_state()
    root_id = "L0"
    g.add_node(root_id, label=f"L{root.lap}\n{root.compound.value}",
               on_path=root.key() in solution_keys)
    frontier = deque([(root, root_id, 0)])
    counter = 0
    seen = {root.key()}
    while frontier:
        state, nid, depth = frontier.popleft()
        if depth >= max_depth:
            continue
        for action in problem.actions(state):
            child = problem.result(state, action)
            counter += 1
            cid = f"n{counter}"
            g.add_node(cid,
                       label=f"L{child.lap}\n{child.compound.value}",
                       on_path=child.key() in solution_keys)
            g.add_edge(nid, cid, label=str(action))
            if child.key() not in seen:
                seen.add(child.key())
                frontier.append((child, cid, depth + 1))

    pos = _hierarchy_pos(g, root_id)
    node_colours = ["#E6194B" if g.nodes[n]["on_path"] else "#A0C4FF"
                    for n in g.nodes]
    fig, ax = plt.subplots(figsize=(16, 10))
    nx.draw_networkx_nodes(g, pos, node_color=node_colours, node_size=700,
                           edgecolors="black", ax=ax)
    nx.draw_networkx_labels(g, pos, labels=nx.get_node_attributes(g, "label"),
                            font_size=6, ax=ax)
    nx.draw_networkx_edges(g, pos, edge_color="#777", arrows=True,
                           arrowsize=8, ax=ax)
    edge_labels = nx.get_edge_attributes(g, "label")
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=5,
                                 ax=ax)
    ax.set_title(f"Search Tree (first {max_depth} levels) — red nodes lie on the "
                 f"{result.algorithm} solution path", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _hierarchy_pos(g: nx.DiGraph, root: str) -> Dict[str, tuple]:
    """Compute a simple top-down tree layout for a DiGraph rooted at ``root``."""
    levels: Dict[str, int] = {root: 0}
    order: Dict[int, List[str]] = {0: [root]}
    from collections import deque
    q = deque([root])
    while q:
        n = q.popleft()
        for child in g.successors(n):
            if child not in levels:
                levels[child] = levels[n] + 1
                order.setdefault(levels[child], []).append(child)
                q.append(child)
    pos: Dict[str, tuple] = {}
    for depth, nodes in order.items():
        count = len(nodes)
        for i, n in enumerate(nodes):
            x = (i - (count - 1) / 2.0)
            pos[n] = (x, -depth)
    return pos
