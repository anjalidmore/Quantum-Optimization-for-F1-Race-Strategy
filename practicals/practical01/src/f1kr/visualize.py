"""
f1kr.visualize
==============

Renders the knowledge graph for human inspection.

Two outputs are produced:

* A **PNG** rendering of a graph using ``matplotlib`` + ``networkx`` spring
  layout, with nodes coloured by category (schema graph) or entity type
  (instance graph).
* A **Mermaid** diagram source (``.mmd``) for the schema graph, suitable for
  embedding in Markdown documentation and rendering on GitHub.

``matplotlib`` uses the non-interactive ``Agg`` backend so this runs headless.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

from .schema import categories  # noqa: E402

# A stable, colour-blind-friendly palette assigned to categories.
_PALETTE = [
    "#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4", "#42D4F4",
    "#F032E6", "#BFEF45", "#FABED4", "#469990", "#DCBEFF", "#9A6324",
    "#800000", "#AAFFC3", "#808000", "#000075",
]


def _category_colours() -> Dict[str, str]:
    return {cat: _PALETTE[i % len(_PALETTE)] for i, cat in enumerate(categories())}


def render_schema_png(schema_graph: nx.MultiDiGraph, path: str | Path) -> Path:
    """Render the schema graph to a PNG, coloured by entity category."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    colours = _category_colours()

    pos = nx.spring_layout(schema_graph, seed=42, k=0.9, iterations=200)
    node_colours = [
        colours.get(schema_graph.nodes[n].get("category", ""), "#888888")
        for n in schema_graph.nodes
    ]

    fig, ax = plt.subplots(figsize=(22, 16))
    nx.draw_networkx_nodes(schema_graph, pos, node_color=node_colours,
                           node_size=900, alpha=0.95, ax=ax)
    nx.draw_networkx_labels(schema_graph, pos, font_size=7, ax=ax)
    nx.draw_networkx_edges(schema_graph, pos, edge_color="#555555", alpha=0.4,
                           arrows=True, arrowsize=8, connectionstyle="arc3,rad=0.08",
                           ax=ax)
    # Legend for categories.
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=cat,
                   markerfacecolor=col, markersize=10)
        for cat, col in colours.items()
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9, title="Entity category")
    ax.set_title("Formula 1 Knowledge Schema Graph (Entities & Relationships)",
                 fontsize=16)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def render_instance_png(instance_graph: nx.MultiDiGraph, path: str | Path) -> Path:
    """Render the populated instance graph to a PNG with edge labels."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pos = nx.spring_layout(instance_graph, seed=7, k=1.2, iterations=200)
    fig, ax = plt.subplots(figsize=(18, 13))
    nx.draw_networkx_nodes(instance_graph, pos, node_color="#4363D8",
                           node_size=1400, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(instance_graph, pos, font_size=7,
                            font_color="white", ax=ax)
    nx.draw_networkx_edges(instance_graph, pos, edge_color="#333333", alpha=0.6,
                           arrows=True, arrowsize=14,
                           connectionstyle="arc3,rad=0.1", ax=ax)
    edge_labels = {
        (u, v): k for u, v, k in instance_graph.edges(keys=True)
    }
    nx.draw_networkx_edge_labels(instance_graph, pos, edge_labels=edge_labels,
                                 font_size=6, ax=ax, label_pos=0.5,
                                 bbox=dict(boxstyle="round", fc="white", ec="none",
                                           alpha=0.7))
    ax.set_title("Formula 1 Knowledge Instance Graph — Monaco GP 2023 (sample)",
                 fontsize=15)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def render_schema_mermaid(schema_graph: nx.MultiDiGraph, path: str | Path) -> Path:
    """Emit a Mermaid ``graph LR`` source file for the schema graph."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def safe(n: str) -> str:
        return n.replace(" ", "_")

    lines = ["graph LR"]
    # Nodes grouped in subgraphs by category for readability.
    by_cat: Dict[str, list] = {}
    for n, d in schema_graph.nodes(data=True):
        by_cat.setdefault(d.get("category", "Other"), []).append(n)
    for cat, nodes in by_cat.items():
        lines.append(f"  subgraph {safe(cat)}")
        for n in nodes:
            lines.append(f"    {safe(n)}[\"{n}\"]")
        lines.append("  end")
    for u, v, k in schema_graph.edges(keys=True):
        lines.append(f"  {safe(u)} -->|{k}| {safe(v)}")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
