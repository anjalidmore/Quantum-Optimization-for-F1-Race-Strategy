"""
f1kr — Formula 1 Knowledge Representation (Phase 1, Task 1)
==========================================================

A production-grade knowledge-representation layer for the *Quantum Optimization
for Formula 1 Race Strategy* project. This package provides:

* ``schema``          — declarative single-source-of-truth domain model.
* ``ontology``        — OWL 2 ontology builder (owlready2).
* ``knowledge_graph`` — NetworkX schema + instance graphs, with RDF export.
* ``validation``      — schema / ontology / instance integrity checks.
* ``visualize``       — PNG and Mermaid renderings of the graphs.
* ``reports``         — Markdown deliverable generators.

Everything is classical: no quantum computing appears in Phase 1.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = [
    "schema",
    "ontology",
    "knowledge_graph",
    "validation",
    "visualize",
    "reports",
]
