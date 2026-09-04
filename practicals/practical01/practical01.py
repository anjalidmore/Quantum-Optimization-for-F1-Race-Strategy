#!/usr/bin/env python3
"""
practical01.py
==============

End-to-end driver for Phase 1 / Task 1 (Knowledge Representation).

Running this script produces *every* Task-1 deliverable into ``outputs/``:

1. Builds the OWL ontology                -> outputs/ontology/formula1.owl
2. Builds schema + instance graphs        -> (in memory) + GraphML/RDF exports
3. Exports the instance graph as RDF/TTL  -> outputs/graph/instance_graph.ttl
4. Renders visualisations (PNG + Mermaid) -> outputs/diagrams/*
5. Runs the full validation suite         -> (drives the validation report)
6. Generates all Markdown reports/tables  -> outputs/reports/*

The script is idempotent and headless; it exits non-zero if any validation
check fails, so it doubles as a CI gate.

Usage
-----
    python practical01.py [--output-dir OUTPUTS]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import networkx as nx

# Ensure ``src`` is importable whether run from repo root or the phase folder.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))

from f1kr import ontology as onto_mod  # noqa: E402
from f1kr import knowledge_graph as kg  # noqa: E402
from f1kr import reports as rep  # noqa: E402
from f1kr import validation as val  # noqa: E402
from f1kr import visualize as viz  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("practical01")


def main(output_dir: Path) -> int:
    output_dir = output_dir.resolve()
    diagrams = output_dir / "diagrams"
    reports = output_dir / "reports"
    onto_dir = output_dir / "ontology"
    graph_dir = output_dir / "graph"
    for d in (diagrams, reports, onto_dir, graph_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- 1. Ontology -------------------------------------------------------
    log.info("Building OWL ontology ...")
    onto = onto_mod.build_ontology(onto_dir / "formula1.owl")
    stats = onto_mod.ontology_statistics(onto)
    log.info("Ontology built: %s classes, %s object props, %s data props",
             stats["classes"], stats["object_properties"], stats["data_properties"])

    # --- 2. Knowledge graphs ----------------------------------------------
    log.info("Building schema and instance knowledge graphs ...")
    schema_graph = kg.build_schema_graph()
    instance_graph = kg.build_instance_graph()
    log.info("Schema graph: %s", kg.graph_summary(schema_graph))
    log.info("Instance graph: %s", kg.graph_summary(instance_graph))

    # GraphML exports (portable, opens in Gephi/yEd).
    nx.write_graphml(schema_graph, graph_dir / "schema_graph.graphml")
    nx.write_graphml(instance_graph, graph_dir / "instance_graph.graphml")

    # --- 3. RDF export -----------------------------------------------------
    log.info("Exporting instance graph as RDF (Turtle) ...")
    kg.export_rdf(graph_dir / "instance_graph.ttl", fmt="turtle")

    # --- 4. Visualisations -------------------------------------------------
    log.info("Rendering visualisations ...")
    viz.render_schema_png(schema_graph, diagrams / "schema_graph.png")
    viz.render_instance_png(instance_graph, diagrams / "instance_graph.png")
    viz.render_schema_mermaid(schema_graph, diagrams / "schema_graph.mmd")

    # --- 5. Validation -----------------------------------------------------
    log.info("Running validation suite ...")
    results = val.run_all_checks(instance_graph=instance_graph, ontology_stats=stats)
    for r in results:
        log.info("  [%s] %s — %s", "PASS" if r.passed else "FAIL", r.name, r.detail)

    # --- 6. Reports --------------------------------------------------------
    log.info("Generating Markdown reports and tables ...")
    written = rep.generate_all(reports, stats, results)
    for name, path in written.items():
        log.info("  wrote %s -> %s", name, path.relative_to(output_dir))

    if not val.all_passed(results):
        log.error("Validation FAILED. See %s", reports / "validation_report.md")
        return 1

    log.info("All Task-1 deliverables generated successfully in %s", output_dir)
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the F1 knowledge representation.")
    p.add_argument("--output-dir", type=Path, default=_HERE / "outputs",
                   help="Directory into which deliverables are written.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(main(args.output_dir))
