"""
Unit tests for the f1kr knowledge-representation package (Phase 1, Task 1).

Run with:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pytest

# Make ``src`` importable regardless of the working directory.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from app.intelligence.knowledge_representation import knowledge_graph as kg
from app.intelligence.knowledge_representation import ontology as onto_mod
from app.intelligence.knowledge_representation import reports as rep
from app.intelligence.knowledge_representation import validation as val
from app.intelligence.knowledge_representation import visualize as viz
from app.intelligence.knowledge_representation.schema import ENTITIES, RELATIONSHIPS, entities_by_name


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

def test_entities_and_relationships_nonempty():
    assert len(ENTITIES) >= 50
    assert len(RELATIONSHIPS) >= 25


def test_entity_names_unique():
    names = [e.name for e in ENTITIES]
    assert len(names) == len(set(names))


def test_relationship_names_unique():
    names = [r.name for r in RELATIONSHIPS]
    assert len(names) == len(set(names))


def test_relationship_endpoints_declared():
    known = set(entities_by_name())
    for r in RELATIONSHIPS:
        assert r.domain in known, f"{r.name} has undeclared domain {r.domain}"
        assert r.range in known, f"{r.name} has undeclared range {r.range}"


def test_parents_are_declared_entities():
    known = set(entities_by_name())
    for e in ENTITIES:
        if e.parent:
            assert e.parent in known


# --------------------------------------------------------------------------- #
# Ontology
# --------------------------------------------------------------------------- #

def test_ontology_builds_and_counts(tmp_path):
    onto = onto_mod.build_ontology(tmp_path / "onto.owl")
    stats = onto_mod.ontology_statistics(onto)
    # One class per entity plus one root per category.
    assert stats["classes"] >= len(ENTITIES)
    assert stats["object_properties"] == len(RELATIONSHIPS)
    assert (tmp_path / "onto.owl").exists()
    assert (tmp_path / "onto.owl").stat().st_size > 0


def test_ontology_no_name_collisions(tmp_path):
    """The Weather/Session/Telemetry/TrackStatus category-vs-entity collision
    must not recur — building must not raise."""
    onto_mod.build_ontology(tmp_path / "onto.owl")  # would raise on cycle


# --------------------------------------------------------------------------- #
# Knowledge graphs
# --------------------------------------------------------------------------- #

def test_schema_graph_structure():
    g = kg.build_schema_graph()
    assert g.number_of_nodes() == len(ENTITIES)
    assert g.number_of_edges() == len(RELATIONSHIPS)
    # Every edge key must be a declared relationship name.
    rel_names = {r.name for r in RELATIONSHIPS}
    for _, _, k in g.edges(keys=True):
        assert k in rel_names


def test_instance_graph_connected():
    g = kg.build_instance_graph()
    assert g.number_of_nodes() > 0
    # The sample scenario is intentionally a single connected component.
    assert nx.number_weakly_connected_components(g) == 1


def test_instance_rdf_export_roundtrip(tmp_path):
    path = kg.export_rdf(tmp_path / "inst.ttl", fmt="turtle")
    assert path.exists()
    from rdflib import Graph as RDFGraph
    reloaded = RDFGraph().parse(str(path), format="turtle")
    assert len(reloaded) > 0


def test_graph_summary_keys():
    g = kg.build_schema_graph()
    s = kg.graph_summary(g)
    assert {"name", "nodes", "edges", "is_dag", "weakly_connected"} <= set(s)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def test_full_validation_passes(tmp_path):
    onto = onto_mod.build_ontology(tmp_path / "onto.owl")
    stats = onto_mod.ontology_statistics(onto)
    inst = kg.build_instance_graph()
    results = val.run_all_checks(instance_graph=inst, ontology_stats=stats)
    assert val.all_passed(results), [
        (r.name, r.detail) for r in results if not r.passed
    ]


def test_validation_detects_bad_instance_edge():
    """A deliberately malformed edge must be caught."""
    g = kg.build_instance_graph()
    # Add an edge using a relationship between incompatible types.
    g.add_node("bogus", entity_type="Circuit")
    g.add_edge("bogus", "driver_verstappen", key="drives_for", name="drives_for")
    results = val.check_instance_graph(g)
    assert any(not r.passed for r in results)


# --------------------------------------------------------------------------- #
# Reports & visualisations
# --------------------------------------------------------------------------- #

def test_report_generation(tmp_path):
    onto = onto_mod.build_ontology(tmp_path / "onto.owl")
    stats = onto_mod.ontology_statistics(onto)
    inst = kg.build_instance_graph()
    results = val.run_all_checks(instance_graph=inst, ontology_stats=stats)
    written = rep.generate_all(tmp_path, stats, results)
    for name, path in written.items():
        assert path.exists(), name
        assert path.stat().st_size > 0, name
    # Entity table must mention a known entity.
    assert "Driver" in (tmp_path / "entity_table.md").read_text()


def test_visualisations_render(tmp_path):
    schema = kg.build_schema_graph()
    inst = kg.build_instance_graph()
    p1 = viz.render_schema_png(schema, tmp_path / "schema.png")
    p2 = viz.render_instance_png(inst, tmp_path / "instance.png")
    p3 = viz.render_schema_mermaid(schema, tmp_path / "schema.mmd")
    for p in (p1, p2, p3):
        assert p.exists() and p.stat().st_size > 0
    assert "graph LR" in p3.read_text()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
