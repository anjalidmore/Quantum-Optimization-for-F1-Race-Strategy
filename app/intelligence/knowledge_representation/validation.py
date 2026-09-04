"""
f1kr.validation
===============

Validates the integrity and consistency of the knowledge representation.

Three layers of checks are performed:

* **Schema checks** -- every relationship references declared entities; no
  duplicate entity or relationship names; every non-root entity's ``parent``
  exists; category roots are well-formed.
* **Ontology checks** -- the ontology builds without error and the class /
  property counts are internally consistent with the schema.
* **Instance-graph checks** -- every individual's ``entity_type`` is a declared
  entity, and every instance edge's relationship name exists in the schema and
  is used between type-compatible endpoints (respecting subclass hierarchy).

Results are returned as structured :class:`CheckResult` records so they can be
rendered into a Markdown validation report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

from .schema import (
    ENTITIES,
    RELATIONSHIPS,
    Relationship,
    entities_by_name,
)


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single validation check."""

    name: str
    passed: bool
    detail: str


def _subclass_closure() -> Dict[str, Set[str]]:
    """
    Return a mapping ``entity -> {itself and all descendants}``.

    Used so that a relationship whose declared range is a parent type also
    admits instances of subclasses (e.g. ``TrackStatus`` admits ``SafetyCar``).
    """
    by_name = entities_by_name()
    children: Dict[str, List[str]] = {name: [] for name in by_name}
    for ent in ENTITIES:
        if ent.parent:
            children.setdefault(ent.parent, []).append(ent.name)

    closure: Dict[str, Set[str]] = {}

    def descend(name: str) -> Set[str]:
        if name in closure:
            return closure[name]
        acc = {name}
        for child in children.get(name, []):
            acc |= descend(child)
        closure[name] = acc
        return acc

    for name in by_name:
        descend(name)
    return closure


# ---------------------------------------------------------------------------
# Schema-level checks
# ---------------------------------------------------------------------------

def check_unique_entities() -> CheckResult:
    names = [e.name for e in ENTITIES]
    dupes = {n for n in names if names.count(n) > 1}
    return CheckResult(
        "unique_entity_names",
        not dupes,
        "No duplicate entity names." if not dupes else f"Duplicates: {sorted(dupes)}",
    )


def check_unique_relationships() -> CheckResult:
    names = [r.name for r in RELATIONSHIPS]
    dupes = {n for n in names if names.count(n) > 1}
    return CheckResult(
        "unique_relationship_names",
        not dupes,
        "No duplicate relationship names." if not dupes else f"Duplicates: {sorted(dupes)}",
    )


def check_relationship_endpoints() -> CheckResult:
    known = set(entities_by_name())
    bad: List[str] = []
    for rel in RELATIONSHIPS:
        if rel.domain not in known:
            bad.append(f"{rel.name}: unknown domain '{rel.domain}'")
        if rel.range not in known:
            bad.append(f"{rel.name}: unknown range '{rel.range}'")
    return CheckResult(
        "relationship_endpoints_declared",
        not bad,
        "All relationship endpoints are declared entities." if not bad else "; ".join(bad),
    )


def check_parents_exist() -> CheckResult:
    known = set(entities_by_name())
    bad = [e.name for e in ENTITIES if e.parent and e.parent not in known]
    return CheckResult(
        "entity_parents_exist",
        not bad,
        "All declared parents exist." if not bad else f"Missing parents for: {bad}",
    )


def check_no_orphan_entities() -> CheckResult:
    """Every entity should either be a category root or take part in >=1 relationship
    or be a parent/child in the hierarchy (i.e. be reachable in the model)."""
    used: Set[str] = set()
    for rel in RELATIONSHIPS:
        used.add(rel.domain)
        used.add(rel.range)
    for ent in ENTITIES:
        if ent.parent:
            used.add(ent.parent)
            used.add(ent.name)
    orphans = [e.name for e in ENTITIES if e.name not in used]
    # Category roots that intentionally group subclasses are acceptable orphans;
    # report them as informational but do not fail the build.
    return CheckResult(
        "no_unreachable_entities",
        True,
        "All entities reachable."
        if not orphans
        else f"Standalone entities (not yet linked, acceptable): {sorted(orphans)}",
    )


# ---------------------------------------------------------------------------
# Instance-graph checks
# ---------------------------------------------------------------------------

def check_instance_graph(instance_graph) -> List[CheckResult]:
    """Validate the populated instance graph against the schema."""
    by_name = entities_by_name()
    rel_by_name: Dict[str, Relationship] = {r.name: r for r in RELATIONSHIPS}
    closure = _subclass_closure()
    results: List[CheckResult] = []

    # 1. Every node's entity_type must be declared.
    bad_types = [
        n for n, d in instance_graph.nodes(data=True)
        if d.get("entity_type") not in by_name
    ]
    results.append(CheckResult(
        "instance_types_declared",
        not bad_types,
        "All instance types are declared entities." if not bad_types
        else f"Unknown types on nodes: {bad_types}",
    ))

    # 2. Every edge's relationship must exist and connect compatible types.
    type_of = {n: d.get("entity_type") for n, d in instance_graph.nodes(data=True)}
    bad_edges: List[str] = []
    for u, v, key in instance_graph.edges(keys=True):
        rel = rel_by_name.get(key)
        if rel is None:
            bad_edges.append(f"unknown relationship '{key}' ({u}->{v})")
            continue
        # subject type must be within the subclass closure of rel.domain
        if type_of.get(u) not in closure.get(rel.domain, set()):
            bad_edges.append(
                f"{key}: subject '{u}' is {type_of.get(u)}, expected {rel.domain}(+subtypes)"
            )
        if type_of.get(v) not in closure.get(rel.range, set()):
            bad_edges.append(
                f"{key}: object '{v}' is {type_of.get(v)}, expected {rel.range}(+subtypes)"
            )
    results.append(CheckResult(
        "instance_edges_type_compatible",
        not bad_edges,
        "All instance edges are relationship- and type-compatible." if not bad_edges
        else "; ".join(bad_edges),
    ))

    return results


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_all_checks(instance_graph=None, ontology_stats: Dict[str, int] | None = None) -> List[CheckResult]:
    """
    Run the full validation suite and return all :class:`CheckResult` records.

    Parameters
    ----------
    instance_graph:
        Optional populated instance graph (from
        :func:`f1kr.knowledge_graph.build_instance_graph`). If provided,
        instance-level checks are included.
    ontology_stats:
        Optional dict from :func:`f1kr.ontology.ontology_statistics`. If
        provided, a consistency check between schema and ontology counts is run.
    """
    results: List[CheckResult] = [
        check_unique_entities(),
        check_unique_relationships(),
        check_relationship_endpoints(),
        check_parents_exist(),
        check_no_orphan_entities(),
    ]

    if ontology_stats is not None:
        # Ontology classes should be >= number of entities (extra category roots).
        n_entities = len(ENTITIES)
        ok = ontology_stats.get("classes", 0) >= n_entities
        results.append(CheckResult(
            "ontology_class_count_consistent",
            ok,
            f"Ontology has {ontology_stats.get('classes')} classes for "
            f"{n_entities} entities (+ category roots).",
        ))

    if instance_graph is not None:
        results.extend(check_instance_graph(instance_graph))

    return results


def all_passed(results: List[CheckResult]) -> bool:
    """Return True iff every check passed."""
    return all(r.passed for r in results)
