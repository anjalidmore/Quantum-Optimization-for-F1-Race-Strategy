"""
Symbolic-reasoning endpoints — Tasks 1, 2 and 3.

These three tasks do not sit in the ML data pipeline, so they had no dashboard
page of their own until Task 9. Everything below is read from the committed
artifacts or computed from the same modules that generated them:

* **Task 1** counts come from ``knowledge_representation.schema``, which is the
  definition the ontology and graph were built from.
* **Task 2** reads ``artifacts/expert_system/rules/rule_base.json``.
* **Task 3** reads ``artifacts/search/reports/comparison.json``.

Where an artifact is missing the response says so, so the dashboard can print
"Not generated yet" instead of a number nobody produced.
"""
from __future__ import annotations

import json

from fastapi import APIRouter

from app.core.paths import (
    EXPERT_SYSTEM_ARTIFACTS_DIR,
    KNOWLEDGE_REPRESENTATION_ARTIFACTS_DIR,
    SEARCH_ARTIFACTS_DIR,
)

router = APIRouter(prefix="/api/reasoning", tags=["reasoning"])

RULE_BASE_JSON = EXPERT_SYSTEM_ARTIFACTS_DIR / "rules" / "rule_base.json"
SEARCH_JSON = SEARCH_ARTIFACTS_DIR / "reports" / "comparison.json"


def _missing(what: str, command: str) -> dict:
    return {"available": False, "reason": f"{what} not generated yet. Run: {command}"}


@router.get("/knowledge")
def knowledge() -> dict:
    """Task 1 — the ontology and knowledge graph, counted from the schema itself."""
    from app.intelligence.knowledge_representation import schema

    entities = schema.entities_by_name()
    if not entities:
        return _missing("Knowledge representation", "python scripts/build_knowledge_base.py")

    by_category: dict[str, int] = {}
    for entity in entities.values():
        by_category[entity.category] = by_category.get(entity.category, 0) + 1

    ontology = KNOWLEDGE_REPRESENTATION_ARTIFACTS_DIR / "ontology" / "formula1.owl"
    graph_dir = KNOWLEDGE_REPRESENTATION_ARTIFACTS_DIR / "graph"
    return {
        "available": True,
        "n_entities": len(entities),
        "n_relationships": len(schema.relationship_names()),
        "n_attributes": sum(len(e.attributes) for e in entities.values()),
        "entities_by_category": by_category,
        "categories": sorted(by_category),
        "relationships": schema.relationship_names(),
        "files": {
            "ontology_owl": ontology.exists(),
            "instance_graph_ttl": (graph_dir / "instance_graph.ttl").exists(),
            "instance_graph_graphml": (graph_dir / "instance_graph.graphml").exists(),
        },
    }


@router.get("/expert-system")
def expert_system() -> dict:
    """Task 2 — the rule base, read from the artifact the build wrote."""
    if not RULE_BASE_JSON.exists():
        return _missing("Expert system rule base", "python scripts/run_expert_system.py")

    rules = json.loads(RULE_BASE_JSON.read_text())
    if isinstance(rules, dict):
        rules = rules.get("rules", [])

    by_category: dict[str, int] = {}
    for rule in rules:
        by_category[rule["category"]] = by_category.get(rule["category"], 0) + 1

    return {
        "available": True,
        "n_rules": len(rules),
        "rules_by_category": by_category,
        "salience_levels": sorted({r["salience"] for r in rules}, reverse=True),
        "rules": [
            {
                "rule_id": r["rule_id"],
                "name": r["name"],
                "category": r["category"],
                "salience": r["salience"],
                "description": r.get("description", ""),
                "n_conditions": len(r.get("conditions", [])),
                "actions": [a["key"] for a in r.get("actions", [])],
            }
            for r in rules
        ],
    }


@router.get("/search")
def search() -> dict:
    """Task 3 — the five algorithms on one race-strategy problem instance."""
    if not SEARCH_JSON.exists():
        return _missing("Search comparison", "python scripts/run_search.py")

    data = json.loads(SEARCH_JSON.read_text())
    plan = data.get("optimal_plan", [])
    return {
        "available": True,
        "problem": data["problem"],
        "algorithms": data["algorithms"],
        "summary": data["summary"],
        # The full 24-lap plan is long; the pit stops are the decision points.
        "pit_stops": [step for step in plan if step["type"] != "RUN"],
        "n_plan_steps": len(plan),
        "final_cost_seconds": plan[-1]["cumulative_cost_seconds"] if plan else None,
    }
