"""
app.api.routers.tasks
========================

A single, honest source of truth for "what has this project actually built
for Task N" — used by the frontend's Dashboard (task cards) and Project
Evidence pages. Every artifact listed here is discovered by scanning the
real ``artifacts/`` directories at request time; nothing is a hard-coded
filename that could silently go stale or claim a deliverable that was never
generated. If a task's output directory doesn't exist, its artifact lists
are simply empty and its status is reported honestly, not padded out.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from app.core.paths import (
    ARTIFACTS_DIR,
    DATA_ENGINEERING_ARTIFACTS_DIR,
    EXPERT_SYSTEM_ARTIFACTS_DIR,
    KNOWLEDGE_REPRESENTATION_ARTIFACTS_DIR,
    ML_METADATA_DIR,
    REPO_ROOT,
    SEARCH_ARTIFACTS_DIR,
    TASK5_FEATURE_METADATA_JSON,
    TASK5_FEATURES_CSV,
)

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def _scan(base: Path) -> dict:
    reports: list[str] = []
    figures: list[str] = []
    other: list[str] = []

    if (base / "reports").is_dir():
        reports = [_rel(p) for p in sorted((base / "reports").glob("*.md"))]

    for sub in ("figures", "diagrams"):
        d = base / sub
        if d.is_dir():
            figures += [_rel(p) for p in sorted(d.glob("*.png"))]

    for sub in ("graph", "ontology", "rules", "clean"):
        d = base / sub
        if d.is_dir():
            other += [_rel(p) for p in sorted(d.iterdir()) if p.is_file()]

    return {"reports": reports, "figures": figures, "other_artifacts": other}


def _task1() -> dict:
    scanned = _scan(KNOWLEDGE_REPRESENTATION_ARTIFACTS_DIR)
    built = bool(scanned["reports"])
    return {
        "id": "task1",
        "number": 1,
        "label": "Knowledge Representation",
        "purpose": "Represent the F1 domain using entities, attributes, relationships, an ontology, and a knowledge graph.",
        "status": "completed" if built else "upcoming",
        **scanned,
    }


def _task2() -> dict:
    scanned = _scan(EXPERT_SYSTEM_ARTIFACTS_DIR)
    built = bool(scanned["reports"])
    return {
        "id": "task2",
        "number": 2,
        "label": "Rule-Based Expert System",
        "purpose": "Forward/backward-chaining inference over production rules for race-strategy decisions, with full explanations.",
        "status": "completed" if built else "upcoming",
        **scanned,
    }


def _task3() -> dict:
    scanned = _scan(SEARCH_ARTIFACTS_DIR)
    built = bool(scanned["reports"])
    return {
        "id": "task3",
        "number": 3,
        "label": "State-Space Search",
        "purpose": "Formulate race strategy as a shortest-path problem and solve it with BFS, DFS, UCS, Greedy Best-First and A*.",
        "status": "completed" if built else "upcoming",
        **scanned,
    }


def _task4() -> dict:
    scanned = _scan(DATA_ENGINEERING_ARTIFACTS_DIR)
    built = bool(scanned["reports"])
    return {
        "id": "task4",
        "number": 4,
        "label": "Data Preparation & EDA",
        "purpose": "Clean, validate and statistically analyse the F1 lap/session data that everything downstream is built on.",
        "status": "completed" if built else "upcoming",
        **scanned,
    }


def _task5() -> dict:
    built = TASK5_FEATURES_CSV.exists() and TASK5_FEATURE_METADATA_JSON.exists()
    other = [_rel(TASK5_FEATURES_CSV), _rel(TASK5_FEATURE_METADATA_JSON)] if built else []
    notebook = REPO_ROOT / "docs" / "notebooks" / "task5_feature_engineering.ipynb"
    if notebook.exists():
        other.append(_rel(notebook))
    return {
        "id": "task5",
        "number": 5,
        "label": "Feature Engineering & Selection",
        "purpose": "Engineer causal, leakage-free features and select a compact, validated set for Task 6's models.",
        "status": "completed" if built else "upcoming",
        "reports": [],
        "figures": [],
        "other_artifacts": other,
    }


def _task6() -> dict:
    built = ML_METADATA_DIR.joinpath("model_registry.json").exists()
    scanned = _scan(ARTIFACTS_DIR)  # figures/ and reports/ live directly under artifacts/ for Task 6
    other = []
    if built:
        other.append(_rel(ML_METADATA_DIR / "model_registry.json"))
    return {
        "id": "task6",
        "number": 6,
        "label": "Machine Learning",
        "purpose": "Train, evaluate, select and persist classical ML models for lap-time regression and pit-decision classification.",
        "status": "completed" if built else "upcoming",
        "reports": scanned["reports"],
        "figures": scanned["figures"],
        "other_artifacts": other,
    }


_UPCOMING = [
    {
        "id": "task7", "number": 7, "label": "Deep Learning",
        "purpose": "Neural models for race-performance prediction.",
        "status": "upcoming", "reports": [], "figures": [], "other_artifacts": [],
    },
    {
        "id": "task8", "number": 8, "label": "Explainable AI",
        "purpose": "SHAP/LIME/counterfactual explanations and trust scores for the trained models.",
        "status": "upcoming", "reports": [], "figures": [], "other_artifacts": [],
    },
    {
        "id": "task9", "number": 9, "label": "System Integration & Deployment",
        "purpose": "A unified strategy engine combining every computational-intelligence capability.",
        "status": "upcoming", "reports": [], "figures": [], "other_artifacts": [],
    },
    {
        "id": "task10", "number": 10, "label": "Evaluation & Responsible AI",
        "purpose": "System-wide evaluation: performance, usability, explainability, robustness, documentation.",
        "status": "upcoming", "reports": [], "figures": [], "other_artifacts": [],
    },
]


@router.get("/evidence")
def get_task_evidence():
    tasks = [_task1(), _task2(), _task3(), _task4(), _task5(), _task6(), *_UPCOMING]
    completed = sum(1 for t in tasks if t["status"] == "completed")
    return {"tasks": tasks, "completed_count": completed, "total_count": len(tasks)}
