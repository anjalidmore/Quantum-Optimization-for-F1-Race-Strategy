"""
Quantum ML endpoints.

Every response is read from the committed run (``artifacts/metrics/qml_metrics.json``),
which ``app.intelligence.qml.pipeline`` produced — the same file the report is
written from, so the dashboard and the report cannot disagree.

Trained weights live under ``artifacts/models/``, which the API does not serve.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query

from app.core.paths import QML_METRICS_JSON

router = APIRouter(prefix="/api/qml", tags=["quantum-ml"])


def _results() -> dict:
    if not QML_METRICS_JSON.exists():
        raise HTTPException(
            status_code=404,
            detail="No quantum results yet. Run scripts/run_qml.py (or build_all.py without --skip-qml).",
        )
    return json.loads(QML_METRICS_JSON.read_text())


@router.get("/summary")
def summary() -> dict:
    """Headline: what was run, on what, and how each model scored."""
    r = _results()
    targets = {}
    for target, t in r["targets"].items():
        key = "pr_auc" if t["task"] == "classification" else "mae"
        rows = []
        for m in t["quantum"] + t["classical_matched"]:
            cv = (m.get("cv_summary") or {}).get(key) or {}
            rows.append({
                "model": m["model"],
                "family": m["family"],
                "description": m["description"],
                "n_parameters": m["n_parameters"],
                "train_seconds": m.get("train_seconds"),
                "cv_mean": cv.get("mean"),
                "cv_std": cv.get("std"),
                "test_metrics": m["test_metrics"],
            })
        ref = t.get("task6_reference") or {}
        targets[target] = {
            "task": t["task"],
            "selection_metric": key,
            "encoding": t["encoding"],
            "n_dev": t["n_dev"],
            "n_test": t["n_test"],
            "n_test_positive": t.get("n_test_positive"),
            "n_folds": t["n_folds"],
            "notes": t["notes"],
            "models": rows,
            "task6_reference": {
                "model": ref.get("model"),
                "test_metrics": ref.get("test_metrics"),
                "cv_summary": ref.get("cv_summary"),
            } if ref.get("available") else None,
        }
    return {
        "generated_at": r["generated_at"],
        "simulator": r["simulator"],
        "framework": r["framework"],
        "seed": r["seed"],
        "search_space": r["search_space"],
        "dataset_source": r["dataset_source"],
        "wall_seconds": r["wall_seconds"],
        "figures": r.get("figures", {}),
        "report": "reports/classical_vs_quantum_report.md",
        "targets": targets,
        "honesty_note": (
            "Noiseless simulation of small circuits on one race. No quantum hardware is involved "
            "and no quantum advantage is claimed."
        ),
    }


@router.get("/training")
def training(target: str = Query(..., description="target_laptime or target_pit_next_lap")) -> dict:
    """Per-epoch training loss for the variational models of one target."""
    r = _results()
    if target not in r["targets"]:
        raise HTTPException(status_code=404, detail=f"Unknown target {target!r}")
    t = r["targets"][target]
    return {
        "target": target,
        "curves": [
            {"model": m["model"], "loss": m["loss_history"], "epochs": len(m["loss_history"]),
             "hyperparameters": m["hyperparameters"]}
            for m in t["quantum"] if m.get("loss_history")
        ],
    }


@router.get("/search")
def search(target: str = Query(...)) -> dict:
    """Every hyperparameter configuration tried, and its cross-validated score."""
    r = _results()
    if target not in r["targets"]:
        raise HTTPException(status_code=404, detail=f"Unknown target {target!r}")
    return {"target": target, "search_space": r["search_space"],
            "trials": r["targets"][target]["search_trials"]}
