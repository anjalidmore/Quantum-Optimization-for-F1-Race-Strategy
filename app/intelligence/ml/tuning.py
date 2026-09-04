"""
app.intelligence.ml.tuning
=============================

A small, explicit expanding-window grid-search engine shared by regression
and classification.

Why not ``sklearn.model_selection.GridSearchCV`` directly: it works fine
with a custom ``cv`` iterable of ``(train_idx, val_idx)`` pairs, but its
NaN-aggregation behaviour when a fold's scorer legitimately cannot be
computed (single-class validation fold — see ``evaluation.py``) is opaque
and version-dependent. Given how small this dataset is (a handful of CV
folds, some of which are expected to be degenerate for the pit-decision
target), an explicit loop is more transparent and testable: every
parameter combination's per-fold outcome — including *why* a fold produced
no score — is retained for the model-selection report, satisfying the
"selection process must be transparent" requirement.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Callable

from sklearn.base import clone

from app.intelligence.ml.splits import Fold


@dataclass
class ParamCandidateResult:
    params: dict
    fold_metrics: list[dict]
    cv_summary: dict
    primary_score: float | None  # None if unusable on every fold


@dataclass
class CVSearchResult:
    model_name: str
    task: str
    best_params: dict
    best_fold_metrics: list[dict]
    best_cv_summary: dict
    all_candidates: list[ParamCandidateResult]
    fit_seconds: float
    final_pipeline: object = field(repr=False)


def _expand_grid(param_grid: dict[str, list]) -> list[dict]:
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    combos = itertools.product(*(param_grid[k] for k in keys))
    return [dict(zip(keys, values)) for values in combos]


def run_expanding_window_search(
    *,
    model_name: str,
    task: str,
    build_pipeline: Callable[[], object],
    param_grid: dict[str, list],
    X_dev,
    y_dev,
    folds: list[Fold],
    metrics_fn: Callable[..., dict],
    metric_keys: list[str],
    primary_metric: str,
    higher_is_better: bool,
    needs_proba: bool,
    aggregate_fn: Callable[[list[dict], list[str]], dict],
) -> CVSearchResult:
    """Try every combination in ``param_grid`` across every fold, pick the
    combination with the best mean primary metric (folds where the metric
    is undefined are excluded from that mean, not treated as zero), then
    refit the winning configuration on the *entire* development set.
    """
    candidates: list[ParamCandidateResult] = []

    for params in _expand_grid(param_grid):
        fold_metrics = []
        for fold in folds:
            X_train, y_train = X_dev.iloc[fold.train_index], y_dev.iloc[fold.train_index]
            X_val, y_val = X_dev.iloc[fold.val_index], y_dev.iloc[fold.val_index]

            pipeline = clone(build_pipeline())
            pipeline.set_params(**{f"model__{k}": v for k, v in params.items()})

            record = {"fold_id": fold.fold_id, "train_rows": len(X_train), "val_rows": len(X_val)}
            try:
                pipeline.fit(X_train, y_train)
            except Exception as exc:  # e.g. a single class in the training fold
                record["error"] = f"{type(exc).__name__}: {exc}"
                for key in metric_keys:
                    record[key] = None
                fold_metrics.append(record)
                continue

            y_pred = pipeline.predict(X_val)
            y_proba = None
            if needs_proba:
                try:
                    y_proba = pipeline.predict_proba(X_val)[:, 1]
                except Exception:
                    y_proba = None

            metrics = metrics_fn(y_val, y_pred, y_proba) if needs_proba else metrics_fn(y_val, y_pred)
            record.update(metrics)
            fold_metrics.append(record)

        cv_summary = aggregate_fn(fold_metrics, metric_keys)
        primary = cv_summary.get(primary_metric, {}).get("mean")
        candidates.append(
            ParamCandidateResult(
                params=params, fold_metrics=fold_metrics, cv_summary=cv_summary, primary_score=primary
            )
        )

    scored = [c for c in candidates if c.primary_score is not None]
    if scored:
        best = max(scored, key=lambda c: c.primary_score if higher_is_better else -c.primary_score)
    else:
        # Every candidate was unusable on every fold (shouldn't happen in
        # practice, but never silently fabricate a winner) — fall back to
        # the first combination so the pipeline can still be persisted, and
        # let the honest (empty) metrics speak for themselves downstream.
        best = candidates[0]

    start = time.perf_counter()
    final_pipeline = clone(build_pipeline())
    final_pipeline.set_params(**{f"model__{k}": v for k, v in best.params.items()})
    final_pipeline.fit(X_dev, y_dev)
    fit_seconds = time.perf_counter() - start

    return CVSearchResult(
        model_name=model_name,
        task=task,
        best_params=best.params,
        best_fold_metrics=best.fold_metrics,
        best_cv_summary=best.cv_summary,
        all_candidates=candidates,
        fit_seconds=fit_seconds,
        final_pipeline=final_pipeline,
    )
