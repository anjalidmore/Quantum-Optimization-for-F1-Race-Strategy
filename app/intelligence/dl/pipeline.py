"""
app.intelligence.dl.pipeline
============================

Task 7 orchestrator: train, tune, evaluate and persist the deep networks, and
compare them against **Task 6's real persisted results** rather than against
baselines re-fitted for the occasion.

That last point is the difference between this module and its ``task-mode``
counterpart. On the practicals branch each practical is self-contained, so
classical baselines are re-fit there. Here Task 6 has already run and written
``artifacts/metrics/*.json`` and ``artifacts/metadata/model_registry.json``, so
the comparison reads those - the same numbers the Machine Learning dashboard
shows. Both sides of the table are therefore the project's own committed
results, not a private re-run.

The split is Task 6's ``chronological_holdout`` over the same feature contract,
so "the same test set" is literally true and not merely intended.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.core.paths import (
    DL_METRICS_JSON,
    DL_MODELS_DIR,
    ML_METRICS_DIR,
    ML_MODEL_REGISTRY_JSON,
    PROCESSED_DATA_SOURCE_JSON,
    ArtifactPaths,
    ensure_dirs,
)
from app.intelligence.dl import models as models_mod
from app.intelligence.dl import persistence, training, tuning, visualize
from app.intelligence.ml.data_contract import build_task_frame, load_and_validate
from app.intelligence.ml.evaluation import classification_metrics, regression_metrics
from app.intelligence.ml.splits import chronological_holdout, expanding_window_folds
from app.intelligence.ml.threshold import DEFAULT_THRESHOLD, tune_threshold
from app.intelligence.ml.threshold import apply as apply_threshold

log = logging.getLogger(__name__)

TARGETS = {
    "target_laptime": "regression",
    "target_pit_next_lap": "classification",
}


def _data_source() -> dict:
    """The provenance marker Task 4/5 propagate. Task 7 and 8 must report the
    same synthetic-vs-real status as Task 6 rather than assuming one."""
    if PROCESSED_DATA_SOURCE_JSON.exists():
        return json.loads(PROCESSED_DATA_SOURCE_JSON.read_text())
    return {"source": "unknown", "reason": f"no marker at {PROCESSED_DATA_SOURCE_JSON}"}


def _numeric_mask(features: list[str], contract) -> np.ndarray:
    binary = set(contract.binary_features_no_scaling_needed)
    return np.array([f not in binary for f in features], dtype=bool)


def _task6_holdout_metrics(target: str) -> dict:
    """Task 6's own committed test metrics for this target, read from the
    artifacts the ML dashboard serves. Returns {} if Task 6 has not run."""
    name = "regression_metrics.json" if target == "target_laptime" else "classification_metrics.json"
    path = ML_METRICS_DIR / name
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out = {}
    for model_name, entry in data.get("models", {}).items():
        if isinstance(entry, dict) and "test_metrics" in entry:
            out[model_name] = entry["test_metrics"]
    return out


def _task6_best(target: str) -> str | None:
    if not ML_MODEL_REGISTRY_JSON.exists():
        return None
    reg = json.loads(ML_MODEL_REGISTRY_JSON.read_text())
    for m in reg.get("models", []):
        if m.get("target") == target and m.get("is_selected_best"):
            return m.get("model_name")
    return None


def train_all(force: bool = False, quick: bool = False, output_root: Path | None = None) -> dict:
    """Train both deep networks end to end. Returns the results dict that the
    reports, registry and API all read from.

    ``output_root`` redirects every write beneath one directory, defaulting to
    the committed ``artifacts/`` layout. The test suite passes a ``tmp_path``
    so running the tests does not rewrite tracked files.
    """
    out = ArtifactPaths.default() if output_root is None else ArtifactPaths(root=Path(output_root))
    out.ensure()
    ensure_dirs()

    training.set_seeds()
    dataset = load_and_validate()
    contract = dataset.contract
    source = _data_source()

    max_epochs = 40 if quick else 200
    patience = 8 if quick else 20
    n_folds = 2 if quick else 4

    results: dict = {}

    for target, task in TARGETS.items():
        log.info("Task 7: training deep network for %s (%s)", target, task)

        frame, features = build_task_frame(dataset, target)
        mask = _numeric_mask(features, contract)
        X = frame[features].to_numpy(dtype="float32")
        y = frame[target].to_numpy(dtype="float32")

        holdout = chronological_holdout(frame, test_fraction=0.2)
        folds = expanding_window_folds(frame.loc[holdout.dev_index], n_folds=n_folds)

        build_fn = models_mod.BUILDERS[target]
        space = tuning.REGRESSION_SPACE if task == "regression" else tuning.CLASSIFICATION_SPACE
        if quick:
            space = tuning.SearchSpace(
                hidden_units=(space.hidden_units[0],), dropout=(space.dropout[0],),
                learning_rate=(space.learning_rate[0],), batch_size=space.batch_size,
            )

        class_weight = None
        if task == "classification":
            class_weight = training.balanced_class_weight(y[holdout.dev_index])

        scale_target = task == "regression"
        best_params, trials = tuning.search(
            build_fn, X, y, folds, mask, space,
            task=task, class_weight=class_weight, scale_target=scale_target,
            max_epochs=max_epochs, patience=patience, log=log,
        )

        # Decision threshold from pooled out-of-fold predictions, for the same
        # reason Task 6 does it: 0.5 left this network predicting "never pit"
        # for every test lap while scoring ROC-AUC 0.92 and accuracy 0.994.
        best_trial = next(
            tr for tr in trials
            if {**tr.params, "hidden_units": list(tr.params["hidden_units"])}
            == {**best_params, "hidden_units": list(best_params["hidden_units"])}
        )
        threshold_choice = None
        if task == "classification":
            threshold_choice = tune_threshold(
                best_trial.oof_y_true, best_trial.oof_y_proba, objective="f1")
            log.info("  tuned decision threshold: %.4f (%s)",
                     threshold_choice.threshold, threshold_choice.note)

        final_fold = folds[-1]
        fit = training.fit_fold(
            build_fn,
            X[holdout.dev_index], y[holdout.dev_index],
            X[final_fold.val_index], y[final_fold.val_index],
            mask,
            hidden_units=best_params["hidden_units"],
            dropout=best_params["dropout"],
            learning_rate=best_params["learning_rate"],
            batch_size=best_params["batch_size"],
            max_epochs=max_epochs, patience=patience,
            class_weight=class_weight, scale_target=scale_target,
        )

        test_pred = training.predict(fit.model, fit.scaler, X[holdout.test_index], mask, fit.y_scaler)
        y_test = y[holdout.test_index]
        if task == "regression":
            dl_metrics = regression_metrics(y_test, test_pred)
        else:
            thr = threshold_choice.threshold if threshold_choice else DEFAULT_THRESHOLD
            dl_metrics = classification_metrics(
                y_test, apply_threshold(test_pred, thr), y_proba=test_pred)
            dl_metrics["decision_threshold"] = round(float(thr), 4)
            # What the default would have produced, so the change is visible.
            dl_metrics["at_default_threshold"] = {
                k: v for k, v in classification_metrics(
                    y_test, apply_threshold(test_pred, DEFAULT_THRESHOLD), y_proba=test_pred
                ).items()
                if k in ("precision", "recall", "f1", "accuracy")
            }

        persistence.save(fit.model, fit.scaler, features, mask, target, out.models_dl, fit.y_scaler)

        fig = visualize.plot_history(
            fit.history,
            f"Task 7 - {target} ({'lap-time regression' if task == 'regression' else 'pit-decision classification'})",
            out.figures / f"dl_{target}_training_history.png",
            best_epoch=fit.best_epoch,
        )

        # --- comparison against Task 6's own committed results ---------------
        classical = _task6_holdout_metrics(target)
        best_classical = _task6_best(target)
        # PR-AUC for classification, matching Task 6's selection metric — a
        # comparison decided on ROC-AUC would rank the models by a quantity
        # neither pipeline now selects on.
        primary = "mae" if task == "regression" else "pr_auc"

        comparison = [{"model": "dnn_mlp", "family": "deep", "metrics": dl_metrics}]
        for name, m in classical.items():
            comparison.append({"model": name, "family": "classical", "metrics": m})

        cmp_fig = None
        rows = [{"model": r["model"], primary: r["metrics"][primary]}
                for r in comparison if r["metrics"].get(primary) is not None]
        if not rows and task == "classification":
            primary = "f1"
            rows = [{"model": r["model"], primary: r["metrics"].get(primary)}
                    for r in comparison if r["metrics"].get(primary) is not None]
        if rows:
            cmp_fig = visualize.plot_model_comparison(
                rows, primary, f"Task 7 - {target}: deep network vs Task 6 classical models",
                out.figures / f"dl_{target}_model_comparison.png",
                lower_is_better=(task == "regression"),
            )

        verdict = _verdict(task, primary, comparison, dl_metrics, best_classical,
                           len(holdout.dev_index), holdout, y_test, classical)

        arch = models_mod.architecture_summary(fit.model)
        n_train = int(len(holdout.dev_index))
        ratio = arch["total_parameters"] / max(n_train, 1)
        capacity_note = (
            f"With {arch['total_parameters']:,} parameters against {n_train} training rows "
            f"(ratio {ratio:.2f}), this network has "
            + ("more parameters than training examples. That is the expected regime for this "
               "dataset and it is why dropout, L2 and early stopping are applied together; it "
               "is also the honest reason to expect a tree ensemble to be competitive here."
               if ratio > 1 else
               "fewer parameters than training examples, which is the intended small-data design.")
        )
        overfit_note = (
            f"Early stopping restored the weights from epoch {fit.best_epoch}. "
            + (f"Training ran {fit.epochs_run} epochs, so {fit.epochs_run - fit.best_epoch} epochs "
               f"of validation-loss deterioration were discarded - the countermeasures did real work."
               if fit.epochs_run > fit.best_epoch else
               "Validation loss was still improving when the epoch cap was reached, so the model "
               "was capacity- or budget-limited rather than overfitting.")
        )

        results[target] = {
            "task": task,
            "features": features,
            "n_features": len(features),
            "identity_features": [f for f in features if f.startswith(("driver_", "team_"))],
            "n_train": int(len(holdout.dev_index)),
            "n_test": int(len(holdout.test_index)),
            "holdout": holdout.to_metadata(),
            "n_folds": len(folds),
            "search_space": space.to_metadata(),
            "trials": [t.to_metadata() for t in trials],
            "best_params": {**best_params, "hidden_units": list(best_params["hidden_units"])},
            "selection_metric": "mae" if task == "regression" else "pr_auc",
            "choice_note": (
                "Ties are impossible here because the grid is fully enumerated and scored "
                "deterministically under a fixed seed."
            ),
            "architecture": arch,
            "capacity_note": capacity_note,
            "overfit_note": overfit_note,
            "epochs_run": fit.epochs_run,
            "best_epoch": fit.best_epoch,
            "max_epochs": max_epochs,
            "patience": patience,
            "test_metrics": dl_metrics,
            "threshold": threshold_choice.to_metadata() if threshold_choice else None,
            "history": fit.history,
            "comparison": comparison,
            "task6_best_model": best_classical,
            "verdict": verdict,
            "dataset_source": source,
            "figures": {
                "training_history": fig.name,
                "model_comparison": cmp_fig.name if cmp_fig else None,
            },
        }

    _write_artifacts(results, source, out)
    return results


def _verdict(task, primary, comparison, dl_metrics, best_classical, n_train,
             holdout, y_test, classical) -> str:
    defined = [r for r in comparison if r["metrics"].get(primary) is not None]
    dl_val = dl_metrics.get(primary)

    if not defined or dl_val is None:
        n_pos = int(np.sum(np.asarray(y_test) == 1)) if task == "classification" else None
        return (
            f"**No verdict is possible from the holdout.** Every model's {primary} - the deep "
            f"network's and Task 6's classical models' alike - is undefined on this test set "
            f"(it contains {n_pos} positive examples; the holdout is laps "
            f"{holdout.test_laps[0]}-{holdout.test_laps[-1]}). That is a property of the data "
            f"and it applies symmetrically to both model families."
        )

    winner = (min if task == "regression" else max)(defined, key=lambda r: r["metrics"][primary])
    if winner["model"] == "dnn_mlp":
        return (
            f"**The deep network wins on {primary}** ({dl_val:.4f}), against "
            f"{len(defined) - 1} of Task 6's classical models evaluated on the same "
            f"chronological holdout."
        )
    best_val = winner["metrics"][primary]
    return (
        f"**Task 6's classical `{winner['model']}` wins on {primary}** ({best_val:.4f} vs the "
        f"deep network's {dl_val:.4f}). Reported as measured. With {n_train} training rows from "
        f"a single race session, a tree ensemble's inductive bias suits this problem better than "
        f"a network's; deep learning's advantage requires substantially more data than exists here."
    )


def _write_artifacts(results: dict, source: dict, out: ArtifactPaths) -> None:
    out.metrics.mkdir(parents=True, exist_ok=True)

    out.dl_metrics_json.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": "Task 7 - Deep Learning Model Development",
        "dataset_source": source,
        "models": {
            t: {
                "task": r["task"],
                "test_metrics": r["test_metrics"],
                "threshold": r.get("threshold"),
                "architecture": r["architecture"],
                "hyperparameters": r["best_params"],
                "n_train": r["n_train"],
                "n_test": r["n_test"],
            } for t, r in results.items()
        },
    }, indent=2))

    out.dl_history_json.write_text(json.dumps({
        t: {
            "epochs_run": r["epochs_run"],
            "best_epoch": r["best_epoch"],
            "early_stopping_patience": r["patience"],
            "max_epochs": r["max_epochs"],
            "hyperparameters": r["best_params"],
            "history": r["history"],
        } for t, r in results.items()
    }, indent=2))

    out.dl_comparison_json.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_source": source,
        "note": (
            "The classical rows are Task 6's own committed holdout metrics, read from "
            "artifacts/metrics/. Both families are evaluated on the identical "
            "chronological holdout produced by app.intelligence.ml.splits."
        ),
        "targets": {
            t: {
                "task": r["task"],
                "selection_metric": r["selection_metric"],
                "task6_best_model": r["task6_best_model"],
                "comparison": r["comparison"],
                "verdict": r["verdict"],
            } for t, r in results.items()
        },
    }, indent=2))

    _extend_registry(results, out)
    _write_reports(results, out)


def _extend_registry(results: dict, out: ArtifactPaths | None = None) -> None:
    """Add Task 7 entries to the **existing** model registry rather than
    creating a parallel one, so the API and dashboard have a single source of
    truth for 'what models exist'."""
    out = out or ArtifactPaths.default()
    registry_path = out.model_registry_json
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
    else:
        registry = {"generated_at": datetime.now(timezone.utc).isoformat(), "models": []}

    # Preserve the original trained_at for a target that is already registered.
    # restore_registry_entries() runs on every build_all skip path, and stamping
    # a fresh timestamp there would make a no-op rebuild dirty the tracked
    # registry - the exact churn this phase is removing elsewhere.
    previous_trained_at = {
        m.get("target"): m.get("trained_at")
        for m in registry.get("models", [])
        if m.get("family") == "deep" and m.get("trained_at")
    }
    registry["models"] = [m for m in registry.get("models", []) if m.get("family") != "deep"]

    for target, r in results.items():
        registry["models"].append({
            "model_name": "dnn_mlp",
            "family": "deep",
            "task_source": "Task 7 - Deep Learning",
            "target": target,
            # Task 6's entries use "task" for the target name and always carry
            # is_selected_best. Match that exactly: a registry with two field
            # conventions is a KeyError waiting to happen in any consumer that
            # iterates it (ModelCache._best_model_name does).
            "task": target,
            "task_type": r["task"],
            "is_selected_best": False,
            "features": r["features"],
            "architecture": r["architecture"],
            "hyperparameters": r["best_params"],
            "metrics": {"test": r["test_metrics"]},
            "artifact": f"models/dl/{target}{persistence.MODEL_EXTENSION}",
            "framework": "keras",
            "model_format": persistence.MODEL_EXTENSION,
            "format_note": (
                f"Reference spec names {persistence.SPEC_EXTENSION_IN_REFERENCE}; Keras 3 "
                f"cannot reload HDF5 models saved this way, so {persistence.MODEL_EXTENSION} "
                f"is used."
            ),
            "training_rows": r["n_train"],
            "test_rows": r["n_test"],
            "dataset": r["dataset_source"],
            "trained_at": previous_trained_at.get(target) or datetime.now(timezone.utc).isoformat(),
        })

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2))


def _write_reports(results: dict, out: ArtifactPaths) -> None:
    from app.intelligence.dl import reports as reports_mod
    reports_mod.evaluation_report(results, out.reports / "dl_evaluation_report.md")
    reports_mod.hyperparameter_report(results, out.reports / "dl_hyperparameter_report.md")


def restore_registry_entries(output_root: Path | None = None) -> int:
    """Rebuild Task 7's registry rows from the committed artifacts, without
    retraining.

    Task 7 extends the shared registry, and anything that rewrites that file
    wholesale can drop those rows. ``write_registry`` now preserves them, but a
    registry written before that fix — or edited by hand — can still be missing
    them. Everything needed is already on disk (``dl_metrics.json`` plus each
    model's ``*_spec.json``), so the rows are reconstructable exactly rather
    than requiring a nine-minute retrain.

    Returns the number of entries restored.
    """
    out = ArtifactPaths.default() if output_root is None else ArtifactPaths(root=Path(output_root))
    if not out.dl_metrics_json.exists():
        return 0

    metrics = json.loads(out.dl_metrics_json.read_text())
    results: dict = {}
    for target, m in metrics.get("models", {}).items():
        spec_path = out.models_dl / f"{target}_spec.json"
        if not spec_path.exists():
            continue
        spec = json.loads(spec_path.read_text())
        results[target] = {
            "task": m["task"],
            "features": spec["features"],
            "architecture": m["architecture"],
            "best_params": m["hyperparameters"],
            "test_metrics": m["test_metrics"],
            "n_train": m["n_train"],
            "n_test": m["n_test"],
            "dataset_source": metrics.get("dataset_source", {}),
        }

    if results:
        _extend_registry(results, out)
    return len(results)


def artifacts_exist() -> bool:
    """True when Task 7 has already produced its outputs, so ``build_all.py``
    can skip the stage unless ``--force`` is passed."""
    return DL_METRICS_JSON.exists() and any(DL_MODELS_DIR.glob(f"*{persistence.MODEL_EXTENSION}"))
