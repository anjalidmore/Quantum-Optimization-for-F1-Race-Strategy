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
import pandas as pd

from app.core.paths import (
    DL_METRICS_JSON,
    DL_MODELS_DIR,
    ML_METRICS_DIR,
    ML_MODEL_REGISTRY_JSON,
    PROCESSED_DATA_SOURCE_JSON,
    TARGET_DIRNAME,
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


def _scaled(scaler, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    Xs = X.astype("float32").copy()
    if mask.any():
        Xs[:, mask] = scaler.transform(X[:, mask])
    return Xs


def _metrics(task: str, y_true, pred, threshold: float | None = None) -> dict:
    if task == "regression":
        return regression_metrics(y_true, pred)
    thr = DEFAULT_THRESHOLD if threshold is None else threshold
    m = classification_metrics(y_true, apply_threshold(pred, thr), y_proba=pred)
    m["decision_threshold"] = round(float(thr), 4)
    return m


def _diagnose(task: str, history: dict, best_epoch: int, train_m: dict, val_m: dict) -> dict:
    """Read the saved model's training curves and name the fit.

    Rules (stated so the verdict can be checked against the plot):
    * overfitting EMERGED during training if, after the best epoch, validation
      loss rose more than 10% above its minimum while training loss kept falling;
    * the SAVED model is overfit if its training/validation gap is large —
      training MAE below 60% of validation MAE (regression), or training PR-AUC
      above validation PR-AUC by more than 0.25 (classification);
    * it is underfit if it has not learned the training data — training R2 below
      0.1 (regression) or training PR-AUC below twice the positive rate
      (classification).
    """
    tl, vl = history["loss"], history["val_loss"]
    b = best_epoch - 1
    rise = (vl[-1] - min(vl)) / min(vl) * 100 if min(vl) > 0 else 0.0
    emerged = rise > 10 and tl[-1] < tl[b]
    if task == "regression":
        underfit = (train_m.get("r2") or 0) < 0.1
        overfit = train_m["mae"] < 0.6 * val_m["mae"]
        gap = {"train_mae": train_m["mae"], "val_mae": val_m["mae"]}
    else:
        pos_rate = val_m["n_positive"] / max(val_m["n"], 1)
        underfit = (train_m.get("pr_auc") or 0) < 2 * pos_rate
        overfit = ((train_m.get("pr_auc") or 0) - (val_m.get("pr_auc") or 0)) > 0.25
        gap = {"train_pr_auc": train_m.get("pr_auc"), "val_pr_auc": val_m.get("pr_auc")}
    verdict = ("UNDERFITTING" if underfit else
               "OVERFITTING (large train/validation gap at the saved weights)" if overfit else
               "REASONABLE FIT")
    return {
        "verdict": verdict, "epochs_run": len(vl), "best_epoch": best_epoch,
        "still_improving_at_stop": best_epoch == len(vl),
        "val_loss_min": float(min(vl)), "val_loss_final": float(vl[-1]),
        "val_loss_rise_after_best_pct": float(rise),
        "train_loss_at_best": float(tl[b]), "train_loss_final": float(tl[-1]),
        "overfitting_emerged_after_best_epoch": bool(emerged), **gap,
    }


def train_all(force: bool = False, quick: bool = False, output_root: Path | None = None) -> dict:
    """Train both deep networks end to end. Returns the results dict that the
    reports, registry and API all read from.

    ``output_root`` redirects every write beneath one directory, defaulting to
    the committed ``artifacts/`` layout. The test suite passes a ``tmp_path``
    so running the tests does not rewrite tracked files.

    Time-awareness, end to end:
    * the test set is ``chronological_holdout`` — the last 20% of laps;
    * hyperparameters are chosen on ``expanding_window_folds`` over the earlier
      laps, each fold validating on the laps just after its training laps;
    * the final network trains on the last fold's training laps and early-stops
      on the block of laps immediately before the test laps. (Before this fix
      the final network early-stopped on rows that were also in its training
      set, so its "validation" curve was a training curve.)
    """
    out = ArtifactPaths.default() if output_root is None else ArtifactPaths(root=Path(output_root))
    out.ensure()
    ensure_dirs()

    training.set_seeds()
    dataset = load_and_validate()
    contract = dataset.contract
    source = _data_source()

    max_epochs = 40 if quick else 300
    patience = 8 if quick else 25
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
        scale_target = task == "regression"

        best_params, trials = tuning.search(
            build_fn, X, y, folds, mask, task=task, scale_target=scale_target,
            max_epochs=max_epochs, patience=patience, quick=quick, log=log)
        best_trial = next(tr for tr in trials if tr.params == best_params)

        threshold_choice = None
        if task == "classification":
            threshold_choice = tune_threshold(best_trial.oof_y_true, best_trial.oof_y_pred, objective="f1")
            log.info("  decision threshold %.4f — %s", threshold_choice.threshold, threshold_choice.note)
        thr = threshold_choice.threshold if threshold_choice else None

        # ---- final, time-aware fit -------------------------------------------
        fin = folds[-1]
        cw = (training.balanced_class_weight(y[fin.train_index])
              if best_params.get("class_weighted") else None)
        fit = training.fit_fold(
            build_fn, X[fin.train_index], y[fin.train_index], X[fin.val_index], y[fin.val_index], mask,
            hidden_units=best_params["hidden_units"], dropout=best_params["dropout"],
            learning_rate=best_params["learning_rate"], batch_size=best_params["batch_size"],
            optimizer=best_params["optimizer"], l2=best_params["l2"], loss=best_params.get("loss"),
            class_weight=cw, scale_target=scale_target, max_epochs=max_epochs, patience=patience,
        )
        pred = {name: training.predict(fit.model, fit.scaler, X[idx], mask, fit.y_scaler)
                for name, idx in (("train", fin.train_index), ("validation", fin.val_index),
                                  ("test", holdout.test_index))}
        y_split = {"train": y[fin.train_index], "validation": y[fin.val_index], "test": y[holdout.test_index]}
        split_metrics = {k: _metrics(task, y_split[k], pred[k], thr) for k in pred}
        dl_metrics = split_metrics["test"]
        if task == "classification":
            dl_metrics["at_default_threshold"] = {
                k: v for k, v in _metrics(task, y_split["test"], pred["test"], DEFAULT_THRESHOLD).items()
                if k in ("precision", "recall", "f1", "accuracy")}
        diag = _diagnose(task, fit.history, fit.best_epoch, split_metrics["train"], split_metrics["validation"])
        log.info("  final fit: %d epochs, best %d — %s | test %s", fit.epochs_run, fit.best_epoch,
                 diag["verdict"], {k: round(v, 4) for k, v in dl_metrics.items()
                                   if k in ("mae", "rmse", "r2", "precision", "recall", "f1", "roc_auc", "pr_auc")
                                   and v is not None})

        saved = persistence.save(fit.model, fit.scaler, features, mask, target, out.models_dl, fit.y_scaler,
                                 X_check=_scaled(fit.scaler, X[holdout.test_index], mask))

        # ---- per-target deliverables ------------------------------------------
        tdir = out.dl_target(target)
        pd.DataFrame({"epoch": np.arange(1, fit.epochs_run + 1), **fit.history}).to_csv(
            tdir / "training_history.csv", index=False)
        label = "lap-time regression" if task == "regression" else "pit-decision classification"
        figures = {"loss_curve": visualize.plot_curve(
            fit.history, "loss", "loss (training objective)", f"Task 7 DNN — {label}: loss",
            tdir / "loss_curve.png", fit.best_epoch).name}
        if task == "regression":
            figures["mae_curve"] = visualize.plot_curve(
                {k: [v * float(fit.y_scaler.scale_[0]) for v in vals] for k, vals in fit.history.items()
                 if "mae" in k}, "mae", "MAE (seconds)", "Task 7 DNN — lap time: MAE",
                tdir / "mae_curve.png", fit.best_epoch).name
            figures["prediction_vs_actual"] = visualize.plot_pred_vs_actual(
                y_split["test"], pred["test"], "Task 7 DNN — test laps", tdir / "prediction_vs_actual.png").name
        else:
            figures["accuracy_curve"] = visualize.plot_curve(
                fit.history, "accuracy", "accuracy (threshold 0.5)", "Task 7 DNN — pit decision: accuracy",
                tdir / "accuracy_curve.png", fit.best_epoch).name
            figures["confusion_matrix"] = visualize.plot_confusion(
                dl_metrics["confusion_matrix"], thr, "Task 7 DNN — test laps", tdir / "confusion_matrix.png").name
            roc = visualize.plot_roc(y_split["test"], pred["test"], "Task 7 DNN — ROC (test laps)",
                                     tdir / "roc_curve.png")
            figures["roc_curve"] = roc.name if roc else None
            if roc is None and (tdir / "roc_curve.png").exists():
                (tdir / "roc_curve.png").unlink()

        # ---- comparison against Task 6's own committed results ----------------
        classical = _task6_holdout_metrics(target)
        best_classical = _task6_best(target)
        primary = "mae" if task == "regression" else "pr_auc"
        comparison = [{"model": "dnn_mlp", "family": "deep", "metrics": dl_metrics}]
        comparison += [{"model": n, "family": "classical", "metrics": m} for n, m in classical.items()]
        rows = [{"model": r["model"], primary: r["metrics"][primary]}
                for r in comparison if r["metrics"].get(primary) is not None]
        if rows:
            figures["model_comparison"] = visualize.plot_model_comparison(
                rows, primary, f"Task 7 — {label}: DNN vs Task 6 models (same test laps)",
                tdir / "model_comparison.png", lower_is_better=(task == "regression")).name
        verdict = _verdict(task, primary, comparison, dl_metrics, best_classical,
                           len(fin.train_index), holdout, y_split["test"], classical)

        arch = models_mod.architecture_summary(fit.model)
        results[target] = {
            "task": task,
            "features": features,
            "n_features": len(features),
            "identity_features": [f for f in features if f.startswith(("driver_", "team_"))],
            "n_train": int(len(fin.train_index)),
            "n_validation": int(len(fin.val_index)),
            "n_test": int(len(holdout.test_index)),
            "n_dev": int(len(holdout.dev_index)),
            "holdout": holdout.to_metadata(),
            "final_fit_split": {"train_laps": fin.train_laps, "validation_laps": fin.val_laps,
                                "test_laps": holdout.test_laps},
            "n_folds": len(folds),
            "search_method": "one-factor-at-a-time over expanding-window folds",
            "stages": [[name, [list(v) if isinstance(v, tuple) else v for v in vals]]
                       for name, vals in tuning.stages_for(task, quick)],
            "trials": [tr.to_metadata() for tr in trials],
            "best_params": {**best_params, "hidden_units": list(best_params["hidden_units"])},
            "selection_metric": primary,
            "class_weights_used": cw,
            "architecture": arch,
            "epochs_run": fit.epochs_run,
            "best_epoch": fit.best_epoch,
            "max_epochs": max_epochs,
            "patience": patience,
            "train_metrics": split_metrics["train"],
            "validation_metrics": split_metrics["validation"],
            "test_metrics": dl_metrics,
            "overfitting": diag,
            "threshold": threshold_choice.to_metadata() if threshold_choice else None,
            "history": fit.history,
            "comparison": comparison,
            "task6_best_model": best_classical,
            "verdict": verdict,
            "dataset_source": source,
            "model_file": {"path": str(saved.model_path.relative_to(out.root.parent))
                           if out.root.parent in saved.model_path.parents else str(saved.model_path),
                           "bytes": saved.size_bytes, "reload_verified": saved.reload_verified},
            "figures": figures,
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
    now = datetime.now(timezone.utc).isoformat()
    out.deep_learning.mkdir(parents=True, exist_ok=True)

    # evaluation_report.json — read by GET /api/dl/metrics (shape unchanged)
    out.dl_metrics_json.write_text(json.dumps({
        "generated_at": now,
        "task": "Task 7 - Deep Learning Model Development",
        "dataset_source": source,
        "evaluation_split": ("chronological holdout: the last 20% of laps, never used for "
                             "hyperparameter selection, threshold tuning or early stopping"),
        "models": {
            t: {
                "task": r["task"],
                "train_metrics": r["train_metrics"],
                "validation_metrics": r["validation_metrics"],
                "test_metrics": r["test_metrics"],
                "overfitting": r["overfitting"],
                "threshold": r.get("threshold"),
                "architecture": r["architecture"],
                "hyperparameters": r["best_params"],
                "class_weights_used": r["class_weights_used"],
                "n_train": r["n_train"], "n_validation": r["n_validation"], "n_test": r["n_test"],
                "final_fit_split": r["final_fit_split"],
                "epochs_run": r["epochs_run"], "best_epoch": r["best_epoch"],
                "model_file": r["model_file"],
                "figures": r["figures"],
            } for t, r in results.items()
        },
    }, indent=2, default=_json_default))

    # training_history.json — read by GET /api/dl/history (shape unchanged)
    out.dl_history_json.write_text(json.dumps({
        t: {"epochs_run": r["epochs_run"], "best_epoch": r["best_epoch"],
            "early_stopping_patience": r["patience"], "max_epochs": r["max_epochs"],
            "hyperparameters": r["best_params"], "history": r["history"]}
        for t, r in results.items()
    }, indent=2, default=_json_default))

    # model_comparison.json / .csv — DL vs Task 6 on the identical test laps
    out.dl_comparison_json.write_text(json.dumps({
        "generated_at": now,
        "dataset_source": source,
        "note": ("The classical rows are Task 6's own committed holdout metrics, read from "
                 "artifacts/metrics/. Both families are evaluated on the identical chronological "
                 "holdout produced by app.intelligence.ml.splits."),
        "targets": {t: {"task": r["task"], "selection_metric": r["selection_metric"],
                        "task6_best_model": r["task6_best_model"], "comparison": r["comparison"],
                        "verdict": r["verdict"]} for t, r in results.items()},
    }, indent=2, default=_json_default))
    cmp_rows = []
    for t, r in results.items():
        keys = (["mae", "rmse", "r2"] if r["task"] == "regression"
                else ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "decision_threshold"])
        for row in r["comparison"]:
            cmp_rows.append({"target": t, "model": row["model"], "family": row["family"],
                             "is_task6_selected": row["model"] == r["task6_best_model"],
                             **{k: row["metrics"].get(k) for k in keys}})
    pd.DataFrame(cmp_rows).to_csv(out.deep_learning / "model_comparison.csv", index=False)

    # hyperparameter_report.csv — every experiment, both targets
    hp_rows = []
    for t, r in results.items():
        metric = r["selection_metric"]
        second = "rmse" if r["task"] == "regression" else "roc_auc"
        for tr in r["trials"]:
            p_ = tr["params"]
            s = tr["cv_summary"]
            hp_rows.append({
                "Target": t,
                "Experiment": tr["experiment"],
                "Varied": tr["varied"],
                "Architecture": "-".join(str(x) for x in (r["n_features"], *p_["hidden_units"], 1)),
                "Hidden Layers": len(p_["hidden_units"]),
                "Parameters": tr["parameters"],
                "Learning Rate": p_["learning_rate"],
                "Batch Size": p_["batch_size"],
                "Dropout": p_["dropout"],
                "Optimizer": "Adam" if p_["optimizer"] == "adam" else "RMSprop",
                "L2": p_["l2"],
                "Loss": p_.get("loss", "binary_crossentropy"),
                "Class Weighting": ("balanced (inside the loss)" if p_.get("class_weighted")
                                    else ("none" if r["task"] == "classification" else "n/a")),
                "Mean Epochs Run": round(tr["mean_epochs_run"], 1),
                "Mean Best Epoch": round(tr["mean_best_epoch"], 1),
                "Validation Metric": f"CV {metric.upper()} (mean of {r['n_folds']} expanding-window folds)",
                "Validation Metric Value": s.get(metric, {}).get("mean"),
                "Validation Metric Std": s.get(metric, {}).get("std"),
                f"CV {second.upper()}": s.get(second, {}).get("mean"),
                "CV F1 @0.5" if r["task"] == "classification" else "CV R2":
                    s.get("f1" if r["task"] == "classification" else "r2", {}).get("mean"),
                "Selected": "YES" if {**p_, "hidden_units": tuple(p_["hidden_units"])} ==
                                     {**r["best_params"], "hidden_units": tuple(r["best_params"]["hidden_units"])}
                            else "",
            })
    pd.DataFrame(hp_rows).to_csv(out.deep_learning / "hyperparameter_report.csv", index=False)

    # model_metadata.json — everything needed to reuse or audit a saved network
    (out.deep_learning / "model_metadata.json").write_text(json.dumps({
        "generated_at": now,
        "task": "Task 7 - Deep Learning Model Development",
        "framework": f"Keras {keras_version()} (backend: {keras_backend()})",
        "dataset_source": source,
        "models": {
            t: {
                "target": t, "task": r["task"], "model_file": r["model_file"],
                "input_features": r["features"], "n_features": r["n_features"],
                "architecture": r["architecture"], "hyperparameters": r["best_params"],
                "class_weights_used": r["class_weights_used"],
                "decision_threshold": (r["threshold"] or {}).get("threshold"),
                "preprocessing": ("StandardScaler fitted on the training laps only; binary indicator "
                                  "columns passed through unscaled"
                                  + ("; lap-time target standardised on the training laps and "
                                     "inverse-transformed before every metric" if r["task"] == "regression" else "")),
                "split": {"test": "chronological holdout (last 20% of laps)",
                          "tuning": f"{r['n_folds']} expanding-window lap-forward folds",
                          "final_fit": r["final_fit_split"],
                          "rows": {"train": r["n_train"], "validation": r["n_validation"], "test": r["n_test"]}},
                "training": {"epochs_run": r["epochs_run"], "best_epoch": r["best_epoch"],
                             "early_stopping": f"val_loss, patience {r['patience']}, restore_best_weights=True",
                             "max_epochs": r["max_epochs"]},
            } for t, r in results.items()
        },
    }, indent=2, default=_json_default))

    _extend_registry(results, out)
    _write_reports(results, out)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def keras_version() -> str:
    import keras
    return keras.__version__


def keras_backend() -> str:
    import keras
    return keras.backend.backend()


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

    # Preserve the original trained_at for a target already registered, so a
    # no-op rebuild (restore_registry_entries on the build_all skip path) does
    # not dirty the tracked registry.
    previous_trained_at = {m.get("target"): m.get("trained_at") for m in registry.get("models", [])
                           if m.get("family") == "deep" and m.get("trained_at")}
    registry["models"] = [m for m in registry.get("models", []) if m.get("family") != "deep"]

    for target, r in results.items():
        registry["models"].append({
            "model_name": "dnn_mlp",
            "family": "deep",
            "task_source": "Task 7 - Deep Learning",
            "target": target,
            "task": target,               # Task 6 entries use "task" for the target name
            "task_type": r["task"],
            "is_selected_best": False,
            "features": r["features"],
            "architecture": r["architecture"],
            "hyperparameters": r["best_params"],
            "metrics": {"test": r["test_metrics"]},
            "artifact": f"models/deep_learning/{TARGET_DIRNAME[target]}/{persistence.MODEL_FILENAME}",
            "framework": "keras",
            "model_format": persistence.MODEL_EXTENSION,
            "training_rows": r["n_train"],
            "test_rows": r["n_test"],
            "dataset": r["dataset_source"],
            "trained_at": previous_trained_at.get(target) or datetime.now(timezone.utc).isoformat(),
        })

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2, default=_json_default))


def _write_reports(results: dict, out: ArtifactPaths) -> None:
    from app.intelligence.dl import reports as reports_mod
    reports_mod.evaluation_report(results, out.deep_learning / "evaluation_report.md")
    reports_mod.hyperparameter_report(results, out.deep_learning / "hyperparameter_report.md")


def restore_registry_entries(output_root: Path | None = None) -> int:
    """Rebuild Task 7's registry rows from the committed artifacts, without
    retraining. Returns the number of entries restored."""
    out = ArtifactPaths.default() if output_root is None else ArtifactPaths(root=Path(output_root))
    if not out.dl_metrics_json.exists():
        return 0
    metrics = json.loads(out.dl_metrics_json.read_text())
    results: dict = {}
    for target, m in metrics.get("models", {}).items():
        spec_path = persistence.target_dir(out.models_dl, target) / "model_spec.json"
        if not spec_path.exists():
            continue
        spec = json.loads(spec_path.read_text())
        results[target] = {
            "task": m["task"], "features": spec["features"], "architecture": m["architecture"],
            "best_params": m["hyperparameters"], "test_metrics": m["test_metrics"],
            "n_train": m["n_train"], "n_test": m["n_test"],
            "dataset_source": metrics.get("dataset_source", {}),
        }
    if results:
        _extend_registry(results, out)
    return len(results)


def artifacts_exist() -> bool:
    """True when Task 7 has already produced its outputs, so ``build_all.py``
    can skip the stage unless ``--force`` is passed."""
    return DL_METRICS_JSON.exists() and all(
        (persistence.target_dir(DL_MODELS_DIR, t) / persistence.MODEL_FILENAME).exists() for t in TARGETS)
