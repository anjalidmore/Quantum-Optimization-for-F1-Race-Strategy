"""
app.intelligence.qml.pipeline
=============================

Runs the quantum experiment end to end and writes every artifact.

The shape of the experiment is deliberately identical to Task 6's and Task 7's,
because that is the only way the comparison means anything:

1. Load the Task 5 feature matrix through Task 6's own data contract.
2. Reserve the same chronological holdout, and use the same expanding-window
   folds on the development laps.
3. Choose hyperparameters on the folds only — layers ∈ {1, 2, 3} × learning
   rate ∈ {0.05, 0.1} — scored on CV MAE (regression) or CV PR-AUC
   (classification), the same metrics Task 6 selects on.
4. Fit the chosen configuration on the development laps and evaluate once on
   the holdout.
5. Tune the classifiers' decision threshold on pooled out-of-fold predictions,
   exactly as Task 6 and Task 7 do, so nothing is compared at an arbitrary 0.5.

**This is a noiseless simulation** (`default.qubit`) of small circuits on one
race's worth of data. Nothing here demonstrates a quantum advantage, and where
the quantum models lose, the report says so.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from app.core.paths import (
    PROCESSED_DATA_SOURCE_JSON,
    QML_METRICS_JSON,
    QML_MODELS_DIR,
    ArtifactPaths,
    ensure_dirs,
)
from app.intelligence.ml import evaluation
from app.intelligence.ml import threshold as threshold_mod
from app.intelligence.qml import baselines, kernel_svm, reports, visualize, vqc, vqr
from app.intelligence.qml import data as qdata

log = logging.getLogger("intelligence.qml")

SEED = 42
LAYER_GRID = (1, 2, 3)
LR_GRID = (0.05, 0.1)
CV_EPOCHS = 40          # shorter during the search; the final fit trains longer
FINAL_EPOCHS = 80
KERNEL_REPS = 2

REG_KEYS = ["mae", "rmse", "r2"]
CLF_KEYS = ["pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy"]


# ---------------------------------------------------------------------------
# Cross-validated hyperparameter search (folds only — never the holdout)
# ---------------------------------------------------------------------------
def _search_vqc(d: qdata.QuantumTaskData) -> tuple[dict, list[dict]]:
    trials = []
    for n_layers in LAYER_GRID:
        for lr in LR_GRID:
            fold_metrics, oof_true, oof_pred = [], [], []
            for fold in d.folds:
                X_tr, y_tr, X_val, y_val = qdata.encode_fold(d, fold)
                model = vqc.train(
                    X_tr, y_tr, n_qubits=d.n_qubits, n_layers=n_layers, learning_rate=lr,
                    epochs=CV_EPOCHS, class_weight=qdata.class_weights(y_tr), seed=SEED,
                )
                p = vqc.predict_proba(model, X_val)
                fold_metrics.append(
                    evaluation.classification_metrics(y_val, (p >= 0.5).astype(int), y_proba=p)
                )
                oof_true.extend(np.asarray(y_val).tolist())
                oof_pred.extend(p.tolist())
            summary = evaluation.aggregate_metrics(fold_metrics, CLF_KEYS)
            trials.append({
                "n_layers": n_layers, "learning_rate": lr, "n_parameters": vqc.n_parameters(d.n_qubits, n_layers),
                "cv_summary": summary, "per_fold": fold_metrics,
                "oof_y_true": oof_true, "oof_y_pred": oof_pred,
            })
            log.info("  VQC layers=%d lr=%.3f -> CV PR-AUC %.4f (± %.4f)", n_layers, lr,
                     summary["pr_auc"]["mean"] or float("nan"), summary["pr_auc"]["std"] or float("nan"))
    best = max(trials, key=lambda t: (t["cv_summary"]["pr_auc"]["mean"] or -np.inf))
    return best, trials


def _search_vqr(d: qdata.QuantumTaskData) -> tuple[dict, list[dict]]:
    trials = []
    for n_layers in LAYER_GRID:
        for lr in LR_GRID:
            fold_metrics = []
            for fold in d.folds:
                X_tr, y_tr, X_val, y_val = qdata.encode_fold(d, fold)
                model = vqr.train(
                    X_tr, y_tr, n_qubits=d.n_qubits, n_layers=n_layers, learning_rate=lr,
                    epochs=CV_EPOCHS, seed=SEED,
                )
                pred = d.inverse_target(vqr.predict_standardised(model, X_val))
                fold_metrics.append(evaluation.regression_metrics(d.inverse_target(y_val), pred))
            summary = evaluation.aggregate_metrics(fold_metrics, REG_KEYS)
            trials.append({
                "n_layers": n_layers, "learning_rate": lr, "n_parameters": vqr.n_parameters(d.n_qubits, n_layers),
                "cv_summary": summary, "per_fold": fold_metrics,
            })
            log.info("  VQR layers=%d lr=%.3f -> CV MAE %.4f s (± %.4f)", n_layers, lr,
                     summary["mae"]["mean"], summary["mae"]["std"])
    best = min(trials, key=lambda t: (t["cv_summary"]["mae"]["mean"] or np.inf))
    return best, trials


def _cv_scores_for(predict_proba_fn, d: qdata.QuantumTaskData, task: str) -> tuple[dict, list, list]:
    """Cross-validated metrics for a model with no hyperparameters to search."""
    fold_metrics, oof_true, oof_pred = [], [], []
    for fold in d.folds:
        X_tr, y_tr, X_val, y_val = qdata.encode_fold(d, fold)
        p = predict_proba_fn(X_tr, y_tr, X_val)
        if task == "classification":
            fold_metrics.append(evaluation.classification_metrics(y_val, (p >= 0.5).astype(int), y_proba=p))
            oof_true.extend(np.asarray(y_val).tolist())
            oof_pred.extend(np.asarray(p).tolist())
        else:
            fold_metrics.append(evaluation.regression_metrics(d.inverse_target(y_val), d.inverse_target(p)))
    keys = CLF_KEYS if task == "classification" else REG_KEYS
    return evaluation.aggregate_metrics(fold_metrics, keys), oof_true, oof_pred


# ---------------------------------------------------------------------------
# One target, end to end
# ---------------------------------------------------------------------------
def run_classification(models_dir: Path) -> dict:
    log.info("Quantum ML: target_pit_next_lap (VQC + quantum-kernel SVM)")
    d = qdata.load_task("target_pit_next_lap", "classification")
    log.info("  %d dev rows / %d test rows | %s", len(d.y_dev), len(d.y_test), d.reduction["method"])

    # --- VQC: search, then final fit on the development laps ----------------
    best, trials = _search_vqc(d)
    log.info("  VQC selected: layers=%d lr=%.3f", best["n_layers"], best["learning_rate"])
    final = vqc.train(
        d.X_dev, d.y_dev, n_qubits=d.n_qubits, n_layers=best["n_layers"],
        learning_rate=best["learning_rate"], epochs=FINAL_EPOCHS,
        class_weight=qdata.class_weights(d.y_dev), seed=SEED, log=log,
    )
    thr = threshold_mod.tune_threshold(np.asarray(best["oof_y_true"]), np.asarray(best["oof_y_pred"]))
    p_test = vqc.predict_proba(final, d.X_test)
    vqc_result = {
        "model": "vqc",
        "family": "quantum",
        "description": "AngleEmbedding + StronglyEntanglingLayers, PauliZ expectation through a sigmoid",
        "hyperparameters": {"n_layers": best["n_layers"], "learning_rate": best["learning_rate"],
                            "epochs": FINAL_EPOCHS, "n_qubits": d.n_qubits},
        "n_parameters": final["n_parameters"],
        "train_seconds": final["train_seconds"],
        "cv_summary": best["cv_summary"],
        "threshold": thr.to_metadata(),
        "test_metrics": evaluation.classification_metrics(
            d.y_test, (p_test >= thr.threshold).astype(int), y_proba=p_test),
        "loss_history": final["loss_history"],
        "class_weight": final["class_weight"],
    }
    np.save(models_dir / "vqc_weights.npy", final["weights"])
    (models_dir / "vqc_config.json").write_text(json.dumps({
        k: v for k, v in final.items() if k != "weights"
    } | {"weights_file": "vqc_weights.npy", "threshold": thr.threshold,
         "encoding": d.reduction, "features": d.feature_names}, indent=2))

    # --- Quantum kernel SVM -------------------------------------------------
    def kernel_fold(X_tr, y_tr, X_val):
        m = kernel_svm.train(X_tr, y_tr, n_qubits=d.n_qubits, reps=KERNEL_REPS, seed=SEED)
        return kernel_svm.predict_proba(m, X_val)

    k_cv, k_oof_true, k_oof_pred = _cv_scores_for(kernel_fold, d, "classification")
    log.info("  quantum kernel SVM -> CV PR-AUC %.4f (± %.4f)",
             k_cv["pr_auc"]["mean"] or float("nan"), k_cv["pr_auc"]["std"] or float("nan"))
    k_final = kernel_svm.train(d.X_dev, d.y_dev, n_qubits=d.n_qubits, reps=KERNEL_REPS, seed=SEED)
    k_thr = threshold_mod.tune_threshold(np.asarray(k_oof_true), np.asarray(k_oof_pred))
    k_p_test = kernel_svm.predict_proba(k_final, d.X_test)
    kernel_result = {
        "model": "quantum_kernel_svm",
        "family": "quantum",
        "description": f"fidelity kernel |<phi(b)|phi(a)>|^2 ({KERNEL_REPS} feature-map reps) + SVC(kernel='precomputed')",
        "hyperparameters": {"reps": KERNEL_REPS, "C": k_final["C"], "n_qubits": d.n_qubits},
        "n_parameters": k_final["n_parameters"],
        "train_seconds": k_final["train_seconds"],
        "cv_summary": k_cv,
        "threshold": k_thr.to_metadata(),
        "test_metrics": evaluation.classification_metrics(
            d.y_test, (k_p_test >= k_thr.threshold).astype(int), y_proba=k_p_test),
        "n_support_vectors": k_final["n_support_vectors"],
        "n_support_per_class": k_final["n_support_per_class"],
        "kernel_matrix_shape": k_final["kernel_matrix_shape"],
    }
    joblib.dump(k_final["svc"], models_dir / "quantum_kernel_svc.joblib")
    np.save(models_dir / "quantum_kernel_train_inputs.npy", k_final["X_train"])
    (models_dir / "quantum_kernel_config.json").write_text(json.dumps({
        "n_qubits": d.n_qubits, "reps": KERNEL_REPS, "C": k_final["C"], "seed": SEED,
        "svc_file": "quantum_kernel_svc.joblib", "train_inputs_file": "quantum_kernel_train_inputs.npy",
        "threshold": k_thr.threshold, "encoding": d.reduction,
    }, indent=2))

    # --- Parameter-matched classical models on the same reduced inputs ------
    matched = baselines.train_matched_classifier(
        d.X_dev, d.y_dev, target_parameters=final["n_parameters"], seed=SEED)
    matched_results = []
    for name, fitted in matched.items():
        def cls_fold(X_tr, y_tr, X_val, _name=name):
            m = baselines.train_matched_classifier(
                X_tr, y_tr, target_parameters=final["n_parameters"], seed=SEED)[_name]
            return baselines.proba(m, X_val)

        cv, oof_true, oof_pred = _cv_scores_for(cls_fold, d, "classification")
        t = threshold_mod.tune_threshold(np.asarray(oof_true), np.asarray(oof_pred))
        p = baselines.proba(fitted, d.X_test)
        matched_results.append({
            "model": f"{name} (reduced inputs)",
            "family": "classical (parameter-matched)",
            "description": f"{fitted['note']}; trained on the same {d.n_qubits} PCA components the circuits see",
            "hyperparameters": {k: v for k, v in fitted.items() if k in ("hidden_units",)},
            "n_parameters": fitted["n_parameters"],
            "train_seconds": fitted["train_seconds"],
            "cv_summary": cv,
            "threshold": t.to_metadata(),
            "test_metrics": evaluation.classification_metrics(d.y_test, (p >= t.threshold).astype(int), y_proba=p),
        })
        log.info("  %-22s -> CV PR-AUC %.4f | %d parameters", name,
                 cv["pr_auc"]["mean"] or float("nan"), fitted["n_parameters"])

    return {
        "target": "target_pit_next_lap",
        "task": "classification",
        "encoding": d.reduction,
        "holdout": d.holdout,
        "n_dev": int(len(d.y_dev)),
        "n_test": int(len(d.y_test)),
        "n_test_positive": int(np.sum(d.y_test == 1)),
        "n_folds": len(d.folds),
        "notes": d.notes,
        "quantum": [vqc_result, kernel_result],
        "classical_matched": matched_results,
        "task6_reference": baselines.task6_reference("classification"),
        "search_trials": [{k: v for k, v in t.items() if k not in ("oof_y_true", "oof_y_pred", "per_fold")}
                          for t in trials],
        "predictions": {"y_test": d.y_test.tolist(), "vqc": p_test.tolist(),
                        "quantum_kernel_svm": k_p_test.tolist(),
                        **{r["model"]: baselines.proba(matched[n], d.X_test).tolist()
                           for n, r in zip(matched, matched_results)}},
    }


def run_regression(models_dir: Path) -> dict:
    log.info("Quantum ML: target_laptime (VQR)")
    d = qdata.load_task("target_laptime", "regression")
    log.info("  %d dev rows / %d test rows | %s", len(d.y_dev), len(d.y_test), d.reduction["method"])

    best, trials = _search_vqr(d)
    log.info("  VQR selected: layers=%d lr=%.3f", best["n_layers"], best["learning_rate"])
    final = vqr.train(
        d.X_dev, d.y_dev, n_qubits=d.n_qubits, n_layers=best["n_layers"],
        learning_rate=best["learning_rate"], epochs=FINAL_EPOCHS, seed=SEED, log=log,
    )
    pred_test = d.inverse_target(vqr.predict_standardised(final, d.X_test))
    vqr_result = {
        "model": "vqr",
        "family": "quantum",
        "description": "AngleEmbedding + StronglyEntanglingLayers, PauliZ expectation scaled to a standardised lap time",
        "hyperparameters": {"n_layers": best["n_layers"], "learning_rate": best["learning_rate"],
                            "epochs": FINAL_EPOCHS, "n_qubits": d.n_qubits},
        "n_parameters": final["n_parameters"],
        "train_seconds": final["train_seconds"],
        "cv_summary": best["cv_summary"],
        "test_metrics": evaluation.regression_metrics(d.y_test_raw, pred_test),
        "loss_history": final["loss_history"],
    }
    np.save(models_dir / "vqr_weights.npy", final["weights"])
    (models_dir / "vqr_config.json").write_text(json.dumps({
        k: v for k, v in final.items() if k != "weights"
    } | {"weights_file": "vqr_weights.npy", "target_mean": d.target_mean, "target_std": d.target_std,
         "encoding": d.reduction, "features": d.feature_names}, indent=2))

    matched = baselines.train_matched_regressor(
        d.X_dev, d.y_dev, target_parameters=final["n_parameters"], seed=SEED)
    matched_results = []
    for name, fitted in matched.items():
        def reg_fold(X_tr, y_tr, X_val, _name=name):
            m = baselines.train_matched_regressor(
                X_tr, y_tr, target_parameters=final["n_parameters"], seed=SEED)[_name]
            return baselines.predict(m, X_val)

        cv, _, _ = _cv_scores_for(reg_fold, d, "regression")
        pred = d.inverse_target(baselines.predict(fitted, d.X_test))
        matched_results.append({
            "model": f"{name} (reduced inputs)",
            "family": "classical (parameter-matched)",
            "description": f"{fitted['note']}; trained on the same {d.n_qubits} PCA components the circuits see",
            "hyperparameters": {k: v for k, v in fitted.items() if k in ("hidden_units",)},
            "n_parameters": fitted["n_parameters"],
            "train_seconds": fitted["train_seconds"],
            "cv_summary": cv,
            "test_metrics": evaluation.regression_metrics(d.y_test_raw, pred),
        })
        log.info("  %-22s -> CV MAE %.4f s | %d parameters", name, cv["mae"]["mean"], fitted["n_parameters"])

    return {
        "target": "target_laptime",
        "task": "regression",
        "encoding": d.reduction,
        "holdout": d.holdout,
        "n_dev": int(len(d.y_dev)),
        "n_test": int(len(d.y_test)),
        "n_folds": len(d.folds),
        "notes": d.notes,
        "quantum": [vqr_result],
        "classical_matched": matched_results,
        "task6_reference": baselines.task6_reference("regression"),
        "search_trials": [{k: v for k, v in t.items() if k != "per_fold"} for t in trials],
        "predictions": {"y_test": np.asarray(d.y_test_raw).tolist(), "vqr": pred_test.tolist(),
                        **{r["model"]: d.inverse_target(baselines.predict(matched[n], d.X_test)).tolist()
                           for n, r in zip(matched, matched_results)}},
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_all(output_root: Path | None = None) -> dict:
    """Run both targets, write every artifact, and return the results dict."""
    out = ArtifactPaths.default() if output_root is None else ArtifactPaths(root=Path(output_root))
    out.ensure()
    ensure_dirs()
    models_dir = out.models_qml
    models_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    results = {
        "target_pit_next_lap": run_classification(models_dir),
        "target_laptime": run_regression(models_dir),
    }
    wall_seconds = time.perf_counter() - t0

    source = json.loads(PROCESSED_DATA_SOURCE_JSON.read_text()) if PROCESSED_DATA_SOURCE_JSON.exists() else {}
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": "Quantum machine learning (PennyLane)",
        "simulator": "default.qubit (noiseless state-vector simulation; no quantum hardware)",
        "framework": {"pennylane": _pennylane_version()},
        "seed": SEED,
        "search_space": {"n_layers": list(LAYER_GRID), "learning_rate": list(LR_GRID),
                         "selected_on": "expanding-window CV folds only"},
        "dataset_source": source,
        "wall_seconds": wall_seconds,
        "targets": results,
    }
    (out.metrics / QML_METRICS_JSON.name).write_text(json.dumps(payload, indent=2, default=_json_default))

    figures = visualize.generate_all(payload, out)
    payload["figures"] = figures
    (out.metrics / QML_METRICS_JSON.name).write_text(json.dumps(payload, indent=2, default=_json_default))
    report = reports.classical_vs_quantum(payload, out.reports / "classical_vs_quantum_report.md")

    log.info("Quantum ML finished in %.1fs; report at %s", wall_seconds, report)
    return payload


def _pennylane_version() -> str:
    import pennylane

    return pennylane.__version__


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def artifacts_exist() -> bool:
    return QML_METRICS_JSON.exists() and (QML_MODELS_DIR / "vqc_weights.npy").exists()


def load_results() -> dict:
    return json.loads(QML_METRICS_JSON.read_text())
