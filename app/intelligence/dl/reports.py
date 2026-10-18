"""
app.intelligence.dl.reports
===========================

Human-readable Task 7 reports, written beside the machine-readable ones in
``artifacts/deep_learning/``. Every number is passed in from the run that wrote
it; an undefined metric is printed as "undefined", never as a stand-in value.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def _f(v, nd: int = 4) -> str:
    if v is None:
        return "_undefined_"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _write(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def hyperparameter_report(results: dict, out_path: Path) -> Path:
    L = ["# Task 7 — Hyperparameter Report", "", f"_Generated {_stamp()}._", "",
         "Selection used **only** the expanding-window lap-forward folds over the development laps. "
         "The chronological test laps were not touched until the final evaluation.", "",
         "Search method: one factor at a time. Each stage varies one hyperparameter while the others hold "
         "the best values so far; every value listed in the specification is tried and every "
         "experiment is recorded (the full table is `hyperparameter_report.csv`).", ""]
    for t, r in results.items():
        m = r["selection_metric"]
        L += [f"## {t} ({r['task']})", "",
              f"Selection metric: **CV {m.upper()}** ({'lower' if m == 'mae' else 'higher'} is better), "
              f"mean of {r['n_folds']} folds. {len(r['trials'])} distinct configurations trained.", "",
              "| Exp | Varied | Layers | LR | Batch | Dropout | Optimizer | L2 | "
              + ("Loss" if r["task"] == "regression" else "Class weighting")
              + f" | CV {m.upper()} (± sd) | Selected |",
              "|---|---|---|---:|---:|---:|---|---:|---|---:|:---:|"]
        best = r["best_params"]
        for tr in r["trials"]:
            p = tr["params"]
            s = tr["cv_summary"].get(m, {})
            sel = "selected" if {**p, "hidden_units": list(p["hidden_units"])} == best else ""
            extra = p.get("loss", "") if r["task"] == "regression" else ("balanced" if p.get("class_weighted") else "none")
            L.append(f"| {tr['experiment']} | {tr['varied']} | {list(p['hidden_units'])} | {p['learning_rate']} | "
                     f"{p['batch_size']} | {p['dropout']} | {p['optimizer']} | {p['l2']} | {extra} | "
                     f"{_f(s.get('mean'))} (± {_f(s.get('std'))}) | {sel} |")
        L += ["", f"**Selected:** `{best}`", "",
              "Read the ± column before reading a winner into small differences: where two configurations "
              "differ by less than their fold-to-fold spread, the data cannot separate them.", ""]
    return _write(out_path, L)


def evaluation_report(results: dict, out_path: Path) -> Path:
    L = ["# Task 7 — Deep Learning Evaluation Report", "", f"_Generated {_stamp()}._", "",
         "Every network below was evaluated **once** on the chronological holdout — the last 20% of laps — "
         "after its hyperparameters, decision threshold and early-stopping epoch had been fixed on earlier laps.",
         ""]
    for t, r in results.items():
        a, o = r["architecture"], r["overfitting"]
        L += [f"## {t} — {r['task']}", "",
              f"Input features: {r['n_features']} · rows: train {r['n_train']}, validation {r['n_validation']}, "
              f"test {r['n_test']} · test laps {r['final_fit_split']['test_laps'][0]}–"
              f"{r['final_fit_split']['test_laps'][-1]}", "",
              "### Architecture", "",
              "| Layer | Type | Units | Activation | Dropout |", "|---|---|---:|---|---:|"]
        for lay in a["layers"]:
            L.append(f"| {lay['name']} | {lay['type']} | {lay.get('units', '')} | {lay.get('activation', '')} | "
                     f"{lay.get('rate', '')} |")
        L += ["", f"Total parameters {a['total_parameters']:,} · trainable {a['trainable_parameters']:,} · "
                  f"optimizer {a['optimizer']} · training loss `{a['loss']}`.", "",
              f"Hyperparameters: `{r['best_params']}`"
              + (f" · class weights {r['class_weights_used']}" if r.get("class_weights_used") else ""), "",
              "### Train / validation / test", ""]
        if r["task"] == "regression":
            L += ["| Split | MAE (s) | RMSE (s) | R² |", "|---|---:|---:|---:|"]
            for name, key in (("Train", "train_metrics"), ("Validation", "validation_metrics"), ("**Test**", "test_metrics")):
                s = r[key]
                L.append(f"| {name} | {_f(s['mae'])} | {_f(s['rmse'])} | {_f(s['r2'])} |")
        else:
            thr = (r.get("threshold") or {}).get("threshold")
            L += [f"Decision threshold {_f(thr)} (tuned on out-of-fold CV predictions, never on the test laps).", "",
                  "| Split | Pit laps | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for name, key in (("Train", "train_metrics"), ("Validation", "validation_metrics"), ("**Test**", "test_metrics")):
                s = r[key]
                L.append(f"| {name} | {s['n_positive']}/{s['n']} | {_f(s['accuracy'])} | {_f(s['precision'])} | "
                         f"{_f(s['recall'])} | {_f(s['f1'])} | {_f(s['roc_auc'])} | {_f(s['pr_auc'])} |")
            cm = r["test_metrics"]["confusion_matrix"]
            L += ["", f"Test confusion matrix (rows actual, columns predicted; 0 = stay out, 1 = pit): "
                      f"`{cm}`."]
            if r["test_metrics"]["n_positive"] < 5:
                L += ["", f"> **Caution.** The chronological test laps contain **{r['test_metrics']['n_positive']} pit event(s)**. "
                          "Precision, recall, F1 and PR-AUC on so few positives are dominated by chance; the "
                          "cross-validated figures in `hyperparameter_report.csv` rest on many more pit laps and "
                          "are the better guide to this model's ranking ability."]
        L += ["", "### Overfitting", "",
              f"Verdict: **{o['verdict']}**. {o['epochs_run']} epochs run; early stopping restored epoch "
              f"{o['best_epoch']}. Validation loss minimum {_f(o['val_loss_min'])} → final {_f(o['val_loss_final'])} "
              f"({o['val_loss_rise_after_best_pct']:+.1f}%); training loss at the best epoch {_f(o['train_loss_at_best'])}"
              f" → final {_f(o['train_loss_final'])}. Overfitting emerged after the best epoch: "
              f"**{'yes' if o['overfitting_emerged_after_best_epoch'] else 'no'}**.", "",
              "### Comparison with Task 6 (same test laps)", ""]
        m = r["selection_metric"]
        keys = ["mae", "rmse", "r2"] if r["task"] == "regression" else ["precision", "recall", "f1", "roc_auc", "pr_auc"]
        L += ["| Model | " + " | ".join(k.upper() for k in keys) + " |", "|---|" + "---:|" * len(keys)]
        for row in r["comparison"]:
            name = row["model"] + (" **(DNN)**" if row["family"] == "deep" else "") + \
                (" (Task 6 selected)" if row["model"] == r["task6_best_model"] else "")
            L.append(f"| {name} | " + " | ".join(_f(row["metrics"].get(k)) for k in keys) + " |")
        L += ["", r["verdict"], "", f"Primary comparison metric: {m.upper()}.", ""]
    return _write(out_path, L)
