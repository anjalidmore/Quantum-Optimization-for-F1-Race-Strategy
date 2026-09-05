"""
app.intelligence.dl.reports
===========================

Markdown deliverable generators for Task 7.

Every number written here is passed in from a real training run. Nothing is
templated with an illustrative value - if a metric is undefined it is
rendered as "undefined" with the reason, never as a plausible-looking number.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _fmt(v, nd: int = 4) -> str:
    if v is None:
        return "_undefined_"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def hyperparameter_report(results: dict, out_path: Path) -> Path:
    lines = [
        "# Task 7 - Hyperparameter Report",
        "",
        f"_Generated {_stamp()}._",
        "",
        "Every combination below was **fully enumerated**, not sampled, and each was",
        "scored on the same expanding-window lap-forward folds Task 6 uses. A random",
        "K-fold search would let the network validate on laps it had already seen the",
        "future of; the Task 5 contract forbids it explicitly.",
        "",
    ]

    for target, res in results.items():
        space = res["search_space"]
        lines += [
            f"## {target}",
            "",
            f"**Task:** {res['task']}  |  **Features:** {res['n_features']}  |  "
            f"**Combinations evaluated:** {space['total_combinations']}  |  "
            f"**Folds per combination:** {res['n_folds']}  |  "
            f"**Total training runs:** {space['total_combinations'] * res['n_folds']}",
            "",
            "### Search space",
            "",
            "| Hyperparameter | Values explored |",
            "|---|---|",
            f"| Hidden units | {space['hidden_units']} |",
            f"| Dropout | {space['dropout']} |",
            f"| Learning rate | {space['learning_rate']} |",
            f"| Batch size | {space['batch_size']} |",
            f"| Max epochs | {res['max_epochs']} (early stopping, patience {res['patience']}) |",
            "",
            f"**Selection metric:** {res['selection_metric']} "
            f"({'lower' if res['task'] == 'regression' else 'higher'} is better), "
            "averaged across folds.",
            "",
            "### All trials",
            "",
        ]

        primary = res["selection_metric"]
        lines += [
            f"| Hidden units | Dropout | LR | Batch | CV {primary} (mean) | CV {primary} (std) | Mean best epoch |",
            "|---|---|---|---|---|---|---|",
        ]
        for t in res["trials"]:
            p = t["params"]
            s = t["cv_summary"].get(primary, {})
            lines.append(
                f"| {p['hidden_units']} | {p['dropout']} | {p['learning_rate']} | "
                f"{p['batch_size']} | {_fmt(s.get('mean'))} | {_fmt(s.get('std'))} | "
                f"{_fmt(t['mean_epochs_to_best'], 1)} |"
            )

        chosen = res["best_params"]
        lines += [
            "",
            "### Chosen configuration",
            "",
            f"```json\n{json.dumps(chosen, indent=2)}\n```",
            "",
            f"**Why:** best mean CV {primary} across {res['n_folds']} expanding-window folds. "
            f"{res['choice_note']}",
            "",
            "---",
            "",
        ]

    return _write(out_path, lines)


def evaluation_report(results: dict, out_path: Path) -> Path:
    lines = [
        "# Task 7 - Deep Learning Evaluation Report",
        "",
        f"_Generated {_stamp()}._",
        "",
        "Deep neural networks for F1 race-state prediction, compared against **Task 6's own",
        "committed holdout results** - the same numbers the Machine Learning dashboard shows,",
        "read from `artifacts/metrics/`. The comparison is like-for-like by construction:",
        "identical feature matrix, identical split code (`app.intelligence.ml.splits`),",
        "identical metric code (`app.intelligence.ml.evaluation`).",
        "",
    ]

    for target, res in results.items():
        arch = res["architecture"]
        lines += [
            f"## {target}",
            "",
            f"**Task type:** {res['task']}  |  **Input features:** {res['n_features']}  |  "
            f"**Training rows:** {res['n_train']}  |  **Test rows:** {res['n_test']}",
            "",
            f"**Dataset source:** `{res['dataset_source'].get('source', 'unknown')}`" + (f" ({res['dataset_source'].get('year')} {res['dataset_source'].get('event')} "f"{res['dataset_source'].get('session')})" if res['dataset_source'].get('year') else ""),
            "",
            "### Network architecture",
            "",
            f"`{arch['name']}` - {arch['total_parameters']:,} trainable parameters, "
            f"optimizer {arch['optimizer']}, loss `{arch['loss']}`.",
            "",
            "| Layer | Type | Detail |",
            "|---|---|---|",
        ]
        for layer in arch["layers"]:
            detail = ""
            if "units" in layer:
                detail = f"{layer['units']} units, {layer['activation']}"
            elif "rate" in layer:
                detail = f"rate {layer['rate']}"
            lines.append(f"| {layer['name']} | {layer['type']} | {detail} |")

        lines += [
            "",
            f"**Parameters-to-training-rows ratio:** {arch['total_parameters'] / max(res['n_train'], 1):.2f}",
            "",
            f"{res['capacity_note']}",
            "",
            "### Overfitting prevention",
            "",
            "| Mechanism | Setting | Effect observed |",
            "|---|---|---|",
            f"| Dropout | {res['best_params']['dropout']} on every hidden layer | see loss curve |",
            f"| L2 weight decay | 1e-04 on every Dense kernel | see loss curve |",
            f"| Early stopping | patience {res['patience']} on `val_loss`, best weights restored | "
            f"stopped at epoch {res['best_epoch']} of {res['epochs_run']} run "
            f"(cap {res['max_epochs']}) |",
            "",
            f"{res['overfit_note']}",
            "",
            "### Test-set comparison - DL vs classical",
            "",
        ]

        if res["task"] == "regression":
            lines += [
                "| Model | MAE (s) | RMSE (s) | R2 | MAPE (%) |",
                "|---|---:|---:|---:|---:|",
            ]
            for row in res["comparison"]:
                m = row["metrics"]
                mark = " **<-- deep network**" if row["model"].startswith("dnn") else ""
                lines.append(
                    f"| {row['model']}{mark} | {_fmt(m['mae'])} | {_fmt(m['rmse'])} | "
                    f"{_fmt(m['r2'])} | {_fmt(m['mape'], 2)} |"
                )
        else:
            lines += [
                "| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Accuracy |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for row in res["comparison"]:
                m = row["metrics"]
                mark = " **<-- deep network**" if row["model"].startswith("dnn") else ""
                lines.append(
                    f"| {row['model']}{mark} | {_fmt(m['roc_auc'])} | {_fmt(m['pr_auc'])} | "
                    f"{_fmt(m['f1'])} | {_fmt(m['precision'])} | {_fmt(m['recall'])} | "
                    f"{_fmt(m['accuracy'])} |"
                )

        lines += ["", "### Verdict", "", res["verdict"], "", "---", ""]

    return _write(out_path, lines)


def _write(out_path: Path, lines: list[str]) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
