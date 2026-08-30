"""
app.intelligence.ml.reports
==============================

Markdown report generation for Task 6. Every table here is built directly
from the ``reg``/``clf`` result dicts produced by ``pipeline.train_all`` —
no numbers are typed in by hand.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.core.paths import ML_REPORTS_DIR


def _fmt(x, digits=4):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int,)):
        return str(x)
    return f"{x:.{digits}f}"


def _regression_table(comparison: list[dict]) -> str:
    lines = [
        "| Model | CV MAE | CV RMSE | CV R² | Test MAE | Test RMSE | Test R² | Selected |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in comparison:
        if row.get("status") == "skipped":
            lines.append(f"| {row['model']} | skipped | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {row['model']} | {_fmt(row['cv_mae'])} | {_fmt(row['cv_rmse'])} | {_fmt(row['cv_r2'])} | "
            f"{_fmt(row['test_mae'])} | {_fmt(row['test_rmse'])} | {_fmt(row['test_r2'])} | "
            f"{'✅' if row.get('selected') else ''} |"
        )
    return "\n".join(lines)


def _classification_table(comparison: list[dict]) -> str:
    lines = [
        "| Model | CV ROC-AUC | CV PR-AUC | CV F1 | Test ROC-AUC | Test PR-AUC | Test F1 | Selected |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in comparison:
        if row.get("status") == "skipped":
            lines.append(f"| {row['model']} | skipped | — | — | — | — | — | — |")
            continue
        test_auc = row["test_roc_auc"]
        test_auc_str = _fmt(test_auc) if test_auc is not None else "undefined*"
        lines.append(
            f"| {row['model']} | {_fmt(row['cv_roc_auc'])} | {_fmt(row['cv_pr_auc'])} | {_fmt(row['cv_f1'])} | "
            f"{test_auc_str} | {_fmt(row['test_pr_auc']) if row['test_pr_auc'] is not None else 'undefined*'} | "
            f"{_fmt(row['test_f1'])} | {'✅' if row.get('selected') else ''} |"
        )
    return "\n".join(lines)


def _fold_table(folds: list[dict]) -> str:
    lines = ["| Fold | Train laps | Val laps | Train rows | Val rows |", "|---|---|---|---:|---:|"]
    for f in folds:
        lines.append(
            f"| {f['fold_id']} | {f['train_laps'][0]}–{f['train_laps'][-1]} | "
            f"{f['val_laps'][0]}–{f['val_laps'][-1]} | {f['train_rows']} | {f['val_rows']} |"
        )
    return "\n".join(lines)


def write_regression_report(dataset, reg: dict, path: Path) -> Path:
    contract = dataset.contract
    content = f"""# Task 6 — Lap-Time Regression Report

Generated: {datetime.now(timezone.utc).isoformat()}

## Objective

Predict `target_laptime` (seconds) from information available before the lap starts.

## Features (Task 5 contract)

{', '.join(f'`{f}`' for f in reg['features'])}

## Validation strategy

Expanding-window lap-forward CV ({len(reg['folds'])} folds) over the development
set, with a chronologically later holdout test set never touched during
model selection.

- Development laps: {reg['holdout']['dev_laps'][0]}–{reg['holdout']['dev_laps'][-1]} ({reg['holdout']['dev_rows']} rows)
- Test laps: {reg['holdout']['test_laps'][0]}–{reg['holdout']['test_laps'][-1]} ({reg['holdout']['test_rows']} rows)

{_fold_table(reg['folds'])}

## Model comparison

Primary ranking metric: **CV MAE** (lower is better).

{_regression_table(reg['comparison'])}

## Best model

**{reg['best_model']}**

## Limitations

{contract.synthetic_caveat or 'None recorded in feature_metadata.json.'}
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def write_classification_report(dataset, clf: dict, path: Path) -> Path:
    contract = dataset.contract
    source = contract.dataset_source
    trained = [r for r in clf["results"] if r.status == "trained"]
    any_undefined = any(r.test_metrics.get("undefined_reason") for r in trained)
    positive_rate = float(dataset.frame[clf["target"]].mean())

    if contract.is_real_data:
        banner = (
            f"## Dataset: Real FastF1 session — {source.get('event')} {source.get('year')} "
            f"({source.get('session')}), {source.get('n_laps')} laps, {source.get('n_drivers')} drivers\n\n"
            f"{contract.synthetic_caveat}\n\n"
            "This is a single real Grand Prix session. Results reflect genuine race strategy "
            "variation but should not be generalised beyond this one race."
        )
    else:
        banner = (
            "## ⚠️ Dataset: Synthetic Demonstration\n\n"
            f"{contract.synthetic_caveat}"
        )

    if any_undefined:
        holdout_note = (
            "- The final chronological holdout test set contains **zero pit events** for at "
            "least one model, so **test-set ROC-AUC/PR-AUC are undefined** there (reported as "
            "`undefined*` below). Model selection is therefore based on cross-validation "
            "metrics, not the holdout test set, for this target."
        )
    else:
        holdout_note = (
            "- The final chronological holdout test set contains pit events, so test-set "
            "ROC-AUC/PR-AUC are defined for every trained model below."
        )

    content = f"""# Task 6 — Pit-Decision Classification Report

Generated: {datetime.now(timezone.utc).isoformat()}

## Objective

Predict `target_pit_next_lap` (should the driver pit at the end of this lap?)
from information available before the lap starts.

{banner}

Pit events are rare in every session. As a direct consequence:

- Some CV folds can have zero pit events in their training split (a classifier
  cannot be fit on a single class there) or zero pit events in their
  validation split (ROC-AUC/PR-AUC are mathematically undefined there) — see
  the per-fold table below for this run.
{holdout_note}

## Features (Task 5 contract)

{', '.join(f'`{f}`' for f in clf['features'])}

## Validation strategy

Expanding-window lap-forward CV ({len(clf['folds'])} folds).

- Development laps: {clf['holdout']['dev_laps'][0]}–{clf['holdout']['dev_laps'][-1]} ({clf['holdout']['dev_rows']} rows)
- Test laps: {clf['holdout']['test_laps'][0]}–{clf['holdout']['test_laps'][-1]} ({clf['holdout']['test_rows']} rows)

{_fold_table(clf['folds'])}

## Model comparison

Primary ranking metrics: **CV ROC-AUC**, then **CV PR-AUC**, then **CV F1**.
Accuracy is not used to rank models — with {positive_rate:.1%} positives, "always predict
no-pit" already scores {1 - positive_rate:.1%} accuracy while being useless.

{_classification_table(clf['comparison'])}

*undefined = ROC-AUC/PR-AUC could not be computed because the split's ground
truth contained only one class.

## Best model

**{clf['best_model']}**
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def write_model_selection_report(reg: dict, clf: dict, path: Path) -> Path:
    content = f"""# Task 6 — Model Selection Report

Generated: {datetime.now(timezone.utc).isoformat()}

## Selection criteria

| Task | Primary | Secondary |
|---|---|---|
| Lap-time regression | CV MAE (lower better) | CV RMSE, CV R², train/test gap |
| Pit-decision classification | CV ROC-AUC (higher better) | CV PR-AUC, CV F1, precision/recall |

Selection uses cross-validated metrics, not a single holdout number, and a
model is only preferred over another when it wins on the primary metric
computed identically across the same lap-forward folds.

## Regression

{_regression_table(reg['comparison'])}

Selected: **{reg['best_model']}**

## Classification

{_classification_table(clf['comparison'])}

Selected: **{clf['best_model']}**

## Notes on skipped models

Any model listed as "skipped" was not trained because an optional
dependency (XGBoost, and/or its native OpenMP runtime) was unavailable in
this environment. No result was fabricated in its place.
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def write_ml_evaluation_report(dataset, reg: dict, clf: dict, path: Path) -> Path:
    contract = dataset.contract
    source = contract.dataset_source
    if contract.is_real_data:
        dataset_line = (
            f"- **DATASET: Real FastF1 session** — {source.get('event')} {source.get('year')} "
            f"({source.get('session')}), fetched {source.get('fetched_at')}, "
            f"{source.get('n_laps')} laps across {source.get('n_drivers')} drivers. Not synthetic."
        )
    else:
        dataset_line = (
            "- **DATASET: Synthetic Demonstration** — generated by a reproducible "
            "synthetic FastF1-like session (Task 4), not real telemetry."
        )
    content = f"""# Task 6 — Machine Learning Model Development

Generated: {datetime.now(timezone.utc).isoformat()}

## 1. Objective

Train and compare classical machine-learning models for two F1 race-strategy
prediction problems: lap-time regression and pit-decision classification.

## 2. Dataset

- Source: `{contract.source_dataset}`
- Feature matrix: `data/processed/f1_features_selected.csv`
- Rows: {reg['holdout']['dev_rows'] + reg['holdout']['test_rows']}
{dataset_line}

## 3. Feature contract

Selected features were produced by Task 5 (Feature Engineering & Feature
Selection) and are consumed here unchanged:

- Regression features: {', '.join(f'`{f}`' for f in reg['features'])}
- Classification features: {', '.join(f'`{f}`' for f in clf['features'])}

## 4. Leakage prevention

Excluded as leakage (Task 5 contract): {', '.join(f'`{c}`' for c in contract.leakage_columns)}.

Reason: {contract.leakage_reason}

The Task 6 data contract (`app.intelligence.ml.data_contract`) asserts none
of these columns are present before any model is trained.

## 5. Validation

Expanding-window, lap-forward cross-validation with a chronologically later,
untouched holdout test set. Random K-fold is never used — see
`app.intelligence.ml.splits`.

## 6. Models

Linear/Logistic Regression, Decision Tree, Random Forest, SVM/SVR, and
XGBoost (optional — see section 9).

## 7-8. Metrics and results

See `regression_report.md`, `classification_report.md`, and
`model_selection_report.md` for full tables. Best regression model:
**{reg['best_model']}**. Best classification model: **{clf['best_model']}**.

## 9. XGBoost availability

{"XGBoost trained successfully." if any(r['model']=='xgboost' and r.get('status')!='skipped' for r in reg['comparison']) else "XGBoost unavailable — skipped."}

## 10. Limitations

{contract.synthetic_caveat}

See `classification_report.md` for the full discussion of how this affects
classification metrics in this run. {"This is real FastF1 telemetry for a single Grand Prix — results should not be generalised to other races or drivers." if contract.is_real_data else "**These are not real-world F1 performance figures.**"}

## 11. Reproducibility

- Random seed: 42 (used for every model with a stochastic component)
- Hyperparameter grids are bounded and recorded per model in
  `artifacts/metadata/model_registry.json`
- Preprocessing (imputation/scaling) is fit inside each CV fold and on the
  development set only before the final holdout evaluation

## 12. Conclusion

Task 6 provides classical ML predictions for lap time and pit timing that
downstream tasks (Expert System, Search, and the strategy simulator) can
consume alongside rule-based and search-based reasoning.
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def generate_all(dataset, reg: dict, clf: dict) -> list[Path]:
    paths = [
        write_regression_report(dataset, reg, ML_REPORTS_DIR / "regression_report.md"),
        write_classification_report(dataset, clf, ML_REPORTS_DIR / "classification_report.md"),
        write_model_selection_report(reg, clf, ML_REPORTS_DIR / "model_selection_report.md"),
    ]
    paths.append(write_ml_evaluation_report(dataset, reg, clf, ML_REPORTS_DIR / "ml_evaluation_report.md"))
    return paths
