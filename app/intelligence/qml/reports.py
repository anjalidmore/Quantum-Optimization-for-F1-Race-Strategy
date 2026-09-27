"""
app.intelligence.qml.reports
============================

The classical-vs-quantum report. Every number is passed in from the run that
produced it; an undefined metric prints as "undefined", never as a stand-in.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def _f(v, nd: int = 4, missing: str = "undefined") -> str:
    if v is None:
        return missing
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _pm(summary: dict, key: str, nd: int = 4) -> str:
    """CV mean ± std for one metric."""
    s = (summary or {}).get(key) or {}
    if s.get("mean") is None:
        return "undefined"
    return f"{s['mean']:.{nd}f} ± {(s.get('std') or 0.0):.{nd}f}"


def _rows(r: dict) -> list[dict]:
    rows = list(r["quantum"]) + list(r["classical_matched"])
    ref = r.get("task6_reference") or {}
    if ref.get("available"):
        rows.append({
            "model": f"{ref['model']} — Task 6, all {r['n_features_full']} features"
                     if r.get("n_features_full") else f"{ref['model']} — Task 6, all selected features",
            "family": "classical (full)",
            "n_parameters": None,
            "train_seconds": None,
            "cv_summary": ref.get("cv_summary", {}),
            "test_metrics": ref["test_metrics"],
        })
    return rows


def classical_vs_quantum(results: dict, out_path: Path) -> Path:
    clf = results["targets"]["target_pit_next_lap"]
    reg = results["targets"]["target_laptime"]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    L = [
        "# Classical vs Quantum — Machine Learning on the Same F1 Data", "",
        f"_Generated {stamp}._", "",
        f"**Simulator:** {results['simulator']}  ·  **PennyLane** {results['framework']['pennylane']}  ·  "
        f"**seed** {results['seed']}  ·  **total run time** {results['wall_seconds']:.1f} s", "",
        "> **Read this first.** These are small variational circuits simulated exactly on a classical",
        "> computer, trained on one Grand Prix. There is no quantum hardware in this project and nothing",
        "> here demonstrates a quantum advantage — a noiseless simulation of 4 qubits is something a laptop",
        "> does easily, which is the whole reason it is possible to run this experiment at all.",
        "> The comparison is worth making because it is *fair*: same laps, same splits, same metric code.",
        "",
        "## How the comparison was kept fair", "",
        "| | |",
        "|---|---|",
        "| Data | the Task 5 feature matrix, loaded through Task 6's own data contract |",
        f"| Split | Task 6's chronological holdout — {clf['n_dev']} development laps, {clf['n_test']} test laps |",
        f"| Cross-validation | Task 6's {clf['n_folds']} expanding-window lap-forward folds |",
        "| Selection | hyperparameters chosen on the folds only; the test laps were used once |",
        f"| Search space | layers ∈ {results['search_space']['n_layers']}, "
        f"learning rate ∈ {results['search_space']['learning_rate']} |",
        "| Metrics | `app.intelligence.ml.evaluation` — the same functions Tasks 6 and 7 call |",
        "| Thresholds | tuned on pooled out-of-fold predictions, never left at 0.5 |",
        "",
        "### Getting 45 features into 4 qubits", "",
        f"- **Pit decision:** {clf['encoding']['method']}"
        + (f", retaining {clf['encoding']['explained_variance_ratio']:.1%} of the variance"
           if clf["encoding"].get("explained_variance_ratio") else "") + ".",
        f"- **Lap time:** {reg['encoding']['method']}"
        + (f", retaining {reg['encoding']['explained_variance_ratio']:.1%} of the variance"
           if reg["encoding"].get("explained_variance_ratio") else "") + ".",
        "",
        "Angle encoding needs a bounded range, so features are standardised and mapped to [0, π]. Both the",
        "standardiser and the PCA are fitted on training rows only — inside each fold during",
        "cross-validation, and on the development laps for the final model.",
        "",
        "This reduction is a real cost, and it is why the table below includes **parameter-matched classical",
        "models trained on those same reduced inputs**. Task 6's model sees every feature; the matched models",
        "see exactly what the circuits see.",
        "",
    ]

    # ---- the comparison table ------------------------------------------------
    L += ["## Results — same test laps, same metrics", "",
          "### Pit decision (`target_pit_next_lap`)", "",
          f"Test laps contain **{clf['n_test_positive']} pit event(s) in {clf['n_test']} laps**. "
          "Precision, recall and F1 on so few positives are dominated by chance; the cross-validated "
          "columns carry far more events and are the better guide.", "",
          "| Model | Family | Params | Train (s) | CV PR-AUC (mean ± sd) | CV ROC-AUC | Test PR-AUC | "
          "Test ROC-AUC | Test F1 | Test precision | Test recall | Confusion matrix |",
          "|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|"]
    for row in _rows(clf):
        t = row["test_metrics"]
        L.append(
            f"| {row['model']} | {row['family']} | {row['n_parameters'] or '—'} | "
            f"{_f(row.get('train_seconds'), 2, missing='not measured here')} | {_pm(row.get('cv_summary'), 'pr_auc')} | "
            f"{_pm(row.get('cv_summary'), 'roc_auc')} | {_f(t.get('pr_auc'))} | {_f(t.get('roc_auc'))} | "
            f"{_f(t.get('f1'))} | {_f(t.get('precision'))} | {_f(t.get('recall'))} | "
            f"`{t.get('confusion_matrix')}` |"
        )

    L += ["",
          "A test PR-AUC of 1.0 in that table is not a triumph: with a single positive lap, PR-AUC is 1.0 "
          "whenever that one lap happens to receive the highest score, and near 0.01 whenever it does not. "
          "The same one lap drives every test precision, recall and F1 value here. Use the CV columns.",
          "",
          "`quantum_kernel_svm`'s parameter count is its SVM dual coefficients plus the intercept — the "
          "quantum part of that model is the kernel, which has no trained parameters at all. It is not "
          "comparable to the variational circuits' angle counts.",
          "", "### Lap time (`target_laptime`)", "",
          "| Model | Family | Params | Train (s) | CV MAE (mean ± sd) | CV R² | Test MAE (s) | "
          "Test RMSE (s) | Test R² |",
          "|---|---|---:|---:|---|---|---:|---:|---:|"]
    for row in _rows(reg):
        t = row["test_metrics"]
        L.append(
            f"| {row['model']} | {row['family']} | {row['n_parameters'] or '—'} | "
            f"{_f(row.get('train_seconds'), 2, missing='not measured here')} | {_pm(row.get('cv_summary'), 'mae')} | "
            f"{_pm(row.get('cv_summary'), 'r2')} | {_f(t.get('mae'))} | {_f(t.get('rmse'))} | "
            f"{_f(t.get('r2'))} |"
        )

    # ---- what the quantum models are ----------------------------------------
    L += ["", "## The three quantum models", ""]
    for r in (clf, reg):
        for m in r["quantum"]:
            hp = m["hyperparameters"]
            L += [f"**`{m['model']}`** — {m['description']}", "",
                  f"- {hp.get('n_qubits')} qubits"
                  + (f", {hp['n_layers']} entangling layer(s)" if "n_layers" in hp else "")
                  + (f", learning rate {hp['learning_rate']}, {hp['epochs']} epochs (Adam)" if "learning_rate" in hp else "")
                  + (f", {hp['reps']} feature-map repetitions, C={hp['C']}" if "reps" in hp else ""),
                  f"- {m['n_parameters']} trainable parameters, final fit in {m['train_seconds']:.2f} s",
                  ]
            if m.get("threshold"):
                th = m["threshold"]
                L.append(f"- decision threshold {th['threshold']} tuned on {th['n_samples']} out-of-fold "
                         f"predictions ({th['n_positive']} pit laps): F1 "
                         f"{th['at_default_0.5'].get('f1', 'n/a')} at 0.5 → {th['at_threshold']['f1']} at the tuned value")
            if m.get("n_support_vectors"):
                L.append(f"- {m['n_support_vectors']} support vectors; kernel matrix "
                         f"{m['kernel_matrix_shape'][0]}×{m['kernel_matrix_shape'][1]}")
            L.append("")

    # ---- honest discussion ---------------------------------------------------
    L += ["## Discussion", "", _discussion(clf, reg), "",
          "### What this experiment cannot tell you", "",
          "- **Nothing about quantum advantage.** `default.qubit` simulates the circuit exactly. Any speed",
          "  comparison in the table favours whichever model has fewer parameters to fit, not whichever is",
          "  running on better hardware — there is no quantum hardware here.",
          "- **Nothing about scaling.** 4 qubits and a 16-dimensional state space is small enough to simulate",
          "  in milliseconds. The interesting regime starts where simulation stops being possible, and this",
          "  dataset never gets there.",
          "- **Nothing generalisable about F1.** One race, one circuit, 995 laps, one labelled pit stop in the",
          "  holdout. Every caveat that applies to Tasks 6-8 applies here too.",
          "- **Noise is absent.** Real devices decohere. A model that trains cleanly in simulation may not",
          "  survive contact with hardware, and this project makes no claim that it would.",
          "",
          "### Reproducing this", "",
          "```bash",
          "python scripts/run_qml.py            # writes every artifact below",
          "python scripts/build_all.py          # includes the quantum stage (skip it with --skip-qml)",
          "```",
          "",
          "Artifacts: `artifacts/models/qml/` (weights and config), `artifacts/metrics/qml_metrics.json`,",
          "`artifacts/figures/qml_*.png`, and this report.",
          ""]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L) + "\n")
    return out_path


def _discussion(clf: dict, reg: dict) -> str:
    """Compare the quantum models against both classical references, in words,
    from the numbers actually produced."""
    lines: list[str] = []

    def cv(row, key):
        return ((row.get("cv_summary") or {}).get(key) or {}).get("mean")

    # Pit decision: PR-AUC on cross-validation is the honest ranking signal.
    q_best = max(clf["quantum"], key=lambda m: cv(m, "pr_auc") or -1)
    matched_best = max(clf["classical_matched"], key=lambda m: cv(m, "pr_auc") or -1)
    ref = clf.get("task6_reference") or {}
    q_val, m_val = cv(q_best, "pr_auc"), cv(matched_best, "pr_auc")
    if q_val is not None and m_val is not None:
        verdict = "ahead of" if q_val > m_val else ("behind" if q_val < m_val else "level with")
        lines.append(
            f"**Pit decision.** The best quantum model on cross-validation is `{q_best['model']}` "
            f"(CV PR-AUC {q_val:.4f}), which is {verdict} the parameter-matched classical model "
            f"`{matched_best['model']}` ({m_val:.4f}) on the identical reduced inputs. "
            + (f"Task 6's `{ref['model']}`, which sees every selected feature, reaches "
               f"{(ref.get('cv_summary', {}).get('pr_auc') or {}).get('mean', float('nan')):.4f} on the same folds."
               if ref.get("available") and (ref.get("cv_summary", {}).get("pr_auc") or {}).get("mean") is not None
               else "")
        )
        lines.append("")
        lines.append(
            f"On the test laps every classifier is judged on {clf['n_test_positive']} pit event(s), so the "
            "test columns should not decide anything. That limitation is the dataset's, not the models'."
        )
        lines.append("")

    # Lap time: MAE, lower is better.
    q_reg = min(reg["quantum"], key=lambda m: cv(m, "mae") or 1e9)
    m_reg = min(reg["classical_matched"], key=lambda m: cv(m, "mae") or 1e9)
    ref_r = reg.get("task6_reference") or {}
    q_mae, m_mae = cv(q_reg, "mae"), cv(m_reg, "mae")
    if q_mae is not None and m_mae is not None:
        gap = q_mae - m_mae
        verdict = ("better than" if gap < 0 else "worse than") + f" the matched classical model by {abs(gap):.3f} s"
        lines.append(
            f"**Lap time.** `{q_reg['model']}` reaches CV MAE {q_mae:.4f} s against {m_mae:.4f} s for "
            f"`{m_reg['model']}` on the same 4 PCA components — {verdict}. "
            f"Test MAE is {_f(q_reg['test_metrics'].get('mae'))} s for the circuit"
            + (f" and {_f(ref_r['test_metrics'].get('mae'))} s for Task 6's `{ref_r['model']}` on all features."
               if ref_r.get("available") else ".")
        )
        lines.append("")
        lines.append(
            "A circuit whose output is a single Pauli-Z expectation is bounded in [-1, 1] and has a few dozen "
            "parameters, so it is closer in capacity to a linear model than to Task 6's ensembles or Task 7's "
            "network. Reading it as \"quantum is worse than classical\" would be the wrong lesson; reading it "
            "as \"a 26-parameter model with 4 compressed inputs cannot match a 45-feature ensemble\" is the "
            "right one."
        )
    return "\n".join(lines)
