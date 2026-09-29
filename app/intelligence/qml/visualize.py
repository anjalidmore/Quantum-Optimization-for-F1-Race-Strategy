"""
app.intelligence.qml.visualize
==============================

Figures for the quantum experiment, all written to ``artifacts/figures/qml_*``.

Every value plotted comes from the results dict the pipeline just produced —
these functions receive numbers, they never compute or assume any.
"""
from __future__ import annotations

from pathlib import Path

from app.intelligence.charts import Axis, Chart, Series, write as write_chart

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pennylane as qml  # noqa: E402
from sklearn.metrics import auc, precision_recall_curve, roc_curve  # noqa: E402

PALETTE = {"quantum": "#7c5cff", "classical (parameter-matched)": "#38bdf8",
           "task6 reference": "#f59e0b"}


def _save(fig, path: Path, tight: bool = True) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def circuit_diagram(n_qubits: int, n_layers: int, out_path: Path) -> str:
    """The actual circuit, drawn by PennyLane from the same code that trains."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x, weights):
        qml.AngleEmbedding(x, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    x = np.linspace(0.2, np.pi - 0.2, n_qubits)
    w = np.zeros(shape)
    # level="device" expands the templates into the gates actually executed,
    # which is the point of showing the circuit at all.
    fig, ax = qml.draw_mpl(circuit, decimals=2, style="pennylane", level="device")(x, w)
    ax.set_title(f"Variational circuit — {n_qubits} qubits, {n_layers} layer(s)", fontsize=10)
    # The same qnode, expanded the same way, as text. The website renders this
    # rather than the picture, so the circuit can be themed and selected.
    write_chart("qml_circuit", Chart(
        kind="circuit",
        title=f"Variational circuit, {n_qubits} qubits, {n_layers} layer(s)",
        caption="AngleEmbedding, then StronglyEntanglingLayers, expanded to the gates the "
                "device actually runs. Drawn by PennyLane from the circuit that trains.",
        x=Axis("", kind="category"), y=Axis("", kind="category"), series=[],
        extra={"text": qml.draw(circuit, decimals=2, level="device")(x, w),
               "n_qubits": int(n_qubits), "n_layers": int(n_layers)},
    ), figure=out_path)
    # qml.draw_mpl lays the figure out itself; tight_layout would fight it.
    return _save(fig, out_path, tight=False)


def loss_curves(results: dict, out_path: Path) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    series = []
    for target, r in results["targets"].items():
        for m in r["quantum"]:
            if m.get("loss_history"):
                ax.plot(range(1, len(m["loss_history"]) + 1), m["loss_history"],
                        label=f"{m['model']} ({target})", linewidth=1.8)
                series.append(Series(f"{m['model']} ({target})",
                                     list(range(1, len(m["loss_history"]) + 1)), m["loss_history"]))
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.set_title("Quantum model training loss (final fit on the development laps)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    write_chart("qml_training_loss", Chart(
        kind="line", title="Quantum model training loss",
        caption="Final fit on the development laps, parameter-shift gradients with Adam.",
        x=Axis("Epoch"), y=Axis("Training loss"), series=series), figure=out_path)
    return _save(fig, out_path)


def metric_comparison(results: dict, out_path: Path) -> str:
    """Cross-validated scores with fold spread, per target.

    Cross-validation rather than the test laps on purpose: the holdout contains
    one pit event, so every test classification metric is decided by a single
    lap. The error bars are the fold-to-fold standard deviation, and where two
    bars overlap within them, this data cannot separate those models.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (target, r) in zip(axes, results["targets"].items()):
        rows = _rows_for(r)
        key = "pr_auc" if r["task"] == "classification" else "mae"
        names, means, stds, colours, families = [], [], [], [], []
        for row in rows:
            s = ((row.get("cv_summary") or {}).get(key) or {})
            if s.get("mean") is None:
                continue
            names.append(f"{row['model']}\n({row['n_parameters']} params)"
                         if row["n_parameters"] != "—" else f"{row['model']}\n(Task 6)")
            means.append(s["mean"])
            stds.append(s.get("std") or 0.0)
            colours.append(PALETTE.get(row["family"], "#94a3b8"))
            families.append(row["family"])
        ax.bar(names, means, yerr=stds, capsize=4, color=colours, edgecolor="white", linewidth=0.8)
        better = "higher is better" if key == "pr_auc" else "lower is better"
        ax.set_ylabel(f"CV {key.upper().replace('_', '-')} ({better})")
        ax.set_title(f"{target} — {r['n_folds']}-fold cross-validation on the development laps", fontsize=10)
        ax.tick_params(axis="x", labelsize=7)
        for label in ax.get_xticklabels():
            label.set_rotation(18)
            label.set_horizontalalignment("right")
        ax.grid(alpha=0.2, axis="y")
        # One chart per target, from the same names/means/stds just plotted.
        write_chart(f"qml_metrics_{target}", Chart(
            kind="bar", title=f"{target}, {r['n_folds']}-fold cross-validation",
            caption=f"CV {key.upper().replace('_', '-')} ({better}) with fold-to-fold spread. "
                    "Cross-validation rather than the test laps, because the holdout holds one pit "
                    "event and a test metric would turn on a single lap. Where two error bars "
                    "overlap, this data cannot separate those models.",
            x=Axis("Model", kind="category"),
            y=Axis(f"CV {key.upper().replace('_', '-')}"),
            series=[Series("mean", [n.replace("\n", " ") for n in names], means)],
            extra={"error": stds, "families": families, "lower_is_better": key != "pr_auc"},
        ), figure=out_path)
    fig.suptitle("Quantum vs parameter-matched classical on the same inputs "
                 "vs Task 6 on all features", fontsize=9)
    return _save(fig, out_path)


def training_time(results: dict, out_path: Path) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    names, times, colours = [], [], []
    for target, r in results["targets"].items():
        for row in _rows_for(r):
            if row["family"] == "task6 reference":
                continue
            names.append(f"{row['model']}\n({target.replace('target_', '')})")
            times.append(row.get("train_seconds") or 0.0)
            colours.append(PALETTE.get(row["family"], "#94a3b8"))
    ax.barh(names, times, color=colours, edgecolor="white")
    ax.set_xlabel("final-fit training time (seconds, wall clock)")
    ax.set_title("Training cost — simulated circuits vs equivalent classical models")
    ax.set_xscale("log")
    for i, t in enumerate(times):
        ax.text(t * 1.05, i, f"{t:.2f}s", va="center", fontsize=8)
    ax.grid(alpha=0.2, axis="x")
    write_chart("qml_training_time", Chart(
        kind="bar-horizontal", title="Training cost",
        caption="Final-fit wall clock. Simulated circuits against the equivalent classical models. "
                "A simulator is not quantum hardware, so this is a cost of simulation, not of quantum computing.",
        x=Axis("Training time", unit="s"), y=Axis("Model", kind="category"),
        series=[Series("Training time", names, times)], extra={"log_x": True}), figure=out_path)
    return _save(fig, out_path)


def roc_pr_curves(r: dict, out_path: Path) -> str:
    """ROC and precision-recall for every classifier, quantum and classical."""
    preds = r["predictions"]
    y = np.asarray(preds["y_test"])
    roc_series: list[Series] = []
    pr_series: list[Series] = []
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, scores in preds.items():
        if name == "y_test":
            continue
        s = np.asarray(scores, dtype=float)
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC {auc(fpr, tpr):.3f})", linewidth=1.8)
        roc_series.append(Series(f"{name} (AUC {auc(fpr, tpr):.3f})", fpr, tpr))
        precision, recall, _ = precision_recall_curve(y, s)
        ax_pr.plot(recall, precision, label=f"{name} (AP {auc(recall, precision):.3f})", linewidth=1.8)
        pr_series.append(Series(f"{name} (AP {auc(recall, precision):.3f})", recall, precision))
    ax_roc.plot([0, 1], [0, 1], "--", color="#64748b", linewidth=1, label="chance")
    ax_roc.set_xlabel("false positive rate")
    ax_roc.set_ylabel("true positive rate")
    ax_roc.set_title(f"ROC — pit decision ({int(y.sum())} pit lap(s) of {len(y)})", fontsize=10)
    ax_roc.legend(fontsize=7)
    ax_roc.grid(alpha=0.2)
    ax_pr.set_xlabel("recall")
    ax_pr.set_ylabel("precision")
    ax_pr.set_title("Precision-recall — pit decision", fontsize=10)
    ax_pr.legend(fontsize=7)
    ax_pr.grid(alpha=0.2)
    write_chart("qml_roc_vs_classical", Chart(
        kind="line", title=f"ROC, pit decision ({int(y.sum())} pit lap(s) of {len(y)})",
        caption="Quantum and classical classifiers on the same reduced inputs. The diagonal is chance.",
        x=Axis("False positive rate"), y=Axis("True positive rate"),
        series=roc_series, extra={"identity_line": [0, 1]}), figure=out_path)
    write_chart("qml_pr_vs_classical", Chart(
        kind="line", title="Precision-recall, pit decision",
        caption="With one pit lap in the holdout, this curve is decided by a single event. Read it as such.",
        x=Axis("Recall"), y=Axis("Precision"), series=pr_series), figure=out_path)
    return _save(fig, out_path)


def predicted_vs_actual(r: dict, out_path: Path) -> str:
    preds = r["predictions"]
    y = np.asarray(preds["y_test"], dtype=float)
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    lo, hi = float(min(y)) - 1, float(max(y)) + 1
    ax.plot([lo, hi], [lo, hi], "--", color="#64748b", linewidth=1, label="perfect")
    for name, p in preds.items():
        if name == "y_test":
            continue
        ax.scatter(y, np.asarray(p, dtype=float), s=16, alpha=0.65, label=name)
    ax.set_xlabel("actual lap time (s)")
    ax.set_ylabel("predicted lap time (s)")
    ax.set_title("Predicted vs actual — test laps", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    write_chart("qml_predicted_vs_actual", Chart(
        kind="scatter", title="Predicted vs actual lap time, test laps",
        caption="Every regressor on the same reduced inputs. The diagonal is a perfect prediction.",
        x=Axis("Actual lap time", unit="s"), y=Axis("Predicted lap time", unit="s"),
        series=[Series(name, y, np.asarray(pv, dtype=float))
                for name, pv in preds.items() if name != "y_test"],
        extra={"identity_line": [lo, hi]}), figure=out_path)
    return _save(fig, out_path)


def _rows_for(r: dict) -> list[dict]:
    """Quantum models, the parameter-matched classical ones, and Task 6's model."""
    rows = list(r["quantum"]) + list(r["classical_matched"])
    ref = r.get("task6_reference") or {}
    if ref.get("available"):
        rows.append({
            "model": f"{ref['model']} (Task 6, all features)",
            "family": "task6 reference",
            "n_parameters": "—",
            "train_seconds": None,
            "cv_summary": ref.get("cv_summary", {}),
            "test_metrics": ref["test_metrics"],
        })
    return rows


def generate_all(results: dict, out) -> dict:
    """Write every figure; return {name: path relative to the artifact root}."""
    figures: dict[str, str] = {}
    fig_dir = out.figures

    clf = results["targets"]["target_pit_next_lap"]
    reg = results["targets"]["target_laptime"]
    vqc_hp = clf["quantum"][0]["hyperparameters"]

    figures["circuit"] = _rel(out, circuit_diagram(
        vqc_hp["n_qubits"], vqc_hp["n_layers"], fig_dir / "qml_circuit.png"))
    figures["loss_curves"] = _rel(out, loss_curves(results, fig_dir / "qml_training_loss.png"))
    figures["metric_comparison"] = _rel(out, metric_comparison(results, fig_dir / "qml_metric_comparison.png"))
    figures["training_time"] = _rel(out, training_time(results, fig_dir / "qml_training_time.png"))
    figures["roc_pr"] = _rel(out, roc_pr_curves(clf, fig_dir / "qml_roc_pr_vs_classical.png"))
    figures["predicted_vs_actual"] = _rel(out, predicted_vs_actual(reg, fig_dir / "qml_predicted_vs_actual.png"))
    return figures


def _rel(out, path: str) -> str:
    return str(Path(path).resolve().relative_to(out.root.resolve()))
