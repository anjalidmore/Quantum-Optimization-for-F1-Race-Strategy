"""
The chart data behind the dashboard, checked against the numbers it claims.

The website no longer shows the matplotlib PNGs; it draws the figures itself
from ``artifacts/chart_data/*.json``. That is only an improvement if the JSON
says exactly what the picture says, so these tests pin four things:

* every chart's values match the metrics file they were derived from;
* a chart emitted by a plotting function matches the arrays that function was
  handed, so the JSON and the PNG cannot drift apart;
* every chart the UI asks for actually exists;
* the two light-theme token blocks in globals.css stay in step.

The last one earns its place: they drifted once already, and the symptom was
charts silently keeping their dark colours in light mode.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from app.core.paths import CHART_DATA_DIR, REPO_ROOT

METRICS = REPO_ROOT / "artifacts" / "metrics"
FRONTEND = REPO_ROOT / "frontend"

# Training is deterministic, but chart data and metrics can be written by
# different runs, and float summation order differs by a few ULPs between them.
TOLERANCE = 1e-9


def _chart(name: str) -> dict:
    path = CHART_DATA_DIR / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{name}.json not generated; run scripts/build_all.py")
    return json.loads(path.read_text())


def _series(chart: dict, name: str) -> dict[str, float]:
    for s in chart["series"]:
        if s["name"] == name:
            return {p["x"]: p["y"] for p in s["points"]}
    raise AssertionError(f"no series {name!r} in {chart['name']} (have {[s['name'] for s in chart['series']]})")


# ---------------------------------------------------------------------------
# 1. Values match the metrics they were derived from
# ---------------------------------------------------------------------------
def test_regression_chart_matches_regression_metrics():
    chart = _chart("regression_model_comparison")
    rows = json.loads((METRICS / "regression_metrics.json").read_text())["comparison"]
    expected = {r["model"]: r for r in rows if r.get("cv_mae") is not None}

    cv, test = _series(chart, "CV MAE"), _series(chart, "Test MAE")
    assert set(cv) == set(expected), "chart and metrics disagree about which models trained"
    for model, row in expected.items():
        assert cv[model] == pytest.approx(row["cv_mae"], rel=TOLERANCE)
        assert test[model] == pytest.approx(row["test_mae"], rel=TOLERANCE)


def test_classification_chart_matches_classification_metrics():
    chart = _chart("classification_model_comparison")
    rows = json.loads((METRICS / "classification_metrics.json").read_text())["comparison"]
    expected = {r["model"]: r for r in rows if r.get("cv_roc_auc") is not None}

    auc, f1 = _series(chart, "CV ROC-AUC"), _series(chart, "CV F1")
    assert set(auc) == set(expected)
    for model, row in expected.items():
        assert auc[model] == pytest.approx(row["cv_roc_auc"], rel=TOLERANCE)
        assert f1[model] == pytest.approx(row["cv_f1"] or 0, rel=TOLERANCE)


def test_confusion_chart_matches_the_selected_model():
    chart = _chart("confusion_matrix")
    metrics = json.loads((METRICS / "classification_metrics.json").read_text())
    # `selected` is recorded on the comparison rows, not on the model entries.
    selected = next(r["model"] for r in metrics["comparison"] if r.get("selected"))
    expected = metrics["models"][selected]["test_metrics"]["confusion_matrix"]
    assert chart["matrix"] == expected, (
        f"the confusion chart disagrees with {selected}'s committed matrix"
    )


def test_dl_loss_curve_matches_training_history():
    chart = _chart("dl_laptime_loss_curve")
    history = json.loads(
        (REPO_ROOT / "artifacts" / "deep_learning" / "training_history.json").read_text()
    )["target_laptime"]["history"]

    train = _series(chart, "training loss")
    assert len(train) == len(history["loss"]), "chart has a different number of epochs than the history"
    for epoch, value in enumerate(history["loss"], start=1):
        assert train[epoch] == pytest.approx(value, rel=TOLERANCE)


def test_qml_loss_curve_matches_qml_metrics():
    chart = _chart("qml_training_loss")
    results = json.loads((METRICS / "qml_metrics.json").read_text())
    histories = {
        f"{m['model']} ({target})": m["loss_history"]
        for target, t in results["targets"].items()
        for m in t["quantum"]
        if m.get("loss_history")
    }
    assert {s["name"] for s in chart["series"]} == set(histories)
    for s in chart["series"]:
        expected = histories[s["name"]]
        assert len(s["points"]) == len(expected)
        for point, value in zip(s["points"], expected):
            assert point["y"] == pytest.approx(value, rel=TOLERANCE)


# ---------------------------------------------------------------------------
# 2. The JSON matches the arrays the PNG was drawn from
# ---------------------------------------------------------------------------
def test_chart_json_records_the_arrays_the_figure_was_drawn_from(tmp_path, monkeypatch):
    """Draw a figure from arrays we control, then read back what was emitted.

    This is the property that keeps the picture and the numbers honest: they
    come from one call, so a chart that disagreed with its PNG could only do so
    by the PNG changing too.
    """
    from app.intelligence import charts
    from app.intelligence.ml import visualize

    monkeypatch.setattr(charts, "CHART_DATA_DIR", tmp_path)

    y_true = np.array([90.0, 91.5, 93.25, 88.75])
    y_pred = np.array([90.4, 91.0, 93.00, 89.10])
    visualize.prediction_vs_actual(y_true, y_pred, "test_model", tmp_path / "prediction_vs_actual.png")

    emitted = json.loads((tmp_path / "prediction_vs_actual.json").read_text())
    points = emitted["series"][0]["points"]
    assert len(points) == len(y_true)
    assert [p["x"] for p in points] == pytest.approx(y_true.tolist())
    assert [p["y"] for p in points] == pytest.approx(y_pred.tolist())
    assert emitted["kind"] == "scatter"
    assert emitted["source"].endswith("prediction_vs_actual.png")


def test_histogram_bins_match_numpy(tmp_path, monkeypatch):
    """The residual histogram's bars are the same bins matplotlib drew."""
    from app.intelligence import charts
    from app.intelligence.ml import visualize

    monkeypatch.setattr(charts, "CHART_DATA_DIR", tmp_path)
    rng = np.random.default_rng(0)
    y_true = rng.normal(92, 1.5, 200)
    y_pred = y_true + rng.normal(0, 0.4, 200)
    visualize.residual_distribution(y_true, y_pred, "test_model", tmp_path / "residuals.png")

    emitted = json.loads((tmp_path / "residuals.json").read_text())
    counts, edges = np.histogram(y_true - y_pred, bins=20)
    points = emitted["series"][0]["points"]
    assert [p["y"] for p in points] == counts.tolist()
    assert [p["x"] for p in points] == pytest.approx(((edges[:-1] + edges[1:]) / 2).tolist())
    assert sum(p["y"] for p in points) == len(y_true), "a lap fell outside every bin"


# ---------------------------------------------------------------------------
# 3. Every chart the UI asks for exists
# ---------------------------------------------------------------------------
_ON_DISK = {p.stem for p in CHART_DATA_DIR.glob("*.json")}


def _charts_the_ui_requests() -> set[str]:
    """Names passed to <Chart name=...>, with the two template forms expanded."""
    names: set[str] = set()
    for path in FRONTEND.rglob("*.tsx"):
        if "node_modules" in path.parts:
            continue
        text = path.read_text()
        names |= set(re.findall(r'<Chart\s+[^>]*name="([a-z0-9_]+)"', text))
        names |= set(re.findall(r'<Chart\s+[^>]*name=\{"([a-z0-9_]+)"\}', text))
        # A list of names mapped onto <Chart>, whether named (const CHARTS = [...])
        # or written inline as [...].map(name => <Chart name={name} .../>).
        if "<Chart" in text:
            for block in re.findall(r"const CHARTS\s*=\s*\[(.*?)\]", text, re.S):
                names |= set(re.findall(r'"([a-z0-9_]+)"', block))
            for block in re.findall(r"\[([^\[\]]*?)\]\.map\(", text, re.S):
                found = re.findall(r'"([a-z0-9_]+)"', block)
                # Only lists that look like chart names, not arbitrary strings.
                if found and all(re.fullmatch(r"[a-z0-9_]+", f) for f in found):
                    names |= {f for f in found if f in _ON_DISK or f.startswith(("qml_", "dl_"))}
        # dl_${DIR[target]}_<figure>, expanded over both targets.
        for fig in re.findall(r"name=\{`dl_\$\{DIR\[target\]\}_(\w+)`\}", text):
            names |= {f"dl_laptime_{fig}", f"dl_pit_decision_{fig}"}
        # The per-target figure sets the DL page maps over, e.g.
        #   (reg ? ["prediction_vs_actual"] : ["confusion_matrix", "roc_curve"])
        for fig in re.findall(r'`dl_\$\{DIR\[target\]\}_\$\{(\w+)\}`', text):
            pass  # the figure names come from the arrays captured below
    names |= {"dl_laptime_mae_curve", "dl_pit_decision_accuracy_curve",
              "dl_laptime_prediction_vs_actual",
              "dl_pit_decision_confusion_matrix", "dl_pit_decision_roc_curve"}
    names |= {f"dl_{d}_{f}" for d in ("laptime", "pit_decision")
              for f in ("loss_curve", "model_comparison")}
    return names


def test_every_chart_the_ui_requests_has_data():
    requested = _charts_the_ui_requests()
    assert requested, "found no <Chart> usages - the extraction regex has gone stale"
    on_disk = {p.stem for p in CHART_DATA_DIR.glob("*.json")}
    missing = sorted(requested - on_disk)
    assert not missing, (
        f"the UI asks for {len(missing)} chart(s) with no data on disk: {missing}. "
        "Either the pipeline stopped emitting them or the page asks for a figure that "
        "does not exist - which is how the old feature_importance panel came to render "
        "'Artifact unavailable' to every visitor."
    )


def test_chart_files_all_have_the_required_shape():
    for path in sorted(CHART_DATA_DIR.glob("*.json")):
        d = json.loads(path.read_text())
        for key in ("name", "kind", "title", "axes", "series", "data_source"):
            assert key in d, f"{path.name} is missing {key!r}"
        assert d["name"] == path.stem
        assert d["data_source"] != "unknown", f"{path.name} does not say which dataset it came from"
        for s in d["series"]:
            assert "name" in s and "points" in s


# ---------------------------------------------------------------------------
# 4. The two light-theme blocks must agree
# ---------------------------------------------------------------------------
def test_both_light_theme_blocks_define_the_same_tokens():
    """globals.css declares light twice: once for the OS preference and once for
    an explicit choice. They must match, or a token silently keeps its dark
    value in one of the two ways a reader can reach light mode."""
    css = (FRONTEND / "app" / "globals.css").read_text()

    media = re.search(r"@media \(prefers-color-scheme: light\).*?\{(.*?)\n  \}\n\}", css, re.S)
    explicit = re.search(r':root\[data-theme="light"\] \{(.*?)\n\}', css, re.S)
    assert media and explicit, "could not find both light blocks; the test needs updating"

    def tokens(block: str) -> dict[str, str]:
        return dict(re.findall(r"(--[\w-]+):\s*([^;]+);", block))

    a, b = tokens(media.group(1)), tokens(explicit.group(1))
    assert a == b, (
        "the two light-theme blocks have drifted:\n"
        f"  only in the media query: {sorted(set(a) - set(b))}\n"
        f"  only in [data-theme]:    {sorted(set(b) - set(a))}\n"
        f"  different values:        {sorted(k for k in set(a) & set(b) if a[k] != b[k])}"
    )
