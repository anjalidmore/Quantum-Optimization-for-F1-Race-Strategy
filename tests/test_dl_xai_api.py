"""
Task 7 & 8 — API endpoint tests.

Asserts the endpoints exist, return real artifact-backed data rather than
placeholders, and fail honestly (404/422) rather than inventing a value when
something is missing.
"""
from __future__ import annotations

import json

from app.core.runtime import prepare_dl_runtime

prepare_dl_runtime()

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.core.paths import DL_METRICS_JSON, ML_METRICS_DIR, XAI_RESULTS_JSON

client = TestClient(app)

_dl = pytest.mark.skipif(not DL_METRICS_JSON.exists(), reason="Task 7 not built")
_xai = pytest.mark.skipif(not XAI_RESULTS_JSON.exists(), reason="Task 8 not built")

TARGETS = ("target_laptime", "target_pit_next_lap")


# --------------------------------------------------------------------------
# Task 7
# --------------------------------------------------------------------------
@_dl
def test_dl_models_returns_only_deep_entries():
    r = client.get("/api/dl/models")
    assert r.status_code == 200
    models = r.json()["models"]
    assert models and all(m["family"] == "deep" for m in models)
    assert {m["target"] for m in models} == set(TARGETS)


@_dl
def test_dl_metrics_match_the_artifact_on_disk():
    """The endpoint must serve the artifact, not a copy that could drift."""
    api = client.get("/api/dl/metrics").json()
    disk = json.loads(DL_METRICS_JSON.read_text())
    assert api == disk


@_dl
def test_dl_comparison_includes_task6_classical_rows():
    body = client.get("/api/dl/comparison").json()
    for target in TARGETS:
        rows = body["targets"][target]["comparison"]
        families = {r["family"] for r in rows}
        assert "deep" in families, f"{target}: no deep row"
        assert "classical" in families, f"{target}: nothing to compare against"


@_dl
def test_dl_comparison_classical_rows_come_from_task6_artifacts():
    """Spot-check the no-fabrication contract: a classical metric shown by the
    DL comparison must be byte-identical to Task 6's own committed file."""
    body = client.get("/api/dl/comparison").json()
    reg = json.loads((ML_METRICS_DIR / "regression_metrics.json").read_text())
    for row in body["targets"]["target_laptime"]["comparison"]:
        if row["family"] != "classical":
            continue
        on_disk = reg["models"][row["model"]]["test_metrics"]
        for key, value in row["metrics"].items():
            if not isinstance(value, (int, float)):
                continue
            # Compared with a tolerance rather than for equality: this suite
            # retrains Task 6, and its metrics move at the ~1e-14 level from
            # floating-point non-determinism between runs. The point of this
            # test is that the numbers come from Task 6's artifacts at all,
            # not that they are bit-identical across a retrain.
            assert value == pytest.approx(on_disk[key], rel=1e-9), f"{row['model']}.{key}"


@_dl
def test_dl_history_reports_real_epoch_counts():
    body = client.get("/api/dl/history").json()
    for target in TARGETS:
        h = body[target]
        assert h["epochs_run"] == len(h["history"]["loss"]) > 0
        assert 1 <= h["best_epoch"] <= h["epochs_run"]


@_dl
def test_dl_artifacts_lists_only_files_that_exist():
    from app.core.paths import ARTIFACTS_DIR

    body = client.get("/api/dl/artifacts").json()
    for rel in body["figures"] + body["models"]:
        assert (ARTIFACTS_DIR / rel).exists(), rel
    assert body["model_format"] == ".keras"


@_dl
def test_dl_predict_rejects_a_feature_mismatch():
    r = client.post("/api/dl/predict/laptime", json={"not_a_real_feature": 1.0})
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "feature mismatch"


@_dl
def test_dl_predict_returns_a_real_number_for_a_valid_payload():
    from app.intelligence.features.contract import load_feature_contract

    contract = load_feature_contract()
    payload = {f: 0.0 for f in contract.selected_features("target_laptime")}
    r = client.post("/api/dl/predict/laptime", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "dnn_mlp"
    assert isinstance(body["prediction"], float)
    assert body["model_format"] == ".keras"
    assert body["data_source"]


# --------------------------------------------------------------------------
# Task 8
# --------------------------------------------------------------------------
@_xai
def test_xai_summary_covers_both_targets():
    body = client.get("/api/xai/summary").json()
    assert set(body["targets"]) == set(TARGETS)
    for t, r in body["targets"].items():
        assert r["explained_rows"], f"{t} has no explained rows"
        assert r["classical_model_explained"]


@_xai
@pytest.mark.parametrize("endpoint", [
    "/api/xai/feature-importance", "/api/xai/shap", "/api/xai/lime",
    "/api/xai/counterfactual", "/api/xai/trust-score", "/api/xai/explanation",
])
def test_xai_endpoints_return_data_for_each_target(endpoint):
    for target in TARGETS:
        r = client.get(f"{endpoint}?target={target}")
        assert r.status_code == 200, f"{endpoint}?target={target} -> {r.status_code}"
        assert r.json()


@_xai
def test_xai_rejects_an_unknown_target():
    r = client.get("/api/xai/shap?target=target_nonsense")
    assert r.status_code == 422


@_xai
def test_xai_rejects_an_unknown_row_label():
    r = client.get("/api/xai/shap?target=target_laptime&row=not_a_row")
    assert r.status_code == 404
    assert "Available" in r.json()["detail"]


@_xai
def test_xai_fairness_matches_the_committed_artifact():
    api = client.get("/api/xai/fairness").json()
    disk = json.loads(XAI_RESULTS_JSON.read_text())["targets"]
    for target in TARGETS:
        assert api[target]["identity_attribution_share"] == pytest.approx(
            disk[target]["fairness"]["identity_attribution_share"])


@_xai
def test_xai_explanation_gives_a_sentence_and_a_trust_band():
    body = client.get("/api/xai/explanation?target=target_pit_next_lap").json()
    for label, row in body["rows"].items():
        assert isinstance(row["narrative"], str) and len(row["narrative"]) > 40
        assert row["trust_band"]["label"] in {"HIGH", "MODERATE", "LOW", "DO NOT ACT"}
        assert row["top_factors"]


@_xai
def test_xai_trust_score_exposes_its_formula_and_weights():
    body = client.get("/api/xai/trust-score?target=target_laptime").json()
    assert "confidence" in body["formula"]
    assert sum(body["weights"].values()) == pytest.approx(1.0)
    assert set(body["bands"]) == {"HIGH", "MODERATE", "LOW", "DO NOT ACT"}


@_xai
def test_xai_counterfactual_reports_reachability_explicitly():
    body = client.get("/api/xai/counterfactual?target=target_pit_next_lap").json()
    for label, row in body["rows"].items():
        assert "reachable" in row["scan"]
        if not row["scan"]["reachable"]:
            assert row["scan"]["searched_range"], "unreachable reported without the range searched"


# --------------------------------------------------------------------------
# Task 6 endpoints must be unaffected by the Task 7 registry extension
# --------------------------------------------------------------------------
def test_existing_ml_endpoints_still_work_after_the_registry_extension():
    for url in ("/api/health", "/api/ml/models", "/api/ml/metrics", "/api/ml/comparison"):
        assert client.get(url).status_code == 200, url


def test_ml_feature_importance_ignores_deep_registry_entries():
    """Task 7 adds entries with family='deep' to the shared registry. Task 6's
    endpoint must not trip over them."""
    r = client.get("/api/ml/feature-importance")
    assert r.status_code == 200


# --------------------------------------------------------------------------
# Task 8 wired into the strategy engine — opt-in, non-breaking
# --------------------------------------------------------------------------
def _race_state(client_):
    o = client_.get("/api/data/options").json()
    return {
        "driver": o["drivers"][0], "team": o["teams"][0],
        "current_lap": 30, "total_laps": 57,
        "tyre_compound": o["compounds"][0], "tyre_age": 22,
        "track_temperature": 30.0,
    }


def test_strategy_response_is_unchanged_when_explain_is_omitted():
    """The flag must be genuinely opt-in: every existing caller sees the
    response shape it saw before."""
    r = client.post("/api/strategy/predict", json=_race_state(client))
    assert r.status_code == 200
    assert "explanation" not in r.json()


@_xai
def test_strategy_explain_true_attaches_a_real_explanation():
    r = client.post("/api/strategy/predict", json={**_race_state(client), "explain": True})
    assert r.status_code == 200
    body = r.json()
    assert "explanation" in body
    for target, e in body["explanation"].items():
        if not e.get("available"):
            continue
        assert e["shap_factors"], f"{target}: no SHAP factors"
        assert len(e["narrative"]) > 40
        assert 0.0 <= e["trust_score"] <= 1.0
        assert e["trust_band"]["label"] in {"HIGH", "MODERATE", "LOW", "DO NOT ACT"}


@_xai
def test_live_explanation_declares_its_reduced_budget():
    """The live explainer is coarser than the committed reports and drops LIME.
    It must say so rather than presenting a differently-computed score under
    the same name."""
    body = client.post("/api/strategy/predict",
                       json={**_race_state(client), "explain": True}).json()
    for target, e in body["explanation"].items():
        if not e.get("available"):
            continue
        assert e["method"]["nsamples"] < 200, "live budget should be smaller than the batch one"
        assert "LIME is not run" in e["method"]["note"]
        assert set(e["trust_components"]) == {"confidence", "model_agreement"}


def test_strategy_rejects_a_deep_model_it_cannot_load():
    """/api/strategy/predict serves sklearn pipelines; a .keras model must be
    refused with a clear 422 rather than a 500 at load time."""
    r = client.post("/api/strategy/predict",
                    json={**_race_state(client), "laptime_model": "dnn_mlp"})
    assert r.status_code == 422
    assert "dnn_mlp" in str(r.json()["detail"])
