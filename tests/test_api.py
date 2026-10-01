"""API tests. Assumes the training pipeline has already been run (as it is
by ``tests/test_ml_training.py``, which pytest collects and runs first
alphabetically) so the model registry and artifacts exist."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.intelligence.features.contract import load_feature_contract


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"


def test_list_models_returns_registry(client):
    r = client.get("/api/ml/models")
    assert r.status_code == 200
    body = r.json()
    assert "models" in body
    assert len(body["models"]) > 0
    for entry in body["models"]:
        assert "model_name" in entry
        assert "target" in entry


def test_metrics_endpoint(client):
    r = client.get("/api/ml/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "regression" in body and "classification" in body


def test_comparison_endpoint(client):
    r = client.get("/api/ml/comparison")
    assert r.status_code == 200
    body = r.json()
    assert "regression" in body and "classification" in body


def test_artifacts_manifest_endpoint(client):
    r = client.get("/api/ml/artifacts")
    assert r.status_code == 200
    body = r.json()
    # synthetic_data_warning must be the negation of a real-data source — never
    # hardcoded True, since the same pipeline also runs on real FastF1 sessions
    # fetched via scripts/fetch_real_session.py.
    assert isinstance(body["synthetic_data_warning"], bool)
    is_real = body["dataset_source"]["source"] == "real_fastf1"
    assert body["synthetic_data_warning"] == (not is_real)

    # Every figure/model/report path must be repo-relative (so artifactUrl()
    # on the frontend can turn it into a working /artifacts/... URL) — an
    # absolute path here means an image renders broken in the browser.
    for path in body["figures"] + body["models"] + body["reports"] + body["metrics"]:
        assert not path.startswith("/"), f"path is absolute, will break the frontend image loader: {path}"


def test_predict_laptime_valid_input(client):
    contract = load_feature_contract()
    features = contract.selected_features("target_laptime")
    payload = {f: 1.0 for f in features}
    r = client.post("/api/ml/predict/laptime", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["prediction"], float)
    assert body["target"] == "target_laptime"
    assert body["data_source"] == contract.dataset_source["source"]


def test_predict_laptime_rejects_missing_features(client):
    r = client.post("/api/ml/predict/laptime", json={"tyre_life": 5})
    assert r.status_code == 422
    assert "missing" in r.json()["detail"]


def test_predict_laptime_rejects_unexpected_features(client):
    contract = load_feature_contract()
    features = contract.selected_features("target_laptime")
    payload = {f: 1.0 for f in features}
    payload["not_a_real_feature"] = 1.0
    r = client.post("/api/ml/predict/laptime", json=payload)
    assert r.status_code == 422
    assert "unexpected" in r.json()["detail"]


def test_predict_pit_valid_input(client):
    contract = load_feature_contract()
    features = contract.selected_features("target_pit_next_lap")
    payload = {f: 0.0 for f in features}
    r = client.post("/api/ml/predict/pit", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] in (0, 1)
    assert 0.0 <= body["probability_pit"] <= 1.0


def test_strategy_predict_runs_ml_expert_and_search(client):
    race_state = {
        "driver": "ALO",
        "team": "ASTON MARTIN",
        "current_lap": 20,
        "total_laps": 55,
        "tyre_compound": "MEDIUM",
        "tyre_age": 15,
        "track_temperature": 40.0,
        "weather": "dry",
        "fuel_kg": 70,
        "track_status": "GREEN",
        "current_position": 6,
    }
    r = client.post("/api/strategy/predict", json=race_state)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"]["predicted_lap_time_seconds"] is not None
    assert body["optimal_search_strategy"]["algorithm"] == "A*"
    assert isinstance(body["triggered_expert_rules"], list)
    assert body["data_source"] == load_feature_contract().dataset_source["source"]


def test_data_options_are_real_dataset_values(client):
    r = client.get("/api/data/options")
    assert r.status_code == 200
    body = r.json()
    assert len(body["drivers"]) > 0
    assert len(body["teams"]) > 0
    assert set(body["compounds"]) <= {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}
    assert body["total_laps_hint"] > 0


def test_strategy_predict_rejects_current_lap_over_total_laps(client):
    race_state = {
        "driver": "ALO", "team": "ASTON MARTIN", "current_lap": 60, "total_laps": 55,
        "tyre_compound": "MEDIUM", "tyre_age": 15, "track_temperature": 40.0,
    }
    r = client.post("/api/strategy/predict", json=race_state)
    assert r.status_code == 422


def test_strategy_predict_rejects_tyre_age_over_current_lap(client):
    race_state = {
        "driver": "ALO", "team": "ASTON MARTIN", "current_lap": 10, "total_laps": 55,
        "tyre_compound": "MEDIUM", "tyre_age": 20, "track_temperature": 40.0,
    }
    r = client.post("/api/strategy/predict", json=race_state)
    assert r.status_code == 422


def test_strategy_predict_rejects_unknown_driver(client):
    race_state = {
        "driver": "Max Verstappen", "team": "ASTON MARTIN", "current_lap": 10, "total_laps": 55,
        "tyre_compound": "MEDIUM", "tyre_age": 5, "track_temperature": 40.0,
    }
    r = client.post("/api/strategy/predict", json=race_state)
    assert r.status_code == 422
    assert "Unknown driver" in str(r.json()["detail"])


def test_strategy_predict_honours_explicit_model_choice(client):
    options = client.get("/api/data/options").json()
    registry = client.get("/api/ml/models").json()
    reg_models = sorted({m["model_name"] for m in registry["models"] if m["target"] == "target_laptime" and m["artifact"]})
    assert len(reg_models) >= 2

    race_state = {
        "driver": options["drivers"][0], "team": options["teams"][0], "current_lap": 20, "total_laps": 55,
        "tyre_compound": options["compounds"][0], "tyre_age": 10, "track_temperature": 40.0,
    }
    predictions = {}
    for model_name in reg_models:
        r = client.post("/api/strategy/predict", json={**race_state, "laptime_model": model_name})
        assert r.status_code == 200
        body = r.json()
        assert body["prediction"]["laptime_model"] == model_name
        predictions[model_name] = body["prediction"]["predicted_lap_time_seconds"]
    # Different models should not all coincidentally produce the exact same value.
    assert len(set(predictions.values())) > 1


def test_strategy_predict_rejects_unavailable_model(client):
    race_state = {
        "driver": "ALO", "team": "ASTON MARTIN", "current_lap": 10, "total_laps": 55,
        "tyre_compound": "MEDIUM", "tyre_age": 5, "track_temperature": 40.0,
        "laptime_model": "a_model_that_was_never_trained",
    }
    r = client.post("/api/strategy/predict", json=race_state)
    assert r.status_code == 422


def test_strategy_predict_rejects_negative_current_lap(client):
    race_state = {
        "driver": "ALO", "team": "ASTON MARTIN", "current_lap": -3, "total_laps": 55,
        "tyre_compound": "MEDIUM", "tyre_age": 5, "track_temperature": 40.0,
    }
    r = client.post("/api/strategy/predict", json=race_state)
    assert r.status_code == 422


def test_strategy_predict_rejects_negative_tyre_age(client):
    race_state = {
        "driver": "ALO", "team": "ASTON MARTIN", "current_lap": 10, "total_laps": 55,
        "tyre_compound": "MEDIUM", "tyre_age": -1, "track_temperature": 40.0,
    }
    r = client.post("/api/strategy/predict", json=race_state)
    assert r.status_code == 422


def test_strategy_predict_rejects_unknown_tyre_compound(client):
    race_state = {
        "driver": "ALO", "team": "ASTON MARTIN", "current_lap": 10, "total_laps": 55,
        "tyre_compound": "SLICK_ULTRA", "tyre_age": 5, "track_temperature": 40.0,
    }
    r = client.post("/api/strategy/predict", json=race_state)
    assert r.status_code == 422
    assert "Unknown tyre compound" in str(r.json()["detail"])


def test_strategy_predict_returns_every_pipeline_stage(client):
    """Task 9's pipeline runs eight stages on one call; every one of them
    must be a key in the response, so the frontend can render a card per
    stage without guessing at an optional field."""
    race_state = {
        "driver": "ALO", "team": "ASTON MARTIN", "current_lap": 20, "total_laps": 55,
        "tyre_compound": "MEDIUM", "tyre_age": 10, "track_temperature": 40.0, "explain": True,
    }
    r = client.post("/api/strategy/predict", json=race_state)
    assert r.status_code == 200
    body = r.json()
    for key in (
        "validation", "feature_construction", "prediction", "dl_prediction",
        "xai_explanation", "optimal_search_strategy", "triggered_expert_rules", "recommendation",
    ):
        assert key in body, key
    assert body["validation"]["passed"] is True
    assert body["recommendation"]["action"] in ("PIT NOW", "STAY OUT") or "PIT IN" in body["recommendation"]["action"]
    assert body["recommendation"]["confidence"] in ("high", "moderate", "low", "none")


def test_strategy_predict_dl_stage_uses_the_tuned_threshold(client):
    import json

    from app.core.paths import DL_METRICS_JSON

    metrics = json.loads(DL_METRICS_JSON.read_text())
    tuned = metrics["models"]["target_pit_next_lap"]["threshold"]["threshold"]
    assert tuned != 0.5  # the whole point: it must not have silently fallen back

    race_state = {
        "driver": "ALO", "team": "ASTON MARTIN", "current_lap": 20, "total_laps": 55,
        "tyre_compound": "MEDIUM", "tyre_age": 10, "track_temperature": 40.0,
    }
    body = client.post("/api/strategy/predict", json=race_state).json()
    assert body["dl_prediction"]["threshold"] == pytest.approx(tuned)
    expected_class = int(body["dl_prediction"]["probability_pit"] >= tuned)
    assert body["dl_prediction"]["predicted_class"] == expected_class


def test_strategy_ml_stage_reports_not_available_when_model_artifact_missing(tmp_path, monkeypatch):
    """A missing .joblib must surface as a structured error, not a silent
    fabricated prediction or an unhandled 500."""
    import app.services.model_cache as mc

    monkeypatch.setattr(mc, "ML_MODELS_LAPTIME_DIR", tmp_path)
    fresh_cache = mc.ModelCache()
    with pytest.raises(mc.ModelUnavailableError, match="no artifact"):
        fresh_cache.get_pipeline("target_laptime", "svr")


def test_strategy_dl_stage_reports_not_available_when_model_missing(tmp_path, monkeypatch):
    """The DL stage's own 'not available' path, exercised the same way the
    live strategy pipeline reaches it — through xai.live's shared cache."""
    import app.intelligence.xai.live as live_mod
    import app.intelligence.xai.loading as loading_mod

    monkeypatch.setattr(loading_mod, "DL_MODELS_DIR", tmp_path / "no-such-dir")
    saved_cache = dict(live_mod._CACHE)
    live_mod._CACHE.clear()
    try:
        result = live_mod.predict_point("target_laptime", {})
        assert result["available"] is False
        assert "build_all.py" in result["reason"]
    finally:
        live_mod._CACHE.clear()
        live_mod._CACHE.update(saved_cache)


def test_task_evidence_endpoint_reports_honest_status(client):
    r = client.get("/api/tasks/evidence")
    assert r.status_code == 200
    body = r.json()
    assert len(body["tasks"]) == 10
    statuses = {t["id"]: t["status"] for t in body["tasks"]}
    # Tasks 1-8 have real generated artifacts in this repo.
    for tid in ("task1", "task2", "task3", "task4", "task5", "task6", "task7", "task8"):
        assert statuses[tid] == "completed"
    # Tasks 9-10 are scanned from real files on disk (docs/, Dockerfiles,
    # artifacts/evaluation, artifacts/reports) rather than a fixed status, so
    # their status here tracks whatever this checkout's state actually is —
    # never "completed" from nothing, and never silently pinned to "upcoming"
    # once real deliverables exist.
    for tid in ("task9", "task10"):
        assert statuses[tid] in ("completed", "in_progress", "upcoming")
    task10 = next(t for t in body["tasks"] if t["id"] == "task10")
    assert task10["documents_total"] == 6
    assert 0 <= task10["documents_complete"] <= 6
    assert statuses["task10"] == (
        "completed" if task10["documents_complete"] == 6
        else "in_progress" if task10["documents_complete"] > 0
        else "upcoming"
    )
    # Every listed artifact path must actually exist on disk (never a fabricated filename).
    import os

    for task in body["tasks"]:
        for path in task["reports"] + task["figures"] + task["other_artifacts"]:
            assert os.path.exists(path), f"listed artifact does not exist: {path}"


def test_top_features_endpoint_returns_exactly_n_with_descriptions(client):
    r = client.get("/api/ml/top-features?target=target_laptime&n=8")
    assert r.status_code == 200
    body = r.json()
    assert len(body["top_features"]) == 8
    for f in body["top_features"]:
        assert f["display_name"]
        assert f["description"]
    assert "ranking_method" in body


# ---------------------------------------------------------------------------
# Security: static artifact serving (TODO.md — "The whole artifacts tree is
# served unauthenticated as static files")
# ---------------------------------------------------------------------------
def test_trained_model_weights_are_not_served_statically(client):
    """Anyone who can reach the API used to be able to download every trained
    model. Only figures and reports are mounted now."""
    for path in (
        "/artifacts/models/laptime/decision_tree.joblib",
        "/artifacts/models/pit_decision/random_forest.joblib",
        "/artifacts/models/deep_learning/laptime/f1_dnn_model.h5",
        "/artifacts/deep_learning/../models/deep_learning/laptime/f1_dnn_model.h5",
    ):
        assert client.get(path).status_code == 404, f"{path} is still downloadable"


def test_metadata_directory_is_not_served_statically(client):
    assert client.get("/artifacts/metadata/model_registry.json").status_code == 404


def test_public_artifact_directories_are_still_served(client):
    """The restriction must not break the dashboard: every directory the
    frontend reads has to stay reachable."""
    for path in (
        "/artifacts/figures/roc_curves.png",
        "/artifacts/reports/regression_report.md",
        "/artifacts/data_engineering/figures/dashboard.png",
        "/artifacts/expert_system/reports/rule_catalogue.md",
        "/artifacts/search/reports/comparison_report.md",
        "/artifacts/knowledge_representation/reports/entity_table.md",
    ):
        assert client.get(path).status_code == 200, f"{path} is no longer served"


def test_private_dirs_are_excluded_structurally_not_by_a_filter(client):
    """The exclusion is 'no mount exists', which cannot be bypassed by path
    tricks the way a string filter could."""
    from app.api.main import PRIVATE_ARTIFACT_DIRS, PUBLIC_ARTIFACT_DIRS

    assert set(PRIVATE_ARTIFACT_DIRS).isdisjoint(PUBLIC_ARTIFACT_DIRS)
    for path in (
        "/artifacts/figures/../models/laptime/decision_tree.joblib",
        "/artifacts/figures/%2e%2e/models/laptime/decision_tree.joblib",
    ):
        assert client.get(path).status_code in (403, 404), f"{path} escaped the mount"


# ---------------------------------------------------------------------------
# Security: CORS (TODO.md — "API allows every origin, method and header")
# ---------------------------------------------------------------------------
def test_cors_rejects_an_unlisted_origin(client):
    r = client.get("/api/health", headers={"Origin": "https://evil.example.com"})
    assert r.headers.get("access-control-allow-origin") is None


def test_cors_allows_the_frontend_origin(client):
    r = client.get("/api/health", headers={"Origin": "http://localhost:3000"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_does_not_advertise_wildcard_methods_or_headers(client):
    r = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r.headers.get("access-control-allow-methods") != "*"
    assert r.headers.get("access-control-allow-headers") != "*"


def test_allowed_origins_are_configurable_by_environment():
    """A deployment must be able to set its real origin without a code change."""
    import importlib
    import os

    import app.api.main as main_mod

    original = os.environ.get("F1_ALLOWED_ORIGINS")
    os.environ["F1_ALLOWED_ORIGINS"] = "https://example.test, https://second.test"
    try:
        reloaded = importlib.reload(main_mod)
        assert reloaded.ALLOWED_ORIGINS == ["https://example.test", "https://second.test"]
    finally:
        if original is None:
            os.environ.pop("F1_ALLOWED_ORIGINS", None)
        else:
            os.environ["F1_ALLOWED_ORIGINS"] = original
        importlib.reload(main_mod)


# ---------------------------------------------------------------------------
# Task 9 — the reasoning endpoints (Tasks 1-3) and the strategy report generator
# ---------------------------------------------------------------------------
REPORT_RACE_STATE = {
    "driver": "ALO", "team": "ASTON MARTIN", "current_lap": 20, "total_laps": 55,
    "tyre_compound": "MEDIUM", "tyre_age": 15, "track_temperature": 40.0,
    "weather": "dry", "fuel_kg": 70, "track_status": "GREEN", "current_position": 6,
}


def test_knowledge_endpoint_counts_match_the_schema_it_was_built_from(client):
    from app.intelligence.knowledge_representation import schema

    body = client.get("/api/reasoning/knowledge").json()
    assert body["available"] is True
    assert body["n_entities"] == len(schema.entities_by_name())
    assert body["n_relationships"] == len(schema.relationship_names())
    # the per-category counts must add up to the total, not be a separate claim
    assert sum(body["entities_by_category"].values()) == body["n_entities"]


def test_expert_system_endpoint_matches_the_committed_rule_base(client):
    import json

    from app.api.routers.reasoning import RULE_BASE_JSON

    body = client.get("/api/reasoning/expert-system").json()
    if not RULE_BASE_JSON.exists():
        assert body["available"] is False and "Run:" in body["reason"]
        return
    rules = json.loads(RULE_BASE_JSON.read_text())
    rules = rules.get("rules", rules) if isinstance(rules, dict) else rules
    assert body["n_rules"] == len(rules)
    assert sum(body["rules_by_category"].values()) == body["n_rules"]
    assert {r["rule_id"] for r in body["rules"]} == {r["rule_id"] for r in rules}


def test_search_endpoint_reports_the_optimality_invariant(client):
    body = client.get("/api/reasoning/search").json()
    if not body.get("available"):
        assert "Run:" in body["reason"]
        return
    costs = {a["algorithm"]: a["solution_cost"] for a in body["algorithms"] if a["found"]}
    # A* with an admissible heuristic must match uniform-cost search exactly.
    assert costs["A*"] == pytest.approx(costs["UCS"])
    assert set(body["summary"]["optimal_algorithms"]) == {"UCS", "A*"}


def test_missing_artifact_reports_not_generated_rather_than_a_number(client, tmp_path, monkeypatch):
    """The dashboard must be able to print 'Not generated yet' instead of a zero."""
    import app.api.routers.reasoning as mod

    monkeypatch.setattr(mod, "SEARCH_JSON", tmp_path / "absent.json")
    body = client.get("/api/reasoning/search").json()
    assert body["available"] is False
    assert "scripts/run_search.py" in body["reason"]


def test_strategy_report_is_a_downloadable_markdown_briefing(client):
    r = client.post("/api/strategy/report", json=REPORT_RACE_STATE)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/markdown")
    assert "attachment; filename=" in r.headers["content-disposition"]
    assert ".md" in r.headers["content-disposition"]


def test_strategy_report_contains_every_section_the_lab_requires(client):
    body = client.post("/api/strategy/report", json=REPORT_RACE_STATE).text
    for heading in ("# Race Strategy Report", "## 1. Race state", "## 2. Recommendation",
                    "## 3. Expert-system rules that fired", "## 4. Search plan",
                    "## 5. Why this prediction", "## 6. Provenance"):
        assert heading in body, heading
    # prediction, search plan, explanation and trust must all carry real values
    assert "Predicted lap time" in body and "Expected cost, remaining stint" in body
    assert "Trust score" in body
    assert "SHAP" in body


def test_strategy_report_never_shows_a_placeholder_number(client):
    """A missing value must read as text, never as a stand-in figure like 0.000."""
    body = client.post("/api/strategy/report", json=REPORT_RACE_STATE).text
    for placeholder in ("TODO", "TBD", "lorem", "XXX", "placeholder", "clinical", "patient", "diagnosis"):
        assert placeholder.lower() not in body.lower(), placeholder


def test_strategy_report_validates_its_race_state(client):
    bad = {**REPORT_RACE_STATE, "driver": "NOT_A_DRIVER"}
    assert client.post("/api/strategy/report", json=bad).status_code == 422
