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


def test_task_evidence_endpoint_reports_honest_status(client):
    r = client.get("/api/tasks/evidence")
    assert r.status_code == 200
    body = r.json()
    assert len(body["tasks"]) == 10
    statuses = {t["id"]: t["status"] for t in body["tasks"]}
    # Tasks 1-6 have real generated artifacts in this repo; 7-10 do not exist yet.
    for tid in ("task1", "task2", "task3", "task4", "task5", "task6"):
        assert statuses[tid] == "completed"
    for tid in ("task7", "task8", "task9", "task10"):
        assert statuses[tid] == "upcoming"
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
