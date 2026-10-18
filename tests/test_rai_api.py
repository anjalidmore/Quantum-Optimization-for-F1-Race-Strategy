"""
Responsible-AI endpoint (/api/rai/summary).

The endpoint recomputes per-lap predictions from the persisted models, so the
strongest check available is that those predictions reproduce the metrics the
training pipelines committed. If they do, the fairness and confidence views
are built on the same models the rest of the dashboard reports.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.core.paths import DL_COMPARISON_JSON, DL_METRICS_JSON

client = TestClient(app)

_built = pytest.mark.skipif(not (DL_METRICS_JSON.exists() and DL_COMPARISON_JSON.exists()),
                            reason="Tasks 6/7 not built")


def _committed(target: str) -> dict:
    blob = json.loads(DL_COMPARISON_JSON.read_text())["targets"][target]
    return {row["model"]: row["metrics"] for row in blob["comparison"]}, blob["task6_best_model"]


@_built
def test_recomputed_laptime_mae_matches_committed_metrics():
    body = client.get("/api/rai/summary", params={"target": "target_laptime"}).json()
    committed, best = _committed("target_laptime")
    assert body["models"]["ml"]["name"] == best
    for key, name in (("simple", "linear_regression"), ("ml", best), ("dnn", "dnn_mlp")):
        assert body["check"][key]["mae"] == pytest.approx(committed[name]["mae"], abs=1e-4)


@_built
def test_recomputed_pit_pr_auc_and_thresholds_match_committed_metrics():
    body = client.get("/api/rai/summary", params={"target": "target_pit_next_lap"}).json()
    committed, best = _committed("target_pit_next_lap")
    for key, name in (("simple", "logistic_regression"), ("ml", best), ("dnn", "dnn_mlp")):
        assert body["check"][key]["pr_auc"] == pytest.approx(committed[name]["pr_auc"], abs=1e-4)
        assert body["models"][key]["threshold"] == pytest.approx(committed[name]["decision_threshold"], abs=1e-4)


@_built
def test_fairness_never_claims_protected_attributes_or_unsupported_gaps():
    body = client.get("/api/rai/summary", params={"target": "target_pit_next_lap"}).json()
    fair = body["fairness"]
    assert fair["protected_attributes_available"] is False
    for per_group in fair["summary"].values():
        for entry in per_group.values():
            # A gap is only reported when at least two groups clear the sample rule.
            if entry["groups_with_enough_pit_laps"] < 2:
                assert entry["equal_opportunity_difference"] is None
            if entry["n_adequate_groups"] < 2:
                assert entry["demographic_parity_difference"] is None


@_built
def test_calibration_is_withheld_when_the_holdout_has_too_few_pit_laps():
    body = client.get("/api/rai/summary", params={"target": "target_pit_next_lap"}).json()
    if body["n_positive"] < 5:
        for m in body["confidence"]["models"].values():
            if m["available"]:
                assert m["calibration"] is None and m["calibration_reason"]


@_built
def test_regression_uncertainty_only_where_a_mechanism_exists():
    body = client.get("/api/rai/summary", params={"target": "target_laptime"}).json()
    for m in body["confidence"]["models"].values():
        assert m["available"] == bool(m.get("mechanism")), m
        if not m["available"]:
            assert m["reason"]


def test_unknown_target_is_rejected():
    assert client.get("/api/rai/summary", params={"target": "nope"}).status_code == 400
