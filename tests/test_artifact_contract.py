"""
The "no fabricated results" contract, enforced.

`docs/PROJECT_REPORT.md` §8 claims every value this platform displays traces to
a generated artifact rather than a hard-coded constant. That claim was verified
by hand during the repository audit — mutate an artifact on disk, confirm the
API returns the mutated value — but nothing enforced it, so a future hard-coded
fallback would have passed CI unnoticed.

These tests do the mutation automatically. Two directions matter:

* **Positive** — write a sentinel into an artifact, assert the endpoint returns
  the sentinel. A matching value alone proves nothing (a constant could
  coincide); it is the endpoint *following* a change that proves the read path
  is real.
* **Negative** — remove the artifact, assert the endpoint fails explicitly
  rather than silently substituting a default. A fabricated zero is worse than
  a 404, because a reader cannot tell it apart from a real result.

Every mutation is made on a copy and restored in a ``finally``, so these tests
never leave the tracked artifacts modified.
"""
from __future__ import annotations

import json
import shutil
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.core.paths import (
    ARTIFACT_MANIFEST_JSON,
    DL_METRICS_JSON,
    ML_METRICS_DIR,
    ML_MODEL_REGISTRY_JSON,
    XAI_RESULTS_JSON,
)

client = TestClient(app)

REGRESSION_METRICS = ML_METRICS_DIR / "regression_metrics.json"

SENTINEL_FLOAT = 777.7777777
SENTINEL_STRING = "SENTINEL-NOT-A-REAL-MODEL"


@contextmanager
def temporarily_patched(path, mutate):
    """Apply ``mutate`` to the JSON at ``path``, restoring the original bytes
    afterwards whatever happens."""
    backup = path.with_suffix(path.suffix + ".contract-test-backup")
    shutil.copy2(path, backup)
    try:
        payload = json.loads(path.read_text())
        mutate(payload)
        path.write_text(json.dumps(payload, indent=2))
        yield
    finally:
        shutil.copy2(backup, path)
        backup.unlink(missing_ok=True)


@contextmanager
def temporarily_removed(path):
    backup = path.with_suffix(path.suffix + ".contract-test-backup")
    shutil.copy2(path, backup)
    try:
        path.unlink()
        yield
    finally:
        shutil.copy2(backup, path)
        backup.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Positive direction: the endpoint follows the artifact
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not REGRESSION_METRICS.exists(), reason="Task 6 not built")
def test_ml_metrics_endpoint_follows_the_artifact():
    def mutate(payload):
        payload["models"]["decision_tree"]["test_metrics"]["mae"] = SENTINEL_FLOAT

    with temporarily_patched(REGRESSION_METRICS, mutate):
        served = client.get("/api/ml/metrics").json()
        assert served["regression"]["models"]["decision_tree"]["test_metrics"]["mae"] == SENTINEL_FLOAT

    restored = client.get("/api/ml/metrics").json()
    assert restored["regression"]["models"]["decision_tree"]["test_metrics"]["mae"] != SENTINEL_FLOAT


@pytest.mark.skipif(not ML_MODEL_REGISTRY_JSON.exists(), reason="Task 6 not built")
def test_ml_models_endpoint_follows_the_registry():
    def mutate(payload):
        for m in payload["models"]:
            if m.get("family") != "deep":
                m["model_name"] = SENTINEL_STRING
                break

    with temporarily_patched(ML_MODEL_REGISTRY_JSON, mutate):
        names = {m["model_name"] for m in client.get("/api/ml/models").json()["models"]}
        assert SENTINEL_STRING in names


@pytest.mark.skipif(not DL_METRICS_JSON.exists(), reason="Task 7 not built")
def test_dl_metrics_endpoint_follows_the_artifact():
    def mutate(payload):
        payload["models"]["target_laptime"]["test_metrics"]["mae"] = SENTINEL_FLOAT

    with temporarily_patched(DL_METRICS_JSON, mutate):
        served = client.get("/api/dl/metrics").json()
        assert served["models"]["target_laptime"]["test_metrics"]["mae"] == SENTINEL_FLOAT


@pytest.mark.skipif(not XAI_RESULTS_JSON.exists(), reason="Task 8 not built")
def test_xai_fairness_endpoint_follows_the_artifact():
    def mutate(payload):
        payload["targets"]["target_laptime"]["fairness"]["identity_attribution_share"] = 0.4242

    with temporarily_patched(XAI_RESULTS_JSON, mutate):
        served = client.get("/api/xai/fairness").json()
        assert served["target_laptime"]["identity_attribution_share"] == 0.4242


# ---------------------------------------------------------------------------
# Negative direction: a missing artifact is an explicit error, not a default
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not REGRESSION_METRICS.exists(), reason="Task 6 not built")
def test_missing_ml_metrics_artifact_is_an_explicit_error():
    with temporarily_removed(REGRESSION_METRICS):
        r = client.get("/api/ml/metrics")
        assert r.status_code == 404, "a missing artifact produced a value instead of an error"
        assert "build_all" in str(r.json()["detail"]), "the error does not say how to fix it"


@pytest.mark.skipif(not DL_METRICS_JSON.exists(), reason="Task 7 not built")
def test_missing_dl_metrics_artifact_is_an_explicit_error():
    with temporarily_removed(DL_METRICS_JSON):
        r = client.get("/api/dl/metrics")
        assert r.status_code == 404
        assert "build_all" in str(r.json()["detail"])


@pytest.mark.skipif(not XAI_RESULTS_JSON.exists(), reason="Task 8 not built")
def test_missing_xai_artifact_is_an_explicit_error():
    with temporarily_removed(XAI_RESULTS_JSON):
        r = client.get("/api/xai/summary")
        assert r.status_code == 404
        assert "build_all" in str(r.json()["detail"])


# ---------------------------------------------------------------------------
# The manifest must never carry absolute paths (the frontend joins them onto
# the /artifacts mount).
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not ARTIFACT_MANIFEST_JSON.exists(), reason="Task 6 not built")
def test_manifest_paths_are_relative():
    manifest = json.loads(ARTIFACT_MANIFEST_JSON.read_text())
    for key in ("models", "metrics", "figures", "reports"):
        for path in manifest.get(key, []):
            assert not str(path).startswith("/"), f"{key}: absolute path {path} in the manifest"


# ---------------------------------------------------------------------------
# These tests mutate tracked files; prove they clean up after themselves.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not REGRESSION_METRICS.exists(), reason="Task 6 not built")
def test_contract_tests_restore_the_artifact_byte_for_byte():
    original = REGRESSION_METRICS.read_bytes()

    def mutate(payload):
        payload["models"]["decision_tree"]["test_metrics"]["mae"] = SENTINEL_FLOAT

    with temporarily_patched(REGRESSION_METRICS, mutate):
        assert REGRESSION_METRICS.read_bytes() != original
    assert REGRESSION_METRICS.read_bytes() == original, "the mutation was not fully restored"


@pytest.mark.skipif(not REGRESSION_METRICS.exists(), reason="Task 6 not built")
def test_no_backup_files_are_left_behind():
    leftovers = list(ML_METRICS_DIR.glob("*.contract-test-backup"))
    assert leftovers == [], f"contract-test backups left on disk: {leftovers}"
