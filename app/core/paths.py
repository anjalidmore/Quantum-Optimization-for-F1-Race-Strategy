"""
app.core.paths
===============

Single source of truth for every filesystem path used across the platform.

No module anywhere in this repository should build a path like
``../../phase1_task4_data_engineering/outputs`` — everything reads and writes
through the constants defined here, so the physical layout can change without
hunting down brittle relative paths module by module.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DATA_DIR = REPO_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

# Task 4 output consumed by Task 5.
FASTF1_LAPS_CLEAN_CSV = DATA_PROCESSED_DIR / "fastf1_laps_clean.csv"

# Provenance markers: written by scripts/fetch_real_session.py (raw) and
# propagated by scripts/run_eda.py (processed) so every downstream stage can
# report "real" vs "synthetic" truthfully instead of assuming one or the other.
RAW_DATA_SOURCE_MARKER = DATA_RAW_DIR / ".data_source.json"
PROCESSED_DATA_SOURCE_JSON = DATA_PROCESSED_DIR / "data_source.json"

# Task 5 outputs — the Task 6 data contract. Do not duplicate these files
# anywhere else in the repository; every downstream module reads them here.
TASK5_FEATURES_CSV = DATA_PROCESSED_DIR / "f1_features_selected.csv"
TASK5_FEATURE_METADATA_JSON = DATA_PROCESSED_DIR / "feature_metadata.json"

# ---------------------------------------------------------------------------
# Artifacts (generated, reproducible outputs of every intelligence module)
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

# Task 1-4 generated reports/diagrams (moved out of the old phase folders).
KNOWLEDGE_REPRESENTATION_ARTIFACTS_DIR = ARTIFACTS_DIR / "knowledge_representation"
EXPERT_SYSTEM_ARTIFACTS_DIR = ARTIFACTS_DIR / "expert_system"
SEARCH_ARTIFACTS_DIR = ARTIFACTS_DIR / "search"
DATA_ENGINEERING_ARTIFACTS_DIR = ARTIFACTS_DIR / "data_engineering"

# Task 6 — Machine Learning.
ML_ARTIFACTS_DIR = ARTIFACTS_DIR / "models"
ML_MODELS_LAPTIME_DIR = ML_ARTIFACTS_DIR / "laptime"
ML_MODELS_PIT_DIR = ML_ARTIFACTS_DIR / "pit_decision"
ML_METRICS_DIR = ARTIFACTS_DIR / "metrics"
ML_FIGURES_DIR = ARTIFACTS_DIR / "figures"
ML_REPORTS_DIR = ARTIFACTS_DIR / "reports"
ML_METADATA_DIR = ARTIFACTS_DIR / "metadata"
ML_MODEL_REGISTRY_JSON = ML_METADATA_DIR / "model_registry.json"
ARTIFACT_MANIFEST_JSON = ARTIFACTS_DIR / "manifest.json"

# Task 7 — Deep Learning. Models live under the existing models/ tree rather
# than a parallel one, and metrics/figures/reports share Task 6's directories
# so the frontend's artifact serving needs no new mount.
DL_MODELS_DIR = ML_ARTIFACTS_DIR / "dl"
DL_METRICS_JSON = ML_METRICS_DIR / "dl_metrics.json"
DL_HISTORY_JSON = ML_METRICS_DIR / "dl_training_history.json"
DL_COMPARISON_JSON = ML_METRICS_DIR / "dl_vs_classical.json"

# Task 8 — Explainable AI.
XAI_RESULTS_JSON = ML_METADATA_DIR / "xai_results.json"


def ensure_dirs() -> None:
    """Create every artifact/data directory this project writes to."""
    for path in (
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        KNOWLEDGE_REPRESENTATION_ARTIFACTS_DIR,
        EXPERT_SYSTEM_ARTIFACTS_DIR,
        SEARCH_ARTIFACTS_DIR,
        DATA_ENGINEERING_ARTIFACTS_DIR,
        ML_MODELS_LAPTIME_DIR,
        ML_MODELS_PIT_DIR,
        ML_METRICS_DIR,
        ML_FIGURES_DIR,
        ML_REPORTS_DIR,
        ML_METADATA_DIR,
        DL_MODELS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
