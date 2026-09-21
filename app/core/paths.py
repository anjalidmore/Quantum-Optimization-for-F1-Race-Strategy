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

from dataclasses import dataclass
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

# Task 7 — Deep Learning.
#   artifacts/models/deep_learning/<laptime|pit_decision>/f1_dnn_model.h5
#       weights + fitted scalers. Kept inside the private models/ tree beside
#       Task 6's pipelines, so trained weights are never served over HTTP.
#   artifacts/deep_learning/
#       every public Task 7 deliverable: training history, hyperparameter and
#       evaluation reports, model comparison, metadata and plots.
DL_MODELS_DIR = ML_ARTIFACTS_DIR / "deep_learning"
DEEP_LEARNING_DIR = ARTIFACTS_DIR / "deep_learning"
DL_METRICS_JSON = DEEP_LEARNING_DIR / "evaluation_report.json"
DL_HISTORY_JSON = DEEP_LEARNING_DIR / "training_history.json"
DL_COMPARISON_JSON = DEEP_LEARNING_DIR / "model_comparison.json"

# Short folder name per target, shared by Task 6's models/ layout.
TARGET_DIRNAME = {"target_laptime": "laptime", "target_pit_next_lap": "pit_decision"}

# Task 8 — Explainable AI. Every Task 8 deliverable lives here; nothing in it
# is a model weight, so the whole directory is safe to serve.
XAI_DIR = ARTIFACTS_DIR / "xai"
XAI_RESULTS_JSON = XAI_DIR / "xai_metadata.json"


@dataclass(frozen=True)
class ArtifactPaths:
    """Every output location the ML/DL/XAI pipelines write to, resolved from a
    single root.

    The module-level constants above remain the default and are what production
    code uses. This object exists so a caller can redirect *all* output
    somewhere else in one argument - which is how the test suite stays
    hermetic: before this, running ``pytest`` retrained Task 6 and overwrote ten
    tracked files under ``artifacts/``, so a clean clone went dirty just from
    running the documented test command.

    ``ArtifactPaths.default()`` returns exactly the committed layout, so passing
    nothing changes no behaviour.
    """

    root: Path

    @classmethod
    def default(cls) -> "ArtifactPaths":
        return cls(root=ARTIFACTS_DIR)

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def models_laptime(self) -> Path:
        return self.models / "laptime"

    @property
    def models_pit(self) -> Path:
        return self.models / "pit_decision"

    @property
    def models_dl(self) -> Path:
        return self.models / "deep_learning"

    @property
    def deep_learning(self) -> Path:
        return self.root / "deep_learning"

    def dl_target(self, target: str) -> Path:
        """Public per-target Task 7 outputs: history CSV and curves."""
        return self.deep_learning / TARGET_DIRNAME[target]

    def dl_model_dir(self, target: str) -> Path:
        """Private per-target Task 7 weights and scalers."""
        return self.models_dl / TARGET_DIRNAME[target]

    @property
    def xai(self) -> Path:
        return self.root / "xai"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def metadata(self) -> Path:
        return self.root / "metadata"

    @property
    def model_registry_json(self) -> Path:
        return self.metadata / "model_registry.json"

    @property
    def manifest_json(self) -> Path:
        return self.root / "manifest.json"

    @property
    def xai_results_json(self) -> Path:
        return self.xai / "xai_metadata.json"

    @property
    def dl_metrics_json(self) -> Path:
        return self.deep_learning / "evaluation_report.json"

    @property
    def dl_history_json(self) -> Path:
        return self.deep_learning / "training_history.json"

    @property
    def dl_comparison_json(self) -> Path:
        return self.deep_learning / "model_comparison.json"

    def ensure(self) -> "ArtifactPaths":
        """Create every directory this object names."""
        for path in (
            self.models_laptime, self.models_pit, self.models_dl,
            self.metrics, self.figures, self.reports, self.metadata,
            self.deep_learning, self.xai,
            *(self.dl_target(t) for t in TARGET_DIRNAME),
            *(self.dl_model_dir(t) for t in TARGET_DIRNAME),
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


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
        DEEP_LEARNING_DIR,
        XAI_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
