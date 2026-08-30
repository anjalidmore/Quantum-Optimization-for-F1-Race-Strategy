"""
app.services.model_cache
===========================

Loads trained Task 6 pipelines and the model registry once, and caches them
in memory. The API never retrains or re-reads a pipeline from disk on every
prediction request — training is a separate, offline operation
(``scripts/build_all.py`` / ``app.intelligence.ml.pipeline.train_all``).
"""
from __future__ import annotations

import logging
from functools import lru_cache

from app.core.paths import ML_MODELS_LAPTIME_DIR, ML_MODELS_PIT_DIR
from app.intelligence.features.contract import load_feature_contract
from app.intelligence.ml.persistence import load_pipeline
from app.intelligence.ml.registry import load_registry

log = logging.getLogger("services.model_cache")


class ModelUnavailableError(RuntimeError):
    pass


class ModelCache:
    def __init__(self) -> None:
        self._pipelines: dict[tuple[str, str], object] = {}

    def registry(self) -> dict | None:
        return load_registry()

    def _best_model_name(self, task: str) -> str | None:
        registry = self.registry()
        if not registry:
            return None
        candidates = [m for m in registry["models"] if m["task"] == task and m["is_selected_best"]]
        return candidates[0]["model_name"] if candidates else None

    def get_pipeline(self, task: str, model_name: str | None = None):
        """``task``: 'target_laptime' or 'target_pit_next_lap'. If
        ``model_name`` is omitted, the registry's selected-best model for
        that task is used."""
        models_dir = ML_MODELS_LAPTIME_DIR if task == "target_laptime" else ML_MODELS_PIT_DIR

        name = model_name or self._best_model_name(task)
        if name is None:
            raise ModelUnavailableError(
                f"No trained model registered for task={task!r}. Run the training "
                "pipeline (scripts/build_all.py) before requesting predictions."
            )

        key = (task, name)
        if key not in self._pipelines:
            path = models_dir / f"{name}.joblib"
            if not path.exists():
                raise ModelUnavailableError(f"Registered model {name!r} for {task!r} has no artifact at {path}")
            log.info("Loading and caching model pipeline: %s / %s", task, name)
            self._pipelines[key] = load_pipeline(path)
        return name, self._pipelines[key]

    def feature_list(self, target: str) -> list[str]:
        return load_feature_contract().selected_features(target)


@lru_cache(maxsize=1)
def get_model_cache() -> ModelCache:
    return ModelCache()
