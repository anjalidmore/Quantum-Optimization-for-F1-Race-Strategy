"""
app.core.runtime
================

Process-level runtime setup that must happen **before** the deep-learning
stack is imported.

The problem this solves is real and was found by a segfault, not anticipated:

Task 6 uses XGBoost and Task 7 uses Keras on the PyTorch backend. On macOS
both ship their own copy of the OpenMP runtime (``libomp.dylib``), and
scikit-learn ships a third. Whichever copy is loaded **first** claims the
process; XGBoost then crashes with a segmentation fault if it has to
initialise against PyTorch's copy:

    import torch          # torch's libomp wins
    import xgboost
    XGBRegressor().fit(X, y)      -> Fatal Python error: Segmentation fault

Import order is the whole fix. With XGBoost first, both work:

    import xgboost        # xgboost/sklearn libomp wins
    import torch
    XGBRegressor().fit(X, y)      -> fine

Note that ``KMP_DUPLICATE_LIB_OK=TRUE`` - the usual folk remedy - does **not**
help here (verified), and is documented as unsafe besides. So this module does
the one thing that does work: it imports XGBoost first, once, at a
deterministic point.

Every module that pulls in Keras/torch calls ``prepare_dl_runtime()`` before
importing it. ``tests/conftest.py`` calls it at collection time so the same
ordering holds no matter which test file pytest happens to load first.

If XGBoost is not installed, this is a no-op - Task 6 already treats XGBoost
as optional and reports "XGBoost unavailable" rather than failing.
"""
from __future__ import annotations

import os

_PREPARED = False


def prepare_dl_runtime() -> dict:
    """Claim the OpenMP runtime for XGBoost, then select the Keras backend.

    Idempotent and safe to call from anywhere. Returns a small dict describing
    what happened, so a caller (or a test) can assert on it rather than trust
    that it worked.
    """
    global _PREPARED

    # Keras 3 reads this at import time. TensorFlow publishes no wheel for the
    # Python version this project targets, so the torch backend is selected
    # unless the environment explicitly asks for something else.
    os.environ.setdefault("KERAS_BACKEND", "torch")

    if _PREPARED:
        return {"prepared": True, "already": True, "backend": os.environ["KERAS_BACKEND"]}

    xgboost_loaded = False
    try:
        import xgboost  # noqa: F401  - imported for its side effect on OpenMP

        xgboost_loaded = True
    except Exception:
        # XGBoost is optional throughout this project. Its absence is not an
        # error; it just means there is no OpenMP conflict to avoid.
        xgboost_loaded = False

    _PREPARED = True
    return {
        "prepared": True,
        "already": False,
        "xgboost_preloaded": xgboost_loaded,
        "backend": os.environ["KERAS_BACKEND"],
    }
