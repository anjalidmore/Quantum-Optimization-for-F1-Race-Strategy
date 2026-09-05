"""
Pytest collection-time setup.

Task 6 uses XGBoost and Task 7 uses Keras on the PyTorch backend. On macOS
both ship their own ``libomp.dylib``, and whichever loads first claims the
process — XGBoost segfaults if it has to initialise against PyTorch's copy.
pytest imports test modules in filename order, so without this the crash
depends on which file happens to be collected first.

``prepare_dl_runtime()`` imports XGBoost before anything can pull in torch,
making the ordering deterministic for the whole session. See
``app/core/runtime.py`` for the full diagnosis.
"""
from __future__ import annotations

from app.core.runtime import prepare_dl_runtime

prepare_dl_runtime()
