"""
app.api.main
==============

F1 Race Strategy Intelligence — backend API.

Serves Task 6 (Machine Learning) and Task 7 (Deep Learning) model
metadata/metrics/predictions, Task 8 (Explainable AI) explanations, the
generated artifact figures and reports, and the cross-task race-strategy
simulator (ML + Task 2 Expert System + Task 3 Search).

Run with: uvicorn app.api.main:app --reload
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routers import data, dl, health, ml, strategy, tasks, xai
from app.core.paths import ARTIFACTS_DIR

app = FastAPI(
    title="F1 Race Strategy Intelligence API",
    description="Backend for the Quantum Optimization for Formula 1 Race Strategy platform.",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
# Previously ``allow_origins=["*"]`` with wildcard methods and headers. A
# wildcard is not harmless just because the API runs on localhost: any website
# the developer visits in the same browser can issue cross-origin requests to
# 127.0.0.1:8000 and read the responses, which includes every prediction
# endpoint and (before the mount fix below) the whole artifacts tree.
#
# Configurable so a deployment can set its real origin without a code change.
DEFAULT_ALLOWED_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("F1_ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    # The API only ever reads (GET) or submits a prediction request (POST).
    allow_methods=["GET", "POST"],
    # The frontend sends JSON bodies and nothing else; a wildcard here would
    # let a cross-origin caller set arbitrary headers.
    allow_headers=["Content-Type"],
)

app.include_router(health.router)
app.include_router(ml.router)
app.include_router(strategy.router)
app.include_router(data.router)
app.include_router(tasks.router)
app.include_router(dl.router)
app.include_router(xai.router)

# ---------------------------------------------------------------------------
# Static artifacts
# ---------------------------------------------------------------------------
# Previously the whole ``artifacts/`` tree was mounted at /artifacts, which
# also exposed ``artifacts/models/`` (every trained .joblib and .keras file)
# and ``artifacts/metadata/``. Anyone who could reach the API could download
# the trained models.
#
# Only the directories the frontend actually reads are mounted, and each is
# mounted individually rather than filtered — a path with no mount simply has
# no route, so exclusion is structural rather than a rule that could be
# bypassed. ``models/`` and ``metadata/`` are deliberately absent; their
# contents are described through /api/ml/models and /api/dl/models instead,
# which return metadata without serving the weights.
PUBLIC_ARTIFACT_DIRS = (
    "figures",                   # Task 6/7/8 figures
    "reports",                   # Task 6/7/8 markdown reports
    "data_engineering",          # Task 4 figures + reports
    "knowledge_representation",  # Task 1 diagrams, ontology, reports
    "expert_system",             # Task 2 reports + rule base
    "search",                    # Task 3 diagrams + reports
)

PRIVATE_ARTIFACT_DIRS = ("models", "metadata")

for _name in PUBLIC_ARTIFACT_DIRS:
    _path = ARTIFACTS_DIR / _name
    if _path.is_dir():
        app.mount(f"/artifacts/{_name}", StaticFiles(directory=str(_path)), name=f"artifacts-{_name}")
