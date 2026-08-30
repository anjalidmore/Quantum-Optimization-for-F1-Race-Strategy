"""
app.api.main
==============

F1 Race Strategy Intelligence — backend API.

Serves Task 6 (Machine Learning) model metadata/metrics/predictions, the
generated artifact figures, and the cross-task race-strategy simulator
(Task 6 ML + Task 2 Expert System + Task 3 Search). See
`TASK6_IMPLEMENTATION_PROMPT.md` section 18 for the endpoint contract.

Run with: uvicorn app.api.main:app --reload
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routers import data, health, ml, strategy, tasks
from app.core.paths import ARTIFACTS_DIR

app = FastAPI(
    title="F1 Race Strategy Intelligence API",
    description="Backend for the Quantum Optimization for Formula 1 Race Strategy platform.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ml.router)
app.include_router(strategy.router)
app.include_router(data.router)
app.include_router(tasks.router)

if ARTIFACTS_DIR.exists():
    app.mount("/artifacts", StaticFiles(directory=str(ARTIFACTS_DIR)), name="artifacts")
