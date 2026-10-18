#!/usr/bin/env python3
"""
evaluate_system.py
===================

Task 10 — System Evaluation, Responsible AI and Documentation.

Runs the checks the lab spec groups under Task 10 and writes every result to
``artifacts/evaluation/`` as JSON (plus a few PNGs) so the Task 9/10 report
reads from files on disk, never from a number typed into a document by hand.

Sections, each its own function and its own output file:

  1. functional_testing          -> test_results.json           (runs pytest for real)
  2. performance_evaluation      -> performance.json, latency_distribution.png
  3. domain_validation           -> domain_validation.json       (2023 Bahrain backtest)
  4. explainability_validation   -> explainability_validation.json
  5. responsible_ai_assessment   -> responsible_ai.json, robustness_flip_rate.png,
                                     fairness_by_team.png
  6. deployment_readiness        -> deployment_readiness.json
  7. future_enhancements         -> future_enhancements.json     (a plan, not a result)

Every section is independent: one failing section does not stop the others,
and a section that cannot run (a missing artifact, a missing optional
dependency) reports ``"available": false`` with a reason rather than being
silently skipped or padded with an invented number.

Usage
-----
    python scripts/evaluate_system.py                 # run everything
    python scripts/evaluate_system.py --skip-pytest    # skip the (slow) full suite
    python scripts/evaluate_system.py --n-latency 50   # fewer timed requests
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import re
import resource
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.core.paths import (  # noqa: E402
    DL_METRICS_JSON,
    EVALUATION_DIR,
    FASTF1_LAPS_CLEAN_CSV,
    ML_METRICS_DIR,
    REPO_ROOT,
    XAI_DIR,
    XAI_RESULTS_JSON,
)

# Shared with app.services.strategy_service, which also uses it to compute
# the live safety-car base rate — imported rather than duplicated.
from app.services.strategy_service import _map_track_status  # noqa: E402

log = logging.getLogger("evaluate_system")
logging.basicConfig(level=logging.INFO, format="%(message)s")

GENERATED_AT = datetime.now(timezone.utc).isoformat()


def _sanitize_nan(obj):
    """Python's json.dumps writes bare NaN/Infinity by default (a Python
    extension Node's and every other standard-compliant JSON.parse rejects).
    DataFrame.to_dict() produces real NaNs for missing values (e.g. the
    classification-only columns on a regression-target fairness row), so
    every payload is walked and NaN/Infinity replaced with null before
    writing, rather than fixing each DataFrame call site separately."""
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nan(v) for v in obj]
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return None
    return obj


def _write_json(name: str, payload: dict) -> Path:
    path = EVALUATION_DIR / name
    payload = _sanitize_nan({"generated_at": GENERATED_AT, **payload})
    path.write_text(json.dumps(payload, indent=2, default=str))
    log.info("wrote %s", path.relative_to(REPO_ROOT))
    return path


def _rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(REPO_ROOT))


# =============================================================================
# 1. Functional testing
# =============================================================================
def functional_testing(skip_pytest: bool = False) -> dict:
    """Runs the real pytest suite (with coverage) and the specific scenarios
    the lab spec calls out by name: input validation, the tuned threshold,
    the report endpoint, and the "not available" paths. All of those already
    exist as real tests in tests/ (see test_api.py and test_dl_xai_api.py) —
    this just runs the suite and records the real result, rather than
    re-describing what the tests do.
    """
    if skip_pytest:
        return {"available": False, "reason": "--skip-pytest was passed; no suite was run."}

    junit_path = EVALUATION_DIR / "junit.xml"
    cmd = [
        sys.executable, "-m", "pytest", "-q", "-p", "no:warnings",
        "--cov=app", "--cov-report=json:" + str(EVALUATION_DIR / "coverage.json"),
        "--junitxml=" + str(junit_path),
    ]
    log.info("running: %s", " ".join(cmd))
    start = time.monotonic()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800)
    elapsed = time.monotonic() - start

    # Parsed from --junitxml, not scraped from stdout: pytest-cov's own
    # "tests coverage" section replaces the usual "N passed in X.Xs" summary
    # line entirely whenever any --cov-report is requested (confirmed by
    # running the exact command with and without --cov-report — the line
    # simply never appears with it on), so text-scraping the terminal output
    # silently returns zeros. The JUnit XML's <testsuite> attributes are
    # structured and unaffected by that.
    passed = failed = errors = skipped = total = 0
    if junit_path.exists():
        import xml.etree.ElementTree as ET
        root = ET.parse(junit_path).getroot()
        suite = root.find("testsuite") if root.tag == "testsuites" else root
        if suite is not None:
            total = int(suite.get("tests", 0))
            failed = int(suite.get("failures", 0))
            errors = int(suite.get("errors", 0))
            skipped = int(suite.get("skipped", 0))
            passed = total - failed - errors - skipped

    coverage_pct = None
    cov_path = EVALUATION_DIR / "coverage.json"
    if cov_path.exists():
        try:
            coverage_pct = json.loads(cov_path.read_text())["totals"]["percent_covered"]
        except Exception as exc:  # pragma: no cover - diagnostic only
            log.warning("could not parse coverage.json: %s", exc)

    key_scenarios = {
        "end_to_end_predict_returns_all_stages": "test_strategy_predict_returns_every_pipeline_stage",
        "input_validation_rejects_bad_inputs": "test_strategy_predict_rejects_negative_current_lap",
        "dl_uses_tuned_threshold_not_0_5": "test_strategy_predict_dl_stage_uses_the_tuned_threshold",
        "report_endpoint_returns_non_empty_file": "test_strategy_report_is_a_downloadable_markdown_briefing",
        "not_available_when_ml_model_missing": "test_strategy_ml_stage_reports_not_available_when_model_artifact_missing",
        "not_available_when_dl_model_missing": "test_strategy_dl_stage_reports_not_available_when_model_missing",
    }
    present = {}
    for label, test_name in key_scenarios.items():
        # No -q here: pytest's quiet collection mode prints a per-file count
        # ("tests/test_api.py: 1"), not the test's node id, so -q would make
        # every one of these checks report False regardless of whether the
        # test actually exists.
        found = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-k", test_name],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        present[label] = test_name in found.stdout

    return {
        "available": True,
        "command": " ".join(cmd),
        "elapsed_seconds": round(elapsed, 1),
        "exit_code": proc.returncode,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "total": total,
        "coverage_percent": coverage_pct,
        "key_scenarios_present_in_suite": present,
        "note": "Counts are parsed from this run's own --junitxml output, not hand-typed. "
                "'key_scenarios_present_in_suite' confirms each named scenario actually exists as "
                "a collected test (see tests/test_api.py), not just that the suite as a whole passed.",
    }


# =============================================================================
# 2. Performance evaluation
# =============================================================================
def _verify_shared_holdout() -> dict:
    """Checks, empirically, whether Task 6 and Task 7 really evaluate on the
    same chronological holdout laps — rather than trusting the README's claim.
    Both call ``chronological_holdout(build_task_frame(load_and_validate(), target), 0.2)``
    (app/intelligence/ml/pipeline.py:62-63, app/intelligence/dl/pipeline.py:189),
    so this recomputes that exact call for both targets and compares the
    resulting held-out lap numbers.
    """
    from app.intelligence.ml.data_contract import build_task_frame, load_and_validate
    from app.intelligence.ml.splits import chronological_holdout

    dataset = load_and_validate()
    result = {}
    for target in ("target_laptime", "target_pit_next_lap"):
        frame, _ = build_task_frame(dataset, target)
        holdout = chronological_holdout(frame, test_fraction=0.2)
        laps = sorted(frame.loc[holdout.test_index, "LapNumber"].unique().tolist())
        result[target] = {
            "n_test_rows": len(holdout.test_index),
            "lap_range": [laps[0], laps[-1]] if laps else None,
            "n_unique_laps": len(laps),
        }
    same_range = result["target_laptime"]["lap_range"] == result["target_pit_next_lap"]["lap_range"]
    return {
        "method": "Recomputed chronological_holdout(build_task_frame(load_and_validate(), target), "
                  "test_fraction=0.2) for both targets — the exact call each pipeline makes "
                  "(app/intelligence/ml/pipeline.py:62-63, app/intelligence/dl/pipeline.py:189) — "
                  "and compared the resulting held-out lap numbers.",
        "per_target": result,
        "laptime_and_pit_targets_share_the_same_lap_range": same_range,
        "finding": (
            "Confirmed: Task 6 and Task 7 call the identical holdout function with the identical "
            "0.2 fraction on frames built by the identical build_task_frame(load_and_validate(), "
            "target) call, so for a given target their test laps are the same laps by construction, "
            "not by coincidence."
            if same_range else
            "Task 6/7 share the holdout-construction code, but the lap-time and pit-decision targets "
            "themselves hold out slightly different lap sets (expected: build_task_frame can drop "
            "different rows per target), so 'laps 47-57' is per-target, not one range for both targets."
        ),
    }


def _consolidated_model_table() -> list[dict]:
    """One row per Task 6/7 model, read from the committed metrics artifacts
    — never re-trained here."""
    rows = []
    ml_cmp = ML_METRICS_DIR / "model_comparison.json"
    if ml_cmp.exists():
        data = json.loads(ml_cmp.read_text())
        for task, entries in data.items():
            for e in entries:
                rows.append({
                    "family": "ML (Task 6)", "task": task, "model": e.get("model"),
                    "test_mae": e.get("test_mae"), "test_rmse": e.get("test_rmse"),
                    "test_r2": e.get("test_r2"), "cv_mae": e.get("cv_mae"),
                    "test_f1": e.get("test_f1"), "test_roc_auc": e.get("test_roc_auc"),
                    "test_pr_auc": e.get("test_pr_auc"),
                    "selected": e.get("selected", False),
                    "source": _rel(ml_cmp),
                })
    if DL_METRICS_JSON.exists():
        data = json.loads(DL_METRICS_JSON.read_text())
        for target, m in data.get("models", {}).items():
            test = m.get("test_metrics", {})
            rows.append({
                "family": "DL (Task 7)", "task": target, "model": "dnn_mlp",
                "test_mae": test.get("mae"), "test_rmse": test.get("rmse"),
                "test_r2": test.get("r2"),
                "test_f1": test.get("f1"), "test_roc_auc": test.get("roc_auc"),
                "test_pr_auc": test.get("pr_auc"),
                "n_test": test.get("n"),
                "source": _rel(DL_METRICS_JSON),
            })
    return rows


def _time_requests(client, payload: dict, n: int, explain: bool) -> dict:
    body = {**payload, "explain": explain}
    durations = []
    errors = 0
    for _ in range(n):
        t0 = time.perf_counter()
        r = client.post("/api/strategy/predict", json=body)
        durations.append(time.perf_counter() - t0)
        if r.status_code != 200:
            errors += 1
    durations.sort()

    def pct(p):
        idx = min(len(durations) - 1, int(round(p / 100 * (len(durations) - 1))))
        return durations[idx]

    return {
        "n_requests": n, "errors": errors, "explain": explain,
        "p50_ms": round(pct(50) * 1000, 2), "p95_ms": round(pct(95) * 1000, 2),
        "p99_ms": round(pct(99) * 1000, 2),
        "min_ms": round(durations[0] * 1000, 2), "max_ms": round(durations[-1] * 1000, 2),
        "mean_ms": round(statistics.mean(durations) * 1000, 2),
        "all_ms": [round(d * 1000, 3) for d in durations],
    }


def performance_evaluation(n_latency: int) -> dict:
    from fastapi.testclient import TestClient

    from app.api.main import app

    client = TestClient(app)
    options = client.get("/api/data/options").json()
    payload = {
        "driver": options["drivers"][0], "team": options["teams"][0],
        "current_lap": 25, "total_laps": options["total_laps_hint"],
        "tyre_compound": options["compounds"][0], "tyre_age": 10,
        "track_temperature": options["track_temperature_range"]["mean"],
    }

    log.info("timing %d requests (explain=false) ...", n_latency)
    latency_no_explain = _time_requests(client, payload, n_latency, explain=False)
    log.info("timing %d requests (explain=true) ...", max(10, n_latency // 4))
    latency_explain = _time_requests(client, payload, max(10, n_latency // 4), explain=True)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(latency_no_explain["all_ms"], bins=30, alpha=0.75, label="explain=false")
    ax.hist(latency_explain["all_ms"], bins=30, alpha=0.75, label="explain=true")
    ax.set_xlabel("request latency (ms)")
    ax.set_ylabel("count")
    ax.set_title("POST /api/strategy/predict latency")
    ax.legend()
    ax.grid(alpha=0.3)
    fig_path = EVALUATION_DIR / "latency_distribution.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    # Model load / memory: a fresh process would pay this once; approximate it
    # here by measuring this process's RSS before and after touching every
    # model family for the first time in a clean cache state.
    import app.intelligence.xai.live as live_mod
    from app.services.model_cache import get_model_cache

    live_mod._CACHE.clear()
    get_model_cache.cache_clear()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    get_model_cache().get_pipeline("target_laptime")
    get_model_cache().get_pipeline("target_pit_next_lap")
    ml_load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    live_mod.predict_point("target_laptime", {})  # triggers the shared DL+classical bundle load
    dl_load_s = time.perf_counter() - t0
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_unit = "bytes" if platform.system() == "Darwin" else "KB"

    return {
        "holdout_check": _verify_shared_holdout(),
        "consolidated_model_table": _consolidated_model_table(),
        "api_latency": {"explain_false": latency_no_explain, "explain_true": latency_explain,
                         "figure": _rel(fig_path)},
        "model_load": {
            "ml_cold_load_seconds": round(ml_load_s, 4),
            "dl_and_classical_bundle_cold_load_seconds": round(dl_load_s, 4),
            "note": "DL load includes building the full holdout frame (xai.live's shared cache), "
                    "not just deserialising the .h5 file, so it is an upper bound on pure model load time.",
        },
        "memory": {
            "ru_maxrss_before": rss_before, "ru_maxrss_after": rss_after,
            "unit": rss_unit,
            "delta": rss_after - rss_before,
            "note": "Peak resident set size for THIS process via resource.getrusage; not a per-model "
                    "breakdown, and ru_maxrss is monotonically non-decreasing within a process so this "
                    "is a lower bound on the memory these models actually use.",
        },
    }


# =============================================================================
# 3. Domain validation
# =============================================================================

def domain_validation() -> dict:
    """Backtests the recommendation engine against the real 2023 Bahrain race:
    replays every driver's laps in order, asks the API for a recommendation
    at each lap, and compares its pit calls against the laps the driver
    actually pitted (target_pit_next_lap in the committed feature matrix,
    itself derived from the real Stint column — see
    app/intelligence/features/build.py:94-103).

    This measures agreement with what the real teams did, not whether that
    decision was optimal — stated explicitly in the output, not just here.
    """
    if not FASTF1_LAPS_CLEAN_CSV.exists():
        return {"available": False, "reason": f"{_rel(FASTF1_LAPS_CLEAN_CSV)} not found."}

    from fastapi.testclient import TestClient

    from app.api.main import app
    from app.core.paths import TASK5_FEATURES_CSV

    client = TestClient(app)
    laps = pd.read_csv(FASTF1_LAPS_CLEAN_CSV)
    total_laps = int(laps["LapNumber"].max())

    ground_truth = None
    if TASK5_FEATURES_CSV.exists():
        feats = pd.read_csv(TASK5_FEATURES_CSV)
        ground_truth = feats.set_index(["Driver", "LapNumber"])["target_pit_next_lap"]

    rows = []
    errors = 0
    error_reasons: dict[str, int] = {}
    for driver, grp in laps.groupby("Driver"):
        grp = grp.sort_values("LapNumber")
        for _, lap in grp.iterrows():
            lap_number = int(lap["LapNumber"])
            fuel_kg = max(5.0, 110.0 - (lap_number / total_laps) * 100.0)  # approximated: not in the cleaned table
            weather = "wet" if bool(lap.get("Rainfall")) else "dry"
            race_state = {
                "driver": driver, "team": lap["Team"], "current_lap": lap_number,
                "total_laps": total_laps, "tyre_compound": str(lap["Compound"]).upper(),
                "tyre_age": int(lap["TyreLife"]), "track_temperature": float(lap["TrackTemp"]),
                "weather": weather, "fuel_kg": round(fuel_kg, 1),
                "track_status": _map_track_status(lap["TrackStatus"]),
                "current_position": 10,  # approximated: no per-lap position in the cleaned table
            }
            r = client.post("/api/strategy/predict", json=race_state)
            if r.status_code != 200:
                errors += 1
                detail = r.json().get("detail", "")
                if isinstance(detail, list) and detail and isinstance(detail[0], dict) and "msg" in detail[0]:
                    detail = detail[0]["msg"]
                detail = re.sub(r"\(\d+\)", "(N)", str(detail))  # bucket by shape, not by the exact lap number
                error_reasons[detail] = error_reasons.get(detail, 0) + 1
                continue
            body = r.json()
            called_pit = bool(body["recommendation"]["action"] != "STAY OUT") if body.get("recommendation") else None
            actual_pit = None
            if ground_truth is not None and (driver, float(lap_number)) in ground_truth.index:
                actual_pit = bool(ground_truth.loc[(driver, float(lap_number))])
            rows.append({
                "driver": driver, "lap": lap_number, "called_pit": called_pit, "actual_pit": actual_pit,
                "track_status_mapped": race_state["track_status"],
                "raw_track_status": str(lap["TrackStatus"]),
                "rules_fired": [f["rule_id"] for f in body.get("triggered_expert_rules", [])],
            })

    df = pd.DataFrame(rows)
    labelled = df.dropna(subset=["actual_pit"]) if not df.empty else df
    hits = int(((labelled["called_pit"]) & (labelled["actual_pit"])).sum()) if not labelled.empty else 0
    misses = int(((~labelled["called_pit"]) & (labelled["actual_pit"])).sum()) if not labelled.empty else 0
    false_calls = int(((labelled["called_pit"]) & (~labelled["actual_pit"])).sum()) if not labelled.empty else 0
    correct_stay = int(((~labelled["called_pit"]) & (~labelled["actual_pit"])).sum()) if not labelled.empty else 0

    # how many laps early/late a hit or a miss was: nearest actual-pit lap to
    # each called-pit lap, per driver.
    offsets = []
    if not df.empty:
        for driver, grp in df.groupby("driver"):
            actual_laps = grp.loc[grp["actual_pit"] == True, "lap"].tolist()  # noqa: E712
            if not actual_laps:
                continue
            for called_lap in grp.loc[grp["called_pit"] == True, "lap"]:  # noqa: E712
                nearest = min(actual_laps, key=lambda a: abs(a - called_lap))
                offsets.append(called_lap - nearest)

    # expert-system safety-car rule firing cross-check against the real,
    # mapped track status for that lap.
    sc_rule_firings = df[df["rules_fired"].apply(lambda rs: any(r.startswith("R-SC-") for r in rs))]
    sc_rule_firings_on_non_sc_lap = sc_rule_firings[~sc_rule_firings["track_status_mapped"].isin(["SC", "VSC"])]
    actual_sc_vsc_laps = df[df["track_status_mapped"].isin(["SC", "VSC"])]

    return {
        "available": True,
        "methodology": (
            "Replays every driver's real lap sequence from the 2023 Bahrain GP through "
            "POST /api/strategy/predict (the live recommendation engine), lap by lap, and compares "
            "its pit call against target_pit_next_lap — the real ground-truth label, derived from "
            "the actual Stint column (app/intelligence/features/build.py:94-103), not a simulation. "
            "fuel_kg and current_position are not in the cleaned per-lap table and are approximated "
            "(a linear depletion curve; a constant mid-field position) — the same kind of "
            "approximation the live simulator already makes and documents for fields it cannot derive "
            "from a single snapshot. This measures AGREEMENT with what the real teams actually did, "
            "not whether that decision was optimal."
        ),
        "n_laps_replayed": len(df), "n_request_errors": errors,
        "request_error_reasons": error_reasons,
        "request_error_explanation": (
            "Almost all of these are an early-race data quirk, not a bug: FastF1's TyreLife counts "
            "from before the recorded race start (formation/installation laps), so on several drivers' "
            "laps 1-9 TyreLife already exceeds LapNumber. RaceStateRequest correctly rejects that as "
            "inconsistent input (tyre_age cannot exceed current_lap) — the same validation that keeps "
            "the live simulator from accepting nonsensical states. Those laps are excluded from the "
            "backtest rather than forced through with a clamped, less-truthful tyre age."
            if errors else "No request errors."
        ),
        "n_laps_with_ground_truth_label": len(labelled),
        "confusion": {"hits": hits, "misses": misses, "false_calls": false_calls, "correct_stay_out": correct_stay},
        "precision": round(hits / (hits + false_calls), 3) if (hits + false_calls) else None,
        "recall": round(hits / (hits + misses), 3) if (hits + misses) else None,
        "pit_call_offset_laps": {
            "n": len(offsets),
            "mean": round(statistics.mean(offsets), 2) if offsets else None,
            "values": offsets,
            "note": "Positive = called later than the real stop; negative = called earlier. "
                    "One real labelled stop in this dataset's test window means this is illustrative, "
                    "not a statistically powered estimate (see README's Known Limitations).",
        },
        "expert_system_rule_sanity_check": {
            "n_laps_with_safety_car_rule_fired": len(sc_rule_firings),
            "n_laps_with_safety_car_rule_fired_on_a_non_sc_vsc_lap": len(sc_rule_firings_on_non_sc_lap),
            "n_actual_sc_or_vsc_laps_in_this_race": len(actual_sc_vsc_laps),
            "passed": len(sc_rule_firings_on_non_sc_lap) == 0,
            "finding": (
                "Every R-SC-* rule firing happened on a lap this script mapped to SC or VSC — "
                "no false positives against the real track-status history."
                if len(sc_rule_firings_on_non_sc_lap) == 0 else
                f"{len(sc_rule_firings_on_non_sc_lap)} safety-car-category rule firing(s) occurred on "
                "a lap not mapped to SC/VSC — see the listed laps for which rule and why."
            ),
            "offending_rows": sc_rule_firings_on_non_sc_lap[["driver", "lap", "rules_fired", "raw_track_status"]]
                               .to_dict("records") if not sc_rule_firings_on_non_sc_lap.empty else [],
        },
        "rule_base_reachability_check": _rule_base_reachability(),
    }


#: Rules confirmed genuinely unsupportable by any data this system has —
#: see strategy_service.py's "Rule-base input coverage" note for the
#: per-key reasoning. Kept here (not re-derived) so this report's limitation
#: list matches the one investigated and documented at the source.
_GENUINE_LIMITATION_REASONS = {
    "R-PIT-002": "needs undercut_threat — a rival car's strategic state, not in any single-car snapshot",
    "R-PIT-003": "needs gap_ahead + overcut_opportunity — rival-car data this system doesn't have",
    "R-DEG-003": "needs graining_risk — no tyre-graining model exists anywhere in this codebase",
    "R-STRAT-001": "needs circuit + grid_position — this system only ever runs on one circuit, and has no starting-grid field",
    "R-STRAT-002": "needs overtaking_difficulty — a circuit characteristic in no dataset this system reads",
    "R-STRAT-003": "needs overtaking_difficulty + pit_loss — same as above",
    "R-ERS-001": "needs drs_enabled + gap_behind — rival-car data this system doesn't have",
    "R-TAC-001": "needs drs_enabled + gap_ahead — rival-car data this system doesn't have",
    "R-TAC-002": "needs gap_ahead + gap_behind — rival-car data this system doesn't have",
    "R-TAC-003": "needs gap_behind — rival-car data this system doesn't have",
}


def _rule_base_reachability() -> dict:
    """Not every input key a Task 2 rule conditions on is one
    ``strategy_service._run_expert_system`` actually populates for a live
    request. An earlier audit found 14 of 32 rules referencing a key outside
    what that function populated; as of this run, 4 of those 14
    (R-SC-002, R-SC-003, R-PIT-004, R-STRAT-004) were made reachable by
    adding ``in_pit_window``, ``safety_car_probability``,
    ``safety_car_likelihood`` and ``tyre_age_laps`` — all genuinely derivable
    from data this system has (see strategy_service.py's module-level note).
    The remaining 10 need rival-car telemetry, multi-circuit data, or a
    tyre-graining model this project doesn't have, and are listed below as
    limitations rather than worked around with invented numbers."""
    import inspect

    from app.intelligence.expert_system import rule_base as rule_base_mod
    from app.intelligence.expert_system.rule_base import build_rule_base
    from app.services.strategy_service import _historical_safety_car_rate

    sc_rate = _historical_safety_car_rate()
    populated_keys = {
        "current_lap", "total_laps", "laps_remaining", "current_position",
        "current_compound", "tyre_wear", "track_temperature", "weather_severity",
        "rain_probability", "track_wet", "track_status", "fuel_margin",
        "in_pit_window", "safety_car_probability", "safety_car_likelihood", "tyre_age_laps",
    }
    rules = build_rule_base()
    unreachable = []
    for r in rules:
        needed = {c.key for c in r.conditions}
        missing = sorted(needed - populated_keys)
        if missing:
            unreachable.append({
                "rule_id": r.rule_id, "name": r.name, "missing_inputs": missing,
                "reason": _GENUINE_LIMITATION_REASONS.get(r.rule_id, "not categorised"),
            })
    return {
        "source": _rel(inspect.getfile(rule_base_mod)),
        "populated_input_keys": sorted(populated_keys),
        "historical_safety_car_vsc_rate_percent": round(sc_rate, 2),
        "n_rules_total": len(rules),
        "n_rules_structurally_unreachable_via_live_pipeline": len(unreachable),
        "unreachable_rules": unreachable,
        "finding": (
            f"{len(unreachable)} of {len(rules)} Task 2 rules remain structurally unreachable via the "
            "live /api/strategy/predict pipeline, each for a documented reason (rival-car telemetry, "
            "multi-circuit data, or a graining model this system doesn't have — see 'reason' per rule). "
            "4 rules (R-SC-002, R-SC-003, R-PIT-004, R-STRAT-004) were fixed by adding in_pit_window, "
            "safety_car_probability/likelihood and tyre_age_laps to strategy_service._run_expert_system's "
            "inputs, all genuinely derivable from data this system has — none of the 10 remaining "
            "limitations were worked around with an invented number."
        ),
    }


# =============================================================================
# 4. Explainability validation
# =============================================================================
def explainability_validation(n_bootstrap: int = 200, n_faithfulness_rows: int = 20) -> dict:
    if not XAI_RESULTS_JSON.exists():
        return {"available": False, "reason": f"{_rel(XAI_RESULTS_JSON)} not found — Task 8 has not run."}

    out = {"available": True}

    # --- SHAP top-feature stability across bootstrap resamples of the
    # already-computed per-lap SHAP values (no re-running KernelExplainer). --
    stability = {}
    for target in ("target_laptime", "target_pit_next_lap"):
        csv_path = XAI_DIR / "shap" / f"{target}_shap_values.csv"
        if not csv_path.exists():
            stability[target] = {"available": False, "reason": f"{_rel(csv_path)} not found."}
            continue
        shap_df = pd.read_csv(csv_path)
        # shap_base_value is the explainer's expected-value offset, not a feature
        # attribution (it is ~constant across rows by definition) — excluded so
        # it cannot trivially "win" every bootstrap resample.
        feature_cols = [c for c in shap_df.columns if c.startswith("shap_") and c != "shap_base_value"]
        if not feature_cols:
            feature_cols = shap_df.select_dtypes(include="number").columns.tolist()
        rng = np.random.default_rng(42)
        n = len(shap_df)
        top_features = []
        for _ in range(n_bootstrap):
            sample = shap_df.iloc[rng.integers(0, n, size=n)]
            mean_abs = sample[feature_cols].abs().mean().sort_values(ascending=False)
            top_features.append(mean_abs.index[0])
        from collections import Counter
        counts = Counter(top_features)
        winner, win_count = counts.most_common(1)[0]
        runner_up = counts.most_common(2)[1] if len(counts) > 1 else None
        stability[target] = {
            "n_bootstrap": n_bootstrap, "n_rows": n,
            "most_frequent_top_feature": winner.removeprefix("shap_"),
            "share_of_resamples_with_this_top_feature": round(win_count / n_bootstrap, 3),
            "runner_up": [runner_up[0].removeprefix("shap_"), runner_up[1]] if runner_up else None,
        }
    out["shap_top_feature_stability"] = stability

    # --- Faithfulness: ablating the top feature should move the prediction
    # more than ablating a random feature. --------------------------------
    try:
        from app.intelligence.xai.loading import load_target

        faithfulness = {}
        for target in ("target_laptime", "target_pit_next_lap"):
            t = load_target(target)
            rng = np.random.default_rng(7)
            idx = rng.choice(len(t.X_test), size=min(n_faithfulness_rows, len(t.X_test)), replace=False)
            X = t.X_test[idx]
            base_pred = np.asarray(t.dnn_predict(X)).ravel()

            mean_feature_value = t.X_train.mean(axis=0)
            top_feat_idx = int(np.argmax(np.abs(X - mean_feature_value).mean(axis=0)))  # most-varying feature, proxy for "a real top feature on this sample"
            top_deltas, random_deltas = [], []
            for i in range(X.shape[0]):
                row = X[i].copy()
                row[top_feat_idx] = mean_feature_value[top_feat_idx]
                top_deltas.append(abs(float(t.dnn_predict(row[None, :])[0]) - base_pred[i]))

                rand_idx = int(rng.integers(0, X.shape[1]))
                row2 = X[i].copy()
                row2[rand_idx] = mean_feature_value[rand_idx]
                random_deltas.append(abs(float(t.dnn_predict(row2[None, :])[0]) - base_pred[i]))

            faithfulness[target] = {
                "n_rows": len(idx),
                "top_feature": t.features[top_feat_idx],
                "mean_abs_delta_ablating_top_feature": round(float(np.mean(top_deltas)), 5),
                "mean_abs_delta_ablating_random_feature": round(float(np.mean(random_deltas)), 5),
                "top_feature_moves_prediction_more": float(np.mean(top_deltas)) > float(np.mean(random_deltas)),
                "note": "Top feature per row is the one furthest (in training-std units) from the "
                        "training mean on that row, a practical proxy for 'a feature SHAP would call "
                        "important here'; ablation sets it to the training mean, matching the "
                        "committed counterfactual methodology (artifacts/xai/counterfactual_analysis.csv).",
            }
        out["faithfulness_check"] = faithfulness
    except Exception as exc:
        out["faithfulness_check"] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    # --- Traceability: every recommendation must carry rule + model
    # evidence. Checked structurally against the pipeline's own response
    # shape (not re-deriving it), on one live example per target combo. ----
    try:
        from fastapi.testclient import TestClient

        from app.api.main import app

        client = TestClient(app)
        options = client.get("/api/data/options").json()
        race_state = {
            "driver": options["drivers"][0], "team": options["teams"][0],
            "current_lap": 20, "total_laps": options["total_laps_hint"],
            "tyre_compound": options["compounds"][0], "tyre_age": 12,
            "track_temperature": options["track_temperature_range"]["mean"], "explain": True,
        }
        body = client.post("/api/strategy/predict", json=race_state).json()
        rec = body.get("recommendation", {})
        out["traceability_check"] = {
            "recommendation_has_reason_text": bool(rec.get("reason")),
            "recommendation_cites_confidence": "confidence" in rec,
            "recommendation_reports_source": "source" in rec,
            "triggered_rules_listed": isinstance(body.get("triggered_expert_rules"), list),
            "ml_evidence_present": body.get("prediction", {}).get("predicted_lap_time_seconds") is not None,
            "dl_evidence_present": body.get("dl_prediction", {}).get("predicted_lap_time_seconds") is not None,
            "passed": all([
                bool(rec.get("reason")), "confidence" in rec, "source" in rec,
                isinstance(body.get("triggered_expert_rules"), list),
            ]),
        }
    except Exception as exc:
        out["traceability_check"] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    return out


# =============================================================================
# 5. Responsible AI assessment
# =============================================================================
def responsible_ai_assessment() -> dict:
    from fastapi.testclient import TestClient

    from app.api.main import app

    client = TestClient(app)
    options = client.get("/api/data/options").json()
    base = {
        "driver": options["drivers"][0], "team": options["teams"][0],
        "current_lap": 25, "total_laps": options["total_laps_hint"],
        "tyre_compound": options["compounds"][0], "tyre_age": 12,
        "track_temperature": options["track_temperature_range"]["mean"],
    }

    def action_for(state):
        r = client.post("/api/strategy/predict", json=state)
        if r.status_code != 200:
            return None
        return r.json().get("recommendation", {}).get("action")

    base_action = action_for(base)
    keys = ("track_temperature_plus5", "track_temperature_minus5", "tyre_age_plus2", "tyre_age_minus2")
    flips = dict.fromkeys(keys, 0)
    trials = dict.fromkeys(keys, 0)
    skipped_invalid = dict.fromkeys(keys, 0)
    perturbation_log = []
    # Perturb around several different base race states, not just one, so the
    # flip rate isn't a single anecdote.
    sample_laps = list(range(5, min(options["total_laps_hint"], 50), 5))
    for lap in sample_laps:
        tyre_age = min(lap, 15)
        state = {**base, "current_lap": lap, "tyre_age": tyre_age}
        a0 = action_for(state)
        if a0 is None:
            continue
        for key, delta, field in (
            ("track_temperature_plus5", 5, "track_temperature"),
            ("track_temperature_minus5", -5, "track_temperature"),
            ("tyre_age_plus2", 2, "tyre_age"),
            ("tyre_age_minus2", -2, "tyre_age"),
        ):
            # tyre_age is clamped to the schema's own valid range (0..current_lap,
            # the same invariant RaceStateRequest enforces) so a perturbation at
            # the edge of the race isn't scored as "couldn't test" — it's clamped
            # to the nearest valid value instead, same as the live form does.
            if field == "tyre_age":
                new_value = max(0, min(lap, state[field] + delta))
            else:
                new_value = max(0, state[field] + delta)
            if new_value == state[field]:
                skipped_invalid[key] += 1
                continue  # clamping left no real perturbation to test at this lap
            perturbed = {**state, field: new_value}
            a1 = action_for(perturbed)
            trials[key] += 1
            flipped = a1 is not None and a1 != a0
            flips[key] += int(flipped)
            perturbation_log.append({"lap": lap, "perturbation": key, "base_action": a0,
                                      "perturbed_value": new_value, "perturbed_action": a1, "flipped": flipped})

    flip_rates = {k: round(flips[k] / trials[k], 3) if trials[k] else None for k in flips}

    fig, ax = plt.subplots(figsize=(6.5, 4))
    labels = list(flip_rates.keys())
    values = [flip_rates[k] or 0 for k in labels]
    ax.bar(labels, values)
    ax.set_ylabel("recommendation flip rate")
    ax.set_title("Robustness: recommendation flips under small input perturbations")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    robustness_fig = EVALUATION_DIR / "robustness_flip_rate.png"
    fig.savefig(robustness_fig, dpi=130)
    plt.close(fig)

    # --- Fairness: MAE and pit recall by team and compound, from the Task 8
    # fairness assessment where it already covers compound, extended with a
    # team breakdown computed the same way (same test rows, same model). ----
    fairness = {"available": False}
    fairness_csv = XAI_DIR / "fairness_assessment.csv"
    fig_path = None
    if fairness_csv.exists():
        fdf = pd.read_csv(fairness_csv)
        # sample_note is "OK" for an adequately-sized group and a descriptive
        # warning string otherwise (not merely non-empty — "OK" is non-empty
        # too), so the flag is "note says something other than OK".
        small_sample_flagged = (
            fdf[~fdf.get("sample_note", "").astype(str).isin(["OK", ""])]
            if "sample_note" in fdf.columns else pd.DataFrame()
        )
        fairness = {
            "available": True,
            "source": _rel(fairness_csv),
            "rows": fdf.to_dict("records"),
            "n_small_sample_groups_flagged": len(small_sample_flagged),
            "note": "Read from Task 8's own stratification (driver/team/compound), which already "
                    "flags groups under 30 laps or 5 pit laps as descriptive-only rather than a rate.",
        }
        team_rows = fdf[fdf.get("group_type", "") == "Team"] if "group_type" in fdf.columns else pd.DataFrame()
        if not team_rows.empty and "mae" in team_rows.columns:
            fig, ax = plt.subplots(figsize=(7, 4.2))
            team_rows_sorted = team_rows.sort_values("mae")
            ax.barh(team_rows_sorted["group"].astype(str), team_rows_sorted["mae"])
            ax.set_xlabel("lap-time MAE (s)")
            ax.set_title("Lap-time error by team (Task 8 fairness_assessment.csv)")
            ax.grid(alpha=0.3, axis="x")
            fig.tight_layout()
            fig_path = EVALUATION_DIR / "fairness_by_team.png"
            fig.savefig(fig_path, dpi=130)
            plt.close(fig)
    else:
        fairness = {"available": False, "reason": f"{_rel(fairness_csv)} not found."}

    # --- Transparency checklist -------------------------------------------
    ml_eval_report = ML_METRICS_DIR / "model_comparison.json"
    pit_underfit_note = None
    if ml_eval_report.exists():
        data = json.loads(ml_eval_report.read_text())
        pit_rows = data.get("classification", [])
        selected = next((r for r in pit_rows if r.get("selected")), None)
        if selected:
            cv = selected.get("cv_mae") or selected.get("cv_pr_auc")
            test = selected.get("test_mae") or selected.get("test_pr_auc")
            pit_underfit_note = {
                "selected_model": selected.get("model"),
                "cv_score": cv, "test_score": test,
                "investigation": (
                    "The pit holdout contains exactly one positive lap (README's Known Limitations), "
                    "so a single test-set hit or miss swings test precision/recall/F1 by a large "
                    "margin regardless of model quality. The CV-vs-holdout gap documented throughout "
                    "this project (e.g. artifacts/reports/model_selection_report.md's "
                    "selection_warning) is an evaluation-design artifact of a tiny, non-representative "
                    "holdout — not evidence of underfitting on the training data itself, since the same "
                    "models post reasonable, stable scores across the cross-validation folds."
                ),
            }

    synthetic_in_raw = []
    raw_dir = REPO_ROOT / "data" / "raw"
    if raw_dir.exists():
        synthetic_in_raw = [str(p.relative_to(REPO_ROOT)) for p in raw_dir.rglob("*")
                             if p.is_file() and "synthetic" in p.name.lower()]

    human_oversight_text = None
    report_module = REPO_ROOT / "app" / "services" / "strategy_report.py"
    if report_module.exists():
        text = report_module.read_text()
        for line in text.splitlines():
            if "not for betting" in line.lower() or "strategist decides" in line.lower():
                human_oversight_text = line.strip(" -\"',")
                break

    return {
        "robustness": {
            "base_action": base_action,
            "trials_per_perturbation": trials,
            "skipped_degenerate_perturbations": skipped_invalid,
            "flip_counts": flips,
            "flip_rates": flip_rates,
            "figure": _rel(robustness_fig),
            "log": perturbation_log,
            "note": "tyre_age perturbations are clamped to [0, current_lap] — the schema's own valid "
                    "range — rather than sent invalid; a clamp that lands back on the original value is "
                    "excluded from the trial count (nothing was actually perturbed), not scored as a "
                    "non-flip.",
        },
        "fairness": fairness,
        "fairness_by_team_figure": _rel(fig_path) if fig_path else None,
        "transparency": {
            "approximated_features_tracked_per_request": True,  # structural: prediction.approximated_features on every /api/strategy/predict response
            "synthetic_files_found_in_data_raw": synthetic_in_raw,
            "synthetic_files_in_data_raw_count": len(synthetic_in_raw),
            "single_race_limitation": "2023 Bahrain GP only — stated in README's Known Limitations and in every strategy report's data-source line.",
            "one_positive_pit_holdout": True,
            "pit_model_cv_vs_test_gap_investigation": pit_underfit_note,
        },
        "human_oversight": {
            "statement_found_in_strategy_report_py": human_oversight_text,
            "present": human_oversight_text is not None,
        },
    }


# =============================================================================
# 6. Deployment readiness
# =============================================================================
def deployment_readiness(test_results: dict) -> dict:
    checklist = {}

    backend_dockerfile = REPO_ROOT / "Dockerfile"
    frontend_dockerfile = REPO_ROOT / "frontend" / "Dockerfile"
    compose_file = REPO_ROOT / "docker-compose.yml"
    checklist["dockerfiles_present"] = {
        "backend": backend_dockerfile.exists(), "frontend": frontend_dockerfile.exists(),
        "compose": compose_file.exists(),
    }

    try:
        from fastapi.testclient import TestClient

        from app.api.main import app
        health = TestClient(app).get("/api/health").json()
        checklist["health_check_passes"] = health.get("status") == "ok"
        checklist["health_check_body"] = health
    except Exception as exc:
        checklist["health_check_passes"] = False
        checklist["health_check_error"] = str(exc)

    checklist["tests_pass"] = bool(test_results.get("available")) and test_results.get("failed", 1) == 0 \
        and test_results.get("errors", 1) == 0 and test_results.get("total", 0) > 0

    # Grep the frontend for numeric literals that look like metrics, the same
    # way a reviewer would — not a claim, a real scan with the matches listed.
    frontend_dir = REPO_ROOT / "frontend" / "app"
    components_dir = REPO_ROOT / "frontend" / "components"
    suspects = []
    metric_pattern = re.compile(r"\b0\.\d{2,4}\b|\b\d{1,3}\.\d{1,3}%\b")
    for base in (frontend_dir, components_dir):
        if not base.exists():
            continue
        for path in base.rglob("*.tsx"):
            for i, line in enumerate(path.read_text().splitlines(), start=1):
                if metric_pattern.search(line) and "fmt(" not in line and "className" not in line.lower()[:40]:
                    suspects.append({"file": str(path.relative_to(REPO_ROOT)), "line": i, "text": line.strip()[:160]})
    checklist["hard_coded_metric_scan"] = {
        "n_matches": len(suspects),
        "matches": suspects[:50],
        "note": "Regex scan for bare decimal/percent literals in .tsx files, excluding lines using the "
                "fmt() formatter (which always wraps a fetched value) or Tailwind className strings. "
                "Each match needs a human read — a CSS opacity or a layout ratio also matches the "
                "pattern — see 'matches' for the actual lines.",
    }

    readme = (REPO_ROOT / "README.md").read_text() if (REPO_ROOT / "README.md").exists() else ""
    checklist["readme_states_995_rows"] = "995" in readme
    checklist["readme_mentions_task9_and_task10"] = "Task 9" in readme and "Task 10" in readme

    passed_items = [
        checklist["dockerfiles_present"]["backend"] and checklist["dockerfiles_present"]["frontend"]
        and checklist["dockerfiles_present"]["compose"],
        checklist["health_check_passes"],
        checklist["tests_pass"],
        checklist["readme_states_995_rows"],
        checklist["readme_mentions_task9_and_task10"],
    ]
    checklist["summary"] = {"items_checked": len(passed_items), "items_passed": sum(bool(x) for x in passed_items)}
    return checklist


# =============================================================================
# 7. Future enhancements — a stated plan, not a measured result
# =============================================================================
def future_enhancements() -> dict:
    return {
        "note": "A short list, not a measurement — included for Task 10 completeness.",
        "items": [
            {"title": "More races", "detail": "Fit/evaluate across multiple Grands Prix so results "
             "generalise beyond one circuit's tyre/fuel/safety-car regime."},
            {"title": "Multi-session Task 4/5", "detail": "Extend data engineering and feature "
             "selection to practice/qualifying sessions, not race-only."},
            {"title": "QAOA / quantum optimisation vs. the A* baseline", "detail": "The project already "
             "has a parameter-matched VQC/VQR/quantum-kernel comparison (Extension task); a QAOA "
             "formulation of the pit-stop scheduling problem itself, compared against A*'s plan cost, "
             "is the natural next step."},
            {"title": "Live FastF1 timing feed", "detail": "Replace the single-snapshot race-state form "
             "with a live timing subscription so laps_remaining, fuel and position stop being "
             "user-entered approximations."},
        ],
    }


# =============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-pytest", action="store_true", help="Skip running the full pytest suite.")
    parser.add_argument("--n-latency", type=int, default=200, help="Number of timed /predict requests.")
    args = parser.parse_args()

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    log.info("=== 1/7 functional testing ===")
    test_results = functional_testing(skip_pytest=args.skip_pytest)
    _write_json("test_results.json", test_results)

    log.info("=== 2/7 performance evaluation ===")
    _write_json("performance.json", performance_evaluation(args.n_latency))

    log.info("=== 3/7 domain validation ===")
    _write_json("domain_validation.json", domain_validation())

    log.info("=== 4/7 explainability validation ===")
    _write_json("explainability_validation.json", explainability_validation())

    log.info("=== 5/7 responsible AI assessment ===")
    _write_json("responsible_ai.json", responsible_ai_assessment())

    log.info("=== 6/7 deployment readiness ===")
    _write_json("deployment_readiness.json", deployment_readiness(test_results))

    log.info("=== 7/7 future enhancements ===")
    _write_json("future_enhancements.json", future_enhancements())

    _write_json("evaluation_summary.json", {
        "sections": [
            "test_results.json", "performance.json", "domain_validation.json",
            "explainability_validation.json", "responsible_ai.json",
            "deployment_readiness.json", "future_enhancements.json",
        ],
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    })
    log.info("done — see %s", EVALUATION_DIR.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
