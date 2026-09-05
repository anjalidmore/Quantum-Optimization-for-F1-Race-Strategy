# TODO — gap analysis

Working backlog for the F1 Race Strategy platform. **This file lives on `proj-mode` only** and is
deliberately kept off `main`: it is useful to whoever is doing the work and noise to someone evaluating
the result.

Every entry below comes from something observed while auditing this repository on **2026-09-05** —
running the documented setup in a clean clone, rebuilding all artifacts, executing the full test suite,
building the frontend, exercising the live API, and scanning dependencies and git history. Nothing here
is speculative.

**Priority counts:** 33 entries — 9 High · 14 Medium · 10 Low

**Updated 2026-09-05 (Phase 2 remediation):** 5 entries closed and 2 partially closed —
see [`todo-complete.md`](todo-complete.md). The measurement that shaped the phase: the
chronological holdout contains **1 pit event in 180 laps**, so every classification change
is tuned and measured on pooled out-of-fold CV predictions (36 positives) and the test-set
numbers are reported with that caveat rather than presented as evidence.

**Updated 2026-09-05 (Phase 1 remediation):** 9 entries closed — see
[`todo-complete.md`](todo-complete.md). Closed entries are struck through below rather than deleted, so
the record of what was wrong stays readable. One entry (the Next.js advisories) is blocked pending a
decision on a major-version bump.

**Updated 2026-09-05 (second pass):** Tasks 7 and 8 are now implemented, so their "not started"
entries are gone and are replaced by what building them actually surfaced. Two earlier entries are
also closed by work done since: CI now exists (`.github/workflows/ci.yml`) and so does `LICENSE`.

---

## Model Quality

### ~~[Priority: High] Pit-decision classifier is unusable at its operating point~~ — ✅ CLOSED (Phase 2)

**See [`todo-complete.md`](todo-complete.md) for what changed, the before/after numbers, and how it was verified.**


**Why:** Pit events are 4.8% of laps (48 of 995). The selected `random_forest` scores test ROC-AUC 0.98 —
which looks excellent and is the number most likely to be quoted — but test PR-AUC is 0.25 and F1 is
0.077, meaning precision ≈ 0.04 at the default 0.5 threshold. The model *ranks* laps well and is
genuinely useless as a decision rule. `/api/strategy/predict` returns `recommended_action` derived partly
from this classifier, so a reviewer trusting the dashboard is trusting a coin-flip dressed as a
prediction.

**How (short):**
1. Add `class_weight="balanced"` to the classifier configs in `app/intelligence/ml/classification.py`,
   and evaluate resampling (SMOTE or simple undersampling) inside the CV folds only.
2. Tune the decision threshold on the validation folds by maximising F1 or expected cost rather than
   defaulting to 0.5; persist the chosen threshold into `artifacts/metadata/model_registry.json`.
3. Make PR-AUC, not ROC-AUC, the primary selection metric for this task in
   `app/intelligence/ml/selection.py` — ROC-AUC is the wrong headline at this prevalence.
4. Surface precision/recall/F1 alongside ROC-AUC on `frontend/app/machine-learning/page.tsx`.

### ~~[Priority: High] Selected regression model has negative test R²~~ — ✅ CLOSED (Phase 2)

**See [`todo-complete.md`](todo-complete.md) for what changed, the before/after numbers, and how it was verified.**


**Why:** `decision_tree` is selected on CV MAE (1.19 s, CV R² 0.34) but scores test R² −0.17 — worse than
predicting the mean. `svr` generalises better on the same holdout (test MAE 0.78 s, R² 0.30) yet is not
selected. The selection procedure is defensible, but shipping a model that underperforms the mean on
held-out data as the platform's lap-time predictor is not.

**How (short):**
1. Add a generalisation guard in `app/intelligence/ml/selection.py`: refuse to select a model whose test
   R² is negative when a candidate with positive test R² exists, or report both and require an explicit
   override.
2. Investigate the CV/test disagreement — the holdout is the final stint of one race, a different
   fuel/tyre regime from training. Consider a grouped or multi-race holdout.
3. Record the CV-versus-test gap explicitly in `artifacts/reports/model_selection_report.md` as a
   selection warning.

### 🟡 [Priority: Medium] Evaluation rests on a single race and a ~180-row holdout — PARTIALLY CLOSED (Phase 2)

**🟡 PARTIALLY CLOSED (Phase 2).** `race_level_holdout()` now exists in `app/intelligence/ml/splits.py`, but the committed dataset is a single session so it has **not been exercised on multiple races** — it raises rather than degrading silently. The multi-session fetch remains open. See [`todo-complete.md`](todo-complete.md).


**Why:** All Task 6 conclusions come from one session (2023 Bahrain GP, 995 usable rows after warm-up).
Variance is high enough that model ranking is not trustworthy, which is the likely root cause of the two
entries above.

**How (short):**
1. Use `scripts/fetch_real_session.py` to pull 5–10 additional 2023 sessions into `data/raw/`.
2. Extend `app/intelligence/ml/splits.py` to hold out whole *races*, not the tail of one race.
3. Re-run `docs/notebooks/task5_feature_engineering.ipynb` and `scripts/build_all.py --force`, then
   compare the model ranking's stability across sessions.

---

## Security

### ~~[Priority: High] API allows every origin, method and header, with no authentication~~ — ✅ CLOSED (Phase 1)

**See [`todo-complete.md`](todo-complete.md) for what changed, the before/after numbers, and how it was verified.**


**Why:** `app/api/main.py:29-34` sets `allow_origins=["*"]`, `allow_methods=["*"]`,
`allow_headers=["*"]`, and no endpoint requires a credential. While the API runs on localhost, any
website the developer visits can call it from their browser. If this is ever deployed as-is, the
prediction endpoints and the whole `artifacts/` tree are open to the internet.

**How (short):**
1. Replace the wildcard in `app/api/main.py` with an origin list read from an env var, defaulting to
   `["http://localhost:3000"]`.
2. Restrict `allow_methods` to `["GET", "POST"]` and drop the header wildcard.
3. Before any non-local deployment, add an auth dependency (API key header at minimum) to the routers in
   `app/api/routers/`.

### [Priority: High] Frontend ships dependencies with high-severity advisories — ⏸ BLOCKED, decision required

**Why:** `npm audit` on `frontend/` reports 2 high-severity vulnerabilities: Next.js 14.2.35 carries 11
advisories (SSRF via rewrites and Server Actions, cache poisoning of RSC responses, cache confusion,
middleware bypass, DoS, unauthenticated disclosure of internal Server Function endpoints), and its bundled
postcss has XSS and arbitrary-file-read issues. The SSRF and disclosure advisories matter as soon as this
is hosted anywhere.

**Phase 1 finding — the premise changed.** There is **no patch bump available**: 14.2.35 is already the
latest 14.2.x. `npm audit --json` reports that for all five advisories the only `fixAvailable` is
`next@16.3.4` / `eslint-config-next@16.3.4`, both `isSemVerMajor: true`. The count has also grown from 2
to **5** high-severity since this entry was written. Awaiting a decision on the major bump.

**Interim:** `npm audit --audit-level=high` now runs in CI as a non-gating step, so the count is visible
on every PR. See [`todo-complete.md`](todo-complete.md#-deferred--decision-required).

**How (short):**
1. Decide on `next@16`. The five pages are already App Router server components, which is most of the
   usual migration work, so the risk is moderate — but it needs a build-and-click pass over all 8 routes.
2. Once done, remove `continue-on-error: true` from the npm audit step in `.github/workflows/ci.yml`.

### ~~[Priority: Medium] The whole artifacts tree is served unauthenticated as static files~~ — ✅ CLOSED (Phase 1)

**See [`todo-complete.md`](todo-complete.md) for what changed, the before/after numbers, and how it was verified.**


**Why:** `app/api/main.py:43` mounts `artifacts/` at `/artifacts` with no filtering. Today that directory
holds only figures and reports, but it also holds `*.joblib` model files — anyone who can reach the API
can download the trained models. Combined with the CORS wildcard, this is exfiltration by default.

**How (short):**
1. Mount only the subdirectories the frontend actually reads (`figures/`, `reports/`, and the per-task
   report dirs) rather than the tree root.
2. Explicitly exclude `artifacts/models/` and `artifacts/metadata/` from static serving.
3. Add a test in `tests/test_api.py` asserting `GET /artifacts/models/laptime/decision_tree.joblib` 404s.

### ~~[Priority: Medium] `run.sh` kills arbitrary processes on ports 8000 and 3000~~ — ✅ CLOSED (Phase 1)

**See [`todo-complete.md`](todo-complete.md) for what changed, the before/after numbers, and how it was verified.**


**Why:** `run.sh:54` issues `kill "$pid"` against whatever is listening on those ports, with no prompt and
no check that the process belongs to this project. A contributor running an unrelated dev server on :3000
loses it, silently, with unsaved state.

**How (short):**
1. In `run.sh`, print the process name and pid and require confirmation before killing, or add a
   `--force-ports` flag and refuse by default.
2. Alternatively read `BACKEND_PORT`/`FRONTEND_PORT` from the environment so a contributor can move off
   the conflict instead of killing it.

### ~~[Priority: Low] `msgpack` 1.1.2 has a known advisory~~ — ✅ CLOSED (Phase 1)

**See [`todo-complete.md`](todo-complete.md) for what changed, the before/after numbers, and how it was verified.**


**Why:** `pip-audit` reports PYSEC-2026-3625 against msgpack 1.1.2, fixed in 1.2.1. It arrives
transitively via `fastf1` → `signalrcore` and is only exercised when fetching live sessions, so exposure
is small — but it is the only flagged Python dependency and the fix is trivial.

**How (short):**
1. Add `msgpack>=1.2.1` to `requirements.txt`.
2. Re-run `pip-audit` to confirm a clean report.

### ~~[Priority: Low] No secret-scanning or dependency-audit automation~~ — ✅ CLOSED (Phase 1)

**See [`todo-complete.md`](todo-complete.md) for what changed, the before/after numbers, and how it was verified.**


**Why:** This audit found no secrets in the working tree or in the full git history across all branches,
and no `.env` or credential file was ever committed — a genuinely clean result. But nothing prevents the
next commit from introducing one, and `pip-audit`/`npm audit` are run only when someone remembers.

**How (short):**
1. Add `pip-audit` and `npm audit --audit-level=high` steps to the CI workflow (see the Testing section).
2. Add a `gitleaks` or `detect-secrets` pre-commit hook.

---

## Testing

### ~~[Priority: High] A non-hermetic test run silently deleted Task 7's registry rows~~ — ✅ CLOSED (Phase 1)

**See [`todo-complete.md`](todo-complete.md) for what changed, the before/after numbers, and how it was verified.**


**Why:** Because `pytest` retrains Task 6, and Task 6's `write_registry` overwrote the shared registry
wholesale, running the test suite deleted the Task 7 entries — leaving `/api/dl/models` returning 404
while the `.keras` files sat on disk. `write_registry` now preserves foreign-family entries and
`dl.pipeline.restore_registry_entries()` rebuilds them from artifacts, so the immediate failure is fixed.
The underlying cause — tests that mutate committed state — is not.

**How (short):**
1. Fix the hermeticity entry below; it is the root cause.
2. Add a test asserting the registry still contains both families after a Task 6 retrain, so a future
   whole-file writer cannot reintroduce this silently.

### ~~[Priority: High] The test suite is not hermetic — it rewrites committed artifacts~~ — ✅ CLOSED (Phase 1)

**See [`todo-complete.md`](todo-complete.md) for what changed, the before/after numbers, and how it was verified.**


**Why:** Running `pytest` retrains models and overwrites 10 tracked files under `artifacts/`
(`metrics/*.json`, `reports/*.md`, `manifest.json`, `metadata/model_registry.json`). A clean clone goes
dirty just from running the documented test command. This trains contributors to `git checkout --
artifacts/` reflexively, which is exactly how a real artifact change gets discarded by accident.

**How (short):**
1. Give the ML tests a `tmp_path` output root — parameterise the writer functions in
   `app/intelligence/ml/persistence.py` and `reports.py` on an output directory, and have
   `tests/test_ml_training.py` pass a fixture-scoped temp dir.
2. Add a CI step that fails if `git status --porcelain` is non-empty after `pytest`.

### [Priority: Medium] CI does not cover Tasks 7 and 8

**Why:** `.github/workflows/ci.yml` now exists and runs ruff, pytest and the frontend build — but it was
written before Tasks 7 and 8, so it does not install `keras`/`torch`/`shap`/`lime`/`dice-ml` explicitly
nor run `scripts/build_all.py --force`. The 193-test suite would still pass there (the DL tests skip
when artifacts are absent), which means CI would go green while never exercising the deep-learning or
explainability paths at all.

**How (short):**
1. Confirm the workflow installs from `requirements.txt` so the new dependencies come in.
2. Add a `python scripts/build_all.py --force` step, or at minimum `--skip-dl` plus a separate job that
   builds Tasks 7-8, so the skipif-guarded tests actually run.
3. Add the artifact-cleanliness check from the hermeticity entry above.

### [Priority: High] ~~No continuous integration exists~~ — RESOLVED

**Why it mattered:** There was no `.github/` directory on any branch. The 120 tests, the `build_all.py --force`
reproducibility check and the frontend build all pass — but only because they were run by hand. Nothing
stops a commit that breaks them, and the README's "120 passing" badge is static and will silently go
stale.

**Resolved by** commit `73de6d8` on `main`, which added `.github/workflows/ci.yml` running ruff, pytest,
npm lint and npm build. See the Testing entry above for what it still does not cover.

### [Priority: Medium] Zero frontend tests

**Why:** 5 pages and 9 components in `frontend/` have no test of any kind — no unit, component or e2e
coverage. `frontend/lib/api.ts` is the single boundary every page depends on, and a change to an API
response shape would surface only as a blank page at runtime.

**How (short):**
1. Add Vitest + React Testing Library; start with `frontend/lib/api.ts` response parsing against fixtures
   captured from the live API.
2. Add a Playwright smoke test that boots the API, loads all 5 routes and asserts no error boundary
   renders.

### ~~[Priority: Medium] Nothing tests the "no fabricated results" contract~~ — ✅ CLOSED (Phase 1)

**See [`todo-complete.md`](todo-complete.md) for what changed, the before/after numbers, and how it was verified.**


**Why:** The contract in `docs/PROJECT_REPORT.md` §8 — that every displayed value traces to a generated
artifact — is the project's central claim. It was verified manually in this audit by mutating
`artifacts/metrics/regression_metrics.json` and confirming the API followed. Nothing enforces it, so a
future hard-coded fallback would pass CI.

**How (short):**
1. Add `tests/test_artifact_contract.py`: write a known sentinel value into a temp copy of the metrics
   artifact, point `app/core/paths.py` at it, and assert `/api/ml/metrics` returns the sentinel.
2. Assert the reverse too — that deleting the artifact produces an explicit error rather than a default.

### ~~[Priority: Low] Test suite emits 133 warnings, including 69 pandas `SettingWithCopyWarning`~~ — ✅ CLOSED (Phase 1)

**See [`todo-complete.md`](todo-complete.md) for what changed, the before/after numbers, and how it was verified.**


**Why:** Real signal is buried. Three of these (`app/intelligence/data/pipeline.py:139`, `:155`, `:181`)
indicate assignments to a DataFrame slice that may or may not propagate — behaviour that changes in pandas
3.x with copy-on-write.

**How (short):**
1. Fix the three `pipeline.py` sites with `.loc[:, col] = …` or an explicit `.copy()` on the slice.
2. Once resolved, add `filterwarnings = ["error::pandas.errors.SettingWithCopyWarning"]` to
   `[tool.pytest.ini_options]` in `pyproject.toml` to prevent regression.

---

## Architecture / Tech Debt

### [Priority: High] `build_all.py`'s Task 5 stage is a presence check, not a build step

**Why:** Under `--force`, the Task 5 stage logs `Task 5 contract present` and finishes in 0.0 s. It never
regenerates the feature matrix. If someone points Task 4 at a new session and runs
`build_all.py --force`, Task 6 silently retrains on the **previous** feature matrix — the pipeline
reports success while producing models fitted to stale features. This is the most dangerous defect found
in the audit because it fails silently.

**How (short):**
1. Have the Task 5 stage in `scripts/build_all.py` execute
   `docs/notebooks/task5_feature_engineering.ipynb` via `nbclient` when `--force` is passed (the
   dependency is already in `requirements.txt`).
2. Failing that, make the stage compare a hash of `artifacts/data_engineering/clean/fastf1_laps_clean.csv`
   against the `source_dataset` provenance recorded in `data/processed/feature_metadata.json`, and abort
   loudly on mismatch.
3. Extract the notebook's logic into `app/intelligence/features/` so it is testable and callable, leaving
   the notebook as the narrative view.

### [Priority: Medium] Every dependency is unpinned, so results drift between environments

**Why:** `requirements.txt` uses `>=` throughout. Re-running the Task 5 notebook on a newer scikit-learn
(1.9.0) reproduced the exported feature matrix byte-identically but moved the recorded validation scores
(MAE 0.2413 → 0.2417, R² 0.5361 → 0.5343, AUC 0.9934 → 1.0) and reordered two tie-broken features. For a
project whose core claim is reproducibility, the environment must be pinned.

**How (short):**
1. Generate `requirements.lock` with `pip freeze` from the verified environment and install from it in
   `run.sh` and CI.
2. Keep `requirements.txt` as the loose declaration; document the split in `README.md`.
3. Record the resolved versions in `artifacts/metadata/model_registry.json` (it already stores
   `software_versions` — extend it to the full set).

### ~~[Priority: Medium] scikit-learn deprecation will break Task 6 at version 1.11~~ — ✅ CLOSED (Phase 2)

**See [`todo-complete.md`](todo-complete.md) for what changed, the before/after numbers, and how it was verified.**


**Why:** `SVC(probability=True)` is deprecated in scikit-learn 1.9 and removed in 1.11. It raises 17
`FutureWarning`s per test run today. Combined with unpinned dependencies, Task 6 will break on a routine
`pip install` at some point without any code change.

**How (short):**
1. Replace `SVC(probability=True)` in `app/intelligence/ml/classification.py` with
   `CalibratedClassifierCV(SVC(), ensemble=False)`.
2. Re-run `python scripts/build_all.py --force` and confirm the classification metrics stay comparable.

### [Priority: Medium] Committed binary artifacts inflate the repository and churn on every build

**Why:** `artifacts/` holds 10 `.joblib` model files and 30+ PNGs under version control. Every
`build_all.py --force` rewrites all of them — mostly embedded timestamps — so they appear in `git status`
constantly and every rebuild adds binary blobs to history.

**How (short):**
1. Decide explicitly: either keep them committed (so the repo is reviewable without a build — the current,
   defensible choice) or move to a release-asset/LFS model.
2. If keeping them, make the report generators omit the `Generated <timestamp>` line, or write it to a
   sidecar file, so a no-op rebuild produces a clean `git diff`.

### [Priority: Low] `docs/task{1,2,3,4}_*.md` were not verified against the code

**Why:** Four per-task documentation files were carried over from the pre-unification layout. This audit
verified the README, FLOW and SHOWCASE, but not these — and README §6 previously drifted from reality in
exactly this way, so the prior probability of staleness is not low.

**How (short):**
1. Read each against its `app/intelligence/` package and correct any references to the retired
   `phase1_taskN/` layout or `src/f1kr`-style package names.
2. Cross-link them from `docs/PROJECT_REPORT.md` so they are visited when it is updated.

### ~~[Priority: Medium] Two OpenMP runtimes coexist in one process~~ — ✅ CLOSED (Phase 2)

**See [`todo-complete.md`](todo-complete.md) for what changed, the before/after numbers, and how it was verified.**


**Why:** Adding `torch` for Task 7 put a third `libomp.dylib` in the process alongside sklearn's and
XGBoost's. XGBoost segfaults outright if it initialises against PyTorch's copy — the full test suite
crashed with `Fatal Python error: Segmentation fault` until import order was forced.
`app/core/runtime.py` fixes it by claiming OpenMP for XGBoost first, and every entry point goes through
`prepare_dl_runtime()`. That works, but it is a discipline the codebase has to keep rather than a
structural guarantee: a new module importing `keras` directly reintroduces the crash.

**How (short):**
1. Add a test that imports `keras` without `prepare_dl_runtime()` in a subprocess and asserts the
   guarded path is the one used — or add an import hook that fails loudly on the unguarded order.
2. Consider running the DL stage in its own process from `build_all.py`, which removes the constraint
   entirely.
3. Note that `KMP_DUPLICATE_LIB_OK=TRUE` does **not** help here (verified) — do not reach for it.

### [Priority: Low] Stray untracked `refernce.png` in the repository root

**Why:** A 1.2 MB screenshot, misspelled, untracked, sitting at the root across every branch. It shows up
in every `git status` and trains contributors to ignore untracked-file output.

**How (short):**
1. Confirm with the author whether it is needed; if so move it to `docs/images/reference.png` and commit
   it, otherwise delete it.

---

## Documentation

### [Priority: Medium] The Task 5 notebook must be re-run by hand, and nothing says so at the point of use

**Why:** `README.md` §13 documents the real-data workflow, but `scripts/build_all.py` — the script a
contributor actually runs — gives no indication that its Task 5 stage skipped the real work. This is the
documentation half of the High-priority `build_all.py` entry above.

**How (short):**
1. Make the Task 5 stage in `scripts/build_all.py` log a `WARNING` naming the notebook whenever it
   short-circuits.
2. Add the same warning to the build summary block at the end of the run.

### [Priority: Low] No `CONTRIBUTING.md`, issue templates or PR template

**Why:** `README.md` has a Contributing section, but a repository intended for outside review has no
structured place for contribution conventions, and no template to prompt contributors to run the three
verification commands before opening a PR.

**How (short):**
1. Extract the README's Contributing section into `CONTRIBUTING.md` and link to it.
2. Add `.github/PULL_REQUEST_TEMPLATE.md` with a checklist: tests pass, `build_all.py --force` exits 0,
   frontend builds, `git status` clean.

### [Priority: Low] No architecture diagram image

**Why:** `FLOW.md` uses an ASCII diagram, which is durable and diffable but reads poorly on GitHub for a
project whose selling point is the composition of five engines.

**How (short):**
1. Add a rendered diagram (Mermaid works natively on GitHub) to `README.md` under Architecture.
2. Keep the ASCII version in `FLOW.md` as the detailed reference.

---

## Planned Features (Tasks 7–10, Quantum)

### [Priority: Medium] Task 9 — three engine dashboard pages are not built

**Why:** Knowledge Representation, the Expert System and Search each produce rich artifacts that are
reachable only through the generic `/evidence` index. The Machine Learning page shows what a dedicated
page looks like; three of the five engines do not have one, which understates the work in Tasks 1–3 to
anyone browsing the dashboard.

**How (short):**
1. Add `frontend/app/knowledge/page.tsx`, `expert-system/page.tsx`, `search/page.tsx`, following the
   pattern in `frontend/app/data-analysis/page.tsx`.
2. They need no backend work — `GET /api/tasks/evidence` and the `/artifacts` mount already serve
   everything required.
3. Add each to `LINKS` in `frontend/components/Nav.tsx`.

### ~~[Priority: High] The Task 7 pit classifier is degenerate at its operating point~~ — ✅ CLOSED (Phase 2)

**See [`todo-complete.md`](todo-complete.md) for what changed, the before/after numbers, and how it was verified.**


**Why:** The deep classifier scores test ROC-AUC 0.9218 and accuracy 0.9944, but **F1, precision and
recall are all exactly 0.0** — at the 0.5 threshold it predicts "never pit" for every test lap and rides
the 4.8% class imbalance. Its PR-AUC is 0.0667. This is the same defect Task 6's classifier has, so the
platform now has *two* pit-decision models that rank acceptably and decide uselessly. Anything consuming
`predicted_class` from either is consuming a constant.

**How (short):**
1. Tune the decision threshold on the validation folds in `app/intelligence/dl/tuning.py`, maximising F1
   or expected cost rather than defaulting to 0.5, and persist it into the model spec.
2. Make PR-AUC the primary selection metric for this task in both `ml/selection.py` and `dl/tuning.py` —
   ROC-AUC is the wrong headline at this prevalence.
3. Surface precision/recall/F1 next to ROC-AUC on `/deep-learning` and `/machine-learning`.

### [Priority: Medium] Task 8 is not wired into the strategy recommendation

**Why:** `POST /api/strategy/predict` returns `triggered_expert_rules` — an auditable symbolic
explanation — but no statistical explanation. Task 8 can now produce SHAP attributions and a trust score
for exactly that prediction, so the recommendation has an explainable half and an unexplained half for no
reason other than that they have not been connected.

**How (short):**
1. Add an optional `explain: bool` field to `RaceStateRequest` in `app/api/schemas.py`, defaulting to
   `false` so the existing response shape is unchanged for current callers.
2. When true, call `app.intelligence.xai.pipeline.explain_target`'s per-row path for the constructed
   feature row and attach `shap_factors`, `trust_score` and `narrative`.
3. Show them in `frontend/components/strategy/` next to the triggered rules.

### 🟡 [Priority: Medium] Deep learning is only ever compared on one race — PARTIALLY CLOSED (Phase 2)

**🟡 PARTIALLY CLOSED (Phase 2).** `race_level_holdout()` now exists in `app/intelligence/ml/splits.py`, but the committed dataset is a single session so it has **not been exercised on multiple races** — it raises rather than degrading silently. The multi-session fetch remains open. See [`todo-complete.md`](todo-complete.md).


**Why:** Task 7's headline result — the network beating every classical model on lap time (MAE 0.5154 vs
0.7815, R² +0.48 vs −0.17) — comes from a single session's 180-row holdout. It is the strongest ML result
in the project and the one most in need of replication before it is relied on.

**How (short):**
1. Fetch several more 2023 sessions with `scripts/fetch_real_session.py`.
2. Extend `app/intelligence/ml/splits.py` with a race-level holdout and re-run
   `scripts/build_all.py --force`.
3. Check whether the deep network's advantage survives across sessions.

### [Priority: Low] Task 7 could use a sequence model

**Why:** Lap time is a sequence problem currently flattened into per-lap rows. An LSTM/GRU or temporal
transformer over stint history is the natural next step and would exploit structure the MLP cannot see.

**How (short):**
1. Add the architecture to `app/intelligence/dl/models.py` behind a new entry in `BUILDERS`.
2. Reuse the existing lap-forward splits — do not invent a new splitting scheme.
3. It will need windowed inputs; add the windowing to `dl/training.py` rather than to the Task 5 contract.

### [Priority: Low] Task 10 — Responsible AI evaluation not started

**Why:** Listed as planned. The relevant concern here is not fairness across people but honest reporting
of uncertainty — which the Model Quality entries show is a live problem, not a theoretical one.

**How (short):**
1. Add calibration analysis (reliability diagrams, Brier score) for the pit classifier.
2. Add prediction intervals for lap-time regression and surface them in the strategy simulator.

### [Priority: Low] Quantum optimisation layer not started

**Why:** The project is named for it. The classical A\* baseline in `app/intelligence/search/` is exactly
the comparison point a QAOA formulation of pit-stop scheduling would need, and Task 5 already keeps the
selected feature count small because each retained feature translates into circuit width — the groundwork
is deliberate.

**How (short):**
1. Formulate pit-strategy selection as a QUBO over the existing state space in
   `app/intelligence/search/problem.py`.
2. Add `app/intelligence/quantum/` with a Qiskit or PennyLane QAOA solver behind the same interface as
   the classical algorithms.
3. Extend `artifacts/search/reports/comparison_report.md` to include the quantum solver in the existing
   head-to-head table — the comparison harness already supports adding algorithms.
