# TODO — gap analysis

Working backlog for the F1 Race Strategy platform. **This file lives on `proj-mode` only** and is
deliberately kept off `main`: it is useful to whoever is doing the work and noise to someone evaluating
the result.

Every entry below comes from something observed while auditing this repository on **2026-09-05** —
running the documented setup in a clean clone, rebuilding all artifacts, executing the full test suite,
building the frontend, exercising the live API, and scanning dependencies and git history. Nothing here
is speculative.

**Priority counts:** 28 entries — 7 High · 10 Medium · 11 Low

---

## Model Quality

### [Priority: High] Pit-decision classifier is unusable at its operating point

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

### [Priority: High] Selected regression model has negative test R²

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

### [Priority: Medium] Evaluation rests on a single race and a ~180-row holdout

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

### [Priority: High] API allows every origin, method and header, with no authentication

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

### [Priority: High] Frontend ships dependencies with 2 high-severity advisories

**Why:** `npm audit` on `frontend/` reports 2 high-severity vulnerabilities: Next.js 14.2.35 carries 11
advisories (SSRF via rewrites and Server Actions, cache poisoning of RSC responses, cache confusion,
middleware bypass, DoS, unauthenticated disclosure of internal Server Function endpoints), and its bundled
postcss has XSS and arbitrary-file-read issues. The SSRF and disclosure advisories matter as soon as this
is hosted anywhere.

**How (short):**
1. `cd frontend && npm audit` to review, then upgrade `next` in `frontend/package.json` — the clean fix is
   `next@16`, which is a breaking change, so budget for the App Router migration.
2. If the major bump has to wait, move to the latest 14.2.x patch and re-audit.
3. Re-run `npm run build` and check all 6 routes still compile.

### [Priority: Medium] The whole artifacts tree is served unauthenticated as static files

**Why:** `app/api/main.py:43` mounts `artifacts/` at `/artifacts` with no filtering. Today that directory
holds only figures and reports, but it also holds `*.joblib` model files — anyone who can reach the API
can download the trained models. Combined with the CORS wildcard, this is exfiltration by default.

**How (short):**
1. Mount only the subdirectories the frontend actually reads (`figures/`, `reports/`, and the per-task
   report dirs) rather than the tree root.
2. Explicitly exclude `artifacts/models/` and `artifacts/metadata/` from static serving.
3. Add a test in `tests/test_api.py` asserting `GET /artifacts/models/laptime/decision_tree.joblib` 404s.

### [Priority: Medium] `run.sh` kills arbitrary processes on ports 8000 and 3000

**Why:** `run.sh:54` issues `kill "$pid"` against whatever is listening on those ports, with no prompt and
no check that the process belongs to this project. A contributor running an unrelated dev server on :3000
loses it, silently, with unsaved state.

**How (short):**
1. In `run.sh`, print the process name and pid and require confirmation before killing, or add a
   `--force-ports` flag and refuse by default.
2. Alternatively read `BACKEND_PORT`/`FRONTEND_PORT` from the environment so a contributor can move off
   the conflict instead of killing it.

### [Priority: Low] `msgpack` 1.1.2 has a known advisory

**Why:** `pip-audit` reports PYSEC-2026-3625 against msgpack 1.1.2, fixed in 1.2.1. It arrives
transitively via `fastf1` → `signalrcore` and is only exercised when fetching live sessions, so exposure
is small — but it is the only flagged Python dependency and the fix is trivial.

**How (short):**
1. Add `msgpack>=1.2.1` to `requirements.txt`.
2. Re-run `pip-audit` to confirm a clean report.

### [Priority: Low] No secret-scanning or dependency-audit automation

**Why:** This audit found no secrets in the working tree or in the full git history across all branches,
and no `.env` or credential file was ever committed — a genuinely clean result. But nothing prevents the
next commit from introducing one, and `pip-audit`/`npm audit` are run only when someone remembers.

**How (short):**
1. Add `pip-audit` and `npm audit --audit-level=high` steps to the CI workflow (see the Testing section).
2. Add a `gitleaks` or `detect-secrets` pre-commit hook.

---

## Testing

### [Priority: High] The test suite is not hermetic — it rewrites committed artifacts

**Why:** Running `pytest` retrains models and overwrites 10 tracked files under `artifacts/`
(`metrics/*.json`, `reports/*.md`, `manifest.json`, `metadata/model_registry.json`). A clean clone goes
dirty just from running the documented test command. This trains contributors to `git checkout --
artifacts/` reflexively, which is exactly how a real artifact change gets discarded by accident.

**How (short):**
1. Give the ML tests a `tmp_path` output root — parameterise the writer functions in
   `app/intelligence/ml/persistence.py` and `reports.py` on an output directory, and have
   `tests/test_ml_training.py` pass a fixture-scoped temp dir.
2. Add a CI step that fails if `git status --porcelain` is non-empty after `pytest`.

### [Priority: High] No continuous integration exists

**Why:** There is no `.github/` directory on any branch. The 120 tests, the `build_all.py --force`
reproducibility check and the frontend build all pass — but only because they were run by hand. Nothing
stops a commit that breaks them, and the README's "120 passing" badge is static and will silently go
stale.

**How (short):**
1. Add `.github/workflows/ci.yml` running, on push and PR: `pip install -r requirements.txt && pip install
   -e .`, `pytest`, `python scripts/build_all.py --force`, and `cd frontend && npm ci && npm run build`.
2. Add the artifact-cleanliness check from the entry above.
3. Replace the static badge in `README.md` with the real workflow status badge.

### [Priority: Medium] Zero frontend tests

**Why:** 5 pages and 9 components in `frontend/` have no test of any kind — no unit, component or e2e
coverage. `frontend/lib/api.ts` is the single boundary every page depends on, and a change to an API
response shape would surface only as a blank page at runtime.

**How (short):**
1. Add Vitest + React Testing Library; start with `frontend/lib/api.ts` response parsing against fixtures
   captured from the live API.
2. Add a Playwright smoke test that boots the API, loads all 5 routes and asserts no error boundary
   renders.

### [Priority: Medium] Nothing tests the "no fabricated results" contract

**Why:** The contract in `docs/PROJECT_REPORT.md` §8 — that every displayed value traces to a generated
artifact — is the project's central claim. It was verified manually in this audit by mutating
`artifacts/metrics/regression_metrics.json` and confirming the API followed. Nothing enforces it, so a
future hard-coded fallback would pass CI.

**How (short):**
1. Add `tests/test_artifact_contract.py`: write a known sentinel value into a temp copy of the metrics
   artifact, point `app/core/paths.py` at it, and assert `/api/ml/metrics` returns the sentinel.
2. Assert the reverse too — that deleting the artifact produces an explicit error rather than a default.

### [Priority: Low] Test suite emits 133 warnings, including 69 pandas `SettingWithCopyWarning`

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

### [Priority: Medium] scikit-learn deprecation will break Task 6 at version 1.11

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

### [Priority: Low] Task 7 — Deep Learning not started

**Why:** Listed as planned in the status table. Sequence models (LSTM/GRU or a temporal transformer over
stint history) are the natural next step for lap-time prediction and would directly address the weak
generalisation flagged in Model Quality.

**How (short):**
1. Add `app/intelligence/deep_learning/` mirroring the `ml/` package interface so the registry, API and
   dashboard pick it up unchanged.
2. Reuse the existing lap-forward splits from `app/intelligence/ml/splits.py` — do not invent a new
   splitting scheme.
3. Register the models in `artifacts/metadata/model_registry.json` alongside the classical ones.

### [Priority: Low] Task 8 — Explainable AI not started

**Why:** The expert system already explains itself with HOW/WHY traces; the ML models are opaque. A
strategy recommendation that mixes both currently has an auditable symbolic half and an unauditable
statistical half.

**How (short):**
1. Add SHAP explanations for the selected regressor and classifier in a new
   `app/intelligence/xai/` package.
2. Extend `POST /api/strategy/predict` to return per-feature attributions next to
   `triggered_expert_rules`, so both halves of the recommendation are explained the same way.

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
