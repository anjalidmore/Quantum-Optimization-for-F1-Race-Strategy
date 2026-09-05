# TODO — completed

Running record of every `TODO.md` entry closed during the four-phase remediation
effort. Lives on `proj-mode` alongside `TODO.md`, which is kept in sync: a closed entry
is struck through there and points here rather than being deleted, so the history of what
was wrong stays visible.

Every "Before → After" below is a measured number from an actual run, not an estimate.

---

# Phase 1 — Security & test-infrastructure foundation

*Completed 2026-09-05. Ten entries attempted; nine closed, one deferred with a decision
required (see the last section).*

## Baseline captured before any change

| Measure | Value |
|---|---|
| Tests | 197 passing |
| Warnings | 1,084 (5 distinct `SettingWithCopyWarning` sites) |
| Tracked files dirtied by one `pytest` run | **10** |
| `pip-audit` | 1 known vulnerability (msgpack 1.1.2) |
| `npm audit --audit-level=high` | 5 high-severity |
| CORS | `allow_origins=["*"]`, `allow_methods=["*"]`, `allow_headers=["*"]` |
| `/artifacts` mount | entire tree, including `models/` and `metadata/` |

---

### ✅ [Was: High] API allows every origin, method and header, with no authentication — closed in Phase 1

**What changed:** `app/api/main.py`. Replaced the three wildcards with an
environment-configured origin allow-list (`F1_ALLOWED_ORIGINS`, defaulting to
`http://localhost:3000,http://127.0.0.1:3000`), `allow_methods=["GET", "POST"]`, and
`allow_headers=["Content-Type"]`. The origin list is read at import so a deployment can
set its real origin without a code change.

**Before → After:**
- Any origin received `access-control-allow-origin` → only listed origins do.
  Measured: `https://evil.example.com` now gets `None`; `http://localhost:3000` gets
  itself.
- `allow_methods`/`allow_headers` were `*` → now `GET, POST` and `Content-Type`.

**Verified by:** 4 new tests in `tests/test_api.py`
(`test_cors_rejects_an_unlisted_origin`, `test_cors_allows_the_frontend_origin`,
`test_cors_does_not_advertise_wildcard_methods_or_headers`,
`test_allowed_origins_are_configurable_by_environment`), plus the full 217-test suite and
a frontend build to confirm the dashboard still reaches the API.

**Not done, deliberately:** no auth dependency was added. Nothing here is deployed, and an
unused auth layer is its own risk surface. The place it would go is a FastAPI dependency
on the routers in `app/api/routers/` — recorded here so the decision is explicit rather
than an oversight.

**Branch(es) touched:** proj-mode. *(No task-mode equivalent — that branch has no API.)*

---

### ✅ [Was: Medium] The whole artifacts tree is served unauthenticated as static files — closed in Phase 1

**What changed:** `app/api/main.py`. Instead of mounting `ARTIFACTS_DIR` at `/artifacts`,
each public subdirectory is mounted individually: `figures`, `reports`,
`data_engineering`, `knowledge_representation`, `expert_system`, `search`. `models/` and
`metadata/` have no mount at all, so the exclusion is **structural** — there is no route
to bypass, as opposed to a string filter that a crafted path might defeat.

**Before → After:**
- `GET /artifacts/models/laptime/decision_tree.joblib` → **200 (downloadable)** → **404**
- `GET /artifacts/models/dl/target_laptime.keras` → **200** → **404**
- `GET /artifacts/metadata/model_registry.json` → **200** → **404**
- Every path the dashboard reads (6 checked across 4 task directories) → **200 → 200**,
  unchanged.

**Verified by:** 4 new tests in `tests/test_api.py`, including
`test_private_dirs_are_excluded_structurally_not_by_a_filter`, which asserts that
`../` and percent-encoded traversal attempts against the mount boundary are refused.

**Branch(es) touched:** proj-mode. *(No task-mode equivalent.)*

---

### ✅ [Was: Medium] `run.sh` kills arbitrary processes on ports 8000 and 3000 — closed in Phase 1

**What changed:** `run.sh`. The script now prints the offending process's name and command
line, asks for confirmation, and **refuses outright in a non-interactive shell**. Added a
`--force-ports` flag for the old behaviour, made `BACKEND_PORT`/`FRONTEND_PORT`
environment-overridable so a collision can be sidestepped rather than resolved by killing,
and added `--help`.

**Before → After:**
- Unconditional `kill "$pid"` with no prompt → confirmation prompt, or exit 1 with a
  message naming the process and suggesting `BACKEND_PORT=8001 FRONTEND_PORT=3001`.

**Verified by:** occupied port 8000 with a real process, ran `./run.sh < /dev/null`, and
observed:
```
! Port 8000 is in use by pid 32131: /opt/homebrew/Cellar/python@3.14/...
Refusing to kill pid 32131 on port 8000 in a non-interactive shell.
Re-run with --force-ports, or set BACKEND_PORT/FRONTEND_PORT to free ports.
```
Also confirmed `--help` works and an unknown argument exits 2.

**Branch(es) touched:** proj-mode. *(No task-mode equivalent — no `run.sh` there.)*

---

### ✅ [Was: Low] `msgpack` 1.1.2 has a known advisory — closed in Phase 1

**What changed:** `requirements.txt` pins `msgpack>=1.2.1`.

**Before → After:** `pip-audit` reported **1 known vulnerability in 1 package**
(msgpack 1.1.2, PYSEC-2026-3625) → **"No known vulnerabilities found"**.

**Trade-off, recorded rather than hidden:** `signalrcore` 1.0.2 hard-pins
`msgpack==1.1.2`, so satisfying this constraint makes pip resolve **signalrcore down to
0.8.8**, which carries no such pin. This is a real downgrade and is not free. Verified
afterwards: `pip check` reports no broken requirements, `pip install -r requirements.txt`
exits 0 with no ERROR lines, and fastf1 3.8.3 still imports — including
`fastf1.livetiming.SignalRClient`, the only consumer of signalrcore. This project reads
historical sessions rather than live timing, so the older signalrcore is not on a code
path it exercises. Documented inline in `requirements.txt`; revisit if signalrcore
relaxes the pin.

**Verified by:** `pip-audit` before/after, `pip check`, and importing fastf1's live-timing
client explicitly.

**Branch(es) touched:** proj-mode. *(task-mode's per-practical `requirements.txt` files do
not depend on fastf1 — practical04 lists it only as an optional commented-out extra — so
msgpack never enters that branch's dependency set.)*

---

### ✅ [Was: Low] No secret-scanning or dependency-audit automation — closed in Phase 1

**What changed:**
- `.github/workflows/ci.yml` gains a `security` job: `pip-audit -r requirements.txt
  --strict`, `npm audit --audit-level=high`, and **gitleaks over the full history**
  (`fetch-depth: 0`).
- New `.pre-commit-config.yaml` with gitleaks, ruff, and `check-added-large-files`
  (5 MB cap).

**Before → After:** audits ran only when someone remembered → they run on every push and
PR, and the secret scan runs before every commit locally.

**One deliberate non-gate:** `npm audit` is `continue-on-error: true`. The only fix for
the current advisories is a major-version bump (see the deferred entry below), which is a
decision to take rather than something CI should force. The comment in the workflow says
to remove the flag once that upgrade lands — so this is a marked temporary state, not a
silently weakened check.

**Why `-r requirements.txt` rather than auditing the environment:** `pip-audit --strict`
against the installed set fails on this repo's own editable package, which is not on PyPI.
Auditing the declared file is both stricter about intent and correct here. Verified:
`pip-audit -r requirements.txt --strict` exits 0 with "No known vulnerabilities found".

**The `check-added-large-files` hook is not theoretical:** a previous session found 9,957
accidentally-committed `node_modules`/`.next` files including a 109 MB binary that GitHub
would have rejected outright. This is the cheap structural guard against a repeat.

**Verified by:** YAML validated and parsed (3 jobs), `pip-audit -r requirements.txt
--strict` run locally and clean.

**Branch(es) touched:** proj-mode. *(No task-mode equivalent — no CI on that branch.)*

---

### ✅ [Was: High] The test suite is not hermetic — it rewrites committed artifacts — closed in Phase 1

**The root cause of the registry-deletion entry below, so it was fixed first.**

**What changed:**
- `app/core/paths.py` gains `ArtifactPaths`: a frozen dataclass resolving every output
  location from one root, with `ArtifactPaths.default()` returning precisely the committed
  `artifacts/` layout.
- `app/intelligence/ml/pipeline.py`, `ml/reports.py`, `dl/pipeline.py` and
  `xai/pipeline.py` thread it through. `train_all()` / `run_all()` gain an optional
  `output_root` defaulting to the committed layout, so **production behaviour is
  unchanged** and the change is purely additive.
- `tests/test_ml_training.py` passes a `tmp_path_factory`-derived root instead of writing
  into tracked `artifacts/`.
- `_rel()` in `ml/pipeline.py` is now root-aware. It builds the manifest's repo-relative
  paths, which is undefined for a tmp root outside the repo; it falls back to a path
  relative to the artifact root, so the "no absolute paths in the manifest" invariant
  holds either way.

**Before → After:** one `pytest` run modified **10 tracked files**
(`artifacts/manifest.json`, `metadata/model_registry.json`, four `metrics/*.json`, four
`reports/*.md`) → **0**.

**Verified by** — this entry demanded demonstration, not assertion:
applied the change set to a **fresh clone**, built a clean venv, and ran the full suite
**twice in a row**. `git status --porcelain artifacts/ data/` was empty after both runs,
and a diff proved the only modified files were exactly the ones in the applied patch —
zero test-generated changes.

**Also fixed here (churn I introduced in the previous session):**
`dl.pipeline.restore_registry_entries()` stamped a fresh `trained_at` on every call, and
`build_all.py` runs it on the skip path — so a no-op rebuild dirtied the tracked registry.
It now preserves the existing timestamp for an already-registered target. Verified: a
repeat restore produces a zero-line diff.

**Branch(es) touched:** proj-mode. *(task-mode verified **already hermetic** — running all
six practical suites leaves 0 tracked files modified, because those tests only read
committed outputs behind `skipif` guards rather than regenerating them.)*

---

### ✅ [Was: High] A non-hermetic test run silently deleted Task 7's registry rows — closed in Phase 1

**What changed:** the underlying fix (`write_registry` preserving non-classical families,
and `restore_registry_entries()`) landed in the previous session; this phase **verified it
holds and added the regression test** the entry asked for, in
`tests/test_dl_training.py`.

`test_task6_retrain_preserves_task7_registry_entries` builds a registry containing one
classical and two deep entries, simulates a Task 6 retrain writing an entirely new
classical set, and asserts both deep rows survive while the stale classical one is
replaced. `test_restore_registry_entries_rebuilds_from_committed_artifacts` proves the
recovery path works without retraining.

**Before → After:** running the suite left `/api/dl/models` returning **404** while the
`.keras` files sat on disk → both deep entries survive a Task 6 retrain, and a future
whole-file writer cannot reintroduce the failure silently.

**Verified by:** the two new tests, plus confirming the deep entry count is still 2 after a
full suite run that retrains Task 6.

**Branch(es) touched:** proj-mode. *(No task-mode equivalent — no shared registry there;
each practical writes its own.)*

---

### ✅ [Was: Medium] Nothing tests the "no fabricated results" contract — closed in Phase 1

**What changed:** new `tests/test_artifact_contract.py`, 10 tests enforcing
`docs/PROJECT_REPORT.md` §8's central claim.

Both directions are covered, because only one of them is actually proof:
- **Positive** — write a sentinel into the artifact, assert the endpoint returns it. A
  *matching* value proves nothing (a hard-coded constant could coincide); it is the
  endpoint **following a change** that proves the read path is real. Covers
  `/api/ml/metrics`, `/api/ml/models`, `/api/dl/metrics`, `/api/xai/fairness`.
- **Negative** — remove the artifact, assert an explicit 404 whose message names
  `build_all`. A fabricated zero is worse than an error, because a reader cannot tell it
  from a real result.

**Before → After:** the contract was verified once by hand during the audit and enforced by
nothing → 10 automated tests, run on every commit.

**Verified by:** the tests themselves, plus two meta-tests
(`test_contract_tests_restore_the_artifact_byte_for_byte`,
`test_no_backup_files_are_left_behind`) confirming these mutation tests do not themselves
reintroduce the non-hermeticity this phase just removed. Confirmed `git status` is clean
after running them.

**Branch(es) touched:** proj-mode. *(No task-mode equivalent — no API to serve values.)*

---

### ✅ [Was: Low] Test suite emits 133 warnings, including 69 `SettingWithCopyWarning` — closed in Phase 1

**What changed:** `app/intelligence/data/pipeline.py` (and the same file in
`practicals/practical04/src/f1data/` on task-mode). `clean_table()` mutated a frame its
callers pass as a **slice** of a larger table, so every column assignment warned. The
function's contract is to *return* a cleaned frame, so it should never have written
through to the caller's data: one explicit `df.copy()` at the top fixes all of it.

**⚠ DEVIATION from the backlog's suggested fix, flagged per Rule 4:** the entry said to use
`.loc[:, col] = …`. **That is wrong here.** The `.loc` form assigns *in place* and
preserves the column's existing dtype, which silently turns the numeric, time and date
coercions into no-ops — it **broke three data-engineering tests** when tried
(`test_statistical_summary_has_moments`, `test_correlation_returns_pairs`, and one more).
Whole-column replacement is the correct operation; owning the frame is what removes the
warning. The reasoning is recorded inline in both copies so the next person does not
repeat the attempt.

**Before → After:**
- proj-mode: **1,084 warnings, 5 `SettingWithCopyWarning` sites** → **995 warnings, 0
  sites**; tests 197 → 217 passing (the increase is this phase's new tests).
- task-mode practical04: **114 warnings, 5 sites** → **25 warnings, 0 sites**; 17 tests
  passing throughout.

**Regression guard:** `pyproject.toml` (proj-mode) and `practicals/practical04/pytest.ini`
(task-mode) now turn `SettingWithCopyWarning` into an **error** under pytest. Verified the
guard actually fires by triggering the pattern deliberately.

**Branch(es) touched:** **both**.

---

## ⏸ Deferred — decision required

### [Priority: High] Frontend ships dependencies with high-severity advisories — NOT closed

**Why it is not closed:** the backlog said to "do the safer one first (patch bump)". I
checked, and **there is no patch bump to do**: `14.2.35` is already the latest `14.2.x`
release. `npm audit --json` reports that for all five advisories
(`next`, `postcss`, `eslint-config-next`, `@next/eslint-plugin-next`, `glob`) the only
`fixAvailable` is `next@16.3.4` / `eslint-config-next@16.3.4`, both flagged
`isSemVerMajor: true`.

The instruction was explicit that a major bump must be flagged before being taken, so this
entry is left open pending that decision. **The advisory count has also grown since the
backlog was written — it is now 5 high-severity, not 2.**

**Interim mitigation applied:** `npm audit --audit-level=high` now runs in CI as a
reporting (non-gating) step, so the count is visible on every PR rather than only when
someone remembers to check.

**What the decision costs:** Next.js 14 → 16 is two major versions. This project's five
pages are already App Router server components, which is the bulk of the usual migration
work, so the risk is moderate rather than severe — but it needs a real build-and-click
verification pass across all 8 routes, and `eslint-config-next` moving in lockstep will
likely surface new lint rules.

**Branch(es) touched:** none yet. *(No task-mode equivalent — no frontend there.)*

---

## Phase 1 exit state

| Measure | Before | After |
|---|---|---|
| Tests passing | 197 | **217** |
| Warnings | 1,084 | **995** |
| `SettingWithCopyWarning` sites | 5 | **0** |
| Tracked files dirtied by `pytest` | 10 | **0** |
| `pip-audit` | 1 vulnerability | **0** |
| `npm audit` high-severity | 5 | 5 *(deferred — see above)* |
| CORS origins allowed | all | 2 configured |
| Model weights downloadable over HTTP | yes | **no** |
| CI jobs | 2 | **3** (+ hermeticity gate, + secret scan) |

`ruff check app/ scripts/ tests/` — clean. Frontend build — 8 routes, 0 errors.
task-mode: 130 tests collected, 124 passing, 6 skipped (artifact-dependent tests behind
`skipif`, since `outputs/` is git-ignored).
