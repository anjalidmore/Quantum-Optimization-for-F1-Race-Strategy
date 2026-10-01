# User manual

For the person using the dashboard — a race engineer or strategist reviewing what the system
concluded about one race — not for the person building it. If you are looking for install
instructions, module layout, or the API contract, see [`deployment_guide.md`](deployment_guide.md),
[`architecture.md`](architecture.md) and [`srs.md`](srs.md) instead.

## Contents

- [What this system is — and is not](#what-this-system-is--and-is-not)
- [Starting it](#starting-it)
- [The six pages](#the-six-pages)
- [Using the Race Strategy Simulator](#using-the-race-strategy-simulator)
- [Reading a downloaded strategy report](#reading-a-downloaded-strategy-report)
- [What "Not available" means when you see it](#what-not-available-means-when-you-see-it)
- [The single-race limitation](#the-single-race-limitation)
- [Glossary](#glossary)

## What this system is — and is not

This is a **decision-support** tool for one real Formula 1 session, the 2023 Bahrain Grand Prix.
It predicts lap time and pit-stop probability from a described race state, shows which symbolic
rules and search plan agree or disagree with that prediction, and explains *why* a prediction came
out the way it did. It does not drive a car, does not talk to a team's timing system, and does not
make a decision for you.

The system's own report generator states this outright, in the last section of every report it
produces (`app/services/strategy_report.py`):

> Decision support for engineers, not an autonomous strategy system, and not for betting.

Two more limits worth knowing before you rely on anything the dashboard shows:

- **Trained on one race.** Every number — predictions, feature rankings, trust scores — reflects
  the 2023 Bahrain GP and should not be read as a general statement about Formula 1 strategy. See
  [The single-race limitation](#the-single-race-limitation) below.
- **The trust score does not predict error.** Task 8's trust score (shown on the Explainability
  page and in every strategy report) is the project's own heuristic for how much supporting
  evidence a prediction has — not a validated probability of being correct. Checked against actual
  error on the test laps, its correlation was −0.09 (not significant). Read its components, never
  the number alone, as a reason to look closer.

## Starting it

From the repository root:

```bash
./run.sh
```

This builds whatever artifacts are missing (Tasks 1–8 plus the quantum models), starts the backend
API and the dashboard, warms the API with a few real predictions so the first page you open shows
numbers instead of a cold error, and opens your browser. Default addresses:

- Dashboard: `http://localhost:3000`
- API: `http://localhost:8000`

Useful flags: `./run.sh --force-retrain` rebuilds every stage first; `./run.sh --force-ports` frees
`:8000`/`:3000` if something else is holding them, without asking. `BACKEND_PORT=8001
FRONTEND_PORT=3001 ./run.sh` moves the whole thing to different ports instead. `./run.sh --help`
prints all of this from the terminal.

If the dashboard loads but shows "Backend unreachable," the API process either isn't running or the
dashboard is pointed at the wrong URL — see the [deployment guide](deployment_guide.md#7-health-checks)
for the checks to run.

## The six pages

The dashboard's navigation bar (`frontend/components/Nav.tsx`) has six pages, in this order:

### Overview (`/`)

The front page. One large figure dominates it: the A\* search's optimal remaining-race cost in
seconds — "the one number this whole platform exists to produce," as the page's own code comment
puts it. Beside it, a task ledger shows how many of the ten lab tasks have real generated output
behind them. Below that: lap-time model error (MAE), how many expert rules are available, how many
trained models are registered, and a table of lap-time error broken down by driver (read from
Task 8's per-driver stratification). If a component's artifact doesn't exist yet, that tile reads
"Not generated yet" with the command that would produce it — never a blank or a zero standing in
for a real number.

### Strategy (`/strategy`)

The Race Strategy Simulator — described in detail in its own section below. This is the page you
use to ask "what would the models say about this specific race situation?"

### Reasoning (`/reasoning`)

The symbolic half of the project — Tasks 1 through 3, which don't sit in the data-training
pipeline and so had no page of their own until Task 9 required every task to be visible:

- **Knowledge** — the F1 ontology and knowledge graph: how many entities, relationships and
  attributes were defined, grouped by category, with the ontology and graph files' existence
  confirmed live.
- **Expert system** — the 32-rule forward-chaining rule base: rule counts by category, salience
  levels, and every rule's id, name, category and the conditions/actions it involves.
- **Search** — the five search algorithms (BFS, DFS, UCS, Greedy, A\*) run on one race-strategy
  problem instance, with the resulting pit-stop plan and its total cost.

Two of these three — the expert system and the search planner — are not just displayed here; they
also run live, alongside the ML models, every time you use the Strategy page.

### Models (`/models`)

Tasks 6 and 7 side by side: the same two prediction problems (lap time, pit decision), solved once
by classical models (logistic/linear regression, decision tree, random forest, SVM/SVR, XGBoost)
and once by Keras neural networks, on the same folds and the same untouched holdout laps — so any
difference you see on this page is the model, not a different test set. It shows model-comparison
tables, ROC/PR curves, confusion matrices, feature importance, the networks' architectures and
training curves, and — when the quantum stage has been run — the simulated quantum models measured
against parameter-matched classical baselines on the same split.

### Explainability (`/explainability`)

Task 8. This page explains the trained Task 7 neural network on the chronological test laps: which
race-state factors moved each prediction (SHAP and LIME, compared against each other), what would
have to change to flip a decision (a tyre-age counterfactual), how much to trust the prediction (the
trust score and its four components), and whether the model performs evenly across drivers, teams
and tyre compounds. A **lap inspector** lets you pick any of the 180 held-out test laps and see its
full explanation; its tyre-age counterfactual is computed live by the saved network rather than read
from a stored file, everything else on this page is read from the committed Task 8 artifacts.

### Data (`/data`)

Task 4's cleaning and EDA figures and reports, followed by the same evidence index shown on the
Overview page's task ledger, but expanded per task: every artifact this project has actually
generated, scanned live from `artifacts/` at request time. Nothing here is a hard-coded filename —
if a task's output directory has nothing in it, that task's section says so.

## Using the Race Strategy Simulator

The Strategy page (`/strategy`) has two tabs: **Full race scenario** (described below) and **Top
features**, a simplified demo that lets you see the ranked features behind whichever model is
currently selected-best, without filling in a full race state.

### Inputs

The full-scenario form is grouped the way an engineer would change it (`RaceStateForm.tsx`):

| Group | Fields |
|---|---|
| Car | Driver, Team (real dropdowns sourced from `GET /api/data/options`, not free text), Fuel on board (kg) |
| Tyre | Compound (SOFT/MEDIUM/HARD/INTERMEDIATE/WET), Tyre age (laps on the current set) |
| Track | Temperature (°C), Weather (dry/damp/wet/extreme), Flag (GREEN/YELLOW/SC/VSC/RED) |
| Race | Current lap, Total laps, Position |
| Model | Best performing (default) or choose a specific trained model for lap time and pit decision separately |

Driver, team and tyre compound are always real dropdown values drawn from the dataset, never
free-text fields — this was a deliberate fix: an earlier version accepted arbitrary text for
driver/team, which usually didn't match any value the models were trained on and so silently had no
effect on the prediction (see [`README.md`](../README.md#what-we-learned)). The form also keeps
itself physically consistent as you edit it: tyre age cannot exceed the current lap, and the current
lap cannot exceed the total.

**Scenario presets** (Normal race, High degradation, Late-race pit call, Safety car, Fresh tyres)
fill the form from the dataset's own real ranges — for example "High degradation" uses the hottest
track temperature this actual session recorded, not a hard-coded guess that might ask the model to
extrapolate wildly beyond anything it ever saw.

### Reading the output

After you click **Predict**, the response is a single POST to `/api/strategy/predict`, which chains
every computational-intelligence component this platform has and returns the result as a set of
cards — one per pipeline stage:

- **Recommendation** (the focal card) — the combined verdict: **Pit now**, **Stay out**, or **Pit in
  N laps**. This is `app/services/strategy_service.py::_combine_recommendation`'s output: a Task 2
  expert-system rule wins outright if one fired for this race state; otherwise the Task 6 (ML) and
  Task 7 (DL) pit-probability predictions are compared. The card shows a **confidence** badge (high /
  moderate / low / none) and, when ML and DL disagree, says so explicitly rather than silently
  picking one — the card states which model it fell back to and why. Below the headline, a stat grid
  shows lap time and pit probability from **both** model families side by side (ML and DL), the A\*
  search's expected remaining-stint cost, and the expert-rule count.
- **1 · Input validation** — confirms the lap/compound/position inputs passed the schema and
  domain-option checks (a 422 would have stopped the request before this point otherwise) and shows
  laps remaining.
- **2 · Feature construction** — an expandable panel showing the exact feature row each model
  received, and which of those features could not be derived from your snapshot and were filled from
  the training data's median instead (features that need multi-lap history a single form snapshot
  can't supply, like a rolling gap or field-pace trend).
- **3 · Expert system** — the full list of Task 2 rule ids and names that fired for this race state.
- **6 · Explainability (Task 8)** — per target, a plain-English narrative of why the **Task 7**
  network predicted what it did, a trust score and band, and the top SHAP factors. Shown automatically
  (the simulator always requests `explain: true`); reads "Not available" with a reason if Task 8's
  artifacts or the DL model aren't present.

Two honesty notes appear when they apply, not as errors but as stated engineering facts:

- **"Extrapolating beyond training data"** — appears when a feature built from your inputs falls
  outside the range the model actually saw during training (most often an unusual track
  temperature). The card lists exactly which feature and what range the model was trained on. A
  prediction built this way is unvalidated, not necessarily wrong.
- **"Show the features sent to each model"** — an expandable panel showing the exact feature row
  each model received, and which of those features could not be derived from your snapshot and were
  filled from the training data's median instead (features that need multi-lap history a single
  form snapshot can't supply, like a rolling gap or field-pace trend).

If driver or team has no effect on the currently selected models (which happens when the trained
model didn't select a one-hot dummy for that particular value), the card says so plainly: "Recorded
as context, not used by these models."

### Downloading a report

**Download strategy report** posts the same race state to `POST /api/strategy/report`, always with
explanations included (a report without them would hide why the call was made), and saves a
self-contained file — **Markdown** or **HTML**, chosen from the dropdown beside the button (the two
are the same content; `render_html` converts `render_markdown`'s output rather than re-deriving it,
so they cannot say different things). It includes the race state, the recommendation (ML and DL
predictions side by side, with a disagreement note when they differ), the expert rules that fired and
what they concluded, the search plan, a full Task 8 explanation of the Task 7 neural network's
prediction on the same state (SHAP factors, trust score and its components, a plain-English
narrative), and a provenance table naming exactly which module produced each section. The report's
own filename encodes the driver, lap and timestamp, e.g. `strategy_VER_lap30_20260927-1412.md` (or
`.html`).

**Read the model names in a downloaded report carefully.** Section 2 ("Recommendation") now shows
**both** the Task 6 classical model's and the Task 7 network's predictions explicitly, labelled. The
Section 5 explanation is specifically of the **Task 7 neural network**, because that is the model
Task 8 explains. How much the two predictions differ is exactly the `model_agreement` term inside the
trust score — and, separately, feeds the recommendation engine's own disagreement flag in Section 2.

## What "Not available" means when it appears

Every page on this dashboard reads its numbers from a generated artifact at request time — nothing
is hard-coded into the frontend. When a page, card or field shows **"Not available"**, **"Not
generated yet"**, or **"undefined"**, it means exactly one thing: the underlying pipeline stage that
would produce that number has not been run in this environment, so there is nothing on disk to
read. It is stated as a design principle throughout the project (`docs/architecture.md`): *"If a
model hasn't been trained, the UI shows 'No trained model available. Run the training pipeline to
generate results' — never an invented number."*

This is never a bug masquerading as a message. The fix is always to run the named command (usually
`python scripts/build_all.py`, sometimes a single stage like `python scripts/run_search.py`), then
reload the page.

A related but distinct case: pit-decision metrics that show as `undefined` rather than a number.
This happens when a chronological holdout split genuinely contains zero pit events for one model —
ROC-AUC and PR-AUC are undefined with no positive examples to score against, and the dashboard says
so rather than printing a misleading `0.000`.

## The single-race limitation

Every prediction, ranking, explanation and trust score on this dashboard comes from one real
session: the **2023 Bahrain Grand Prix (Race)**, 1,055 laps across 20 drivers, 995 of them usable
after the warm-up trim (`data/processed/feature_metadata.json`). Stated plainly, because it changes
how every number on this dashboard should be read:

- A model trained on one race learns that race's specific pattern of tyre degradation, pit timing
  and field pace — not a general theory of Formula 1 strategy. Applying its numbers to a different
  circuit, weather pattern or competitive order is extrapolation the project has not validated.
- The pit-decision holdout (the 180 most recent laps) contains **exactly one** labelled real pit
  stop. Precision, recall and F1 computed on one event are close to noise — the dashboard and every
  report lean on cross-validated figures instead and say so wherever the number appears.
- There is a second, unlabelled pit stop in the same holdout: FastF1 had no stint data for one
  driver (NOR), so Task 4's cleaning filled it with the median rather than the real value, meaning
  that driver's real stop on lap 47 is recorded as "no pit." Both models rank that lap low, so no
  conclusion in the project changes because of it — but it is a data gap, not a model failure, and
  it is documented rather than hidden.

If you want to see the system trained on a different session, see
[`deployment_guide.md`](deployment_guide.md#6-training-on-a-different-race) — `run.sh` and the
dashboard need no code changes, only a re-fetch and a rebuild.

## Glossary

Terms that appear on the dashboard without much room to define themselves:

| Term | Meaning here |
|---|---|
| MAE | Mean absolute error, in seconds, for lap-time predictions. Lower is better. |
| R² | Fraction of lap-time variance the model explains. Can be negative on a hard holdout. |
| PR-AUC | Area under the precision-recall curve for the pit-decision classifier — preferred over ROC-AUC here because pit events are only 4.8% of laps, where ROC-AUC stays high even for a model that rarely fires. |
| SHAP | Shapley-value feature attribution — how much each input pushed one specific prediction up or down. |
| LIME | A local linear surrogate fitted around one prediction — a second, different way of asking what mattered locally; the dashboard's own analysis found it unreliable on this dataset for one specific feature (`track_status`, constant across all test laps), and reports that finding rather than hiding it. |
| Trust score | A project-defined 0–1 score combining confidence, agreement between the two model families, SHAP/LIME explanation stability, and whether the inputs are inside the training range. Not a validated probability of correctness. |
| Counterfactual | "What tyre age would change this call?" — a scan over the training range that shows how far the input would have to move before the prediction flips. |
| Data source badge | Confirms every number on the page came from the real 2023 Bahrain FastF1 session, not a synthetic demo dataset the codebase can also run on. |
