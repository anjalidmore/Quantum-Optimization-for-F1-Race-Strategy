# F1 Race Strategy — Practicals (`task-mode`)

Individual computational-intelligence practical submissions for the
**Quantum Optimization for Formula 1 Race Strategy** project.

This branch is the **per-practical archive**. Each practical is a
self-contained, independently runnable submission with its own source package,
tests, dependency list and README. If you want the unified production
application that these were later merged into — one `app/` package, a FastAPI
backend and a Next.js dashboard — see the **`main`** branch.

| | `task-mode` (this branch) | `main` |
|---|---|---|
| Shape | 5 standalone practicals | 1 unified application |
| Run | one script per practical | `./run.sh` |
| Audience | marking each task in isolation | using / reviewing the product |

---

## The practicals

| # | Topic | Entry point | Tests |
|---|---|---|---|
| 01 | Knowledge Representation — OWL 2 ontology + knowledge graph over 61 F1 entities and 29 relationships | `practicals/practical01/practical01.py` | 15 |
| 02 | Rule-Based Expert System — 32 curated production rules, forward + backward chaining, HOW/WHY explanations | `practicals/practical02/practical02.py` | 21 |
| 03 | State-Space Search — pit strategy as shortest path, solved with BFS, DFS, UCS, Greedy and A\* | `practicals/practical03/practical03.py` | 18 |
| 04 | Data Engineering & EDA — load → clean → transform → analyse → visualise → report, with a full cleaning audit | `practicals/practical04/practical04.py` | 17 |
| 05 | Feature Engineering & Selection — a four-stage selection funnel producing the modelling matrix | `practicals/practical05/practical05.ipynb` | — (notebook) |

**71 tests across practicals 01–04, all passing.**

Read [FLOW.md](FLOW.md) for how they build on each other, and
[SHOWCASE.md](SHOWCASE.md) for what to look at first.

---

## Layout

```text
practicals/
├── practical01/           # Knowledge Representation
│   ├── practical01.py     #   entry point — regenerates every deliverable
│   ├── README.md          #   full task write-up
│   ├── requirements.txt   #   dependencies for THIS practical only
│   ├── pytest.ini
│   ├── src/f1kr/          #   implementation package
│   ├── tests/
│   └── outputs/           #   generated (git-ignored)
├── practical02/           # Expert System        → src/f1es/
├── practical03/           # State-Space Search   → src/f1search/
├── practical04/           # Data Engineering     → src/f1data/
│   ├── data/raw/          #   committed input CSVs
│   └── outputs/clean/fastf1_laps_clean.csv   ← committed, feeds practical05
└── practical05/           # Feature Engineering
    ├── practical05.ipynb
    └── outputs/           #   committed — the Task 5 contract
```

**Why per-practical folders rather than flat `practical01.py` files at the
root:** each practical is not a single script. Practicals 01–04 each ship a
source package, a test suite, and their own pinned dependency list, and each
writes into its own `outputs/` tree. Flattening them would collide four
different `src/`, `tests/` and `outputs/` directories and destroy the ability to
run, test and mark any one practical in isolation — which is the entire point of
this branch. The folders keep that isolation; the sequential `practicalNN`
naming gives the consistency.

**Ordering:** by the task number the practicals declare for themselves.
Practical 01's README states "Phase 1 · Task 1", practical 02 "Task 2", and so
on through practical 05 "Phase 2 · Task 5". This ordering is unambiguous, it
matches the chronological commit order, and it matches the dependency order —
practical 05 consumes practical 04's output, and nothing depends on anything
later than itself.

**Not renamed, deliberately:** the `src/` packages (`f1kr`, `f1es`, `f1search`,
`f1data`) and the test files (`test_expert_system.py`, …) keep their descriptive
names. They are implementation, not submissions, and `test_practical02.py` would
tell a reader strictly less than `test_expert_system.py` does.

---

## Prerequisites

* **Python 3.10+** (verified on 3.14.6)
* No network access required — practical 04 generates realistic synthetic data
  matching the real Kaggle and FastF1 schemas, so every practical runs offline.

---

## Running a practical

Each practical is independent. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

cd practicals/practical01            # or 02, 03, 04
pip install -r requirements.txt
python practical01.py                # writes everything into ./outputs/
pytest -q                            # run this practical's tests
```

Each entry point accepts `--output-dir`; practical 03 also takes `--laps N`, and
practical 04 takes `--data-dir` (to point at real CSVs instead of synthetic) and
`--regenerate`. Run any of them with `--help` for the full list.

### Practical 05 (notebook)

Practical 05 reads practical 04's cleaned dataset, so **run practical 04
first** — or just let the notebook do it: if
`practical04/outputs/clean/fastf1_laps_clean.csv` is missing, cell 4 invokes
`practical04.py` for you.

```bash
source .venv/bin/activate
pip install -r practicals/practical04/requirements.txt   # p05 adds no new deps
cd practicals/practical05
jupyter lab practical05.ipynb        # then "Run All"
```

### Running everything at once

```bash
source .venv/bin/activate
for n in 01 02 03 04; do
  ( cd practicals/practical$n && pip install -q -r requirements.txt && \
    python practical$n.py && pytest -p no:warnings -q )
done
```

---

## Generated outputs

Every `outputs/` directory is reproducible from its entry point and is
git-ignored, with two committed exceptions that are **inputs to later work**:

* `practicals/practical04/outputs/clean/fastf1_laps_clean.csv` — consumed by practical 05
* `practicals/practical05/outputs/` — the feature matrix and metadata contract consumed downstream

---

## Verification status

Everything below was executed on this branch, not assumed:

| Practical | `pytest` | Entry point |
|---|---|---|
| 01 | 15 passed | exit 0 — all Task-1 deliverables written |
| 02 | 21 passed | exit 0 — all Task-2 deliverables written |
| 03 | 18 passed | exit 0 — optimality invariant holds (A\* cost == UCS cost == 2262.42 s) |
| 04 | 17 passed | exit 0 — all Task-4 deliverables written |
| 05 | — | notebook executed end to end; 550 laps × 20 cols → 520 × 19 feature matrix |

Practical 04's cleaned CSV and practical 05's feature matrix both regenerated
**byte-identical** to the committed copies. Practical 05's recorded validation
scores do drift slightly with the scikit-learn version — see
[FLOW.md](FLOW.md#a-note-on-reproducibility).

---

## Licence

MIT — see [LICENSE](LICENSE).
