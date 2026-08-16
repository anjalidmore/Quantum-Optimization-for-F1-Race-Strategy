# Phase 1 · Task 4 — Data Engineering & EDA

Part of **Quantum Optimization for Formula 1 Race Strategy**.

> **Scope of this module:** classical data engineering and exploratory data
> analysis. No machine-learning modelling, no quantum computing. This task
> builds the clean, analysed data foundation that Phase 2 (ML/DL/QML) trains on.

---

## 1. What this module does

A complete, auditable data pipeline over the two project data sources:

**Load → Clean → Transform → Analyse → Visualise → Report**

| Stage | What happens |
|-------|--------------|
| **Load** | Read the CSVs, validate schema columns, convert the Ergast `\N` null sentinel to NaN |
| **Clean** | Remove duplicates; coerce dtypes; **parse `m:ss.mmm` lap times to seconds**; normalise inconsistent categorical spellings; impute missing values (median/mode by column kind); detect & optionally cap IQR outliers — with a full audit trail |
| **Transform** | Label / one-hot encode categoricals; standardise numeric features for downstream ML |
| **Analyse** | Statistical summaries (incl. skew/kurtosis), correlation analysis, a data-quality scorecard, and **nine domain analyses** |
| **Visualise** | Correlation heatmap, driver/constructor bars, lap-time distribution, pit-duration box, tyre-degradation trend, and a composite **dashboard** |
| **Report** | Five Markdown reports + cleaned & encoded CSVs |

The nine domain analyses are: driver, constructor, circuit, pit-stop, tyre,
lap-time, weather, season, and safety-car / track-status.

---

## 2. Data sources & the synthetic generator

The pipeline targets:

1. **Historical Kaggle dataset** — *Formula 1 World Championship (1950–2020)*
   (`rohanrao/formula-1-world-championship-1950-2020`), Ergast-derived CSVs.
2. **FastF1** — session laps, stints, tyre, weather and telemetry summaries.

`src/f1data/schemas.py` encodes the **real column names** of both sources, so the
loader works unchanged on a genuine download.

To keep this module **fully runnable without a multi-gigabyte download or live
network calls**, `src/f1data/synthetic.py` generates realistic,
internally-consistent sample data matching those exact schemas — and
deliberately injects the data-quality problems (missing values, duplicate rows,
outliers, inconsistent spellings, `\N` sentinels) that the cleaning stage is
built to fix. The tyre model even bakes in the correct pace-versus-durability
physics, so the EDA reveals SOFT < MEDIUM < HARD degradation.

### Running on the real data instead

1. Download the Kaggle CSVs and place them in `data/raw/` (keep the original
   filenames: `races.csv`, `drivers.csv`, `results.csv`, …).
2. Optionally `pip install fastf1` and export a session's laps to
   `data/raw/fastf1_laps.csv` with the columns in `FASTF1_LAPS`.
3. Run `python run_eda.py` **without** `--regenerate`. Because the schemas
   match, every stage runs identically on the real data.

---

## 3. Installation & execution

Requires **Python 3.10+**.

```bash
cd phase1_task4_data_engineering
pip install -r requirements.txt
python run_eda.py                 # generates synthetic data on first run
python run_eda.py --regenerate    # force fresh synthetic data
python run_eda.py --data-dir /path/to/real/csvs   # use the real dataset
```

Expected tail:

```
Cleaned results        rows 242 -> 240, quality 100.0/100
Cleaned pit_stops      rows 479 -> 475, quality 89.3/100
Cleaned lap_times      rows 13495 -> 13363, quality 100.0/100
Rendering figures ...
All Task-4 deliverables generated successfully in .../outputs
```

---

## 4. Deliverables (written to `outputs/`)

| Deliverable | Path |
|-------------|------|
| EDA report (9 domain analyses) | `outputs/reports/eda_report.md` |
| Correlation report | `outputs/reports/correlation_report.md` |
| Data-quality report | `outputs/reports/data_quality_report.md` |
| Cleaning audit | `outputs/reports/cleaning_audit.md` |
| Statistical summary | `outputs/reports/statistical_summary.md` |
| Visual analytics dashboard | `outputs/figures/dashboard.png` |
| Individual figures | `outputs/figures/*.png` |
| Cleaned datasets | `outputs/clean/*_clean.csv` |
| Encoded + scaled feature frame | `outputs/clean/fastf1_laps_encoded_scaled.csv` |

---

## 5. Tests

```bash
pytest -q
```

17 tests cover lap-time parsing (incl. null handling), synthetic generation &
schema conformance, null-sentinel conversion, primary-key validation, duplicate
removal + imputation, time-column parsing, outlier capping, the statistical /
correlation / quality analyses, and encoding + scaling (verifying scaled columns
have ~zero mean and that the tyre physics — SOFT degrades faster than HARD —
holds in the analysis).

---

## 6. Folder structure

```
phase1_task4_data_engineering/
├── run_eda.py                  # end-to-end driver
├── requirements.txt
├── pytest.ini
├── README.md
├── src/f1data/
│   ├── __init__.py
│   ├── schemas.py              # real Kaggle + FastF1 column schemas
│   ├── synthetic.py            # runnable sample-data generator
│   ├── pipeline.py             # load / clean / encode / scale + audit
│   ├── eda.py                  # statistics, correlation, quality, domain analyses
│   ├── visualize.py            # figures + dashboard
│   └── reports.py              # Markdown generators
├── tests/
│   └── test_data_engineering.py
├── data/raw/                   # generated (or real) CSVs
└── outputs/                    # generated on run
    ├── reports/*.md
    ├── figures/*.png
    └── clean/*.csv
```

---

## 7. Relationship to other tasks

* **Completes classical Phase 1.** Together with Knowledge Representation
  (Task 1), the Expert System (Task 2) and State-Space Search (Task 3), this
  task delivers the full classical foundation.
* **Feeds Phase 2 directly.** The cleaned, encoded, scaled feature frames and the
  correlation / feature statistics produced here are the inputs to Feature
  Engineering (Task 5) and the ML / DL / QML models (Tasks 6–8).

---

## 8. Phase 1 is now complete

All four classical tasks are implemented, documented, tested and validated:
knowledge representation, the rule-based expert system, state-space search, and
this data-engineering & EDA foundation. Phase 2 introduces machine learning,
deep learning, explainable AI and — for the first time — quantum machine
learning.
