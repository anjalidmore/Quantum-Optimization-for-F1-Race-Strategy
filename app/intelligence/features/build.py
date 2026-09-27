"""
app.intelligence.features.build
===============================

Task 5 — feature engineering and feature selection. Turns Task 4's cleaned lap
table into the two files every later task reads as its contract:

    data/processed/f1_features_selected.csv     rows: identifiers + features + targets
    data/processed/feature_metadata.json        which features, why, and how to use them

This used to live only in ``docs/notebooks/task5_feature_engineering.ipynb``, so
``scripts/build_all.py`` could check the contract but never rebuild it. The logic
now lives here, the notebook imports it as a walkthrough, and
``scripts/build_features.py`` runs it as a pipeline stage.

Two rules hold throughout, because every downstream task depends on them:

* **No leakage.** Sector times sum to the lap time, and speed traps and
  ``IsPersonalBest`` are only knowable once the lap is over. They are excluded
  before any feature is built, and the export asserts they never reappear.
* **No scaling here.** Scaling belongs inside each cross-validation fold, which
  is Task 6's job. Writing scaled features to disk would leak the full dataset's
  statistics into every later fold.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from app.core.paths import (
    DATA_PROCESSED_DIR,
    FASTF1_LAPS_CLEAN_CSV,
    REPO_ROOT,
    TASK5_FEATURE_METADATA_JSON,
    TASK5_FEATURES_CSV,
)

RANDOM_STATE = 42

# Known only during or after the lap being predicted.
LEAKAGE_COLS = ["Sector1Time", "Sector2Time", "Sector3Time",
                "SpeedFL", "SpeedST", "IsPersonalBest"]
TARGETS = ["target_laptime", "target_pit_next_lap", "target_laptime_fuel_corrected"]
ID_COLS = ["Driver", "LapNumber", "Stint", "Compound", "Team"]

# Features that need earlier laps to exist; their first rows are dropped.
HISTORY_COLS = ["gap_roll3_mean", "gap_roll3_std", "form_vs_baseline",
                "field_median_lag1", "field_pace_trend"]

REFERENCE_COMPOUND = "HARD"   # baseline level for the compound contrasts
CORRELATION_CUT = 0.95
VIF_THRESHOLD = 10.0
TOP_K_STABILITY = 12          # how many top features each fold votes on
K_CLASSIFICATION = 8          # pit-decision feature count (few positives; keep it small)


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def load_clean_laps(clean_csv: Path = FASTF1_LAPS_CLEAN_CSV) -> pd.DataFrame:
    """Task 4's cleaned laps as a driver-by-lap panel, sorted and complete."""
    if not Path(clean_csv).exists():
        raise FileNotFoundError(
            f"{clean_csv} not found. Run Task 4 first: python scripts/run_eda.py"
        )
    laps = pd.read_csv(clean_csv).sort_values(["Driver", "LapNumber"]).reset_index(drop=True)
    assert not laps.duplicated(["Driver", "LapNumber"]).any(), "Driver/LapNumber is not unique"
    assert laps.notna().all().all(), "Task 4 should have delivered a complete frame"
    return laps


def leakage_evidence(laps: pd.DataFrame) -> dict:
    """Measured proof that the sector times are leakage, not features."""
    sector_sum = laps[["Sector1Time", "Sector2Time", "Sector3Time"]].sum(axis=1)
    residual = (laps["LapTime"] - sector_sum).abs()
    return {"max_residual_s": float(residual.max()), "median_residual_s": float(residual.median())}


# ---------------------------------------------------------------------------
# 2. Targets
# ---------------------------------------------------------------------------
def add_targets(laps: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Add the three targets. Returns the frame and the fitted fuel coefficient.

    ``target_pit_next_lap`` is derived from the stint index increasing on the
    next lap, so it marks the lap *before* the stop — the decision point.
    """
    laps = laps.copy()
    grp = laps.groupby("Driver", sort=False)

    laps["target_laptime"] = laps["LapTime"].astype(float)

    next_stint = grp["Stint"].shift(-1)
    laps["target_pit_next_lap"] = ((next_stint > laps["Stint"]) & next_stint.notna()).astype(int)

    # Fuel burn-off makes every car quicker as the race goes on. Removing that
    # trend leaves a target that isolates tyre degradation.
    design = pd.concat(
        [laps[["LapNumber", "TyreLife"]].astype(float),
         pd.get_dummies(laps["Compound"], prefix="cmp", dtype=float)],
        axis=1,
    )
    fuel_coef = float(LinearRegression().fit(design, laps["target_laptime"]).coef_[0])
    laps["target_laptime_fuel_corrected"] = (
        laps["target_laptime"] - fuel_coef * (laps["LapNumber"] - 1)
    )
    return laps, fuel_coef


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(laps: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Build every candidate feature, with the reason each one exists.

    ``laps`` is modified in place with two helper columns (``_field_median``,
    ``_gap_to_field``); they are inputs to the pace features, never exported.
    """
    total_laps = int(laps["LapNumber"].max())
    field_median = laps.groupby("LapNumber")["LapTime"].median()
    laps["_field_median"] = laps["LapNumber"].map(field_median)
    laps["_gap_to_field"] = laps["LapTime"] - laps["_field_median"]

    F = pd.DataFrame(index=laps.index)
    provenance: dict[str, str] = {}

    def add(name: str, values, why: str) -> None:
        F[name] = values
        provenance[name] = why

    # Block A — tyre and stint dynamics
    add("tyre_life", laps["TyreLife"].astype(float),
        "Laps completed on the current tyre set - the primary degradation driver.")
    add("tyre_life_sq", laps["TyreLife"].astype(float) ** 2,
        "Quadratic term: degradation accelerates once the tyre passes its cliff.")
    add("stint_number", laps["Stint"].astype(float),
        "Which stint of the race this is - proxies for race phase and strategy state.")
    add("is_fresh_tyre", laps["FreshTyre"].astype(int),
        "Whether the set was new when fitted (scrubbed sets behave differently).")
    add("tyrelife_x_soft", laps["TyreLife"] * (laps["Compound"] == "SOFT"),
        "Interaction: lets the model learn SOFT's steeper degradation slope vs HARD.")
    add("tyrelife_x_medium", laps["TyreLife"] * (laps["Compound"] == "MEDIUM"),
        "Interaction: MEDIUM degradation slope relative to the HARD baseline.")

    # Block B — fuel load and race progress
    add("race_progress", laps["LapNumber"] / total_laps,
        "Fraction of the race completed; proxies fuel burn-off and race phase. "
        "Canonical axis - lap number / fuel load / laps remaining are affine transforms of it.")

    # Block C — own recent pace, in gap space and strictly causal (shift(1) first)
    gap_shift = laps.groupby("Driver", sort=False)["_gap_to_field"].shift(1)
    by_driver = gap_shift.groupby(laps["Driver"])
    roll3_mean = by_driver.rolling(3, min_periods=3).mean().reset_index(level=0, drop=True)
    roll3_std = by_driver.rolling(3, min_periods=3).std().reset_index(level=0, drop=True)
    expanding = by_driver.expanding(min_periods=1).mean().reset_index(level=0, drop=True)

    add("gap_roll3_mean", roll3_mean,
        "Mean gap to the field over the previous 3 laps - short-run pace.")
    add("gap_roll3_std", roll3_std,
        "Volatility of that gap - unstable laps often precede a tyre cliff or traffic.")
    add("gap_expanding", expanding,
        "Running mean gap over all previous laps - the driver's baseline competitiveness.")
    add("form_vs_baseline", gap_shift - expanding,
        "Last lap's gap relative to the driver's own baseline: are they gaining or losing form?")

    # Block D — field-level pace
    add("field_median_lag1", laps["LapNumber"].sub(1).map(field_median),
        "Field median lap time on the previous lap - the ambient pace level "
        "(absorbs track evolution, safety cars, fuel effects common to everyone).")
    add("field_pace_trend",
        laps["LapNumber"].sub(1).map(field_median) - laps["LapNumber"].sub(2).map(field_median),
        "Change in field median between the last two laps - is the track speeding up or slowing?")

    # Block E — environment
    track_temp = laps["TrackTemp"].astype(float)
    air_temp = laps["AirTemp"].astype(float)
    add("air_temp", air_temp, "Ambient air temperature (deg C).")
    add("track_air_delta", track_temp - air_temp,
        "Track minus air temperature - how much the surface has heated beyond ambient. "
        "Paired with air_temp instead of raw track temp to avoid an exact identity.")
    add("humidity", laps["Humidity"].astype(float), "Relative humidity (%).")
    add("wind_speed", laps["WindSpeed"].astype(float), "Wind speed (m/s); affects aero balance.")
    add("tracktemp_dev_x_tyrelife", (track_temp - track_temp.mean()) * laps["TyreLife"],
        "Interaction: degradation accelerates on a hotter-than-average surface. "
        "Track temp is mean-centred so this is not a rescaled tyre_life.")

    # Block F — categorical encoding. One level of each category is the
    # reference, so the dummies are not collinear with an intercept.
    add("compound_soft", (laps["Compound"] == "SOFT").astype(int),
        f"Compound contrast vs the {REFERENCE_COMPOUND} reference level.")
    add("compound_medium", (laps["Compound"] == "MEDIUM").astype(int),
        f"Compound contrast vs the {REFERENCE_COMPOUND} reference level.")
    for team in sorted(laps["Team"].unique())[1:]:
        add(f"team_{team.lower().replace(' ', '_')}", (laps["Team"] == team).astype(int),
            "Team identity (car performance).")
    for driver in sorted(laps["Driver"].unique())[1:]:
        add(f"driver_{driver.lower()}", (laps["Driver"] == driver).astype(int),
            "Driver identity (skill / style).")

    # Constant in this session; kept so the near-zero-variance filter removes
    # them on the record rather than them being dropped by hand.
    add("track_status", laps["TrackStatus"].astype(float),
        "Track status flag - constant in this session; retained so the NZV filter can remove it.")
    add("is_rainfall", laps["Rainfall"].astype(int),
        "Rain flag - constant in this session; retained so the NZV filter can remove it.")

    return F.astype(float), provenance


def trim_warmup(laps: pd.DataFrame, F: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """Drop the opening laps, where the history-based features are undefined.

    Returns ``(laps, F, n_dropped, first_usable_lap)``, both frames re-indexed
    together so row *i* of one still describes row *i* of the other.
    """
    complete = F[HISTORY_COLS].notna().all(axis=1)
    n_dropped = int((~complete).sum())
    F = F[complete].reset_index(drop=True)
    laps = laps[complete].reset_index(drop=True)
    assert F.notna().all().all(), "unexpected NaNs after the warm-up trim"
    return laps, F, n_dropped, int(laps["LapNumber"].min())


# ---------------------------------------------------------------------------
# 4. Validation splits — the same shape Task 6 uses
# ---------------------------------------------------------------------------
def lap_forward_splits(lap_numbers: pd.Series, n_splits: int = 4,
                       min_train_frac: float = 0.4):
    """Expanding-window splits over lap number; whole laps stay in one fold.

    Never a random K-fold: this is a time-ordered panel, so a shuffled split
    would train on later laps and validate on earlier ones.
    """
    unique_laps = np.sort(lap_numbers.unique())
    bounds = np.linspace(int(len(unique_laps) * min_train_frac),
                         len(unique_laps), n_splits + 1).astype(int)
    for i in range(n_splits):
        train_laps = unique_laps[: bounds[i]]
        test_laps = unique_laps[bounds[i]: bounds[i + 1]]
        if len(test_laps) == 0:
            continue
        yield (np.where(lap_numbers.isin(train_laps))[0],
               np.where(lap_numbers.isin(test_laps))[0])


# ---------------------------------------------------------------------------
# 5. Selection funnel: variance -> correlation -> VIF -> importance
# ---------------------------------------------------------------------------
def near_zero_variance(frame: pd.DataFrame, freq_cut: float = 0.99) -> list[str]:
    """Columns that are constant, or one value at least ``freq_cut`` of the time."""
    dropped = []
    for col in frame.columns:
        top_share = frame[col].value_counts(normalize=True).iloc[0]
        if frame[col].nunique() <= 1 or top_share >= freq_cut:
            dropped.append(col)
    return dropped


def prune_correlated(frame: pd.DataFrame, y: np.ndarray,
                     cut: float = CORRELATION_CUT) -> tuple[list[str], list[dict]]:
    """Of each pair correlated above ``cut``, drop the one less informative about ``y``.

    Mutual information decides which survives, so the choice is measured rather
    than alphabetical.
    """
    mi_scores = pd.Series(
        mutual_info_regression(frame, y, random_state=RANDOM_STATE), index=frame.columns
    )
    corr = frame.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    dropped: list[str] = []
    pairs: list[dict] = []
    for col in upper.columns:
        for row in upper.index:
            r = upper.loc[row, col]
            if pd.notna(r) and r > cut:
                loser = col if mi_scores[col] < mi_scores[row] else row
                keeper = row if loser == col else col
                if loser not in dropped:
                    dropped.append(loser)
                    pairs.append({"kept": keeper, "dropped": loser, "|r|": round(r, 4)})
    return dropped, pairs


def prune_by_vif(frame: pd.DataFrame, cols: list[str],
                 threshold: float = VIF_THRESHOLD) -> tuple[list[str], list[dict]]:
    """Drop the worst multicollinear column until every VIF is under ``threshold``.

    Binary dummies are excluded by the caller: a rare dummy always looks
    collinear with the rest of its group, so VIF is not the right test for it.
    """
    cols = list(cols)
    Z = pd.DataFrame(StandardScaler().fit_transform(frame[cols]), columns=cols)
    removed: list[dict] = []
    while len(cols) > 2:
        vifs = {}
        for c in cols:
            others = [x for x in cols if x != c]
            r2 = LinearRegression().fit(Z[others], Z[c]).score(Z[others], Z[c])
            vifs[c] = 1.0 / max(1e-12, 1.0 - r2)
        worst = max(vifs, key=vifs.get)
        if vifs[worst] <= threshold:
            break
        cols.remove(worst)
        removed.append({"dropped": worst, "VIF": round(min(vifs[worst], 1e6), 1)})
    return cols, removed


def rank_features(X: pd.DataFrame, y: np.ndarray, task: str = "reg") -> pd.DataFrame:
    """Rank features by mutual information, tree importance and L1 coefficient.

    Three different notions of importance, averaged by rank: a feature only
    finishes high if more than one of them agrees.
    """
    Xs = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    if task == "reg":
        mi = mutual_info_regression(X, y, random_state=RANDOM_STATE)
        forest = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE,
                                       n_jobs=-1).fit(X, y)
        linear = np.abs(LassoCV(cv=5, random_state=RANDOM_STATE,
                                max_iter=5000).fit(Xs, y).coef_)
    else:
        mi = mutual_info_classif(X, y, random_state=RANDOM_STATE)
        forest = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                        class_weight="balanced", n_jobs=-1).fit(X, y)
        linear = np.abs(LogisticRegression(penalty="l1", solver="liblinear", C=0.5,
                                           class_weight="balanced",
                                           max_iter=5000).fit(Xs, y).coef_[0])
    out = pd.DataFrame({"mutual_info": mi, "tree_importance": forest.feature_importances_,
                        "l1_coef": linear}, index=X.columns)
    out["avg_rank"] = out.rank(ascending=False).mean(axis=1)
    return out.sort_values("avg_rank")


def rank_with_stability(F_sel: pd.DataFrame, y_reg: np.ndarray, splits: list) -> pd.DataFrame:
    """Full-data ranking, plus how often each feature makes a fold's top group.

    A feature that only ranks highly on the whole dataset may be fitting one
    stretch of the race; ``stability`` says how many folds agree.
    """
    ranking = rank_features(F_sel, y_reg)
    hits = pd.Series(0, index=F_sel.columns, dtype=int)
    for train_idx, _ in splits:
        fold_rank = rank_features(F_sel.iloc[train_idx], y_reg[train_idx])
        hits[fold_rank.head(TOP_K_STABILITY).index] += 1
    ranking["stability"] = hits / len(splits)
    return ranking


def cv_regression(F_sel: pd.DataFrame, y_reg: np.ndarray, splits: list,
                  cols: list[str]) -> tuple[float, float]:
    """Mean CV MAE and R² for one candidate feature set."""
    maes, r2s = [], []
    for train_idx, test_idx in splits:
        model = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(F_sel.iloc[train_idx][cols], y_reg[train_idx])
        pred = model.predict(F_sel.iloc[test_idx][cols])
        maes.append(mean_absolute_error(y_reg[test_idx], pred))
        r2s.append(r2_score(y_reg[test_idx], pred))
    return float(np.mean(maes)), float(np.mean(r2s))


def cv_classification(F_sel: pd.DataFrame, y_clf: np.ndarray, splits: list,
                      cols: list[str]) -> float:
    """Mean CV ROC-AUC, skipping folds whose test split has only one class."""
    aucs = []
    for train_idx, test_idx in splits:
        if len(np.unique(y_clf[test_idx])) < 2:
            continue
        model = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                       class_weight="balanced", n_jobs=-1)
        model.fit(F_sel.iloc[train_idx][cols], y_clf[train_idx])
        proba = model.predict_proba(F_sel.iloc[test_idx][cols])[:, 1]
        aucs.append(roc_auc_score(y_clf[test_idx], proba))
    return float(np.mean(aucs)) if aucs else float("nan")


def choose_regression_features(F_sel: pd.DataFrame, y_reg: np.ndarray, splits: list,
                               ranking: pd.DataFrame) -> tuple[list[str], pd.DataFrame, int]:
    """Sweep how many top-ranked features to keep; take the smallest set within
    2% of the best CV MAE — the simplest model the data cannot distinguish from
    the best one."""
    ranked_order = ranking.index.tolist()
    ks = sorted({k for k in (4, 6, 8, 10, 12, 15, 20, len(ranked_order)) if k <= len(ranked_order)})
    sweep = {k: cv_regression(F_sel, y_reg, splits, ranked_order[:k]) for k in ks}
    sweep_df = pd.DataFrame(
        [{"K": k, "cv_MAE_s": round(m, 4), "cv_R2": round(r, 4)} for k, (m, r) in sweep.items()]
    ).set_index("K")
    best_mae = sweep_df["cv_MAE_s"].min()
    k_star = int(sweep_df.index[sweep_df["cv_MAE_s"] <= best_mae * 1.02].min())
    return ranked_order[:k_star], sweep_df, k_star


def choose_classification_features(F_sel: pd.DataFrame, y_clf: np.ndarray, splits: list
                                   ) -> tuple[list[str], dict, pd.DataFrame]:
    """Pit-decision features. K is fixed at 8: with ~5% positives, a sweep would
    be choosing between numbers that are mostly noise."""
    ranking = rank_features(F_sel, y_clf, task="clf")
    order = ranking.index.tolist()
    sweep = {k: cv_classification(F_sel, y_clf, splits, order[:k]) for k in (4, 6, 8, 10)}
    return order[:K_CLASSIFICATION], sweep, ranking


# ---------------------------------------------------------------------------
# 6. Export
# ---------------------------------------------------------------------------
def build_export(laps: pd.DataFrame, F: pd.DataFrame, selected_union: list[str]) -> pd.DataFrame:
    """Identifiers, the selected features and the targets, in that order."""
    export = pd.concat(
        [laps[ID_COLS].reset_index(drop=True),
         F[selected_union].reset_index(drop=True),
         laps[TARGETS].reset_index(drop=True)],
        axis=1,
    )
    assert export.notna().all().all(), "export contains NaNs"
    assert not export.columns.duplicated().any(), "duplicate columns in export"
    for leak in LEAKAGE_COLS:
        assert leak not in export.columns, f"leakage column {leak} reached the export"
    return export


def pit_schedule_caveat(laps: pd.DataFrame) -> str:
    """Describe how the pit events are spread, rather than assuming.

    A tightly clustered pit schedule inflates classification AUC and can leave a
    chronological holdout with no pit events at all — true of synthetic and real
    sessions alike, so it is measured either way.
    """
    pit_laps = laps.loc[laps["target_pit_next_lap"] == 1, "LapNumber"]
    n_events = int(pit_laps.shape[0])
    n_distinct = int(pit_laps.nunique())
    max_share = float(pit_laps.value_counts(normalize=True).max()) if n_events else float("nan")
    if n_events == 0:
        return "No pit events observed in this dataset; classification metrics cannot be validated."
    if n_distinct <= 3 or max_share >= 0.5:
        return (
            f"Pit events cluster tightly: {n_events} pit event(s) across only "
            f"{n_distinct} distinct lap(s) (the busiest lap accounts for "
            f"{max_share:.0%} of all pits). This concentration will inflate classification "
            f"AUC relative to a session with more strategic pit-timing variation, and can "
            f"leave a chronological holdout with zero pit events."
        )
    return (
        f"{n_events} pit event(s) observed across {n_distinct} distinct laps "
        f"(the busiest lap accounts for {max_share:.0%} of all pits) - a reasonably realistic "
        f"spread of strategic pit timing. This remains a single session, though, and results "
        f"should not be generalised beyond it."
    )


def dataset_source(out_dir: Path) -> dict:
    """Real-vs-synthetic provenance, propagated by Task 4 rather than assumed."""
    marker = Path(out_dir) / "data_source.json"
    return json.loads(marker.read_text()) if marker.exists() else {"source": "synthetic"}


def build_metadata(*, clean_csv: Path, out_dir: Path, export: pd.DataFrame, F: pd.DataFrame,
                   provenance: dict, selected_regression: list[str],
                   selected_classification: list[str], selected_union: list[str],
                   fuel_coef: float, n_dropped: int, first_usable_lap: int,
                   nzv_dropped: list[str], corr_dropped: list[str], vif_removed: list[dict],
                   k_star: int, mae_sel: float, r2_sel: float, clf_auc: float,
                   caveat: str) -> dict:
    """The contract Task 6 onwards reads: which features, why, and how to use them."""
    binary_selected = [c for c in selected_union if F[c].isin([0, 1]).all()]
    return {
        "task": "Phase 2 / Task 5 - Feature Engineering & Feature Selection",
        "source_dataset": str(Path(clean_csv).relative_to(REPO_ROOT)),
        "dataset_source": dataset_source(out_dir),
        "rows": int(len(export)),
        "random_state": RANDOM_STATE,
        "identifier_columns": ID_COLS,
        "targets": {
            "target_laptime": "Lap time in seconds - primary regression target (Task 6).",
            "target_pit_next_lap": "1 if the driver pits at the end of this lap - classification.",
            "target_laptime_fuel_corrected":
                f"Lap time with the fuel-burn trend ({fuel_coef:+.4f} s/lap) removed, "
                "isolating tyre degradation.",
        },
        "selected_features": {
            "target_laptime": selected_regression,
            "target_pit_next_lap": selected_classification,
            "union_exported": selected_union,
        },
        "feature_provenance": {f: provenance[f] for f in selected_union},
        "numeric_features_requiring_scaling":
            [c for c in selected_union if c not in binary_selected],
        "binary_features_no_scaling_needed": binary_selected,
        "preprocessing_contract": {
            "scaling": "NOT applied here. Fit the scaler inside each CV fold to avoid leakage.",
            "validation": "Expanding-window lap-forward split; keep whole laps in one fold. "
                          "Do not use random K-fold - this is a time-ordered panel.",
            "warmup_rows_dropped": n_dropped,
            "first_usable_lap": first_usable_lap,
        },
        "selection_funnel": {
            "1_near_zero_variance_dropped": nzv_dropped,
            "2_correlation_dropped": corr_dropped,
            "3_vif_dropped": [d["dropped"] for d in vif_removed],
            "4_importance_and_stability": f"ranked by mutual information + tree importance + "
                                          f"L1 coefficient; K*={k_star} chosen within 2% of best CV MAE",
        },
        "excluded_as_leakage": {
            "columns": LEAKAGE_COLS,
            "reason": "Sector times sum exactly to LapTime; speed traps and IsPersonalBest are "
                      "only knowable during or after the lap being predicted.",
        },
        "validation_scores": {
            "regression_cv_MAE_s": round(mae_sel, 4),
            "regression_cv_R2": round(r2_sel, 4),
            "classification_cv_AUC": round(clf_auc, 4),
            "caveat": caveat,
        },
    }


# ---------------------------------------------------------------------------
# 7. The whole stage
# ---------------------------------------------------------------------------
def build(clean_csv: Path = FASTF1_LAPS_CLEAN_CSV, out_dir: Path = DATA_PROCESSED_DIR,
          write: bool = True) -> dict:
    """Run Task 5 end to end and write the two contract files.

    Returns a summary dict (also used by the notebook and the build script).
    Every stochastic step is seeded with ``RANDOM_STATE``, so the same cleaned
    input produces the same features and the same selected sets.
    """
    np.random.seed(RANDOM_STATE)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    laps = load_clean_laps(clean_csv)
    laps, fuel_coef = add_targets(laps)
    F, provenance = engineer_features(laps)
    laps, F, n_dropped, first_usable_lap = trim_warmup(laps, F)

    splits = list(lap_forward_splits(laps["LapNumber"]))
    y_reg = laps["target_laptime"].to_numpy()
    y_clf = laps["target_pit_next_lap"].to_numpy()

    nzv_dropped = near_zero_variance(F)
    F_nzv = F.drop(columns=nzv_dropped)

    corr_dropped, corr_pairs = prune_correlated(F_nzv, y_reg)
    F_corr = F_nzv.drop(columns=corr_dropped)

    binary_cols = [c for c in F_corr.columns if F_corr[c].isin([0, 1]).all()]
    continuous_cols = [c for c in F_corr.columns if c not in binary_cols]
    kept_continuous, vif_removed = prune_by_vif(F_corr, continuous_cols)
    F_sel = F_corr[kept_continuous + binary_cols]

    ranking = rank_with_stability(F_sel, y_reg, splits)
    selected_regression, sweep_df, k_star = choose_regression_features(F_sel, y_reg, splits, ranking)
    selected_classification, clf_sweep, ranking_clf = choose_classification_features(
        F_sel, y_clf, splits
    )

    mae_sel, r2_sel = cv_regression(F_sel, y_reg, splits, selected_regression)
    mae_all, r2_all = cv_regression(F_sel, y_reg, splits, list(F_sel.columns))

    selected_union = sorted(set(selected_regression) | set(selected_classification))
    export = build_export(laps, F, selected_union)
    caveat = pit_schedule_caveat(laps)
    metadata = build_metadata(
        clean_csv=clean_csv, out_dir=out_dir, export=export, F=F, provenance=provenance,
        selected_regression=selected_regression, selected_classification=selected_classification,
        selected_union=selected_union, fuel_coef=fuel_coef, n_dropped=n_dropped,
        first_usable_lap=first_usable_lap, nzv_dropped=nzv_dropped, corr_dropped=corr_dropped,
        vif_removed=vif_removed, k_star=k_star, mae_sel=mae_sel, r2_sel=r2_sel,
        clf_auc=clf_sweep[K_CLASSIFICATION], caveat=caveat,
    )

    csv_path = out_dir / TASK5_FEATURES_CSV.name
    json_path = out_dir / TASK5_FEATURE_METADATA_JSON.name
    if write:
        export.to_csv(csv_path, index=False)
        json_path.write_text(json.dumps(metadata, indent=2))

    return {
        "export": export,
        "metadata": metadata,
        "csv_path": csv_path,
        "json_path": json_path,
        "features_all": F,
        "features_post_funnel": F_sel,
        "laps": laps,
        "splits": splits,
        "ranking": ranking,
        "ranking_classification": ranking_clf,
        "regression_sweep": sweep_df,
        "classification_sweep": clf_sweep,
        "correlation_pairs": corr_pairs,
        "vif_removed": vif_removed,
        "fuel_coef": fuel_coef,
        "cv_selected": (mae_sel, r2_sel),
        "cv_all_features": (mae_all, r2_all),
    }
