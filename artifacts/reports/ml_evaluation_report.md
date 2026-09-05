# Task 6 — Machine Learning Model Development

Generated: 2026-09-05T01:46:15.654297+00:00

## 1. Objective

Train and compare classical machine-learning models for two F1 race-strategy
prediction problems: lap-time regression and pit-decision classification.

## 2. Dataset

- Source: `data/processed/fastf1_laps_clean.csv`
- Feature matrix: `data/processed/f1_features_selected.csv`
- Rows: 995
- **DATASET: Real FastF1 session** — Bahrain 2023 (R), fetched 2026-08-30T12:34:08.172714+00:00, 1055 laps across 20 drivers. Not synthetic.

## 3. Feature contract

Selected features were produced by Task 5 (Feature Engineering & Feature
Selection) and are consumed here unchanged:

- Regression features: `field_median_lag1`, `stint_number`, `gap_expanding`, `track_status`, `tracktemp_dev_x_tyrelife`, `form_vs_baseline`, `field_pace_trend`, `gap_roll3_std`, `wind_speed`, `tyrelife_x_soft`, `tyre_life`, `gap_roll3_mean`, `team_red_bull_racing`, `driver_dev`, `driver_oco`, `team_aston_martin`, `driver_per`, `driver_rus`, `compound_soft`, `humidity`, `team_mclaren`, `driver_gas`, `driver_zho`, `driver_alo`, `driver_bot`, `driver_ver`, `team_mercedes`, `driver_str`, `team_alpine`, `is_fresh_tyre`, `team_williams`, `driver_nor`, `driver_pia`, `team_alphatauri`, `driver_sar`, `driver_lec`, `tyrelife_x_medium`, `team_haas_f1_team`, `driver_mag`, `driver_ham`, `driver_hul`, `driver_sai`, `driver_tsu`, `team_ferrari`, `compound_medium`
- Classification features: `tyre_life`, `tracktemp_dev_x_tyrelife`, `form_vs_baseline`, `field_median_lag1`, `field_pace_trend`, `tyrelife_x_soft`, `gap_roll3_mean`, `compound_soft`

## 4. Leakage prevention

Excluded as leakage (Task 5 contract): `Sector1Time`, `Sector2Time`, `Sector3Time`, `SpeedFL`, `SpeedST`, `IsPersonalBest`.

Reason: Sector times sum exactly to LapTime; speed traps and IsPersonalBest are only knowable during or after the lap being predicted.

The Task 6 data contract (`app.intelligence.ml.data_contract`) asserts none
of these columns are present before any model is trained.

## 5. Validation

Expanding-window, lap-forward cross-validation with a chronologically later,
untouched holdout test set. Random K-fold is never used — see
`app.intelligence.ml.splits`.

## 6. Models

Linear/Logistic Regression, Decision Tree, Random Forest, SVM/SVR, and
XGBoost (optional — see section 9).

## 7-8. Metrics and results

See `regression_report.md`, `classification_report.md`, and
`model_selection_report.md` for full tables. Best regression model:
**decision_tree**. Best classification model: **random_forest**.

## 9. XGBoost availability

XGBoost trained successfully.

## 10. Limitations

48 pit event(s) observed across 21 distinct laps (the busiest lap accounts for 12% of all pits) - a reasonably realistic spread of strategic pit timing. This remains a single session, though, and results should not be generalised beyond it.

See `classification_report.md` for the full discussion of how this affects
classification metrics in this run. This is real FastF1 telemetry for a single Grand Prix — results should not be generalised to other races or drivers.

## 11. Reproducibility

- Random seed: 42 (used for every model with a stochastic component)
- Hyperparameter grids are bounded and recorded per model in
  `artifacts/metadata/model_registry.json`
- Preprocessing (imputation/scaling) is fit inside each CV fold and on the
  development set only before the final holdout evaluation

## 12. Conclusion

Task 6 provides classical ML predictions for lap time and pit timing that
downstream tasks (Expert System, Search, and the strategy simulator) can
consume alongside rule-based and search-based reasoning.
