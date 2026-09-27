# Visualization Report

_Generated 2026-09-27 17:42 UTC — 7 figures._

Every figure below was rendered by `app/intelligence/data/visualize.py` from the cleaned tables in this same run. None is a stock image.

| Figure | File | What it shows |
|---|---|---|
| Correlation heatmap | `figures/correlation_heatmap.png` | Pairwise correlation across the numeric columns. The block that matters is the sector times against lap time: they are near-perfectly correlated, which is the evidence for excluding them as leakage in Task 5. |
| Points by driver | `figures/driver_points.png` | Championship points per driver in the loaded season tables — a sanity check that the results table joins correctly to drivers. |
| Points by constructor | `figures/constructor_points.png` | The same aggregation at team level. |
| Lap-time distribution | `figures/laptime_distribution.png` | How lap times are spread. The right tail is in-laps, out-laps and traffic, which is why Task 4 caps IQR outliers rather than deleting the rows. |
| Pit-stop duration | `figures/pit_duration.png` | Box plot of stationary time. Useful for the pit-loss constant the Task 3 search uses. |
| Tyre degradation | `figures/tyre_degradation.png` | Lap time against tyre age per compound — the physical relationship every later model is trying to learn. |
| Composite dashboard | `figures/dashboard.png` | All of the above on one canvas, for a single-glance overview. |

## How to read them together

The correlation heatmap explains a modelling decision (the leakage exclusions). The tyre-degradation plot shows the effect the models are asked to predict. The lap-time distribution explains why the cleaning step caps outliers instead of dropping rows. The remaining three are data-integrity checks on the season tables.

Figures cover 10 drivers and 6 constructors from the loaded tables.

