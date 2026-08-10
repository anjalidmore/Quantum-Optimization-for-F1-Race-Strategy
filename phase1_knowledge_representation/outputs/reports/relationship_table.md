# Relationship Table

_Generated 2026-08-03 04:22 UTC — 29 relationships._

| # | Relationship | Domain | → | Range | Cardinality | Description |
|---|--------------|--------|---|-------|-------------|-------------|
| 1 | `drives_for` | Driver | → | Constructor | *..1 | A driver competes on behalf of a constructor. |
| 2 | `owns` | Constructor | → | Car | 1..* | A constructor owns / fields cars. |
| 3 | `participates_in` | Driver | → | Race | *..* | A driver takes part in a race. |
| 4 | `held_at` | Race | → | Circuit | *..1 | A race is held at a circuit. |
| 5 | `belongs_to_season` | Race | → | Season | *..1 | A race belongs to a season. |
| 6 | `instance_of_gp` | Race | → | GrandPrix | *..1 | A race is an instance of a recurring Grand Prix. |
| 7 | `contains_lap` | Race | → | Lap | 1..* | A race contains laps. |
| 8 | `lap_of_driver` | Lap | → | Driver | *..1 | A lap is driven by a driver. |
| 9 | `lap_in_session` | Lap | → | Session | *..1 | A lap belongs to a session. |
| 10 | `lap_has_telemetry` | Lap | → | Telemetry | 1..* | A lap carries telemetry samples. |
| 11 | `lap_has_sector` | Lap | → | Sector | 1..* | A lap is divided into sectors. |
| 12 | `sector_has_time` | Sector | → | SectorTime | 1..1 | A sector has a measured sector time. |
| 13 | `performs_pitstop` | Driver | → | PitStop | 1..* | A driver performs pit stops. |
| 14 | `pitstop_changes_tyre` | PitStop | → | TyreCompound | *..1 | A pit stop changes to a tyre compound. |
| 15 | `pitstop_begins_stint` | PitStop | → | Stint | 1..1 | A pit stop begins a new stint. |
| 16 | `stint_uses_compound` | Stint | → | TyreCompound | *..1 | A stint runs on a tyre compound. |
| 17 | `weather_affects_strategy` | Weather | → | Strategy | *..* | Weather conditions influence strategy. |
| 18 | `safetycar_affects_pitwindow` | SafetyCar | → | PitWindow | *..* | Safety-car deployment shifts the optimal pit window. |
| 19 | `trackstatus_affects_strategy` | TrackStatus | → | Strategy | *..* | Track status influences strategy. |
| 20 | `telemetry_influences_prediction` | Telemetry | → | Prediction | *..* | Telemetry features feed predictions. |
| 21 | `strategy_recommends_pitstop` | Strategy | → | PitStop | 1..* | A strategy recommends pit stops. |
| 22 | `strategy_recommends_tyre` | Strategy | → | TyreCompound | 1..* | A strategy recommends tyre compounds. |
| 23 | `strategy_predicts_position` | Strategy | → | Position | 1..1 | A strategy predicts a finish position. |
| 24 | `prediction_generates_recommendation` | Prediction | → | Recommendation | 1..* | A prediction yields recommendations. |
| 25 | `driver_has_result` | Driver | → | Result | 1..* | A driver has classified results. |
| 26 | `result_in_race` | Result | → | Race | *..1 | A result is recorded for a race. |
| 27 | `driver_has_standing` | Driver | → | DriverStanding | 1..* | A driver has championship standings. |
| 28 | `constructor_has_standing` | Constructor | → | ConstructorStanding | 1..* | A constructor has championship standings. |
| 29 | `optimization_yields_strategy` | Optimization | → | Strategy | 1..* | Optimisation produces candidate strategies. |
