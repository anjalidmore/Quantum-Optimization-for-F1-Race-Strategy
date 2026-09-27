# Decision Table

_Generated 2026-09-27 17:42 UTC — 32 rules, 25 input keys, 10 output keys._

Each row is one rule. A condition cell shows the test that input must pass; an action cell shows what the rule asserts. Empty means the rule ignores that key.

## Conditions (inputs)

| Rule | Salience | `circuit` | `current_compound` | `current_position` | `drs_enabled` | `fuel_margin` | `gap_ahead` | `gap_behind` | `graining_risk` | `grid_position` | `in_pit_window` | `laps_remaining` | `overcut_opportunity` | `overtaking_difficulty` | `pit_loss` | `rain_probability` | `safety_car_likelihood` | `safety_car_probability` | `total_laps` | `track_status` | `track_temperature` | `track_wet` | `tyre_age_laps` | `tyre_wear` | `undercut_threat` | `weather_severity` |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `R-WX-001` | 100 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | > 70 |  |  |  |  |  |  |  |  |  |  |
| `R-WX-002` | 100 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | <= 90 |  |  |  |  |  | is_true |  |  |  |  |
| `R-WX-003` | 100 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | == extreme |
| `R-WX-004` | 100 |  | in ('INTERMEDIATE', 'WET') |  |  |  |  |  |  |  |  |  |  |  |  | < 30 |  |  |  |  |  | is_false |  |  |  |  |
| `R-SC-001` | 80 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | == SC |  |  |  | >= 40 |  |  |
| `R-SC-002` | 80 |  |  |  |  |  |  |  |  |  | is_true |  |  |  |  |  |  |  |  | == VSC |  |  |  |  |  |  |
| `R-SC-003` | 80 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | >= 60 |  | == GREEN |  |  |  | < 55 |  |  |
| `R-SC-004` | 80 |  |  | <= 3 |  |  |  |  |  |  |  | <= 8 |  |  |  |  |  |  |  | == SC |  |  |  |  |  |  |
| `R-DEG-001` | 60 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | > 40 |  |  |  |  |  |
| `R-DEG-002` | 60 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | < 25 |  |  | < 60 |  |  |
| `R-DEG-003` | 60 |  |  |  |  |  |  |  | == high |  |  |  |  |  |  |  |  |  |  |  |  |  |  | >= 50 |  |  |
| `R-PIT-001` | 60 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | >= 80 |  |  |
| `R-PIT-002` | 60 |  |  |  |  |  |  |  |  |  | is_true |  |  |  |  |  |  |  |  |  |  |  |  |  | is_true |  |
| `R-PIT-003` | 60 |  |  |  |  |  | > 2.0 |  |  |  |  |  | is_true |  |  |  |  |  |  |  |  |  |  | < 65 |  |  |
| `R-PIT-004` | 60 |  |  |  |  |  |  |  |  |  | is_true |  |  |  |  |  |  |  |  |  |  |  |  | < 80 |  |  |
| `R-STRAT-001` | 60 | == Monaco |  |  |  |  |  |  |  | <= 3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `R-STRAT-002` | 40 |  |  |  |  |  |  |  |  |  |  |  |  | == high |  |  |  |  |  |  |  |  |  |  |  |  |
| `R-STRAT-003` | 40 |  |  |  |  |  |  |  |  |  |  |  |  | != high | < 20 |  |  |  |  |  | > 40 |  |  |  |  |  |
| `R-TYRE-001` | 40 |  | == SOFT |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | >= 60 |  |  |
| `R-TYRE-002` | 40 |  |  |  |  |  |  |  |  |  |  | > 30 |  |  |  |  |  |  |  |  |  | is_false |  |  |  |  |
| `R-TYRE-003` | 40 |  |  |  |  |  |  |  |  |  |  | <= 15 |  |  |  |  |  |  |  |  |  | is_false |  | >= 40 |  |  |
| `R-TYRE-004` | 40 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | == damp |
| `R-TYRE-005` | 40 |  | == SOFT |  |  |  |  |  |  |  |  | > 20 |  |  |  |  |  |  |  |  | > 45 |  |  |  |  |  |
| `R-ERS-001` | 30 |  |  |  | is_true |  |  | < 1.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `R-FUEL-001` | 30 |  |  |  |  | < 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `R-FUEL-002` | 30 |  | == SOFT |  |  | > 1.5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `R-STRAT-004` | 30 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | == low |  | > 50 |  |  |  |  |  |  |  |
| `R-TAC-001` | 20 |  |  |  | is_true |  | < 1.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | < 10 |  |  |  |
| `R-TAC-002` | 20 |  |  |  |  |  | > 3.0 | > 3.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `R-TAC-003` | 20 |  |  |  |  |  |  | < 1.5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | >= 60 |  |  |
| `R-RISK-001` | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | > 50 |  |  |  |  |  |  |  | >= 75 |  |  |
| `R-RISK-002` | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | == GREEN |  |  |  | < 50 |  | == dry |

## Actions (conclusions)

| Rule | `defend_advice` | `engine_mode_advice` | `fuel_advice` | `notes` | `pit_decision` | `push_advice` | `recommended_tyre` | `risk_level` | `strategy_stops` | `tyre_deg_adjustment` |
|---|---|---|---|---|---|---|---|---|---|---|
| `R-WX-001` |  |  |  | High rain probability: switch to intermediates. |  |  | INTERMEDIATE | high |  |  |
| `R-WX-002` |  |  |  |  |  | conserve | INTERMEDIATE |  |  |  |
| `R-WX-003` |  |  |  | Extreme conditions: full wet tyres, expect SC/red flag. |  | conserve | WET | high |  |  |
| `R-WX-004` |  |  |  | Track drying: crossover to slicks approaching. | PIT_SOON |  | SOFT |  |  |  |
| `R-SC-001` |  |  |  | SC out: cheap pit stop, pit immediately. | PIT_NOW |  |  |  |  |  |
| `R-SC-002` |  |  |  | VSC reduces pit loss: take the stop within the window. | PIT_NOW |  |  |  |  |  |
| `R-SC-003` |  |  |  | Bank on an imminent SC for a cheaper stop. | DELAY_PIT |  |  |  |  |  |
| `R-SC-004` | control_restart |  |  | Track position outweighs tyre delta this late. | STAY_OUT |  |  |  |  |  |
| `R-DEG-001` |  |  |  | Hot track: expect elevated thermal degradation. |  |  |  |  |  | increase |
| `R-DEG-002` |  |  |  | Cool track favours longer stints. | STAY_OUT |  |  |  |  | decrease |
| `R-DEG-003` |  |  |  | Manage graining: reduce sliding, plan an earlier stop. | PIT_SOON | conserve |  |  |  |  |
| `R-PIT-001` |  |  |  |  | PIT_NOW |  |  | high |  |  |
| `R-PIT-002` |  |  |  | Cover the undercut before the rival gains free air. | PIT_NOW |  |  |  |  |  |
| `R-PIT-003` |  |  |  | Overcut: push in clear air, pit later than rival. | STAY_OUT | push |  |  |  |  |
| `R-PIT-004` |  |  |  | Within window and wearing: prepare to stop. | PIT_SOON |  |  |  |  |  |
| `R-STRAT-001` |  |  |  | Monaco: track position is king, minimise stops. |  |  |  |  | 1 |  |
| `R-STRAT-002` | protect_position |  |  |  |  |  |  |  | 1 |  |
| `R-STRAT-003` |  |  |  | Cheap stops + high deg reward an aggressive 2-stop. |  | push |  |  | 2 |  |
| `R-TYRE-001` |  |  |  |  |  |  | MEDIUM |  |  |  |
| `R-TYRE-002` |  |  |  |  |  |  | HARD |  |  |  |
| `R-TYRE-003` |  |  |  |  |  | push | SOFT |  |  |  |
| `R-TYRE-004` |  |  |  |  |  | conserve | INTERMEDIATE |  |  |  |
| `R-TYRE-005` |  |  |  | Very hot track: avoid softs for a long middle stint. |  |  | MEDIUM |  |  |  |
| `R-ERS-001` | cover_inside_line | deploy_on_straights |  | Under DRS threat: deploy ERS to defend the straights. |  |  |  |  |  |  |
| `R-FUEL-001` |  |  | lift_and_coast | Under fuel target: save via lift-and-coast zones. |  | conserve |  | medium |  |  |
| `R-FUEL-002` |  |  | normal |  |  | push |  |  |  |  |
| `R-STRAT-004` |  |  |  | Low SC prior: commit to a planned strategy, fewer reactive stops. |  |  |  |  |  |  |
| `R-TAC-001` |  |  |  | Fresh tyres + DRS: attack the car ahead. |  | push |  |  |  |  |
| `R-TAC-002` |  |  |  | Clear air: manage tyres, no need to push. |  | conserve |  |  |  |  |
| `R-TAC-003` | defensive_lines |  |  | Worn tyres under pressure: defend, protect braking zones. |  | conserve |  |  |  |  |
| `R-RISK-001` |  |  |  | Worn tyres and rain risk compound: high uncertainty. |  |  |  | high |  |  |
| `R-RISK-002` |  |  |  |  |  |  |  | low |  |  |

## Conclusions more than one rule can assert

Conflict resolution is by salience, then specificity, then rule id (see `app/intelligence/expert_system/inference.py`).

| Conclusion | Rules that can assert it |
|---|---|
| `defend_advice` | `R-ERS-001`, `R-SC-004`, `R-STRAT-002`, `R-TAC-003` |
| `fuel_advice` | `R-FUEL-001`, `R-FUEL-002` |
| `notes` | `R-DEG-001`, `R-DEG-002`, `R-DEG-003`, `R-ERS-001`, `R-FUEL-001`, `R-PIT-002`, `R-PIT-003`, `R-PIT-004`, `R-RISK-001`, `R-SC-001`, `R-SC-002`, `R-SC-003`, `R-SC-004`, `R-STRAT-001`, `R-STRAT-003`, `R-STRAT-004`, `R-TAC-001`, `R-TAC-002`, `R-TAC-003`, `R-TYRE-005`, `R-WX-001`, `R-WX-003`, `R-WX-004` |
| `pit_decision` | `R-DEG-002`, `R-DEG-003`, `R-PIT-001`, `R-PIT-002`, `R-PIT-003`, `R-PIT-004`, `R-SC-001`, `R-SC-002`, `R-SC-003`, `R-SC-004`, `R-WX-004` |
| `push_advice` | `R-DEG-003`, `R-FUEL-001`, `R-FUEL-002`, `R-PIT-003`, `R-STRAT-003`, `R-TAC-001`, `R-TAC-002`, `R-TAC-003`, `R-TYRE-003`, `R-TYRE-004`, `R-WX-002`, `R-WX-003` |
| `recommended_tyre` | `R-TYRE-001`, `R-TYRE-002`, `R-TYRE-003`, `R-TYRE-004`, `R-TYRE-005`, `R-WX-001`, `R-WX-002`, `R-WX-003`, `R-WX-004` |
| `risk_level` | `R-FUEL-001`, `R-PIT-001`, `R-RISK-001`, `R-RISK-002`, `R-WX-001`, `R-WX-003` |
| `strategy_stops` | `R-STRAT-001`, `R-STRAT-002`, `R-STRAT-003` |
| `tyre_deg_adjustment` | `R-DEG-001`, `R-DEG-002` |

