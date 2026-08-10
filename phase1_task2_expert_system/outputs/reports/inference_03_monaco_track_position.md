# Inference Report — Monaco track position

_Generated 2026-08-03 04:26 UTC._

## Inputs (GIVEN facts)

| Fact | Value |
|------|-------|
| `circuit` | 'Monaco' |
| `grid_position` | 2 |
| `overtaking_difficulty` | 'high' |
| `tyre_wear` | 20 |
| `weather_severity` | 'dry' |
| `track_status` | 'GREEN' |
| `current_compound` | 'MEDIUM' |

## Firing sequence

| # | Iter | Rule | Name | Asserted |
|---|------|------|------|----------|
| 1 | 1 | `R-STRAT-001` | Monaco + front row -> one stop | strategy_stops=1, notes='Monaco: track position is king, minimise stops.' |
| 2 | 1 | `R-STRAT-002` | High overtaking difficulty -> minimise stops | defend_advice='protect_position' |
| 3 | 1 | `R-RISK-002` | Stable dry conditions mid-race -> low risk baseline | risk_level='low' |

## Conclusions (derived decisions)

| Decision | Value |
|----------|-------|
| `strategy_stops` | 1 |
| `notes` | 'Monaco: track position is king, minimise stops.' |
| `defend_advice` | 'protect_position' |
| `risk_level` | 'low' |

## WHY — narrative explanation

```
The engine fired 3 rule(s) over 2 iteration(s) and reached a stable conclusion.
  [iter 1] R-STRAT-001 «Monaco + front row -> one stop» fired, asserting strategy_stops=1, notes='Monaco: track position is king, minimise stops.' — Overtaking at Monaco is exceptionally hard, so protecting track position with a single stop is standard.
  [iter 1] R-STRAT-002 «High overtaking difficulty -> minimise stops» fired, asserting defend_advice='protect_position' — Where passing is hard, every pit stop risks a position that cannot be won back on track.
  [iter 1] R-RISK-002 «Stable dry conditions mid-race -> low risk baseline» fired, asserting risk_level='low' — Green-flag dry running on healthy tyres is the low-risk baseline against which deviations are judged.
Final decisions:
  • strategy_stops = 1
  • notes = 'Monaco: track position is king, minimise stops.'
  • defend_advice = 'protect_position'
  • risk_level = 'low'
```

## HOW — per-decision justification

**`strategy_stops` = 1**  — via R-STRAT-001 «Monaco + front row -> one stop»
  - because `circuit == 'Monaco'`
  - because `grid_position <= 3`

**`notes` = 'Monaco: track position is king, minimise stops.'**  — via R-STRAT-001 «Monaco + front row -> one stop»
  - because `circuit == 'Monaco'`
  - because `grid_position <= 3`

**`defend_advice` = 'protect_position'**  — via R-STRAT-002 «High overtaking difficulty -> minimise stops»
  - because `overtaking_difficulty == 'high'`

**`risk_level` = 'low'**  — via R-RISK-002 «Stable dry conditions mid-race -> low risk baseline»
  - because `weather_severity == 'dry'`
  - because `track_status == 'GREEN'`
  - because `tyre_wear < 50`

## Audit trail

```
circuit := 'Monaco'  (input)
grid_position := 2  (input)
overtaking_difficulty := 'high'  (input)
tyre_wear := 20  (input)
weather_severity := 'dry'  (input)
track_status := 'GREEN'  (input)
current_compound := 'MEDIUM'  (input)
strategy_stops := 1  (rule R-STRAT-001)
notes := 'Monaco: track position is king, minimise stops.'  (rule R-STRAT-001)
defend_advice := 'protect_position'  (rule R-STRAT-002)
risk_level := 'low'  (rule R-RISK-002)
```

