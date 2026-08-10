# Inference Report — Undercut battle in the heat

_Generated 2026-08-03 04:26 UTC._

## Inputs (GIVEN facts)

| Fact | Value |
|------|-------|
| `in_pit_window` | True |
| `undercut_threat` | True |
| `track_temperature` | 46 |
| `pit_loss` | 18 |
| `overtaking_difficulty` | 'medium' |
| `tyre_wear` | 58 |
| `current_compound` | 'SOFT' |
| `weather_severity` | 'dry' |
| `track_status` | 'GREEN' |
| `laps_remaining` | 25 |

## Firing sequence

| # | Iter | Rule | Name | Asserted |
|---|------|------|------|----------|
| 1 | 1 | `R-PIT-004` | Approaching window edge -> prime pit crew | pit_decision='PIT_SOON', notes='Within window and wearing: prepare to stop.' |
| 2 | 1 | `R-PIT-002` | In window + undercut threat -> pit now | pit_decision='PIT_NOW', notes='Cover the undercut before the rival gains free air.' |
| 3 | 1 | `R-DEG-001` | High track temp -> increased degradation | tyre_deg_adjustment='increase', notes='Hot track: expect elevated thermal degradation.' |
| 4 | 1 | `R-STRAT-003` | High deg circuit + low pit loss -> two stops | strategy_stops=2, push_advice='push', notes='Cheap stops + high deg reward an aggressive 2-stop.' |
| 5 | 1 | `R-TYRE-005` | Medium mid-race baseline in the heat | recommended_tyre='MEDIUM', notes='Very hot track: avoid softs for a long middle stint.' |

## Conclusions (derived decisions)

| Decision | Value |
|----------|-------|
| `pit_decision` | 'PIT_NOW' |
| `notes` | 'Within window and wearing: prepare to stop. | Cover the undercut before the rival gains free air. | Hot track: expect elevated thermal degradation. | Cheap stops + high deg reward an aggressive 2-stop. | Very hot track: avoid softs for a long middle stint.' |
| `tyre_deg_adjustment` | 'increase' |
| `strategy_stops` | 2 |
| `push_advice` | 'push' |
| `recommended_tyre` | 'MEDIUM' |

## WHY — narrative explanation

```
The engine fired 5 rule(s) over 2 iteration(s) and reached a stable conclusion.
  [iter 1] R-PIT-004 «Approaching window edge -> prime pit crew» fired, asserting pit_decision='PIT_SOON', notes='Within window and wearing: prepare to stop.' — Signals the crew to be ready without committing, keeping flexibility for a safety car.
  [iter 1] R-PIT-002 «In window + undercut threat -> pit now» fired, asserting pit_decision='PIT_NOW', notes='Cover the undercut before the rival gains free air.' — Reacting to an undercut requires pitting on the same lap or the fresh-tyre advantage flips positions.
  [iter 1] R-DEG-001 «High track temp -> increased degradation» fired, asserting tyre_deg_adjustment='increase', notes='Hot track: expect elevated thermal degradation.' — Track temperatures above ~40°C accelerate thermal degradation, shortening viable stint length.
  [iter 1] R-STRAT-003 «High deg circuit + low pit loss -> two stops» fired, asserting strategy_stops=2, push_advice='push', notes='Cheap stops + high deg reward an aggressive 2-stop.' — When pit loss is small and degradation high, a two-stop on fresher tyres beats nursing one set.
  [iter 1] R-TYRE-005 «Medium mid-race baseline in the heat» fired, asserting recommended_tyre='MEDIUM', notes='Very hot track: avoid softs for a long middle stint.' — In extreme heat, softs overheat quickly; mediums give a more stable operating window for a long stint.
Final decisions:
  • pit_decision = 'PIT_NOW'
  • notes = 'Within window and wearing: prepare to stop. | Cover the undercut before the rival gains free air. | Hot track: expect elevated thermal degradation. | Cheap stops + high deg reward an aggressive 2-stop. | Very hot track: avoid softs for a long middle stint.'
  • tyre_deg_adjustment = 'increase'
  • strategy_stops = 2
  • push_advice = 'push'
  • recommended_tyre = 'MEDIUM'
```

## HOW — per-decision justification

**`pit_decision` = 'PIT_NOW'**  — via R-PIT-002 «In window + undercut threat -> pit now»
  - because `in_pit_window is_true`
  - because `undercut_threat is_true`

**`notes` = 'Within window and wearing: prepare to stop. | Cover the undercut before the rival gains free air. | Hot track: expect elevated thermal degradation. | Cheap stops + high deg reward an aggressive 2-stop. | Very hot track: avoid softs for a long middle stint.'**  — via R-TYRE-005 «Medium mid-race baseline in the heat»
  - because `current_compound == 'SOFT'`
  - because `track_temperature > 45`
  - because `laps_remaining > 20`

**`tyre_deg_adjustment` = 'increase'**  — via R-DEG-001 «High track temp -> increased degradation»
  - because `track_temperature > 40`

**`strategy_stops` = 2**  — via R-STRAT-003 «High deg circuit + low pit loss -> two stops»
  - because `track_temperature > 40`
  - because `pit_loss < 20`
  - because `overtaking_difficulty != 'high'`

**`push_advice` = 'push'**  — via R-STRAT-003 «High deg circuit + low pit loss -> two stops»
  - because `track_temperature > 40`
  - because `pit_loss < 20`
  - because `overtaking_difficulty != 'high'`

**`recommended_tyre` = 'MEDIUM'**  — via R-TYRE-005 «Medium mid-race baseline in the heat»
  - because `current_compound == 'SOFT'`
  - because `track_temperature > 45`
  - because `laps_remaining > 20`

## Audit trail

```
in_pit_window := True  (input)
undercut_threat := True  (input)
track_temperature := 46  (input)
pit_loss := 18  (input)
overtaking_difficulty := 'medium'  (input)
tyre_wear := 58  (input)
current_compound := 'SOFT'  (input)
weather_severity := 'dry'  (input)
track_status := 'GREEN'  (input)
laps_remaining := 25  (input)
pit_decision := 'PIT_SOON'  (rule R-PIT-004)
notes := 'Within window and wearing: prepare to stop.'  (rule R-PIT-004)
pit_decision := 'PIT_NOW'  (rule R-PIT-002)
notes := 'Within window and wearing: prepare to stop. | Cover the undercut before the rival gains free air.'  (rule R-PIT-002)
tyre_deg_adjustment := 'increase'  (rule R-DEG-001)
notes := 'Within window and wearing: prepare to stop. | Cover the undercut before the rival gains free air. | Hot track: expect elevated thermal degradation.'  (rule R-DEG-001)
strategy_stops := 2  (rule R-STRAT-003)
push_advice := 'push'  (rule R-STRAT-003)
notes := 'Within window and wearing: prepare to stop. | Cover the undercut before the rival gains free air. | Hot track: expect elevated thermal degradation. | Cheap stops + high deg reward an aggressive 2-stop.'  (rule R-STRAT-003)
recommended_tyre := 'MEDIUM'  (rule R-TYRE-005)
notes := 'Within window and wearing: prepare to stop. | Cover the undercut before the rival gains free air. | Hot track: expect elevated thermal degradation. | Cheap stops + high deg reward an aggressive 2-stop. | Very hot track: avoid softs for a long middle stint.'  (rule R-TYRE-005)
```

