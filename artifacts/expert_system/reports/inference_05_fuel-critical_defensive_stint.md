# Inference Report — Fuel-critical defensive stint

_Generated 2026-09-05 01:24 UTC._

## Inputs (GIVEN facts)

| Fact | Value |
|------|-------|
| `fuel_margin` | -0.8 |
| `gap_behind` | 0.7 |
| `drs_enabled` | True |
| `tyre_wear` | 64 |
| `weather_severity` | 'dry' |
| `track_status` | 'GREEN' |
| `current_compound` | 'HARD' |
| `laps_remaining` | 10 |

## Firing sequence

| # | Iter | Rule | Name | Asserted |
|---|------|------|------|----------|
| 1 | 1 | `R-ERS-001` | Defending with DRS train -> deploy ERS on straights | engine_mode_advice='deploy_on_straights', defend_advice='cover_inside_line', notes='Under DRS threat: deploy ERS to defend the straights.' |
| 2 | 1 | `R-FUEL-001` | Fuel short -> lift and coast | fuel_advice='lift_and_coast', push_advice='conserve', risk_level='medium', notes='Under fuel target: save via lift-and-coast zones.' |
| 3 | 1 | `R-TAC-003` | Under pressure + worn tyres -> defensive lines | defend_advice='defensive_lines', notes='Worn tyres under pressure: defend, protect braking zones.' |

## Conclusions (derived decisions)

| Decision | Value |
|----------|-------|
| `engine_mode_advice` | 'deploy_on_straights' |
| `defend_advice` | 'defensive_lines' |
| `notes` | 'Under DRS threat: deploy ERS to defend the straights. | Under fuel target: save via lift-and-coast zones. | Worn tyres under pressure: defend, protect braking zones.' |
| `fuel_advice` | 'lift_and_coast' |
| `push_advice` | 'conserve' |
| `risk_level` | 'medium' |

## WHY — narrative explanation

```
The engine fired 3 rule(s) over 2 iteration(s) and reached a stable conclusion.
  [iter 1] R-ERS-001 «Defending with DRS train -> deploy ERS on straights» fired, asserting engine_mode_advice='deploy_on_straights', defend_advice='cover_inside_line', notes='Under DRS threat: deploy ERS to defend the straights.' — When a rival is within DRS range, biasing ERS deployment to the straights protects against a slipstream pass.
  [iter 1] R-FUEL-001 «Fuel short -> lift and coast» fired, asserting fuel_advice='lift_and_coast', push_advice='conserve', risk_level='medium', notes='Under fuel target: save via lift-and-coast zones.' — A negative fuel margin means the car will not reach the flag at full pace; lift-and-coast recovers margin.
  [iter 1] R-TAC-003 «Under pressure + worn tyres -> defensive lines» fired, asserting defend_advice='defensive_lines', notes='Worn tyres under pressure: defend, protect braking zones.' — Worn tyres cannot match a fresher pursuer's pace, so positioning and braking-zone control matter most.
Final decisions:
  • engine_mode_advice = 'deploy_on_straights'
  • defend_advice = 'defensive_lines'
  • notes = 'Under DRS threat: deploy ERS to defend the straights. | Under fuel target: save via lift-and-coast zones. | Worn tyres under pressure: defend, protect braking zones.'
  • fuel_advice = 'lift_and_coast'
  • push_advice = 'conserve'
  • risk_level = 'medium'
```

## HOW — per-decision justification

**`engine_mode_advice` = 'deploy_on_straights'**  — via R-ERS-001 «Defending with DRS train -> deploy ERS on straights»
  - because `gap_behind < 1.0`
  - because `drs_enabled is_true`

**`defend_advice` = 'defensive_lines'**  — via R-TAC-003 «Under pressure + worn tyres -> defensive lines»
  - because `gap_behind < 1.5`
  - because `tyre_wear >= 60`

**`notes` = 'Under DRS threat: deploy ERS to defend the straights. | Under fuel target: save via lift-and-coast zones. | Worn tyres under pressure: defend, protect braking zones.'**  — via R-TAC-003 «Under pressure + worn tyres -> defensive lines»
  - because `gap_behind < 1.5`
  - because `tyre_wear >= 60`

**`fuel_advice` = 'lift_and_coast'**  — via R-FUEL-001 «Fuel short -> lift and coast»
  - because `fuel_margin < 0`

**`push_advice` = 'conserve'**  — via R-FUEL-001 «Fuel short -> lift and coast»
  - because `fuel_margin < 0`

**`risk_level` = 'medium'**  — via R-FUEL-001 «Fuel short -> lift and coast»
  - because `fuel_margin < 0`

## Audit trail

```
fuel_margin := -0.8  (input)
gap_behind := 0.7  (input)
drs_enabled := True  (input)
tyre_wear := 64  (input)
weather_severity := 'dry'  (input)
track_status := 'GREEN'  (input)
current_compound := 'HARD'  (input)
laps_remaining := 10  (input)
engine_mode_advice := 'deploy_on_straights'  (rule R-ERS-001)
defend_advice := 'cover_inside_line'  (rule R-ERS-001)
notes := 'Under DRS threat: deploy ERS to defend the straights.'  (rule R-ERS-001)
fuel_advice := 'lift_and_coast'  (rule R-FUEL-001)
push_advice := 'conserve'  (rule R-FUEL-001)
risk_level := 'medium'  (rule R-FUEL-001)
notes := 'Under DRS threat: deploy ERS to defend the straights. | Under fuel target: save via lift-and-coast zones.'  (rule R-FUEL-001)
defend_advice := 'defensive_lines'  (rule R-TAC-003)
notes := 'Under DRS threat: deploy ERS to defend the straights. | Under fuel target: save via lift-and-coast zones. | Worn tyres under pressure: defend, protect braking zones.'  (rule R-TAC-003)
```

