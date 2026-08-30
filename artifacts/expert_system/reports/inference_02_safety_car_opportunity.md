# Inference Report — Safety car opportunity

_Generated 2026-08-30 11:23 UTC._

## Inputs (GIVEN facts)

| Fact | Value |
|------|-------|
| `track_status` | 'SC' |
| `tyre_wear` | 52 |
| `in_pit_window` | True |
| `current_position` | 5 |
| `laps_remaining` | 30 |
| `weather_severity` | 'dry' |

## Firing sequence

| # | Iter | Rule | Name | Asserted |
|---|------|------|------|----------|
| 1 | 1 | `R-SC-001` | Safety car deployed + worn tyres -> pit now | pit_decision='PIT_NOW', notes='SC out: cheap pit stop, pit immediately.' |

## Conclusions (derived decisions)

| Decision | Value |
|----------|-------|
| `pit_decision` | 'PIT_NOW' |
| `notes` | 'SC out: cheap pit stop, pit immediately.' |

## WHY — narrative explanation

```
The engine fired 1 rule(s) over 2 iteration(s) and reached a stable conclusion.
  [iter 1] R-SC-001 «Safety car deployed + worn tyres -> pit now» fired, asserting pit_decision='PIT_NOW', notes='SC out: cheap pit stop, pit immediately.' — Under a safety car the pit-loss shrinks dramatically; stopping with worn tyres is near free time.
Final decisions:
  • pit_decision = 'PIT_NOW'
  • notes = 'SC out: cheap pit stop, pit immediately.'
```

## HOW — per-decision justification

**`pit_decision` = 'PIT_NOW'**  — via R-SC-001 «Safety car deployed + worn tyres -> pit now»
  - because `track_status == 'SC'`
  - because `tyre_wear >= 40`

**`notes` = 'SC out: cheap pit stop, pit immediately.'**  — via R-SC-001 «Safety car deployed + worn tyres -> pit now»
  - because `track_status == 'SC'`
  - because `tyre_wear >= 40`

## Audit trail

```
track_status := 'SC'  (input)
tyre_wear := 52  (input)
in_pit_window := True  (input)
current_position := 5  (input)
laps_remaining := 30  (input)
weather_severity := 'dry'  (input)
pit_decision := 'PIT_NOW'  (rule R-SC-001)
notes := 'SC out: cheap pit stop, pit immediately.'  (rule R-SC-001)
```

