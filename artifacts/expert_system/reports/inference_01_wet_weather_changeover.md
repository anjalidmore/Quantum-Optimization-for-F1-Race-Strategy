# Inference Report — Wet weather changeover

_Generated 2026-09-05 01:24 UTC._

## Inputs (GIVEN facts)

| Fact | Value |
|------|-------|
| `rain_probability` | 85 |
| `track_wet` | True |
| `current_compound` | 'SOFT' |
| `tyre_wear` | 30 |
| `weather_severity` | 'wet' |
| `track_status` | 'GREEN' |
| `current_lap` | 12 |
| `total_laps` | 57 |
| `laps_remaining` | 45 |

## Firing sequence

| # | Iter | Rule | Name | Asserted |
|---|------|------|------|----------|
| 1 | 1 | `R-WX-002` | Track already wet -> intermediates minimum | recommended_tyre='INTERMEDIATE', push_advice='conserve' |
| 2 | 1 | `R-WX-001` | Heavy rain forecast -> wet tyres | risk_level='high', notes='High rain probability: switch to intermediates.' |

## Conclusions (derived decisions)

| Decision | Value |
|----------|-------|
| `recommended_tyre` | 'INTERMEDIATE' |
| `push_advice` | 'conserve' |
| `risk_level` | 'high' |
| `notes` | 'High rain probability: switch to intermediates.' |

## WHY — narrative explanation

```
The engine fired 2 rule(s) over 2 iteration(s) and reached a stable conclusion.
  [iter 1] R-WX-002 «Track already wet -> intermediates minimum» fired, asserting recommended_tyre='INTERMEDIATE', push_advice='conserve' — A wet track with moderate rain calls for intermediates and reduced pace until conditions clarify.
  [iter 1] R-WX-001 «Heavy rain forecast -> wet tyres» fired, asserting risk_level='high', notes='High rain probability: switch to intermediates.' — If rain probability exceeds 70%, intermediates are the safe baseline; grip on slicks collapses once the track wets.
Final decisions:
  • recommended_tyre = 'INTERMEDIATE'
  • push_advice = 'conserve'
  • risk_level = 'high'
  • notes = 'High rain probability: switch to intermediates.'
```

## HOW — per-decision justification

**`recommended_tyre` = 'INTERMEDIATE'**  — via R-WX-002 «Track already wet -> intermediates minimum»
  - because `track_wet is_true`
  - because `rain_probability <= 90`

**`push_advice` = 'conserve'**  — via R-WX-002 «Track already wet -> intermediates minimum»
  - because `track_wet is_true`
  - because `rain_probability <= 90`

**`risk_level` = 'high'**  — via R-WX-001 «Heavy rain forecast -> wet tyres»
  - because `rain_probability > 70`

**`notes` = 'High rain probability: switch to intermediates.'**  — via R-WX-001 «Heavy rain forecast -> wet tyres»
  - because `rain_probability > 70`

## Audit trail

```
rain_probability := 85  (input)
track_wet := True  (input)
current_compound := 'SOFT'  (input)
tyre_wear := 30  (input)
weather_severity := 'wet'  (input)
track_status := 'GREEN'  (input)
current_lap := 12  (input)
total_laps := 57  (input)
laps_remaining := 45  (input)
recommended_tyre := 'INTERMEDIATE'  (rule R-WX-002)
push_advice := 'conserve'  (rule R-WX-002)
risk_level := 'high'  (rule R-WX-001)
notes := 'High rain probability: switch to intermediates.'  (rule R-WX-001)
```

