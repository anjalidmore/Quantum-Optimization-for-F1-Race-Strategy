# Rule Tree

_Generated 2026-09-27 17:42 UTC — 32 rules across 10 categories._

Read it as: **category → rule (salience) → conditions that must hold → what it asserts**.
Higher salience fires first when several rules match.

```text
RULE BASE
├── degradation  (3 rules)
│   ├── R-DEG-001  salience 60  [ALL]
│   │   │   IF   track_temperature > 40
│   │   ├── THEN tyre_deg_adjustment = 'increase'
│   │   └── THEN notes = 'Hot track: expect elevated thermal degradation.'
│   ├── R-DEG-002  salience 60  [ALL]
│   │   │   IF   track_temperature < 25
│   │   │   IF   tyre_wear < 60
│   │   ├── THEN tyre_deg_adjustment = 'decrease'
│   │   ├── THEN pit_decision = 'STAY_OUT'
│   │   └── THEN notes = 'Cool track favours longer stints.'
│   └── R-DEG-003  salience 60  [ALL]
│       │   IF   graining_risk == 'high'
│       │   IF   tyre_wear >= 50
│       ├── THEN pit_decision = 'PIT_SOON'
│       ├── THEN push_advice = 'conserve'
│       └── THEN notes = 'Manage graining: reduce sliding, plan an earlier stop.'
├── energy  (1 rules)
│   └── R-ERS-001  salience 30  [ALL]
│       │   IF   gap_behind < 1.0
│       │   IF   drs_enabled is_true
│       ├── THEN engine_mode_advice = 'deploy_on_straights'
│       ├── THEN defend_advice = 'cover_inside_line'
│       └── THEN notes = 'Under DRS threat: deploy ERS to defend the straights.'
├── fuel  (2 rules)
│   ├── R-FUEL-001  salience 30  [ALL]
│   │   │   IF   fuel_margin < 0
│   │   ├── THEN fuel_advice = 'lift_and_coast'
│   │   ├── THEN push_advice = 'conserve'
│   │   ├── THEN risk_level = 'medium'
│   │   └── THEN notes = 'Under fuel target: save via lift-and-coast zones.'
│   └── R-FUEL-002  salience 30  [ALL]
│       │   IF   fuel_margin > 1.5
│       │   IF   current_compound == 'SOFT'
│       ├── THEN fuel_advice = 'normal'
│       └── THEN push_advice = 'push'
├── pit  (4 rules)
│   ├── R-PIT-001  salience 60  [ALL]
│   │   │   IF   tyre_wear >= 80
│   │   ├── THEN pit_decision = 'PIT_NOW'
│   │   └── THEN risk_level = 'high'
│   ├── R-PIT-002  salience 60  [ALL]
│   │   │   IF   in_pit_window is_true
│   │   │   IF   undercut_threat is_true
│   │   ├── THEN pit_decision = 'PIT_NOW'
│   │   └── THEN notes = 'Cover the undercut before the rival gains free air.'
│   ├── R-PIT-003  salience 60  [ALL]
│   │   │   IF   overcut_opportunity is_true
│   │   │   IF   gap_ahead > 2.0
│   │   │   IF   tyre_wear < 65
│   │   ├── THEN pit_decision = 'STAY_OUT'
│   │   ├── THEN push_advice = 'push'
│   │   └── THEN notes = 'Overcut: push in clear air, pit later than rival.'
│   └── R-PIT-004  salience 60  [ALL]
│       │   IF   in_pit_window is_true
│       │   IF   tyre_wear >= 55
│       │   IF   tyre_wear < 80
│       ├── THEN pit_decision = 'PIT_SOON'
│       └── THEN notes = 'Within window and wearing: prepare to stop.'
├── risk  (2 rules)
│   ├── R-RISK-001  salience 0  [ALL]
│   │   │   IF   tyre_wear >= 75
│   │   │   IF   rain_probability > 50
│   │   ├── THEN risk_level = 'high'
│   │   └── THEN notes = 'Worn tyres and rain risk compound: high uncertainty.'
│   └── R-RISK-002  salience 0  [ALL]
│       │   IF   weather_severity == 'dry'
│       │   IF   track_status == 'GREEN'
│       │   IF   tyre_wear < 50
│       └── THEN risk_level = 'low'
├── safety_car  (4 rules)
│   ├── R-SC-001  salience 80  [ALL]
│   │   │   IF   track_status == 'SC'
│   │   │   IF   tyre_wear >= 40
│   │   ├── THEN pit_decision = 'PIT_NOW'
│   │   └── THEN notes = 'SC out: cheap pit stop, pit immediately.'
│   ├── R-SC-002  salience 80  [ALL]
│   │   │   IF   track_status == 'VSC'
│   │   │   IF   in_pit_window is_true
│   │   ├── THEN pit_decision = 'PIT_NOW'
│   │   └── THEN notes = 'VSC reduces pit loss: take the stop within the window.'
│   ├── R-SC-003  salience 80  [ALL]
│   │   │   IF   safety_car_probability >= 60
│   │   │   IF   tyre_wear < 55
│   │   │   IF   track_status == 'GREEN'
│   │   ├── THEN pit_decision = 'DELAY_PIT'
│   │   └── THEN notes = 'Bank on an imminent SC for a cheaper stop.'
│   └── R-SC-004  salience 80  [ALL]
│       │   IF   track_status == 'SC'
│       │   IF   laps_remaining <= 8
│       │   IF   current_position <= 3
│       ├── THEN pit_decision = 'STAY_OUT'
│       ├── THEN defend_advice = 'control_restart'
│       └── THEN notes = 'Track position outweighs tyre delta this late.'
├── strategy  (4 rules)
│   ├── R-STRAT-001  salience 60  [ALL]
│   │   │   IF   circuit == 'Monaco'
│   │   │   IF   grid_position <= 3
│   │   ├── THEN strategy_stops = 1
│   │   └── THEN notes = 'Monaco: track position is king, minimise stops.'
│   ├── R-STRAT-002  salience 40  [ALL]
│   │   │   IF   overtaking_difficulty == 'high'
│   │   ├── THEN strategy_stops = 1
│   │   └── THEN defend_advice = 'protect_position'
│   ├── R-STRAT-003  salience 40  [ALL]
│   │   │   IF   track_temperature > 40
│   │   │   IF   pit_loss < 20
│   │   │   IF   overtaking_difficulty != 'high'
│   │   ├── THEN strategy_stops = 2
│   │   ├── THEN push_advice = 'push'
│   │   └── THEN notes = 'Cheap stops + high deg reward an aggressive 2-stop.'
│   └── R-STRAT-004  salience 30  [ALL]
│       │   IF   safety_car_likelihood == 'low'
│       │   IF   total_laps > 50
│       └── THEN notes = 'Low SC prior: commit to a planned strategy, fewer reactive stops.'
├── tactics  (3 rules)
│   ├── R-TAC-001  salience 20  [ALL]
│   │   │   IF   gap_ahead < 1.0
│   │   │   IF   tyre_age_laps < 10
│   │   │   IF   drs_enabled is_true
│   │   ├── THEN push_advice = 'push'
│   │   └── THEN notes = 'Fresh tyres + DRS: attack the car ahead.'
│   ├── R-TAC-002  salience 20  [ALL]
│   │   │   IF   gap_ahead > 3.0
│   │   │   IF   gap_behind > 3.0
│   │   ├── THEN push_advice = 'conserve'
│   │   └── THEN notes = 'Clear air: manage tyres, no need to push.'
│   └── R-TAC-003  salience 20  [ALL]
│       │   IF   gap_behind < 1.5
│       │   IF   tyre_wear >= 60
│       ├── THEN defend_advice = 'defensive_lines'
│       ├── THEN push_advice = 'conserve'
│       └── THEN notes = 'Worn tyres under pressure: defend, protect braking zones.'
├── tyre  (5 rules)
│   ├── R-TYRE-001  salience 40  [ALL]
│   │   │   IF   current_compound == 'SOFT'
│   │   │   IF   tyre_wear >= 60
│   │   └── THEN recommended_tyre = 'MEDIUM'
│   ├── R-TYRE-002  salience 40  [ALL]
│   │   │   IF   laps_remaining > 30
│   │   │   IF   track_wet is_false
│   │   └── THEN recommended_tyre = 'HARD'
│   ├── R-TYRE-003  salience 40  [ALL]
│   │   │   IF   laps_remaining <= 15
│   │   │   IF   track_wet is_false
│   │   │   IF   tyre_wear >= 40
│   │   ├── THEN recommended_tyre = 'SOFT'
│   │   └── THEN push_advice = 'push'
│   ├── R-TYRE-004  salience 40  [ALL]
│   │   │   IF   weather_severity == 'damp'
│   │   ├── THEN recommended_tyre = 'INTERMEDIATE'
│   │   └── THEN push_advice = 'conserve'
│   └── R-TYRE-005  salience 40  [ALL]
│       │   IF   current_compound == 'SOFT'
│       │   IF   track_temperature > 45
│       │   IF   laps_remaining > 20
│       ├── THEN recommended_tyre = 'MEDIUM'
│       └── THEN notes = 'Very hot track: avoid softs for a long middle stint.'
└── weather  (4 rules)
    ├── R-WX-001  salience 100  [ALL]
    │   │   IF   rain_probability > 70
    │   ├── THEN recommended_tyre = 'INTERMEDIATE'
    │   ├── THEN risk_level = 'high'
    │   └── THEN notes = 'High rain probability: switch to intermediates.'
    ├── R-WX-002  salience 100  [ALL]
    │   │   IF   track_wet is_true
    │   │   IF   rain_probability <= 90
    │   ├── THEN recommended_tyre = 'INTERMEDIATE'
    │   └── THEN push_advice = 'conserve'
    ├── R-WX-003  salience 100  [ALL]
    │   │   IF   weather_severity == 'extreme'
    │   ├── THEN recommended_tyre = 'WET'
    │   ├── THEN risk_level = 'high'
    │   ├── THEN push_advice = 'conserve'
    │   └── THEN notes = 'Extreme conditions: full wet tyres, expect SC/red flag.'
    └── R-WX-004  salience 100  [ALL]
        │   IF   current_compound in ('INTERMEDIATE', 'WET')
        │   IF   track_wet is_false
        │   IF   rain_probability < 30
        ├── THEN pit_decision = 'PIT_SOON'
        ├── THEN recommended_tyre = 'SOFT'
        └── THEN notes = 'Track drying: crossover to slicks approaching.'
```

