# Rule Catalogue

_Generated 2026-09-05 06:37 UTC — 32 rules across 10 categories._

Salience convention: 100 weather-critical · 80 safety-car · 60 pit/deg · 40 tyre · 20-30 tactical · 0 advisory.

## Category: `degradation` (3 rules)

### R-DEG-001 — High track temp -> increased degradation

- **Salience:** 60 · **Specificity:** 1 · **Connective:** all
- **Logic:** `IF track_temperature > 40 THEN set tyre_deg_adjustment = 'increase'; set notes = 'Hot track: expect elevated thermal degradation.'`
- **Rationale:** Track temperatures above ~40°C accelerate thermal degradation, shortening viable stint length.

### R-DEG-002 — Cool track -> reduced degradation, extend stint

- **Salience:** 60 · **Specificity:** 2 · **Connective:** all
- **Logic:** `IF track_temperature < 25 AND tyre_wear < 60 THEN set tyre_deg_adjustment = 'decrease'; set pit_decision = 'STAY_OUT'; set notes = 'Cool track favours longer stints.'`
- **Rationale:** Lower surface temperatures reduce degradation, so a longer stint (fewer stops) becomes viable.

### R-DEG-003 — High graining risk + high wear -> pit soon

- **Salience:** 60 · **Specificity:** 2 · **Connective:** all
- **Logic:** `IF graining_risk == 'high' AND tyre_wear >= 50 THEN set pit_decision = 'PIT_SOON'; set push_advice = 'conserve'; set notes = 'Manage graining: reduce sliding, plan an earlier stop.'`
- **Rationale:** Graining scrubs grip nonlinearly; conserving and stopping earlier avoids a lap-time collapse.

## Category: `energy` (1 rules)

### R-ERS-001 — Defending with DRS train -> deploy ERS on straights

- **Salience:** 30 · **Specificity:** 2 · **Connective:** all
- **Logic:** `IF gap_behind < 1.0 AND drs_enabled is_true THEN set engine_mode_advice = 'deploy_on_straights'; set defend_advice = 'cover_inside_line'; set notes = 'Under DRS threat: deploy ERS to defend the straights.'`
- **Rationale:** When a rival is within DRS range, biasing ERS deployment to the straights protects against a slipstream pass.

## Category: `fuel` (2 rules)

### R-FUEL-001 — Fuel short -> lift and coast

- **Salience:** 30 · **Specificity:** 1 · **Connective:** all
- **Logic:** `IF fuel_margin < 0 THEN set fuel_advice = 'lift_and_coast'; set push_advice = 'conserve'; set risk_level = 'medium'; set notes = 'Under fuel target: save via lift-and-coast zones.'`
- **Rationale:** A negative fuel margin means the car will not reach the flag at full pace; lift-and-coast recovers margin.

### R-FUEL-002 — Comfortable fuel + soft tyres -> push

- **Salience:** 30 · **Specificity:** 2 · **Connective:** all
- **Logic:** `IF fuel_margin > 1.5 AND current_compound == 'SOFT' THEN set fuel_advice = 'normal'; set push_advice = 'push'`
- **Rationale:** With fuel in hand and grippy tyres, converting the surplus into pace is worthwhile.

## Category: `pit` (4 rules)

### R-PIT-001 — Critical tyre wear -> pit now

- **Salience:** 60 · **Specificity:** 1 · **Connective:** all
- **Logic:** `IF tyre_wear >= 80 THEN set pit_decision = 'PIT_NOW'; set risk_level = 'high'`
- **Rationale:** Beyond ~80% wear, lap-time loss and failure risk rise sharply; stop regardless of window.

### R-PIT-002 — In window + undercut threat -> pit now

- **Salience:** 60 · **Specificity:** 2 · **Connective:** all
- **Logic:** `IF in_pit_window is_true AND undercut_threat is_true THEN set pit_decision = 'PIT_NOW'; set notes = 'Cover the undercut before the rival gains free air.'`
- **Rationale:** Reacting to an undercut requires pitting on the same lap or the fresh-tyre advantage flips positions.

### R-PIT-003 — Overcut opportunity + clear air -> extend stint

- **Salience:** 60 · **Specificity:** 3 · **Connective:** all
- **Logic:** `IF overcut_opportunity is_true AND gap_ahead > 2.0 AND tyre_wear < 65 THEN set pit_decision = 'STAY_OUT'; set push_advice = 'push'; set notes = 'Overcut: push in clear air, pit later than rival.'`
- **Rationale:** With serviceable tyres and clear air, staying out and pushing can leapfrog a rival who stops early.

### R-PIT-004 — Approaching window edge -> prime pit crew

- **Salience:** 60 · **Specificity:** 3 · **Connective:** all
- **Logic:** `IF in_pit_window is_true AND tyre_wear >= 55 AND tyre_wear < 80 THEN set pit_decision = 'PIT_SOON'; set notes = 'Within window and wearing: prepare to stop.'`
- **Rationale:** Signals the crew to be ready without committing, keeping flexibility for a safety car.

## Category: `risk` (2 rules)

### R-RISK-001 — Multiple high-risk factors -> flag high risk

- **Salience:** 0 · **Specificity:** 2 · **Connective:** all
- **Logic:** `IF tyre_wear >= 75 AND rain_probability > 50 THEN set risk_level = 'high'; set notes = 'Worn tyres and rain risk compound: high uncertainty.'`
- **Rationale:** Concurrent worn tyres and rain risk multiply strategic uncertainty and warrant a conservative posture.

### R-RISK-002 — Stable dry conditions mid-race -> low risk baseline

- **Salience:** 0 · **Specificity:** 3 · **Connective:** all
- **Logic:** `IF weather_severity == 'dry' AND track_status == 'GREEN' AND tyre_wear < 50 THEN set risk_level = 'low'`
- **Rationale:** Green-flag dry running on healthy tyres is the low-risk baseline against which deviations are judged.

## Category: `safety_car` (4 rules)

### R-SC-001 — Safety car deployed + worn tyres -> pit now

- **Salience:** 80 · **Specificity:** 2 · **Connective:** all
- **Logic:** `IF track_status == 'SC' AND tyre_wear >= 40 THEN set pit_decision = 'PIT_NOW'; set notes = 'SC out: cheap pit stop, pit immediately.'`
- **Rationale:** Under a safety car the pit-loss shrinks dramatically; stopping with worn tyres is near free time.

### R-SC-002 — VSC + in pit window -> pit now

- **Salience:** 80 · **Specificity:** 2 · **Connective:** all
- **Logic:** `IF track_status == 'VSC' AND in_pit_window is_true THEN set pit_decision = 'PIT_NOW'; set notes = 'VSC reduces pit loss: take the stop within the window.'`
- **Rationale:** A virtual safety car slows the whole field, cutting the relative time lost in the pit lane.

### R-SC-003 — High SC probability + marginal tyres -> delay pit

- **Salience:** 80 · **Specificity:** 3 · **Connective:** all
- **Logic:** `IF safety_car_probability >= 60 AND tyre_wear < 55 AND track_status == 'GREEN' THEN set pit_decision = 'DELAY_PIT'; set notes = 'Bank on an imminent SC for a cheaper stop.'`
- **Rationale:** When a safety car looks likely and tyres are not yet critical, delaying can convert a full-price stop into a discounted one.

### R-SC-004 — SC in final laps + leading -> stay out to keep track position

- **Salience:** 80 · **Specificity:** 3 · **Connective:** all
- **Logic:** `IF track_status == 'SC' AND laps_remaining <= 8 AND current_position <= 3 THEN set pit_decision = 'STAY_OUT'; set defend_advice = 'control_restart'; set notes = 'Track position outweighs tyre delta this late.'`
- **Rationale:** Late in a race, losing track position to a pit stop is rarely recoverable; hold position and manage the restart.

## Category: `strategy` (4 rules)

### R-STRAT-001 — Monaco + front row -> one stop

- **Salience:** 60 · **Specificity:** 2 · **Connective:** all
- **Logic:** `IF circuit == 'Monaco' AND grid_position <= 3 THEN set strategy_stops = 1; set notes = 'Monaco: track position is king, minimise stops.'`
- **Rationale:** Overtaking at Monaco is exceptionally hard, so protecting track position with a single stop is standard.

### R-STRAT-002 — High overtaking difficulty -> minimise stops

- **Salience:** 40 · **Specificity:** 1 · **Connective:** all
- **Logic:** `IF overtaking_difficulty == 'high' THEN set strategy_stops = 1; set defend_advice = 'protect_position'`
- **Rationale:** Where passing is hard, every pit stop risks a position that cannot be won back on track.

### R-STRAT-003 — High deg circuit + low pit loss -> two stops

- **Salience:** 40 · **Specificity:** 3 · **Connective:** all
- **Logic:** `IF track_temperature > 40 AND pit_loss < 20 AND overtaking_difficulty != 'high' THEN set strategy_stops = 2; set push_advice = 'push'; set notes = 'Cheap stops + high deg reward an aggressive 2-stop.'`
- **Rationale:** When pit loss is small and degradation high, a two-stop on fresher tyres beats nursing one set.

### R-STRAT-004 — Low SC likelihood + long stint -> plan fixed strategy

- **Salience:** 30 · **Specificity:** 2 · **Connective:** all
- **Logic:** `IF safety_car_likelihood == 'low' AND total_laps > 50 THEN set notes = 'Low SC prior: commit to a planned strategy, fewer reactive stops.'`
- **Rationale:** With little chance of a race-altering SC, a pre-planned strategy usually outperforms reactive gambling.

## Category: `tactics` (3 rules)

### R-TAC-001 — Close behind + faster tyres -> attack

- **Salience:** 20 · **Specificity:** 3 · **Connective:** all
- **Logic:** `IF gap_ahead < 1.0 AND tyre_age_laps < 10 AND drs_enabled is_true THEN set push_advice = 'push'; set notes = 'Fresh tyres + DRS: attack the car ahead.'`
- **Rationale:** A fresher-tyred car within DRS range should press the advantage before the tyre delta erodes.

### R-TAC-002 — Big gap both ways -> manage tyres in clear air

- **Salience:** 20 · **Specificity:** 2 · **Connective:** all
- **Logic:** `IF gap_ahead > 3.0 AND gap_behind > 3.0 THEN set push_advice = 'conserve'; set notes = 'Clear air: manage tyres, no need to push.'`
- **Rationale:** With no immediate threat or target, conserving tyres extends strategic options later.

### R-TAC-003 — Under pressure + worn tyres -> defensive lines

- **Salience:** 20 · **Specificity:** 2 · **Connective:** all
- **Logic:** `IF gap_behind < 1.5 AND tyre_wear >= 60 THEN set defend_advice = 'defensive_lines'; set push_advice = 'conserve'; set notes = 'Worn tyres under pressure: defend, protect braking zones.'`
- **Rationale:** Worn tyres cannot match a fresher pursuer's pace, so positioning and braking-zone control matter most.

## Category: `tyre` (5 rules)

### R-TYRE-001 — Soft worn -> step to medium

- **Salience:** 40 · **Specificity:** 2 · **Connective:** all
- **Logic:** `IF current_compound == 'SOFT' AND tyre_wear >= 60 THEN set recommended_tyre = 'MEDIUM'`
- **Rationale:** When softs degrade, mediums trade a little peak grip for durability over the next stint.

### R-TYRE-002 — Long final stint -> hard compound

- **Salience:** 40 · **Specificity:** 2 · **Connective:** all
- **Logic:** `IF laps_remaining > 30 AND track_wet is_false THEN set recommended_tyre = 'HARD'`
- **Rationale:** A long run to the flag favours the hard compound's durability, avoiding a further stop.

### R-TYRE-003 — Short final stint + dry -> soft for pace

- **Salience:** 40 · **Specificity:** 3 · **Connective:** all
- **Logic:** `IF laps_remaining <= 15 AND track_wet is_false AND tyre_wear >= 40 THEN set recommended_tyre = 'SOFT'; set push_advice = 'push'`
- **Rationale:** For a short sprint to the flag, the soft's peak grip outweighs its lower durability.

### R-TYRE-004 — Damp but not wet -> intermediates over slicks

- **Salience:** 40 · **Specificity:** 1 · **Connective:** all
- **Logic:** `IF weather_severity == 'damp' THEN set recommended_tyre = 'INTERMEDIATE'; set push_advice = 'conserve'`
- **Rationale:** A damp track lacks the grip for slicks but not the water for full wets; intermediates bridge the gap.

### R-TYRE-005 — Medium mid-race baseline in the heat

- **Salience:** 40 · **Specificity:** 3 · **Connective:** all
- **Logic:** `IF current_compound == 'SOFT' AND track_temperature > 45 AND laps_remaining > 20 THEN set recommended_tyre = 'MEDIUM'; set notes = 'Very hot track: avoid softs for a long middle stint.'`
- **Rationale:** In extreme heat, softs overheat quickly; mediums give a more stable operating window for a long stint.

## Category: `weather` (4 rules)

### R-WX-001 — Heavy rain forecast -> wet tyres

- **Salience:** 100 · **Specificity:** 1 · **Connective:** all
- **Logic:** `IF rain_probability > 70 THEN set recommended_tyre = 'INTERMEDIATE'; set risk_level = 'high'; set notes = 'High rain probability: switch to intermediates.'`
- **Rationale:** If rain probability exceeds 70%, intermediates are the safe baseline; grip on slicks collapses once the track wets.

### R-WX-002 — Track already wet -> intermediates minimum

- **Salience:** 100 · **Specificity:** 2 · **Connective:** all
- **Logic:** `IF track_wet is_true AND rain_probability <= 90 THEN set recommended_tyre = 'INTERMEDIATE'; set push_advice = 'conserve'`
- **Rationale:** A wet track with moderate rain calls for intermediates and reduced pace until conditions clarify.

### R-WX-003 — Standing water / extreme -> full wets

- **Salience:** 100 · **Specificity:** 1 · **Connective:** all
- **Logic:** `IF weather_severity == 'extreme' THEN set recommended_tyre = 'WET'; set risk_level = 'high'; set push_advice = 'conserve'; set notes = 'Extreme conditions: full wet tyres, expect SC/red flag.'`
- **Rationale:** Standing water demands full wet tyres for aquaplaning resistance; expect race-control intervention.

### R-WX-004 — Drying track after rain -> prepare slick changeover

- **Salience:** 100 · **Specificity:** 3 · **Connective:** all
- **Logic:** `IF current_compound in ('INTERMEDIATE', 'WET') AND track_wet is_false AND rain_probability < 30 THEN set pit_decision = 'PIT_SOON'; set recommended_tyre = 'SOFT'; set notes = 'Track drying: crossover to slicks approaching.'`
- **Rationale:** On a drying line, the intermediate 'cliff' arrives fast; plan the crossover to slicks before losing seconds/lap.

