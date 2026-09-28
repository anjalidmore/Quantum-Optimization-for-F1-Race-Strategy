# Backward-Chaining Demonstration

Goal: prove `pit_decision == 'PIT_NOW'` for the safety-car scenario.

```
[proven] pit_decision == 'PIT_NOW'  [rule R-SC-001]
  [proven] track_status == 'SC'  [given]
  [proven] tyre_wear >= 40  [given]
```
