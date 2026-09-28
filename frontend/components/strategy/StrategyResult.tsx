"use client";

import { StrategyResponse } from "@/lib/api";
import { TyreBadge } from "@/components/TyreBadge";
import { fmt } from "@/lib/format";

/**
 * One recommendation, from all three reasoning paradigms: the ML prediction, the
 * A* search cost, and which expert rules fired. The extrapolation warning comes
 * first on purpose — a prediction from outside the training range should be read
 * with that in mind, not after.
 *
 * It reads as one column, not a grid of equal cards: the action is the headline,
 * the two model outputs sit under it, and the provenance is at the bottom.
 */
export function StrategyResult({
  result,
  showInputs,
  setShowInputs,
}: {
  result: StrategyResponse;
  showInputs: boolean;
  setShowInputs: (v: boolean) => void;
}) {
  const p = result.prediction;
  const regContextOnly = p.context_only["target_laptime"] ?? {};
  const clfContextOnly = p.context_only["target_pit_next_lap"] ?? {};
  const outOfRange = Array.from(
    new Map(
      [
        ...(p.out_of_range["target_laptime"] ?? []),
        ...(p.out_of_range["target_pit_next_lap"] ?? []),
      ].map((o) => [o.feature, o]),
    ).values(),
  );

  const next = result.optimal_search_strategy.next_action;
  // The verdict comes from the Task 2 rule base, and only falls back to the
  // classifier when no rule fired. Saying which one spoke matters more than the
  // word itself, because the Task 3 planner answers a different question below.
  const fromRules = result.triggered_expert_rules.length > 0;
  const ACTION: Record<string, string> = {
    PIT_NOW: "Pit now",
    STAY_OUT: "Stay out",
  };
  const action = result.recommended_action;
  const contextNotes = [
    regContextOnly.driver && "driver (lap time)",
    regContextOnly.team && "team (lap time)",
    clfContextOnly.driver && "driver (pit decision)",
    clfContextOnly.team && "team (pit decision)",
  ].filter(Boolean) as string[];

  return (
    <div className="space-y-4" aria-live="polite">
      {outOfRange.length > 0 && (
        <div className="card border-accent" role="note">
          <span className="badge badge-warning">Extrapolating beyond training data</span>
          <p className="t-body mt-2.5 text-[13px]">
            At least one feature is outside the range the model was trained on. Its behaviour out there is
            unvalidated and can be arbitrary.
          </p>
          <ul className="mt-2.5 space-y-1 text-[12px] text-paper-500">
            {outOfRange.map((o) => (
              <li key={o.feature} className="tabular-nums">
                <span className="t-code text-paper-700">{o.feature}</span> = {fmt(o.value, 2)}, trained on{" "}
                {fmt(o.training_min, 2)}–{fmt(o.training_max, 2)}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="card card-focal">
        <p className="stat-label">
          Recommended action, {fromRules ? "Task 2 expert rules" : "Task 6 pit classifier"}
        </p>
        <p className="t-display-md mt-1">{action ? (ACTION[action] ?? action) : "—"}</p>
        <p className="t-micro mt-2.5 max-w-[42ch]">
          The Task 3 {result.optimal_search_strategy.algorithm} planner answers a different question — the cheapest
          remaining race, not this lap — and its first move is{" "}
          {next?.type === "PIT" ? (
            <>
              to pit for <TyreBadge compound={next.compound} />
            </>
          ) : (
            "to stay out"
          )}
          . Where the two disagree, they are disagreeing about the horizon.
        </p>

        <dl className="mt-5 grid grid-cols-2 gap-x-6 gap-y-5 border-t border-track-300 pt-5">
          <div>
            <dt className="stat-label">Predicted lap time</dt>
            <dd className="stat-value">
              {p.predicted_lap_time_seconds !== null ? fmt(p.predicted_lap_time_seconds, 3) : "—"}
              <span className="unit"> s</span>
            </dd>
            <p className="t-micro mt-1">{p.laptime_model}</p>
          </div>
          <div>
            <dt className="stat-label">Pit probability</dt>
            <dd className="stat-value">
              {p.probability_pit !== null ? fmt(p.probability_pit * 100, 1) : "—"}
              <span className="unit"> %</span>
            </dd>
            <p className="t-micro mt-1">{p.pit_model}</p>
          </div>
          <div>
            <dt className="stat-label">Expected cost, remaining stint</dt>
            <dd className="stat-value">
              {result.expected_cost_seconds !== null ? fmt(result.expected_cost_seconds, 1) : "—"}
              <span className="unit"> s</span>
            </dd>
          </div>
          <div>
            <dt className="stat-label">Expert rules fired</dt>
            <dd className="stat-value tabular-nums">{result.triggered_expert_rules.length}</dd>
          </div>
        </dl>

        {contextNotes.length > 0 && (
          <p className="t-micro mt-4 border-t border-track-300 pt-3">
            Recorded as context, not used by these models: {contextNotes.join(", ")}.
          </p>
        )}
      </div>

      <div className="card">
        <h3 className="t-title text-[15px]">Triggered expert rules</h3>
        {result.triggered_expert_rules.length === 0 ? (
          <p className="t-body mt-2 text-[13px] text-paper-500">No rules fired for this race state.</p>
        ) : (
          <ul className="mt-2.5 space-y-1.5 text-[13px] text-paper-700">
            {result.triggered_expert_rules.map((r) => (
              <li key={r.rule_id}>
                <span className="t-code text-[12px] text-paper-900">{r.rule_id}</span> {r.name}
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="card">
        <button
          type="button"
          onClick={() => setShowInputs(!showInputs)}
          aria-expanded={showInputs}
          aria-controls="feature-rows"
          className="text-[13px] font-medium text-paper-900 underline decoration-edge underline-offset-[3px] hover:decoration-paper-900"
        >
          {showInputs ? "Hide" : "Show"} the features sent to each model
        </button>
        {showInputs && (
          <div id="feature-rows" className="mt-3.5 space-y-3.5">
            {(["target_laptime", "target_pit_next_lap"] as const).map((target) => (
              <div key={target}>
                <p className="stat-label">
                  {target === "target_laptime" ? "Lap-time" : "Pit-decision"} model inputs
                </p>
                <pre className="t-code mt-1.5 max-h-[220px] overflow-auto rounded-sm bg-track-050 p-2.5 text-[11px] leading-[1.5] text-paper-700">
                  {JSON.stringify(p.feature_rows[target], null, 2)}
                </pre>
                {p.approximated_features[target]?.length > 0 && (
                  <p className="t-micro mt-1.5">
                    Filled from training-data medians, since they need multi-lap history this snapshot cannot
                    supply: {p.approximated_features[target].join(", ")}
                  </p>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
