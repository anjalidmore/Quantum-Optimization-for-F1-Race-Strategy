"use client";

import { StrategyResponse } from "@/lib/api";

/**
 * One recommendation, from all three reasoning paradigms: the ML prediction, the
 * A* search cost, and which expert rules fired. The extrapolation warning comes
 * first on purpose — a prediction from outside the training range should be read
 * with that in mind, not after.
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
  const outOfRange = [
    ...(p.out_of_range["target_laptime"] ?? []),
    ...(p.out_of_range["target_pit_next_lap"] ?? []),
  ];

  return (
    <div className="space-y-4">
      {outOfRange.length > 0 && (
        <div className="card border-amber-500/30 bg-amber-500/5">
          <div className="badge badge-warning mb-2">Extrapolating beyond training data</div>
          <p className="text-sm text-white/70">
            These inputs push at least one model feature outside the range the model was actually trained on.
            Its behaviour out there is unvalidated and can be arbitrary — treat this prediction with caution,
            not as a reliable answer.
          </p>
          <ul className="text-xs text-white/50 mt-2 space-y-1">
            {Array.from(new Map(outOfRange.map((o) => [o.feature, o])).values()).map((o) => (
              <li key={o.feature}>
                <span className="text-white/70 font-mono">{o.feature}</span> = {o.value.toFixed(2)} (trained on{" "}
                {o.training_min.toFixed(2)} – {o.training_max.toFixed(2)})
              </li>
            ))}
          </ul>
        </div>
      )}

      <div>
        <div className="text-xs uppercase tracking-wider text-white/40 mb-2">
          Machine Learning Prediction — what is likely to happen?
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="card">
            <div className="stat-label">Predicted lap time</div>
            <div className="stat-value">{p.predicted_lap_time_seconds?.toFixed(3) ?? "—"}s</div>
            <div className="text-xs text-white/40 mt-1">Model: {p.laptime_model}</div>
          </div>
          <div className="card">
            <div className="stat-label">Pit probability</div>
            <div className="stat-value">
              {p.probability_pit !== null ? `${(p.probability_pit * 100).toFixed(1)}%` : "—"}
            </div>
            <div className="text-xs text-white/40 mt-1">Model: {p.pit_model}</div>
          </div>
        </div>
        {(regContextOnly.driver || regContextOnly.team || clfContextOnly.driver || clfContextOnly.team) && (
          <p className="text-xs text-white/40 mt-2">
            Context only — not used by this model:{" "}
            {[
              regContextOnly.driver && "driver (lap time)",
              regContextOnly.team && "team (lap time)",
              clfContextOnly.driver && "driver (pit decision)",
              clfContextOnly.team && "team (pit decision)",
            ]
              .filter(Boolean)
              .join(", ")}
          </p>
        )}
      </div>

      <div>
        <div className="text-xs uppercase tracking-wider text-white/40 mb-2">
          Strategy Optimisation — what should we do?
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div className="card">
            <div className="stat-label">Recommended action</div>
            <div className="text-lg font-semibold text-white">{result.recommended_action ?? "—"}</div>
          </div>
          <div className="card">
            <div className="stat-label">Expected cost (remaining stint)</div>
            <div className="stat-value">
              {result.expected_cost_seconds ? `${result.expected_cost_seconds.toFixed(1)}s` : "—"}
            </div>
            <div className="text-xs text-white/40 mt-1">{result.optimal_search_strategy.algorithm}</div>
          </div>
          <div className="card">
            <div className="stat-label">Next search action</div>
            <div className="text-lg font-semibold text-white">
              {result.optimal_search_strategy.next_action?.type === "PIT"
                ? `PIT → ${result.optimal_search_strategy.next_action.compound}`
                : "RUN"}
            </div>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="font-semibold text-white mb-2">Triggered expert rules</div>
        {result.triggered_expert_rules.length === 0 ? (
          <div className="text-sm text-white/50">No rules fired for this race state.</div>
        ) : (
          <ul className="text-sm text-white/70 space-y-1">
            {result.triggered_expert_rules.map((r) => (
              <li key={r.rule_id}>
                <span className="text-white font-medium">{r.rule_id}</span> — {r.name}
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="card">
        <button onClick={() => setShowInputs(!showInputs)} className="text-sm text-sky-400 hover:underline">
          {showInputs ? "Hide" : "Show"} prediction details (features sent to the model)
        </button>
        {showInputs && (
          <div className="mt-3 space-y-3 text-xs">
            {(["target_laptime", "target_pit_next_lap"] as const).map((target) => (
              <div key={target}>
                <div className="text-white/50 mb-1">
                  {target === "target_laptime" ? "Lap-time" : "Pit-decision"} model inputs:
                </div>
                <pre className="text-white/70 whitespace-pre-wrap bg-panel2 rounded p-2">
                  {JSON.stringify(p.feature_rows[target], null, 2)}
                </pre>
                {p.approximated_features[target]?.length > 0 && (
                  <p className="text-white/40 mt-1">
                    Auto-filled from training-data medians (they need multi-lap history this snapshot cannot
                    supply): {p.approximated_features[target].join(", ")}
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
