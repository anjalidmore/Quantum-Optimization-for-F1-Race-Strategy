"use client";

import { StrategyResponse } from "@/lib/api";
import { TyreBadge } from "@/components/TyreBadge";
import { fmt } from "@/lib/format";

const CONFIDENCE_BADGE: Record<string, string> = {
  high: "badge-success",
  moderate: "badge-info",
  low: "badge-warning",
  none: "badge-danger",
};

/**
 * The full pipeline, stage by stage: input validation, feature construction,
 * the expert system, ML and DL predictions, the search plan and the
 * recommendation engine that combines all four into one call.
 *
 * The recommendation stays the one focal card — action, confidence, and the
 * ML/DL/search evidence behind it read as one decision, not four things that
 * happen to agree — while validation, feature construction, the expert
 * system and the XAI explanation get their own cards below, since each is a
 * genuinely separate thing a reader might want to inspect on its own.
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
  const dl = result.dl_prediction;
  const rec = result.recommendation;
  const validation = result.validation;
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
  const fromRules = result.triggered_expert_rules.length > 0;
  const ACTION: Record<string, string> = {
    PIT_NOW: "Pit now",
    STAY_OUT: "Stay out",
  };
  const action = rec?.action ?? (result.recommended_action ? (ACTION[result.recommended_action] ?? result.recommended_action) : "—");
  const confidenceClass = rec ? CONFIDENCE_BADGE[rec.confidence] ?? "badge-info" : "badge-info";
  const contextNotes = [
    regContextOnly.driver && "driver (lap time)",
    regContextOnly.team && "team (lap time)",
    clfContextOnly.driver && "driver (pit decision)",
    clfContextOnly.team && "team (pit decision)",
  ].filter(Boolean) as string[];

  const explanation = result.xai_explanation ?? {};
  const explanationTargets = Object.entries(explanation).filter(([k]) => k.startsWith("target_"));

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
        <div className="flex items-center justify-between gap-3">
          <p className="stat-label">Recommendation — rules + ML + DL + search, combined</p>
          {rec && <span className={`badge ${confidenceClass}`}>{rec.confidence} confidence</span>}
        </div>
        <p className="t-display-md mt-1">{action}</p>
        {rec ? (
          <p className="t-micro mt-2.5 max-w-[52ch]">{rec.reason}</p>
        ) : (
          <p className="t-micro mt-2.5 max-w-[42ch]">
            Verdict from {fromRules ? "Task 2 expert rules" : "Task 6 pit classifier"}.
          </p>
        )}
        {rec?.disagreement && (
          <p className="t-micro mt-2 text-accent">
            ML and DL disagree on this call — see the ML vs. DL figures below.
          </p>
        )}
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
          .
        </p>

        <dl className="mt-5 grid grid-cols-2 gap-x-6 gap-y-5 border-t border-track-300 pt-5">
          <div>
            <dt className="stat-label">Lap time — ML (Task 6)</dt>
            <dd className="stat-value">
              {p.predicted_lap_time_seconds !== null ? fmt(p.predicted_lap_time_seconds, 3) : "—"}
              <span className="unit"> s</span>
            </dd>
            <p className="t-micro mt-1">{p.laptime_model}</p>
          </div>
          <div>
            <dt className="stat-label">Lap time — DL (Task 7)</dt>
            <dd className="stat-value">
              {dl?.predicted_lap_time_seconds != null ? fmt(dl.predicted_lap_time_seconds, 3) : "—"}
              <span className="unit"> s</span>
            </dd>
            <p className="t-micro mt-1">dnn_mlp</p>
          </div>
          <div>
            <dt className="stat-label">Pit probability — ML</dt>
            <dd className="stat-value">
              {p.probability_pit !== null ? fmt(p.probability_pit * 100, 1) : "—"}
              <span className="unit"> %</span>
            </dd>
            <p className="t-micro mt-1">{p.pit_model}</p>
          </div>
          <div>
            <dt className="stat-label">Pit probability — DL</dt>
            <dd className="stat-value">
              {dl?.probability_pit != null ? fmt(dl.probability_pit * 100, 1) : "—"}
              <span className="unit"> %</span>
            </dd>
            <p className="t-micro mt-1">threshold {dl?.threshold != null ? fmt(dl.threshold, 4) : "—"} (tuned)</p>
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

      {validation && (
        <div className="card">
          <h3 className="t-title text-[15px]">1 · Input validation</h3>
          <p className="t-body mt-2 text-[13px] text-paper-500">
            Lap {validation.current_lap} of {validation.total_laps} — {validation.laps_remaining} remaining, on{" "}
            {validation.tyre_compound}. {validation.note}
          </p>
        </div>
      )}

      <div className="card">
        <button
          type="button"
          onClick={() => setShowInputs(!showInputs)}
          aria-expanded={showInputs}
          aria-controls="feature-rows"
          className="text-[13px] font-medium text-paper-900 underline decoration-edge underline-offset-[3px] hover:decoration-paper-900"
        >
          2 · Feature construction — {showInputs ? "hide" : "show"} the features sent to each model
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

      <div className="card">
        <h3 className="t-title text-[15px]">3 · Expert system</h3>
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
        <h3 className="t-title text-[15px]">6 · Explainability (Task 8)</h3>
        {explanationTargets.length === 0 ? (
          <p className="t-body mt-2 text-[13px] text-paper-500">
            {explanation.reason ?? "Not requested for this run."}
          </p>
        ) : (
          <div className="mt-2.5 space-y-4">
            {explanationTargets.map(([target, e]: [string, any]) => (
              <div key={target} className="border-t border-track-300 pt-3 first:border-t-0 first:pt-0">
                <p className="stat-label">{target === "target_laptime" ? "Lap time" : "Pit decision"}</p>
                {!e.available ? (
                  <p className="t-body mt-1.5 text-[13px] text-paper-500">Not available: {e.reason}</p>
                ) : (
                  <>
                    <p className="t-body mt-1.5 text-[13px] text-paper-700">{e.narrative}</p>
                    <p className="t-micro mt-1.5">
                      Trust score {fmt(e.trust_score, 3)} — {e.trust_band?.label ?? "unknown"}
                    </p>
                    {Array.isArray(e.shap_factors) && e.shap_factors.length > 0 && (
                      <ul className="t-micro mt-1.5 space-y-0.5">
                        {e.shap_factors.slice(0, 4).map((f: any) => (
                          <li key={f.feature}>
                            <span className="t-code">{f.feature}</span>: {f.shap_value >= 0 ? "+" : ""}
                            {fmt(f.shap_value, 4)} ({f.direction})
                          </li>
                        ))}
                      </ul>
                    )}
                  </>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
