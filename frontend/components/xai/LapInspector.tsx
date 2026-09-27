"use client";

import { useEffect, useState } from "react";
import { api, type XaiLap, type XaiLapRow } from "@/lib/api";
import { fmt } from "@/lib/format";

const TARGETS: Record<string, string> = {
  target_pit_next_lap: "Pit decision",
  target_laptime: "Lap time",
};

/**
 * Inspect any Task 7 test lap: race state, the DNN's prediction, its SHAP
 * attribution, LIME's top-3, the trust assessment, and a tyre-age
 * counterfactual that the backend computes live with the saved network.
 * Categorical choices are dropdowns; nothing here is typed in by hand.
 */
export default function LapInspector() {
  const [target, setTarget] = useState("target_pit_next_lap");
  const [laps, setLaps] = useState<XaiLapRow[]>([]);
  const [rowIndex, setRowIndex] = useState(0);
  const [lap, setLap] = useState<XaiLap | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setError(null);
    api
      .xaiLaps(target)
      .then((r) => {
        setLaps(r.laps);
        setRowIndex(0);
      })
      .catch((e) => setError(String(e?.message ?? e)));
  }, [target]);

  useEffect(() => {
    if (!laps.length) return;
    setLoading(true);
    api
      .xaiLap(target, rowIndex)
      .then(setLap)
      .catch((e) => setError(String(e?.message ?? e)))
      .finally(() => setLoading(false));
  }, [target, rowIndex, laps.length]);

  const isPit = target === "target_pit_next_lap";
  const row = lap?.lap;
  const cf = lap?.counterfactual ?? {};
  const maxAbs = Math.max(...(lap?.shap_factors ?? []).map((f) => Math.abs(f.shap_value)), 1e-12);

  return (
    <div className="card space-y-4">
      <div className="flex flex-wrap gap-3 items-end">
        <label className="text-sm text-white/60">
          Model
          <select className="block mt-1 bg-white/5 rounded px-2 py-1 text-white" value={target}
                  onChange={(e) => setTarget(e.target.value)}>
            {Object.entries(TARGETS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
          </select>
        </label>
        <label className="text-sm text-white/60 grow">
          Test lap
          <select className="block mt-1 w-full bg-white/5 rounded px-2 py-1 text-white" value={rowIndex}
                  onChange={(e) => setRowIndex(Number(e.target.value))}>
            {laps.map((l) => (
              <option key={l.row_index} value={l.row_index}>
                {l.driver} ({l.team}, {l.compound}) — lap {l.lap} · trust {l.trust_band}
              </option>
            ))}
          </select>
        </label>
      </div>

      {error && <p className="text-sm text-red-400">{error}</p>}
      {loading && <p className="text-sm text-white/40">Loading…</p>}

      {lap && row && (
        <div className="grid gap-4 lg:grid-cols-3">
          <div className="space-y-2">
            <h4 className="text-xs uppercase tracking-wide text-white/40">Prediction</h4>
            <p className="text-2xl font-bold text-white">
              {isPit ? `P(pit) ${fmt(row.dnn_prediction, 4)}` : `${fmt(row.dnn_prediction, 3)} s`}
            </p>
            <p className="text-sm text-white/60">
              {isPit
                ? <>Decision at threshold {fmt(lap.decision_threshold, 4)}: <strong className="text-white">{row.dnn_decision}</strong> · actually {row.actual === 1 ? "pitted" : "stayed out"}</>
                : <>Actual {fmt(row.actual, 3)} s · error {fmt(row.dnn_abs_error_s, 3)} s</>}
            </p>
            <p className="text-sm text-white/60">
              Task 6 {isPit ? "P(pit)" : "prediction"}: {fmt(row.classical_prediction, isPit ? 4 : 3)}
            </p>
            <h4 className="text-xs uppercase tracking-wide text-white/40 pt-2">Trust (project-defined)</h4>
            <p className="text-white"><strong>{fmt(row.trust_score, 3)}</strong> — {row.trust_band}</p>
            <dl className="grid grid-cols-2 gap-x-3 text-xs text-white/50">
              {Object.entries(row).filter(([k]) => k.startsWith("component_")).map(([k, v]) => (
                <div key={k} className="contents">
                  <dt>{k.replace("component_", "").replace(/_/g, " ")}</dt>
                  <dd className="text-right text-white/70">{fmt(v as number, 3)}</dd>
                </div>
              ))}
            </dl>
          </div>

          <div className="space-y-2">
            <h4 className="text-xs uppercase tracking-wide text-white/40">SHAP — what moved this prediction</h4>
            <ul className="space-y-1.5">
              {lap.shap_factors.slice(0, 8).map((f) => (
                <li key={f.feature} className="text-sm">
                  <div className="flex justify-between gap-2">
                    <span><code className="text-white/70">{f.feature}</code> <span className="text-white/40">= {fmt(f.value, 3)}</span></span>
                    <span className={f.shap_value > 0 ? "text-red-400 tabular-nums" : "text-sky-400 tabular-nums"}>
                      {f.shap_value > 0 ? "+" : ""}{fmt(f.shap_value, 4)}
                    </span>
                  </div>
                  <div className="h-1 rounded-full bg-white/5 mt-0.5">
                    <div className={`h-full rounded-full ${f.shap_value > 0 ? "bg-red-400/70" : "bg-sky-400/70"}`}
                         style={{ width: `${(Math.abs(f.shap_value) / maxAbs) * 100}%` }} />
                  </div>
                </li>
              ))}
            </ul>
            <p className="text-xs text-white/40">
              Base value {fmt(lap.shap_base_value, 4)}. SHAP top 3: <code>{row.shap_top3}</code> · LIME top 3:{" "}
              <code>{row.lime_top3}</code>
            </p>
          </div>

          <div className="space-y-2">
            <h4 className="text-xs uppercase tracking-wide text-white/40">Counterfactual — tyre age</h4>
            {cf.available === false ? (
              <p className="text-sm text-white/50">Unavailable: {cf.reason}</p>
            ) : isPit ? (
              <p className="text-sm text-white/70">
                Tyre age now {fmt(cf.original_value, 1)} laps.{" "}
                {cf.reachable
                  ? <>The model&rsquo;s decision flips at <strong className="text-white">{fmt(cf.crossing_value, 1)} laps</strong> ({cf.delta_required > 0 ? "+" : ""}{fmt(cf.delta_required, 1)}).</>
                  : <>No tyre age in the training range ({cf.searched_range?.join("–")} laps) flips the decision.</>}
              </p>
            ) : (
              <ul className="text-sm text-white/70 space-y-1">
                {(cf.regression_effects ?? []).map((e: any) => (
                  <li key={e.tyre_age_change_laps}>
                    {e.tyre_age_change_laps > 0 ? "+" : ""}{e.tyre_age_change_laps} laps of tyre age →{" "}
                    {e.change_in_prediction > 0 ? "+" : ""}{fmt(e.change_in_prediction, 3)} s predicted
                  </li>
                ))}
              </ul>
            )}
            {cf.derived_features_recomputed && (
              <p className="text-xs text-white/40">
                Recomputed with tyre age: <code>{cf.derived_features_recomputed.join(", ")}</code>. Compound, set
                freshness and track temperature held fixed. This shows model sensitivity, not a strategy instruction.
              </p>
            )}
            <h4 className="text-xs uppercase tracking-wide text-white/40 pt-2">Race state</h4>
            <dl className="grid grid-cols-2 gap-x-3 text-xs text-white/50 max-h-40 overflow-y-auto">
              {Object.entries(lap.race_state).map(([k, v]) => (
                <div key={k} className="contents">
                  <dt className="truncate">{k}</dt>
                  <dd className="text-right text-white/70 tabular-nums">{fmt(v, 3)}</dd>
                </div>
              ))}
            </dl>
          </div>
        </div>
      )}
    </div>
  );
}
