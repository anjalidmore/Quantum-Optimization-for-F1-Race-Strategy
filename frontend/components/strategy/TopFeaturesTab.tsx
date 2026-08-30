"use client";

import { useEffect, useState } from "react";
import { api, ApiError, FeatureDescriptor, Registry, TopFeaturesResponse } from "@/lib/api";
import { Tooltip } from "@/components/Tooltip";

const TARGETS: { value: string; label: string }[] = [
  { value: "target_laptime", label: "Lap-Time Regression" },
  { value: "target_pit_next_lap", label: "Pit-Decision Classification" },
];

export function TopFeaturesTab({ registry }: { registry: Registry }) {
  const [target, setTarget] = useState("target_laptime");
  const [data, setData] = useState<TopFeaturesResponse | null>(null);
  const [values, setValues] = useState<Record<string, string>>({});
  const [showAuto, setShowAuto] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [result, setResult] = useState<{ prediction: number; model: string } | { probability_pit: number; predicted_class: number; model: string } | null>(null);
  const [predictError, setPredictError] = useState<string | null>(null);
  const [predicting, setPredicting] = useState(false);

  useEffect(() => {
    setResult(null);
    setPredictError(null);
    api
      .topFeatures(target, 8)
      .then((d) => {
        setData(d);
        setValues(Object.fromEntries(d.top_features.map((f) => [f.feature, String(f.median ?? 0)])));
        setLoadError(null);
      })
      .catch((e) => setLoadError(e instanceof ApiError ? e.message : "Could not load top features."));
  }, [target]);

  async function submit() {
    if (!data) return;
    setPredicting(true);
    setPredictError(null);
    setResult(null);
    try {
      const autoFilled = Object.fromEntries(data.remaining_features.map((f) => [f.feature, f.median ?? 0]));
      const userValues = Object.fromEntries(data.top_features.map((f) => [f.feature, Number(values[f.feature])]));
      const payload = { ...autoFilled, ...userValues };

      if (target === "target_laptime") {
        const res = await api.predictLaptime(payload);
        setResult({ prediction: res.prediction, model: res.model });
      } else {
        const res = await api.predictPit(payload);
        setResult({ probability_pit: res.probability_pit, predicted_class: res.predicted_class, model: res.model });
      }
    } catch (e) {
      setPredictError(e instanceof ApiError ? e.message : "Prediction failed.");
    } finally {
      setPredicting(false);
    }
  }

  return (
    <div className="space-y-6">
      <div className="card">
        <div className="font-semibold text-white mb-2">Why these 8?</div>
        <p className="text-sm text-white/60">
          These are the highest-ranked features for the currently selected-best model on this task. Ranking
          method: <span className="text-white/80">{data?.ranking_method ?? "loading…"}</span>
        </p>
        <div className="mt-3">
          <label className="text-xs text-white/60 block">
            Target
            <select className="input mt-1" value={target} onChange={(e) => setTarget(e.target.value)}>
              {TARGETS.map((t) => (
                <option key={t.value} value={t.value}>
                  {t.label}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>

      {loadError && <div className="card border-red-500/30 bg-red-500/5 text-red-400 text-sm">{loadError}</div>}

      {data && (
        <>
          <div className="card">
            <div className="font-semibold text-white mb-3">Top {data.top_features.length} Features</div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {data.top_features.map((f) => (
                <FeatureInput key={f.feature} descriptor={f} value={values[f.feature] ?? ""} onChange={(v) => setValues((s) => ({ ...s, [f.feature]: v }))} />
              ))}
            </div>
            <p className="text-xs text-white/40 mt-3">
              The values above are user-controlled. Other required model inputs are automatically populated
              for demonstration purposes (see below).
            </p>
          </div>

          {data.remaining_features.length > 0 && (
            <div className="card">
              <button onClick={() => setShowAuto(!showAuto)} className="text-sm text-sky-400 hover:underline">
                {showAuto ? "Hide" : "Show"} automatically populated inputs ({data.remaining_features.length})
              </button>
              {showAuto && (
                <div className="mt-3 grid grid-cols-2 md:grid-cols-3 gap-2 text-xs">
                  {data.remaining_features.map((f) => (
                    <div key={f.feature} className="bg-panel2 rounded p-2">
                      <div className="text-white/70">{f.display_name}</div>
                      <div className="text-white/40">Auto-filled: {f.median?.toFixed(3)}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          <button
            onClick={submit}
            disabled={predicting}
            className="bg-f1red hover:bg-f1red/80 text-white font-semibold px-6 py-3 rounded disabled:opacity-50"
          >
            {predicting ? "Running model…" : "Predict"}
          </button>

          {predictError && <div className="card border-red-500/30 bg-red-500/5 text-red-400 text-sm">{predictError}</div>}

          {result && (
            <div className="card">
              {"prediction" in result ? (
                <>
                  <div className="stat-label">Predicted Lap Time</div>
                  <div className="stat-value">{result.prediction.toFixed(3)}s</div>
                </>
              ) : (
                <>
                  <div className="stat-label">Pit Decision Probability</div>
                  <div className="stat-value">{(result.probability_pit * 100).toFixed(1)}%</div>
                  <div className="text-sm text-white/60 mt-1">Predicted: {result.predicted_class === 1 ? "PIT" : "NO PIT"}</div>
                </>
              )}
              <div className="text-xs text-white/40 mt-2">Model: {result.model}</div>
              <div className="text-xs text-white/40">Data: Task 5 selected features, top-8 demo mode</div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function FeatureInput({
  descriptor,
  value,
  onChange,
}: {
  descriptor: FeatureDescriptor;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="bg-panel2 rounded-lg p-3">
      <div className="text-sm text-white font-medium inline-flex items-center">
        {descriptor.display_name}
        <Tooltip text={descriptor.description} />
      </div>
      <div className="text-xs text-white/40 mb-2">{descriptor.description}</div>
      <div className="flex items-center gap-2">
        <input
          type="number"
          step="any"
          className="input"
          value={value}
          min={descriptor.min}
          max={descriptor.max}
          onChange={(e) => onChange(e.target.value)}
        />
        {descriptor.unit && <span className="text-xs text-white/40 whitespace-nowrap">{descriptor.unit}</span>}
      </div>
      {descriptor.min !== undefined && descriptor.max !== undefined && (
        <div className="text-[10px] text-white/30 mt-1">
          typical range: {descriptor.min.toFixed(2)} – {descriptor.max.toFixed(2)}
        </div>
      )}
    </div>
  );
}
