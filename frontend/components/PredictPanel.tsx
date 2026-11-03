"use client";

import { useState } from "react";
import { api, ApiError } from "@/lib/api";

/**
 * Calls a prediction endpoint with raw feature values. Collapsed by default:
 * it exposes every model input by its contract name, which is for checking the
 * API by hand. The Strategy page is where a race situation is described.
 */
function RawPredictPanel({
  title,
  action,
  features,
  predict,
}: {
  title: string;
  action: string;
  features: string[];
  predict: (payload: Record<string, number>) => Promise<React.ReactNode>;
}) {
  const [values, setValues] = useState<Record<string, string>>(Object.fromEntries(features.map((f) => [f, "0"])));
  const [result, setResult] = useState<React.ReactNode>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function submit() {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await predict(Object.fromEntries(features.map((f) => [f, Number(values[f])]))));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Prediction failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <details className="card">
      <summary className="cursor-pointer font-semibold text-paper-900">{title}</summary>
      <p className="mb-3 mt-2 text-xs text-paper-500">
        Raw model inputs ({features.length}), all starting at 0. To describe a race situation, use the Strategy page.
      </p>
      <div className="grid grid-cols-2 gap-2">
        {features.map((f) => (
          <label key={f} className="text-xs text-paper-500">
            {f}
            <input
              className="mt-1 w-full rounded border border-track-300 bg-track-200 px-2 py-1 text-sm text-paper-900"
              value={values[f]}
              onChange={(e) => setValues((v) => ({ ...v, [f]: e.target.value }))}
            />
          </label>
        ))}
      </div>
      <button onClick={submit} disabled={loading} className="btn-primary mt-3">
        {loading ? "Predicting…" : action}
      </button>
      {result && <div className="mt-3 text-sm text-paper-700">{result}</div>}
      {error && <div className="mt-3 text-sm text-accent">{error}</div>}
    </details>
  );
}

export function LaptimePredictPanel({ features }: { features: string[] }) {
  return (
    <RawPredictPanel
      title="Try a lap-time prediction"
      action="Predict lap time"
      features={features}
      predict={async (payload) => {
        const res = await api.predictLaptime(payload);
        return (
          <>
            Model <span className="font-medium text-paper-900">{res.model}</span> predicts{" "}
            <span className="font-semibold text-paper-900">{res.prediction.toFixed(3)}s</span>
          </>
        );
      }}
    />
  );
}

export function PitPredictPanel({ features }: { features: string[] }) {
  return (
    <RawPredictPanel
      title="Try a pit-decision prediction"
      action="Predict pit decision"
      features={features}
      predict={async (payload) => {
        const res = await api.predictPit(payload);
        return (
          <>
            Model <span className="font-medium text-paper-900">{res.model}</span> — probability of pit{" "}
            <span className="font-semibold text-paper-900">{(res.probability_pit * 100).toFixed(1)}%</span> (predicted
            class: {res.predicted_class === 1 ? "PIT" : "NO PIT"})
          </>
        );
      }}
    />
  );
}
