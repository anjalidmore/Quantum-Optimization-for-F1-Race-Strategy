"use client";

import { useState } from "react";
import { api, ApiError } from "@/lib/api";

export function LaptimePredictPanel({ features }: { features: string[] }) {
  const [values, setValues] = useState<Record<string, string>>(
    Object.fromEntries(features.map((f) => [f, "0"]))
  );
  const [result, setResult] = useState<{ model: string; prediction: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function submit() {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const payload = Object.fromEntries(features.map((f) => [f, Number(values[f])]));
      const res = await api.predictLaptime(payload);
      setResult(res);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Prediction failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="card">
      <div className="font-semibold text-white mb-3">Try a lap-time prediction</div>
      <div className="grid grid-cols-2 gap-2">
        {features.map((f) => (
          <label key={f} className="text-xs text-white/60">
            {f}
            <input
              className="mt-1 w-full bg-panel2 border border-white/10 rounded px-2 py-1 text-white text-sm"
              value={values[f]}
              onChange={(e) => setValues((v) => ({ ...v, [f]: e.target.value }))}
            />
          </label>
        ))}
      </div>
      <button
        onClick={submit}
        disabled={loading}
        className="mt-3 bg-f1red hover:bg-f1red/80 text-white text-sm font-semibold px-4 py-2 rounded disabled:opacity-50"
      >
        {loading ? "Predicting…" : "Predict lap time"}
      </button>
      {result && (
        <div className="mt-3 text-sm text-white/80">
          Model <span className="text-white font-medium">{result.model}</span> predicts{" "}
          <span className="text-white font-semibold">{result.prediction.toFixed(3)}s</span>
        </div>
      )}
      {error && <div className="mt-3 text-sm text-red-400">{error}</div>}
    </div>
  );
}

export function PitPredictPanel({ features }: { features: string[] }) {
  const [values, setValues] = useState<Record<string, string>>(
    Object.fromEntries(features.map((f) => [f, "0"]))
  );
  const [result, setResult] = useState<{ model: string; probability_pit: number; predicted_class: number } | null>(
    null
  );
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function submit() {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const payload = Object.fromEntries(features.map((f) => [f, Number(values[f])]));
      const res = await api.predictPit(payload);
      setResult(res);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Prediction failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="card">
      <div className="font-semibold text-white mb-3">Try a pit-decision prediction</div>
      <div className="grid grid-cols-2 gap-2">
        {features.map((f) => (
          <label key={f} className="text-xs text-white/60">
            {f}
            <input
              className="mt-1 w-full bg-panel2 border border-white/10 rounded px-2 py-1 text-white text-sm"
              value={values[f]}
              onChange={(e) => setValues((v) => ({ ...v, [f]: e.target.value }))}
            />
          </label>
        ))}
      </div>
      <button
        onClick={submit}
        disabled={loading}
        className="mt-3 bg-f1red hover:bg-f1red/80 text-white text-sm font-semibold px-4 py-2 rounded disabled:opacity-50"
      >
        {loading ? "Predicting…" : "Predict pit decision"}
      </button>
      {result && (
        <div className="mt-3 text-sm text-white/80">
          Model <span className="text-white font-medium">{result.model}</span> —{" "}
          probability of pit <span className="text-white font-semibold">{(result.probability_pit * 100).toFixed(1)}%</span>{" "}
          (predicted class: {result.predicted_class === 1 ? "PIT" : "NO PIT"})
        </div>
      )}
      {error && <div className="mt-3 text-sm text-red-400">{error}</div>}
    </div>
  );
}
