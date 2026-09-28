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
      <div className="font-semibold text-paper-900 mb-3">Try a lap-time prediction</div>
      <div className="grid grid-cols-2 gap-2">
        {features.map((f) => (
          <label key={f} className="text-xs text-paper-500">
            {f}
            <input
              className="mt-1 w-full bg-track-200 border border-track-300 rounded px-2 py-1 text-paper-900 text-sm"
              value={values[f]}
              onChange={(e) => setValues((v) => ({ ...v, [f]: e.target.value }))}
            />
          </label>
        ))}
      </div>
      <button
        onClick={submit}
        disabled={loading}
        className="mt-3 btn-primary mt-0"
      >
        {loading ? "Predicting…" : "Predict lap time"}
      </button>
      {result && (
        <div className="mt-3 text-sm text-paper-700">
          Model <span className="text-paper-900 font-medium">{result.model}</span> predicts{" "}
          <span className="text-paper-900 font-semibold">{result.prediction.toFixed(3)}s</span>
        </div>
      )}
      {error && <div className="mt-3 text-sm text-accent">{error}</div>}
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
      <div className="font-semibold text-paper-900 mb-3">Try a pit-decision prediction</div>
      <div className="grid grid-cols-2 gap-2">
        {features.map((f) => (
          <label key={f} className="text-xs text-paper-500">
            {f}
            <input
              className="mt-1 w-full bg-track-200 border border-track-300 rounded px-2 py-1 text-paper-900 text-sm"
              value={values[f]}
              onChange={(e) => setValues((v) => ({ ...v, [f]: e.target.value }))}
            />
          </label>
        ))}
      </div>
      <button
        onClick={submit}
        disabled={loading}
        className="mt-3 btn-primary mt-0"
      >
        {loading ? "Predicting…" : "Predict pit decision"}
      </button>
      {result && (
        <div className="mt-3 text-sm text-paper-700">
          Model <span className="text-paper-900 font-medium">{result.model}</span> —{" "}
          probability of pit <span className="text-paper-900 font-semibold">{(result.probability_pit * 100).toFixed(1)}%</span>{" "}
          (predicted class: {result.predicted_class === 1 ? "PIT" : "NO PIT"})
        </div>
      )}
      {error && <div className="mt-3 text-sm text-accent">{error}</div>}
    </div>
  );
}
