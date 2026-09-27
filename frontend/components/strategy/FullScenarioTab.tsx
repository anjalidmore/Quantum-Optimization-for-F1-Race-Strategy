"use client";

import { useMemo, useState } from "react";
import { api, ApiError, DataOptions, RaceState, Registry, StrategyResponse } from "@/lib/api";
import { buildScenarios, clampForm, RaceStateForm, Scenario } from "@/components/strategy/RaceStateForm";
import { StrategyResult } from "@/components/strategy/StrategyResult";

/**
 * The full race-strategy simulator: hold the race state, POST it to
 * /api/strategy/predict, and show what came back. The form lives in
 * RaceStateForm and the answer in StrategyResult; this component is the state.
 */
export function FullScenarioTab({ options, registry }: { options: DataOptions; registry: Registry }) {
  const scenarios = useMemo(() => buildScenarios(options), [options]);

  const [form, setForm] = useState<RaceState>(() =>
    clampForm({
      driver: options.drivers[0],
      team: options.teams[0],
      current_lap: Math.round(options.total_laps_hint * 0.4),
      total_laps: options.total_laps_hint,
      tyre_compound: options.compounds[0],
      tyre_age: 8,
      track_temperature: Math.round(options.track_temperature_range.mean),
      weather: "dry",
      fuel_kg: 70,
      track_status: "GREEN",
      current_position: 6,
    }),
  );

  const regModels = registry.models.filter((m) => m.target === "target_laptime" && m.artifact);
  const clfModels = registry.models.filter((m) => m.target === "target_pit_next_lap" && m.artifact);

  const [modelMode, setModelMode] = useState<"best" | "select">("best");
  const [laptimeModel, setLaptimeModel] = useState(regModels[0]?.model_name ?? "");
  const [pitModel, setPitModel] = useState(clfModels[0]?.model_name ?? "");

  const [result, setResult] = useState<StrategyResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [showInputs, setShowInputs] = useState(false);

  function set<K extends keyof RaceState>(key: K, value: RaceState[K]) {
    setForm((f) => clampForm({ ...f, [key]: value }));
  }

  function applyScenario(scenario: Scenario) {
    setForm((f) => clampForm({ ...f, ...scenario.values }));
    setResult(null);
  }

  async function submit() {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const payload: RaceState = {
        ...form,
        laptime_model: modelMode === "select" ? laptimeModel : null,
        pit_model: modelMode === "select" ? pitModel : null,
      };
      setResult(await api.strategyPredict(payload));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Prediction failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <RaceStateForm
        options={options}
        registry={registry}
        form={form}
        set={set}
        scenarios={scenarios}
        applyScenario={applyScenario}
        modelMode={modelMode}
        setModelMode={setModelMode}
        laptimeModel={laptimeModel}
        setLaptimeModel={setLaptimeModel}
        pitModel={pitModel}
        setPitModel={setPitModel}
      />

      <button
        onClick={submit}
        disabled={loading}
        className="bg-f1red hover:bg-f1red/80 text-white font-semibold px-6 py-3 rounded disabled:opacity-50"
      >
        {loading ? "Running model…" : "Predict"}
      </button>

      {error && <div className="card border-red-500/30 bg-red-500/5 text-red-400 text-sm">{error}</div>}

      {result && <StrategyResult result={result} showInputs={showInputs} setShowInputs={setShowInputs} />}
    </div>
  );
}
