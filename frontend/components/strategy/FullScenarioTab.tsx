"use client";

import { useMemo, useState } from "react";
import { api, ApiError, DataOptions, RaceState, Registry, StrategyResponse } from "@/lib/api";
import { buildScenarios, clampForm, RaceStateForm, Scenario } from "@/components/strategy/RaceStateForm";
import { StrategyResult } from "@/components/strategy/StrategyResult";

/**
 * The full race-strategy simulator: hold the race state, POST it to
 * /api/strategy/predict, and show what came back. The form lives in
 * RaceStateForm and the answer in StrategyResult; this component is the state.
 *
 * Layout: inputs left, answer right and sticky. The answer is the focal element,
 * so it stays in view while the inputs beside it change.
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
  const [reporting, setReporting] = useState(false);
  const [activeScenario, setActiveScenario] = useState<string | null>(null);

  function set<K extends keyof RaceState>(key: K, value: RaceState[K]) {
    setForm((f) => clampForm({ ...f, [key]: value }));
    setActiveScenario(null); // an edited preset is no longer that preset
  }

  function applyScenario(scenario: Scenario) {
    setForm((f) => clampForm({ ...f, ...scenario.values }));
    setActiveScenario(scenario.name);
    setResult(null);
  }

  function payload(): RaceState {
    return {
      ...form,
      laptime_model: modelMode === "select" ? laptimeModel : null,
      pit_model: modelMode === "select" ? pitModel : null,
    };
  }

  /** Task 9: ask the backend for a full Markdown briefing on this race state. */
  async function downloadReport() {
    setReporting(true);
    setError(null);
    try {
      const { filename, markdown } = await api.strategyReport(payload());
      const url = URL.createObjectURL(new Blob([markdown], { type: "text/markdown" }));
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Report generation failed.");
    } finally {
      setReporting(false);
    }
  }

  async function submit() {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await api.strategyPredict(payload()));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Prediction failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid gap-x-12 gap-y-10 lg:grid-cols-[1fr_minmax(360px,420px)]">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <RaceStateForm
          options={options}
          registry={registry}
          form={form}
          set={set}
          scenarios={scenarios}
          applyScenario={applyScenario}
          activeScenario={activeScenario}
          modelMode={modelMode}
          setModelMode={setModelMode}
          laptimeModel={laptimeModel}
          setLaptimeModel={setLaptimeModel}
          pitModel={pitModel}
          setPitModel={setPitModel}
        />

        <div className="mt-7 flex flex-wrap items-center gap-3 border-t border-track-300 pt-6">
          <button type="submit" disabled={loading} className="btn-primary">
            {loading ? "Running model…" : "Predict"}
          </button>
          <button type="button" onClick={downloadReport} disabled={reporting} className="btn-secondary">
            {reporting ? "Generating report…" : "Download strategy report"}
          </button>
          <p className="t-micro max-w-[30ch]">
            The report is one Markdown file: prediction, triggered rules, search plan, SHAP explanation and trust
            score.
          </p>
        </div>

        {error && (
          <div className="card mt-6 border-accent" role="alert">
            <span className="badge badge-danger">Request failed</span>
            <p className="t-body mt-2.5 text-[13px]">{error}</p>
          </div>
        )}
      </form>

      <div className="lg:sticky lg:top-[76px] lg:self-start">
        {result ? (
          <StrategyResult result={result} showInputs={showInputs} setShowInputs={setShowInputs} />
        ) : (
          <div className="card card-focal" aria-live="polite">
            <p className="stat-label">Recommendation</p>
            <p className="t-display-md mt-1.5 text-paper-500">{loading ? "Running…" : "Not run yet"}</p>
            <p className="t-body mt-3 text-[13px] text-paper-500">
              {loading
                ? "Asking the lap-time model, the pit classifier, the rule base and the search planner."
                : "Set the race state, then predict. Nothing is computed until you do."}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
