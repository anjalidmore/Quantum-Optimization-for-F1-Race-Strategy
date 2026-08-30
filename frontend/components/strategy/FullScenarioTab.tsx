"use client";

import { useMemo, useState } from "react";
import { api, ApiError, DataOptions, RaceState, Registry, StrategyResponse } from "@/lib/api";
import { Tooltip } from "@/components/Tooltip";

const WEATHER_OPTIONS = ["dry", "damp", "wet", "extreme"];
const TRACK_STATUS_OPTIONS = ["GREEN", "YELLOW", "SC", "VSC", "RED"];

const FIELD_HELP: Record<string, string> = {
  driver: "The driver whose race-state this prediction is for.",
  team: "The driver's constructor/team for this race.",
  current_lap: "The lap the car is currently on.",
  total_laps: "Total race distance in laps.",
  tyre_compound: "The tyre compound currently fitted: Soft (fastest, wears quickest), Medium, or Hard (slowest, most durable).",
  tyre_age: "How many laps the current tyre set has completed.",
  track_temperature: "Track surface temperature — higher temperatures generally accelerate tyre degradation.",
  weather: "Current weather severity. Wetter conditions favour intermediate/wet tyres.",
  fuel_kg: "Estimated fuel remaining on board — affects car weight and pace.",
  track_status: "Race-control flag state: Green (racing), Yellow (caution), SC (Safety Car), VSC (Virtual Safety Car), Red (stopped).",
  current_position: "The car's current position in the race order.",
};

type Scenario = { name: string; description: string; values: Partial<RaceState> };

function buildScenarios(options: DataOptions): Scenario[] {
  const laps = options.total_laps_hint;
  const mid = Math.max(1, Math.round(laps * 0.4));
  const late = Math.max(1, Math.round(laps * 0.85));
  // Track temperature presets are pinned to the REAL range this model was
  // trained on (never a hard-coded "hot track" guess) — using a value the
  // model never saw during training (e.g. 48°C for a session that only
  // ever reached 31°C) makes it extrapolate wildly and produce a
  // meaningless/arbitrary prediction rather than a genuine "hotter track"
  // answer. "High Tyre Degradation" uses the hottest track temperature
  // actually observed in this session, not an exaggerated fabricated one.
  const { mean: tMean, max: tMax } = options.track_temperature_range;
  const tNormal = Math.round(tMean);
  const tHot = Math.round(tMax);
  return [
    {
      name: "Normal Race",
      description: `Lap ${mid}/${laps} · Medium · 8 laps old · ${tNormal}°C, Dry, Green`,
      values: { current_lap: mid, total_laps: laps, tyre_compound: "MEDIUM", tyre_age: 8, track_temperature: tNormal, weather: "dry", track_status: "GREEN" },
    },
    {
      name: "High Tyre Degradation",
      description: `Lap ${mid}/${laps} · Soft · ${Math.min(mid, 22)} laps old · ${tHot}°C (hottest this session saw)`,
      values: { current_lap: mid, total_laps: laps, tyre_compound: "SOFT", tyre_age: Math.min(mid, 22), track_temperature: tHot, weather: "dry", track_status: "GREEN" },
    },
    {
      name: "Late-Race Pit Decision",
      description: `Lap ${late}/${laps} · Hard · 25 laps old · ${tNormal}°C`,
      values: { current_lap: late, total_laps: laps, tyre_compound: "HARD", tyre_age: Math.min(late, 25), track_temperature: tNormal, weather: "dry", track_status: "GREEN" },
    },
    {
      name: "Safety-Car Scenario",
      description: `Lap ${mid}/${laps} · Medium · Safety Car out`,
      values: { current_lap: mid, total_laps: laps, tyre_compound: "MEDIUM", tyre_age: 12, track_temperature: tNormal, track_status: "SC" },
    },
    {
      name: "Fresh Tyres",
      description: `Lap ${Math.min(mid + 1, laps)}/${laps} · Soft · 1 lap old (just pitted) · ${tNormal}°C`,
      values: { current_lap: Math.min(mid + 1, laps), total_laps: laps, tyre_compound: "SOFT", tyre_age: 1, track_temperature: tNormal },
    },
  ];
}

function clampForm(form: RaceState): RaceState {
  const total_laps = Math.max(1, form.total_laps);
  const current_lap = Math.min(Math.max(1, form.current_lap), total_laps);
  const tyre_age = Math.min(Math.max(0, form.tyre_age), current_lap);
  return { ...form, total_laps, current_lap, tyre_age };
}

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
    })
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
      const res = await api.strategyPredict(payload);
      setResult(res);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Prediction failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      {/* Quick scenarios */}
      <div className="card">
        <div className="font-semibold text-white mb-3">Quick Scenarios</div>
        <div className="flex flex-wrap gap-2">
          {scenarios.map((s) => (
            <button
              key={s.name}
              title={s.description}
              onClick={() => applyScenario(s)}
              className="px-3 py-1.5 rounded-full text-sm border border-white/15 text-white/70 hover:text-white hover:border-white/30 transition"
            >
              {s.name}
            </button>
          ))}
        </div>
        <p className="text-xs text-white/40 mt-2">
          Presets only fill in the form below — click Predict to run the real model on those inputs.
        </p>
      </div>

      {/* Race state form */}
      <div className="card grid grid-cols-2 md:grid-cols-4 gap-3">
        <Field label="Driver" help={FIELD_HELP.driver}>
          <select className="input" value={form.driver} onChange={(e) => set("driver", e.target.value)}>
            {options.drivers.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Team" help={FIELD_HELP.team}>
          <select className="input" value={form.team} onChange={(e) => set("team", e.target.value)}>
            {options.teams.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Total Laps" help={FIELD_HELP.total_laps}>
          <input
            type="number"
            min={1}
            className="input"
            value={form.total_laps}
            onChange={(e) => set("total_laps", Number(e.target.value) || 1)}
          />
        </Field>
        <Field label="Current Lap" help={FIELD_HELP.current_lap}>
          <input
            type="number"
            min={1}
            max={form.total_laps}
            className="input"
            value={form.current_lap}
            onChange={(e) => set("current_lap", Number(e.target.value) || 1)}
          />
        </Field>
        <Field label="Tyre Compound" help={FIELD_HELP.tyre_compound}>
          <select className="input" value={form.tyre_compound} onChange={(e) => set("tyre_compound", e.target.value)}>
            {options.compounds.map((c) => (
              <option key={c} value={c}>
                {c.charAt(0) + c.slice(1).toLowerCase()}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Tyre Age (laps)" help={FIELD_HELP.tyre_age}>
          <input
            type="number"
            min={0}
            max={form.current_lap}
            className="input"
            value={form.tyre_age}
            onChange={(e) => set("tyre_age", Number(e.target.value) || 0)}
          />
        </Field>
        <Field label="Track Temperature (°C)" help={FIELD_HELP.track_temperature}>
          <input
            type="number"
            className="input"
            value={form.track_temperature}
            onChange={(e) => set("track_temperature", Number(e.target.value) || 0)}
          />
        </Field>
        <Field label="Weather" help={FIELD_HELP.weather}>
          <select className="input" value={form.weather} onChange={(e) => set("weather", e.target.value)}>
            {WEATHER_OPTIONS.map((w) => (
              <option key={w} value={w}>
                {w.charAt(0).toUpperCase() + w.slice(1)}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Fuel State (kg)" help={FIELD_HELP.fuel_kg}>
          <input
            type="number"
            min={0}
            className="input"
            value={form.fuel_kg}
            onChange={(e) => set("fuel_kg", Number(e.target.value) || 0)}
          />
        </Field>
        <Field label="Track Status" help={FIELD_HELP.track_status}>
          <select className="input" value={form.track_status} onChange={(e) => set("track_status", e.target.value)}>
            {TRACK_STATUS_OPTIONS.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Current Position" help={FIELD_HELP.current_position}>
          <input
            type="number"
            min={1}
            max={24}
            className="input"
            value={form.current_position}
            onChange={(e) => set("current_position", Number(e.target.value) || 1)}
          />
        </Field>
      </div>

      {/* Model selection */}
      <div className="card">
        <div className="font-semibold text-white mb-3">Prediction Model</div>
        <div className="flex items-center gap-6 mb-3 text-sm">
          <label className="flex items-center gap-2 text-white/80">
            <input type="radio" checked={modelMode === "best"} onChange={() => setModelMode("best")} />
            Best Performing Model
          </label>
          <label className="flex items-center gap-2 text-white/80">
            <input type="radio" checked={modelMode === "select"} onChange={() => setModelMode("select")} />
            Select Model
          </label>
        </div>
        {modelMode === "select" && (
          <div className="grid grid-cols-2 gap-3">
            <Field label="Lap-Time Model">
              <select className="input" value={laptimeModel} onChange={(e) => setLaptimeModel(e.target.value)}>
                {regModels.map((m) => (
                  <option key={m.model_name} value={m.model_name}>
                    {m.model_name}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Pit-Decision Model">
              <select className="input" value={pitModel} onChange={(e) => setPitModel(e.target.value)}>
                {clfModels.map((m) => (
                  <option key={m.model_name} value={m.model_name}>
                    {m.model_name}
                  </option>
                ))}
              </select>
            </Field>
          </div>
        )}
      </div>

      <button
        onClick={submit}
        disabled={loading}
        className="bg-f1red hover:bg-f1red/80 text-white font-semibold px-6 py-3 rounded disabled:opacity-50"
      >
        {loading ? "Running model…" : "Predict"}
      </button>

      {error && <div className="card border-red-500/30 bg-red-500/5 text-red-400 text-sm">{error}</div>}

      {result && <ResultPanel result={result} showInputs={showInputs} setShowInputs={setShowInputs} />}
    </div>
  );
}

function Field({ label, help, children }: { label: string; help?: string; children: React.ReactNode }) {
  return (
    <label className="text-xs text-white/60 block">
      <span className="inline-flex items-center">
        {label}
        {help && <Tooltip text={help} />}
      </span>
      <div className="mt-1">{children}</div>
    </label>
  );
}

function ResultPanel({
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
  const regOutOfRange = p.out_of_range["target_laptime"] ?? [];
  const clfOutOfRange = p.out_of_range["target_pit_next_lap"] ?? [];

  return (
    <div className="space-y-4">
      {(regOutOfRange.length > 0 || clfOutOfRange.length > 0) && (
        <div className="card border-amber-500/30 bg-amber-500/5">
          <div className="badge badge-warning mb-2">Extrapolating beyond training data</div>
          <p className="text-sm text-white/70">
            These inputs push at least one model feature outside the range the model was actually trained
            on. Its behaviour out there is unvalidated and can be arbitrary — treat this prediction with
            caution, not as a reliable answer.
          </p>
          <ul className="text-xs text-white/50 mt-2 space-y-1">
            {Array.from(new Map([...regOutOfRange, ...clfOutOfRange].map((o) => [o.feature, o])).values()).map((o) => (
              <li key={o.feature}>
                <span className="text-white/70 font-mono">{o.feature}</span> = {o.value.toFixed(2)} (trained on{" "}
                {o.training_min.toFixed(2)} – {o.training_max.toFixed(2)})
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* ML Prediction */}
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
            <div className="stat-value">{p.probability_pit !== null ? `${(p.probability_pit * 100).toFixed(1)}%` : "—"}</div>
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

      {/* Strategy Optimisation */}
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
            <div className="stat-value">{result.expected_cost_seconds ? `${result.expected_cost_seconds.toFixed(1)}s` : "—"}</div>
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
            <div>
              <div className="text-white/50 mb-1">Lap-time model inputs:</div>
              <pre className="text-white/70 whitespace-pre-wrap bg-panel2 rounded p-2">
                {JSON.stringify(p.feature_rows["target_laptime"], null, 2)}
              </pre>
              {p.approximated_features["target_laptime"]?.length > 0 && (
                <p className="text-white/40 mt-1">
                  Auto-filled from training-data medians (require multi-lap history this snapshot can&apos;t
                  supply): {p.approximated_features["target_laptime"].join(", ")}
                </p>
              )}
            </div>
            <div>
              <div className="text-white/50 mb-1">Pit-decision model inputs:</div>
              <pre className="text-white/70 whitespace-pre-wrap bg-panel2 rounded p-2">
                {JSON.stringify(p.feature_rows["target_pit_next_lap"], null, 2)}
              </pre>
              {p.approximated_features["target_pit_next_lap"]?.length > 0 && (
                <p className="text-white/40 mt-1">
                  Auto-filled from training-data medians: {p.approximated_features["target_pit_next_lap"].join(", ")}
                </p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
