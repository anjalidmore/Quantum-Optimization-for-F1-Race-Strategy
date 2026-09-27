"use client";

import { DataOptions, RaceState, Registry } from "@/lib/api";
import { Tooltip } from "@/components/Tooltip";

const WEATHER_OPTIONS = ["dry", "damp", "wet", "extreme"];
const TRACK_STATUS_OPTIONS = ["GREEN", "YELLOW", "SC", "VSC", "RED"];

const FIELD_HELP: Record<string, string> = {
  driver: "The driver whose race-state this prediction is for.",
  team: "The driver's constructor/team for this race.",
  current_lap: "The lap the car is currently on.",
  total_laps: "Total race distance in laps.",
  tyre_compound:
    "The tyre compound currently fitted: Soft (fastest, wears quickest), Medium, or Hard (slowest, most durable).",
  tyre_age: "How many laps the current tyre set has completed.",
  track_temperature:
    "Track surface temperature — higher temperatures generally accelerate tyre degradation.",
  weather: "Current weather severity. Wetter conditions favour intermediate/wet tyres.",
  fuel_kg: "Estimated fuel remaining on board — affects car weight and pace.",
  track_status:
    "Race-control flag state: Green (racing), Yellow (caution), SC (Safety Car), VSC (Virtual Safety Car), Red (stopped).",
  current_position: "The car's current position in the race order.",
};

export type Scenario = { name: string; description: string; values: Partial<RaceState> };

/**
 * Presets built from the real data's own ranges.
 *
 * Track temperature is pinned to the range this model was trained on, never a
 * hard-coded "hot track" guess: a value the model never saw (48°C for a session
 * that peaked at 31°C) makes it extrapolate and return something arbitrary that
 * still looks like an answer.
 */
export function buildScenarios(options: DataOptions): Scenario[] {
  const laps = options.total_laps_hint;
  const mid = Math.max(1, Math.round(laps * 0.4));
  const late = Math.max(1, Math.round(laps * 0.85));
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

/** Keep the form physically possible: a lap cannot exceed the race, tyres cannot be older than the stint. */
export function clampForm(form: RaceState): RaceState {
  const total_laps = Math.max(1, form.total_laps);
  const current_lap = Math.min(Math.max(1, form.current_lap), total_laps);
  const tyre_age = Math.min(Math.max(0, form.tyre_age), current_lap);
  return { ...form, total_laps, current_lap, tyre_age };
}

export function Field({ label, help, children }: { label: string; help?: string; children: React.ReactNode }) {
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

/** The race-state inputs: preset buttons, the eleven fields, and which model to run. */
export function RaceStateForm({
  options,
  registry,
  form,
  set,
  scenarios,
  applyScenario,
  modelMode,
  setModelMode,
  laptimeModel,
  setLaptimeModel,
  pitModel,
  setPitModel,
}: {
  options: DataOptions;
  registry: Registry;
  form: RaceState;
  set: <K extends keyof RaceState>(key: K, value: RaceState[K]) => void;
  scenarios: Scenario[];
  applyScenario: (s: Scenario) => void;
  modelMode: "best" | "select";
  setModelMode: (m: "best" | "select") => void;
  laptimeModel: string;
  setLaptimeModel: (m: string) => void;
  pitModel: string;
  setPitModel: (m: string) => void;
}) {
  const regModels = registry.models.filter((m) => m.target === "target_laptime" && m.artifact);
  const clfModels = registry.models.filter((m) => m.target === "target_pit_next_lap" && m.artifact);

  return (
    <>
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

      <div className="card grid grid-cols-2 md:grid-cols-4 gap-3">
        <Field label="Driver" help={FIELD_HELP.driver}>
          <select className="input" value={form.driver} onChange={(e) => set("driver", e.target.value)}>
            {options.drivers.map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        </Field>
        <Field label="Team" help={FIELD_HELP.team}>
          <select className="input" value={form.team} onChange={(e) => set("team", e.target.value)}>
            {options.teams.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </Field>
        <Field label="Total Laps" help={FIELD_HELP.total_laps}>
          <input type="number" min={1} className="input" value={form.total_laps}
                 onChange={(e) => set("total_laps", Number(e.target.value) || 1)} />
        </Field>
        <Field label="Current Lap" help={FIELD_HELP.current_lap}>
          <input type="number" min={1} max={form.total_laps} className="input" value={form.current_lap}
                 onChange={(e) => set("current_lap", Number(e.target.value) || 1)} />
        </Field>
        <Field label="Tyre Compound" help={FIELD_HELP.tyre_compound}>
          <select className="input" value={form.tyre_compound} onChange={(e) => set("tyre_compound", e.target.value)}>
            {options.compounds.map((c) => (
              <option key={c} value={c}>{c.charAt(0) + c.slice(1).toLowerCase()}</option>
            ))}
          </select>
        </Field>
        <Field label="Tyre Age (laps)" help={FIELD_HELP.tyre_age}>
          <input type="number" min={0} max={form.current_lap} className="input" value={form.tyre_age}
                 onChange={(e) => set("tyre_age", Number(e.target.value) || 0)} />
        </Field>
        <Field label="Track Temperature (°C)" help={FIELD_HELP.track_temperature}>
          <input type="number" className="input" value={form.track_temperature}
                 onChange={(e) => set("track_temperature", Number(e.target.value) || 0)} />
        </Field>
        <Field label="Weather" help={FIELD_HELP.weather}>
          <select className="input" value={form.weather} onChange={(e) => set("weather", e.target.value)}>
            {WEATHER_OPTIONS.map((w) => (
              <option key={w} value={w}>{w.charAt(0).toUpperCase() + w.slice(1)}</option>
            ))}
          </select>
        </Field>
        <Field label="Fuel State (kg)" help={FIELD_HELP.fuel_kg}>
          <input type="number" min={0} className="input" value={form.fuel_kg}
                 onChange={(e) => set("fuel_kg", Number(e.target.value) || 0)} />
        </Field>
        <Field label="Track Status" help={FIELD_HELP.track_status}>
          <select className="input" value={form.track_status} onChange={(e) => set("track_status", e.target.value)}>
            {TRACK_STATUS_OPTIONS.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </Field>
        <Field label="Current Position" help={FIELD_HELP.current_position}>
          <input type="number" min={1} max={24} className="input" value={form.current_position}
                 onChange={(e) => set("current_position", Number(e.target.value) || 1)} />
        </Field>
      </div>

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
                  <option key={m.model_name} value={m.model_name}>{m.model_name}</option>
                ))}
              </select>
            </Field>
            <Field label="Pit-Decision Model">
              <select className="input" value={pitModel} onChange={(e) => setPitModel(e.target.value)}>
                {clfModels.map((m) => (
                  <option key={m.model_name} value={m.model_name}>{m.model_name}</option>
                ))}
              </select>
            </Field>
          </div>
        )}
      </div>
    </>
  );
}
