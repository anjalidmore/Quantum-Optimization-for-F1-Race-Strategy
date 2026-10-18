"use client";

import { DataOptions, RaceState, Registry } from "@/lib/api";
import { TyreBadge } from "@/components/TyreBadge";

const WEATHER_OPTIONS = ["dry", "damp", "wet", "extreme"];
const TRACK_STATUS_OPTIONS = ["GREEN", "YELLOW", "SC", "VSC", "RED"];

/* Only the four inputs whose meaning is genuinely not obvious keep a note.
   There were eleven question-mark buttons; the rest said what the label said. */
const HELP: Record<string, string> = {
  tyre_age: "Laps completed on the current set — the main degradation driver.",
  track_temperature: "Surface temperature. Hotter surfaces accelerate wear.",
  fuel_kg: "Fuel remaining on board; affects car weight and pace.",
  track_status: "Race-control flag: SC is a safety car, VSC a virtual one.",
};

export type Scenario = { name: string; description: string; values: Partial<RaceState> };

/**
 * Presets built from the real data's own ranges.
 *
 * Track temperature is pinned to the range the model was trained on, never a
 * hard-coded "hot track" guess: a value it never saw makes it extrapolate and
 * return something arbitrary that still looks like an answer.
 */
export function buildScenarios(options: DataOptions): Scenario[] {
  const laps = options.total_laps_hint;
  const mid = Math.max(1, Math.round(laps * 0.4));
  const late = Math.max(1, Math.round(laps * 0.85));
  const { mean: tMean, max: tMax } = options.track_temperature_range;
  const tNormal = Math.round(tMean);
  const tHot = Math.round(tMax);
  return [
    { name: "Normal race", description: `Lap ${mid} of ${laps}, medium, 8 laps old, ${tNormal}°C`,
      values: { current_lap: mid, total_laps: laps, tyre_compound: "MEDIUM", tyre_age: 8, track_temperature: tNormal, weather: "dry", track_status: "GREEN" } },
    { name: "High degradation", description: `Soft, ${Math.min(mid, 22)} laps old, ${tHot}°C — the hottest this session saw`,
      values: { current_lap: mid, total_laps: laps, tyre_compound: "SOFT", tyre_age: Math.min(mid, 22), track_temperature: tHot, weather: "dry", track_status: "GREEN" } },
    { name: "Late-race pit call", description: `Lap ${late} of ${laps}, hard, 25 laps old`,
      values: { current_lap: late, total_laps: laps, tyre_compound: "HARD", tyre_age: Math.min(late, 25), track_temperature: tNormal, weather: "dry", track_status: "GREEN" } },
    { name: "Safety car", description: `Lap ${mid}, medium, safety car deployed`,
      values: { current_lap: mid, total_laps: laps, tyre_compound: "MEDIUM", tyre_age: 12, track_temperature: tNormal, track_status: "SC" } },
    { name: "Fresh tyres", description: `Lap ${Math.min(mid + 1, laps)}, soft, 1 lap old — just pitted`,
      values: { current_lap: Math.min(mid + 1, laps), total_laps: laps, tyre_compound: "SOFT", tyre_age: 1, track_temperature: tNormal } },
  ];
}

/** Keep the form physically possible: a lap cannot exceed the race, tyres cannot outlive the stint. */
export function clampForm(form: RaceState): RaceState {
  const total_laps = Math.max(1, form.total_laps);
  const current_lap = Math.min(Math.max(1, form.current_lap), total_laps);
  const tyre_age = Math.min(Math.max(0, form.tyre_age), current_lap);
  return { ...form, total_laps, current_lap, tyre_age };
}

function Field({
  label,
  htmlFor,
  help,
  width = "full",
  children,
}: {
  label: string;
  htmlFor: string;
  help?: string;
  width?: "num" | "med" | "full";
  children: React.ReactNode;
}) {
  // Inputs are sized to their content: a two-digit lap number does not need 380px.
  const w = width === "num" ? "w-[92px]" : width === "med" ? "w-[150px]" : "w-full max-w-[220px]";
  return (
    <div>
      <label htmlFor={htmlFor} className="stat-label block">
        {label}
      </label>
      <div className={`mt-1.5 ${w}`}>{children}</div>
      {help && <p className="t-micro mt-1.5 max-w-[28ch]">{help}</p>}
    </div>
  );
}

function Group({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <fieldset className="border-t border-track-300 pt-5">
      <legend className="sr-only">{title}</legend>
      <p className="font-display text-[15px] font-semibold text-paper-900">{title}</p>
      <div className="mt-4 flex flex-wrap gap-x-8 gap-y-5">{children}</div>
    </fieldset>
  );
}

/** The race-state inputs, grouped the way an engineer changes them. */
export function RaceStateForm({
  options,
  registry,
  form,
  set,
  scenarios,
  applyScenario,
  activeScenario,
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
  activeScenario: string | null;
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
    <div className="space-y-7">
      <div>
        <p className="font-display text-[15px] font-semibold text-paper-900">Start from a scenario</p>
        <p className="t-micro mt-1">Presets fill the fields below; nothing runs until you predict.</p>
        <div className="mt-3 flex flex-wrap gap-2">
          {scenarios.map((s) => {
            const active = activeScenario === s.name;
            return (
              <button
                key={s.name}
                type="button"
                aria-pressed={active}
                onClick={() => applyScenario(s)}
                className={`rounded-sm border px-3 py-1.5 text-[13px] transition-colors duration-[120ms] ${
                  active
                    ? "border-accent text-paper-900"
                    : "border-track-300 text-paper-500 hover:border-edge hover:text-paper-900"
                }`}
              >
                {s.name}
              </button>
            );
          })}
        </div>
        {/* The description was a hover-only title attribute; it is visible text now. */}
        <p className="t-micro mt-2.5 min-h-[1.2em]">
          {scenarios.find((s) => s.name === activeScenario)?.description ?? ""}
        </p>
      </div>

      <Group title="Car">
        <Field label="Driver" htmlFor="f-driver" width="med">
          <select id="f-driver" className="input" value={form.driver} onChange={(e) => set("driver", e.target.value)}>
            {options.drivers.map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        </Field>
        <Field label="Team" htmlFor="f-team">
          <select id="f-team" className="input" value={form.team} onChange={(e) => set("team", e.target.value)}>
            {options.teams.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </Field>
        <Field label="Fuel on board" htmlFor="f-fuel" help={HELP.fuel_kg} width="num">
          <input id="f-fuel" type="number" min={0} className="input" value={form.fuel_kg}
                 onChange={(e) => set("fuel_kg", Number(e.target.value) || 0)} />
        </Field>
      </Group>

      <Group title="Tyre">
        <Field label="Compound" htmlFor="f-compound" width="med">
          <select id="f-compound" className="input" value={form.tyre_compound}
                  onChange={(e) => set("tyre_compound", e.target.value)}>
            {options.compounds.map((c) => (
              <option key={c} value={c}>{c.charAt(0) + c.slice(1).toLowerCase()}</option>
            ))}
          </select>
        </Field>
        <div className="self-start pt-[26px]">
          <TyreBadge compound={form.tyre_compound} />
        </div>
        <Field label="Tyre age" htmlFor="f-age" help={HELP.tyre_age} width="num">
          <input id="f-age" type="number" min={0} max={form.current_lap} className="input" value={form.tyre_age}
                 onChange={(e) => set("tyre_age", Number(e.target.value) || 0)} />
        </Field>
      </Group>

      <Group title="Track">
        <Field label="Temperature °C" htmlFor="f-temp" help={HELP.track_temperature} width="num">
          <input id="f-temp" type="number" className="input" value={form.track_temperature}
                 onChange={(e) => set("track_temperature", Number(e.target.value) || 0)} />
        </Field>
        <Field label="Weather" htmlFor="f-weather" width="med">
          <select id="f-weather" className="input" value={form.weather} onChange={(e) => set("weather", e.target.value)}>
            {WEATHER_OPTIONS.map((w) => (
              <option key={w} value={w}>{w.charAt(0).toUpperCase() + w.slice(1)}</option>
            ))}
          </select>
        </Field>
        <Field label="Flag" htmlFor="f-status" help={HELP.track_status} width="med">
          <select id="f-status" className="input" value={form.track_status}
                  onChange={(e) => set("track_status", e.target.value)}>
            {TRACK_STATUS_OPTIONS.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </Field>
      </Group>

      <Group title="Race">
        <Field label="Current lap" htmlFor="f-lap" width="num">
          <input id="f-lap" type="number" min={1} max={form.total_laps} className="input" value={form.current_lap}
                 onChange={(e) => set("current_lap", Number(e.target.value) || 1)} />
        </Field>
        <Field label="Total laps" htmlFor="f-total" width="num">
          <input id="f-total" type="number" min={1} className="input" value={form.total_laps}
                 onChange={(e) => set("total_laps", Number(e.target.value) || 1)} />
        </Field>
        <Field label="Position" htmlFor="f-pos" width="num">
          <input id="f-pos" type="number" min={1} max={24} className="input" value={form.current_position}
                 onChange={(e) => set("current_position", Number(e.target.value) || 1)} />
        </Field>
      </Group>

      <Group title="Model">
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-[13px]">
          <label className="flex items-center gap-2 text-paper-700">
            <input type="radio" name="model-mode" checked={modelMode === "best"} onChange={() => setModelMode("best")} />
            Best performing
          </label>
          <label className="flex items-center gap-2 text-paper-700">
            <input type="radio" name="model-mode" checked={modelMode === "select"} onChange={() => setModelMode("select")} />
            Choose a model
          </label>
        </div>
        {modelMode === "select" && (
          <div className="flex w-full flex-wrap gap-x-8 gap-y-5">
            <Field label="Lap-time model" htmlFor="f-regmodel">
              <select id="f-regmodel" className="input" value={laptimeModel} onChange={(e) => setLaptimeModel(e.target.value)}>
                {regModels.map((m) => (
                  <option key={m.model_name} value={m.model_name}>{m.model_name}</option>
                ))}
              </select>
            </Field>
            <Field label="Pit-decision model" htmlFor="f-clfmodel">
              <select id="f-clfmodel" className="input" value={pitModel} onChange={(e) => setPitModel(e.target.value)}>
                {clfModels.map((m) => (
                  <option key={m.model_name} value={m.model_name}>{m.model_name}</option>
                ))}
              </select>
            </Field>
          </div>
        )}
      </Group>
    </div>
  );
}
