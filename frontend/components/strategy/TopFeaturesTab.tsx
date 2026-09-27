"use client";

import { useEffect, useState } from "react";
import { api, ApiError, FeatureDescriptor, Registry, TopFeaturesResponse } from "@/lib/api";
import { fmt } from "@/lib/format";

const TARGETS: { value: string; label: string }[] = [
  { value: "target_laptime", label: "Lap time" },
  { value: "target_pit_next_lap", label: "Pit decision" },
];

/**
 * The narrow simulator: the eight highest-ranked features only, every other
 * model input filled from its training median. Same layout as the full
 * scenario — inputs left, answer right — so the two tabs read the same way.
 */
export function TopFeaturesTab({ registry }: { registry: Registry }) {
  const [target, setTarget] = useState("target_laptime");
  const [data, setData] = useState<TopFeaturesResponse | null>(null);
  const [values, setValues] = useState<Record<string, string>>({});
  const [showAuto, setShowAuto] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [result, setResult] = useState<
    { prediction: number; model: string } | { probability_pit: number; predicted_class: number; model: string } | null
  >(null);
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
    <div className="grid gap-x-12 gap-y-10 lg:grid-cols-[1fr_minmax(360px,420px)]">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <div className="flex flex-wrap items-end gap-x-8 gap-y-4">
          <div>
            <label htmlFor="tf-target" className="stat-label block">
              Target
            </label>
            <div className="mt-1.5 w-[170px]">
              <select id="tf-target" className="input" value={target} onChange={(e) => setTarget(e.target.value)}>
                {TARGETS.map((t) => (
                  <option key={t.value} value={t.value}>
                    {t.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <p className="t-micro max-w-[40ch]">
            The eight features ranked highest for this task&rsquo;s best model. Ranking method:{" "}
            <span className="text-paper-500">{data?.ranking_method ?? "loading…"}</span>
          </p>
        </div>

        {loadError && (
          <div className="card mt-6 border-accent" role="alert">
            <span className="badge badge-danger">Could not load features</span>
            <p className="t-body mt-2.5 text-[13px]">{loadError}</p>
          </div>
        )}

        {data && (
          <>
            <fieldset className="mt-7 border-t border-track-300 pt-5">
              <legend className="sr-only">Feature values</legend>
              <p className="font-display text-[15px] font-semibold text-paper-900">
                Top {data.top_features.length} features
              </p>
              <div className="mt-4 grid gap-x-8 gap-y-5 sm:grid-cols-2">
                {data.top_features.map((f) => (
                  <FeatureInput
                    key={f.feature}
                    descriptor={f}
                    value={values[f.feature] ?? ""}
                    onChange={(v) => setValues((s) => ({ ...s, [f.feature]: v }))}
                  />
                ))}
              </div>
            </fieldset>

            {data.remaining_features.length > 0 && (
              <div className="mt-6 border-t border-track-300 pt-5">
                <button
                  type="button"
                  onClick={() => setShowAuto(!showAuto)}
                  aria-expanded={showAuto}
                  aria-controls="auto-filled"
                  className="text-[13px] font-medium text-paper-900 underline decoration-edge underline-offset-[3px] hover:decoration-paper-900"
                >
                  {showAuto ? "Hide" : "Show"} the {data.remaining_features.length} inputs filled from training
                  medians
                </button>
                {showAuto && (
                  <table id="auto-filled" className="t-table mt-3.5">
                    <thead>
                      <tr>
                        <th scope="col">Feature</th>
                        <th scope="col" className="num">Median used</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.remaining_features.map((f) => (
                        <tr key={f.feature}>
                          <td>{f.display_name}</td>
                          <td className="num">{f.median !== undefined ? fmt(f.median, 3) : "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            )}

            <div className="mt-7 border-t border-track-300 pt-6">
              <button type="submit" disabled={predicting} className="btn-primary">
                {predicting ? "Running model…" : "Predict"}
              </button>
            </div>

            {predictError && (
              <div className="card mt-6 border-accent" role="alert">
                <span className="badge badge-danger">Request failed</span>
                <p className="t-body mt-2.5 text-[13px]">{predictError}</p>
              </div>
            )}
          </>
        )}
      </form>

      <div className="lg:sticky lg:top-[76px] lg:self-start">
        <div className="card card-focal" aria-live="polite">
          {result ? (
            "prediction" in result ? (
              <>
                <p className="stat-label">Predicted lap time</p>
                <p className="t-display-md mt-1">
                  {fmt(result.prediction, 3)}
                  <span className="ml-1.5 font-display text-[20px] font-semibold text-paper-500">s</span>
                </p>
              </>
            ) : (
              <>
                <p className="stat-label">Pit probability</p>
                <p className="t-display-md mt-1">
                  {fmt(result.probability_pit * 100, 1)}
                  <span className="ml-1.5 font-display text-[20px] font-semibold text-paper-500">%</span>
                </p>
                <p className="t-body mt-2 text-[13px]">
                  At the tuned threshold this is {result.predicted_class === 1 ? "a pit call" : "stay out"}.
                </p>
              </>
            )
          ) : (
            <>
              <p className="stat-label">Prediction</p>
              <p className="t-display-md mt-1 text-paper-500">{predicting ? "Running…" : "Not run yet"}</p>
            </>
          )}
          <p className="t-micro mt-4 border-t border-track-300 pt-3">
            {result
              ? `${result.model} — Task 5 selected features, eight of them set by hand.`
              : `${registry.models.length} models registered. Set the eight values, then predict.`}
          </p>
        </div>
      </div>
    </div>
  );
}

/**
 * One feature input. The description is visible text, not a question-mark
 * tooltip: it was the same sentence twice, and the tooltip hid it behind a click.
 */
function FeatureInput({
  descriptor,
  value,
  onChange,
}: {
  descriptor: FeatureDescriptor;
  value: string;
  onChange: (v: string) => void;
}) {
  const id = `tf-${descriptor.feature}`;
  const hasRange = descriptor.min !== undefined && descriptor.max !== undefined;
  return (
    <div>
      <label htmlFor={id} className="stat-label block">
        {descriptor.display_name}
        {descriptor.unit && <span className="text-paper-400"> ({descriptor.unit})</span>}
      </label>
      <div className="mt-1.5 flex items-center gap-2.5">
        <div className="w-[110px]">
          <input
            id={id}
            type="number"
            step="any"
            className="input"
            value={value}
            min={descriptor.min}
            max={descriptor.max}
            aria-describedby={`${id}-help`}
            onChange={(e) => onChange(e.target.value)}
          />
        </div>
        {hasRange && (
          <span className="t-micro tabular-nums whitespace-nowrap">
            {fmt(descriptor.min!, 2)}–{fmt(descriptor.max!, 2)}
          </span>
        )}
      </div>
      <p id={`${id}-help`} className="t-micro mt-1.5 max-w-[34ch]">
        {descriptor.description}
      </p>
    </div>
  );
}
