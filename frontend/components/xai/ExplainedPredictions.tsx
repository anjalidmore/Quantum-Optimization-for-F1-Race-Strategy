import { artifactUrl, XaiExplanation, XaiTrust } from "@/lib/api";
import { ArtifactImage } from "@/components/ArtifactImage";
import { fmt, TARGET_LABEL } from "@/lib/format";

const BAND_CLASS: Record<string, string> = {
  HIGH: "badge-success",
  MODERATE: "badge-warning",
  LOW: "badge-warning",
  "DO NOT ACT": "badge-danger",
};

/** One card per explained lap: the sentence, its SHAP factors, trust, and the counterfactual. */
export function ExplainedPredictions({
  targets,
  explanations,
  trust,
}: {
  targets: string[];
  explanations: (XaiExplanation | null)[];
  trust: (XaiTrust | null)[];
}) {
  return (
    <section>
      <h2 className="text-lg font-semibold text-paper-900 mb-1">Explained predictions</h2>
      <p className="text-sm text-paper-500 mb-3">
        The sentence a race engineer would read, with the SHAP factors behind it and what would have to change
        to flip the call.
      </p>
      <div className="space-y-6">
        {targets.map((target, i) => {
          const exp = explanations[i];
          const tr = trust[i];
          if (!exp) return null;
          return (
            <div key={target} className="space-y-3">
              <h3 className="font-semibold text-paper-900">
                {TARGET_LABEL[target] ?? target}
                <span className="text-paper-400 font-normal text-sm ml-2">
                  deep network vs {exp.classical_model_explained}
                </span>
              </h3>
              <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                {Object.entries(exp.rows).map(([label, row]) => {
                  const comps = tr?.rows?.[label]?.components;
                  return (
                    <div key={label} className="card">
                      <div className="flex items-start justify-between gap-2 flex-wrap">
                        <span className="text-sm text-paper-500">
                          {label.replace(/_/g, " ")} · {row.driver} ({row.team}, {row.compound}) · lap {row.lap}
                        </span>
                        <span className={`badge ${BAND_CLASS[row.trust_band.label] ?? ""}`}>
                          trust {row.trust_score.toFixed(2)} · {row.trust_band.label}
                        </span>
                      </div>
                      <p className="text-paper-900 text-sm mt-3">{row.narrative}</p>

                      {comps && (
                        <dl className="grid grid-cols-2 gap-x-3 text-xs mt-3 text-paper-500">
                          {Object.entries(comps).map(([k, v]) => (
                            <div key={k} className="contents">
                              <dt>{k.replace(/_/g, " ")}</dt>
                              <dd className="text-right text-paper-700">{fmt(v as number)}</dd>
                            </div>
                          ))}
                        </dl>
                      )}

                      <div className="mt-3">
                        <div className="text-xs text-paper-400 uppercase tracking-wide mb-1">Top factors (SHAP)</div>
                        <ul className="text-sm space-y-1">
                          {row.top_factors.map((f) => (
                            <li key={f.feature} className="flex justify-between gap-2">
                              <code className="text-paper-700">{f.feature}</code>
                              <span
                                className={f.shap_value > 0 ? "data-pos tabular-nums" : "data-neg tabular-nums"}
                              >
                                {f.shap_value > 0 ? "+" : ""}
                                {f.shap_value.toFixed(4)}
                              </span>
                            </li>
                          ))}
                        </ul>
                      </div>

                      {row.lime_top3 && (
                        <p className="text-xs text-paper-500 mt-3">
                          LIME top 3: <code className="text-paper-700">{row.lime_top3.join(", ")}</code>
                          {row.lime?.local_r2 !== undefined && ` (surrogate R² ${fmt(row.lime.local_r2)})`} · SHAP
                          top 3: <code className="text-paper-700">{(row.shap_top3 ?? []).join(", ")}</code>
                        </p>
                      )}

                      <p className="text-xs text-paper-500 mt-3 border-t border-track-300 pt-2">
                        {row.counterfactual_sentence}
                      </p>

                      {row.figures && (
                        <details className="mt-2">
                          <summary className="text-xs text-paper-500 cursor-pointer hover:text-paper-900">
                            SHAP · LIME · counterfactual plots
                          </summary>
                          <div className="space-y-2 mt-2">
                            {Object.entries(row.figures).map(([k, src]) => (
                              <ArtifactImage
                                key={k}
                                src={artifactUrl(src)}
                                alt={`${label} ${k}`}
                                className="w-full rounded"
                              />
                            ))}
                          </div>
                        </details>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
