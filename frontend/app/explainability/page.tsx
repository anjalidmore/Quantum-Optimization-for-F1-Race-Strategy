import { api, ApiError, artifactUrl } from "@/lib/api";
import { DatasetBadge } from "@/components/DatasetBadge";
import { ArtifactImage } from "@/components/ArtifactImage";

function fmt(x: number | null | undefined, digits = 3): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return x.toFixed(digits);
}

function pct(x: number | null | undefined, digits = 1): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return `${(x * 100).toFixed(digits)}%`;
}

const TARGET_LABEL: Record<string, string> = {
  target_laptime: "Lap-time regression",
  target_pit_next_lap: "Pit-decision classification",
};

const BAND_CLASS: Record<string, string> = {
  HIGH: "badge-success",
  MODERATE: "badge-warning",
  LOW: "badge-warning",
  "DO NOT ACT": "badge-danger",
};

function TrustPill({ score, band }: { score: number; band: string }) {
  return (
    <span className={`badge ${BAND_CLASS[band] ?? ""}`}>
      trust {score.toFixed(2)} · {band}
    </span>
  );
}

export default async function ExplainabilityPage() {
  let summary = null;
  let fairness = null;
  let error: string | null = null;

  try {
    [summary, fairness] = await Promise.all([api.xaiSummary(), api.xaiFairness()]);
  } catch (e) {
    error = e instanceof ApiError ? e.message : "Could not reach the backend API.";
  }

  if (error || !summary || !fairness) {
    return (
      <div className="card border-red-500/30 bg-red-500/5">
        <div className="badge badge-warning">No explanations available</div>
        <p className="text-sm text-white/70 mt-2">
          {error ?? "Run the explainability stage to generate results."}
        </p>
        <p className="text-sm text-white/50 mt-1">
          <code className="text-white/80">python scripts/build_all.py --force</code>
        </p>
      </div>
    );
  }

  const targets = Object.keys(summary.targets);
  const [explanations, shap, trust] = await Promise.all([
    Promise.all(targets.map((t) => api.xaiExplanation(t).catch(() => null))),
    Promise.all(targets.map((t) => api.xaiShap(t).catch(() => null))),
    Promise.all(targets.map((t) => api.xaiTrust(t).catch(() => null))),
  ]);

  return (
    <div className="space-y-8">
      <section>
        <div className="flex items-center gap-3 flex-wrap">
          <h1 className="text-2xl font-bold text-white">Explainability</h1>
          <DatasetBadge source={summary.dataset_source} />
          <span className="badge">Task 8</span>
        </div>
        <p className="text-white/60 mt-1 max-w-3xl">
          Every prediction traced to named race-state factors, cross-checked between two
          independent explanation methods, and scored for how much it should be trusted.
          Nothing here is a black box.
        </p>
      </section>

      {/* --- fairness ------------------------------------------------------ */}
      <section>
        <h2 className="text-lg font-semibold text-white mb-1">
          Fairness — race state, or who is driving?
        </h2>
        <p className="text-sm text-white/50 mb-3 max-w-3xl">
          Task 5 kept one-hot driver and team dummies. A model leaning on them has learned
          &ldquo;this driver laps like this&rdquo; rather than &ldquo;a tyre this old on a track
          this hot laps like this&rdquo; — it cannot generalise to an unseen driver, and would
          give two cars in an identical race state different calls purely because of the name on
          the car. The measurement is identity&rsquo;s share of total attribution against what an
          even spread would give.
        </p>
        <div className="grid gap-4 md:grid-cols-2">
          {Object.entries(fairness).map(([target, f]) => {
            const healthy = (f.concentration_ratio ?? 0) < 1.2;
            return (
              <div key={target} className="card">
                <div className="flex items-center justify-between gap-2 flex-wrap">
                  <h3 className="font-semibold text-white">{TARGET_LABEL[target] ?? target}</h3>
                  <span className={`badge ${healthy ? "badge-success" : "badge-warning"}`}>
                    {f.concentration_ratio === null
                      ? "no identity features"
                      : `${f.concentration_ratio}× concentration`}
                  </span>
                </div>
                <div className="mt-3 h-3 w-full rounded-full overflow-hidden bg-white/5 flex">
                  <div
                    className="bg-red-500/70"
                    style={{ width: `${f.identity_attribution_share * 100}%` }}
                    title={`identity: ${pct(f.identity_attribution_share)}`}
                  />
                  <div
                    className="bg-emerald-500/70"
                    style={{ width: `${f.race_state_attribution_share * 100}%` }}
                    title={`race state: ${pct(f.race_state_attribution_share)}`}
                  />
                </div>
                <div className="flex justify-between text-xs text-white/50 mt-1">
                  <span>identity {pct(f.identity_attribution_share)}</span>
                  <span>race state {pct(f.race_state_attribution_share)}</span>
                </div>
                <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm mt-4">
                  <dt className="text-white/50">Identity features</dt>
                  <dd className="text-white/80 text-right">
                    {f.n_identity_features} of {f.n_features}
                  </dd>
                  <dt className="text-white/50">Expected if uniform</dt>
                  <dd className="text-white/80 text-right">{pct(f.expected_share_if_uniform)}</dd>
                </dl>
                <p className="text-sm text-white/60 mt-3">{f.reading.replace(/\*\*/g, "")}</p>
                {f.figure && (
                  <div className="mt-3">
                    <ArtifactImage
                      src={artifactUrl(`figures/${f.figure}`)}
                      alt={`${target} fairness`}
                      className="w-full rounded"
                    />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </section>

      {/* --- per-prediction explanations ------------------------------------ */}
      <section>
        <h2 className="text-lg font-semibold text-white mb-1">Explained predictions</h2>
        <p className="text-sm text-white/50 mb-3">
          The sentence a race engineer would read, with the SHAP factors behind it and what would
          have to change to flip the call.
        </p>
        <div className="space-y-6">
          {targets.map((target, i) => {
            const exp = explanations[i];
            const tr = trust[i];
            if (!exp) return null;
            return (
              <div key={target} className="space-y-3">
                <h3 className="font-semibold text-white">
                  {TARGET_LABEL[target] ?? target}
                  <span className="text-white/40 font-normal text-sm ml-2">
                    deep network vs {exp.classical_model_explained}
                  </span>
                </h3>
                <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                  {Object.entries(exp.rows).map(([label, row]) => {
                    const comps = tr?.rows?.[label]?.components;
                    return (
                      <div key={label} className="card">
                        <div className="flex items-start justify-between gap-2 flex-wrap">
                          <span className="text-sm text-white/50">
                            {label.replace(/_/g, " ")} · lap {row.lap}
                          </span>
                          <TrustPill score={row.trust_score} band={row.trust_band.label} />
                        </div>
                        <p className="text-white/90 text-sm mt-3">{row.narrative}</p>

                        {comps && (
                          <dl className="grid grid-cols-2 gap-x-3 text-xs mt-3 text-white/50">
                            <dt>confidence</dt>
                            <dd className="text-right text-white/70">{fmt(comps.confidence)}</dd>
                            <dt>model agreement</dt>
                            <dd className="text-right text-white/70">{fmt(comps.model_agreement)}</dd>
                            <dt>explanation stability</dt>
                            <dd className="text-right text-white/70">
                              {fmt(comps.explanation_stability)}
                            </dd>
                          </dl>
                        )}

                        <div className="mt-3">
                          <div className="text-xs text-white/40 uppercase tracking-wide mb-1">
                            Top factors (SHAP)
                          </div>
                          <ul className="text-sm space-y-1">
                            {row.top_factors.map((f) => (
                              <li key={f.feature} className="flex justify-between gap-2">
                                <code className="text-white/70">{f.feature}</code>
                                <span
                                  className={
                                    f.shap_value > 0 ? "text-red-400 tabular-nums" : "text-sky-400 tabular-nums"
                                  }
                                >
                                  {f.shap_value > 0 ? "+" : ""}
                                  {f.shap_value.toFixed(4)}
                                </span>
                              </li>
                            ))}
                          </ul>
                        </div>

                        <p className="text-xs text-white/50 mt-3 border-t border-white/5 pt-2">
                          {row.counterfactual_sentence}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* --- global SHAP ---------------------------------------------------- */}
      <section>
        <h2 className="text-lg font-semibold text-white mb-3">Global SHAP attribution</h2>
        <div className="grid gap-4 md:grid-cols-2">
          {targets.map((target, i) => {
            const s = shap[i];
            if (!s) return null;
            const top = s.deep_network.ranking.slice(0, 10);
            const max = Math.max(...top.map((r) => r.mean_abs_shap), 1e-12);
            return (
              <div key={target} className="card">
                <h3 className="font-semibold text-white">{TARGET_LABEL[target] ?? target}</h3>
                <p className="text-xs text-white/40 mt-1">
                  deep network: {s.deep_network.explainer}
                  {s.deep_network.exact ? " (exact)" : " (sampled)"} · classical:{" "}
                  {s.classical.explainer ?? "—"} on {s.classical.model}
                </p>
                <ul className="mt-3 space-y-1.5">
                  {top.map((r) => (
                    <li key={r.feature} className="text-sm">
                      <div className="flex justify-between gap-2">
                        <code className="text-white/70 truncate">{r.feature}</code>
                        <span className="text-white/50 tabular-nums">{r.mean_abs_shap.toFixed(5)}</span>
                      </div>
                      <div className="h-1.5 rounded-full bg-white/5 mt-1">
                        <div
                          className="h-full rounded-full bg-sky-500/70"
                          style={{ width: `${(r.mean_abs_shap / max) * 100}%` }}
                        />
                      </div>
                    </li>
                  ))}
                </ul>
                {s.figure && (
                  <div className="mt-4">
                    <ArtifactImage
                      src={artifactUrl(`figures/${s.figure}`)}
                      alt={`${target} SHAP summary`}
                      className="w-full rounded"
                    />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </section>

      {/* --- the trust formula ---------------------------------------------- */}
      {trust[0] && (
        <section className="card">
          <h2 className="font-semibold text-white mb-2">How the trust score works</h2>
          <code className="text-sm text-sky-300">{trust[0].formula}</code>
          <dl className="grid md:grid-cols-3 gap-4 mt-4 text-sm">
            <div>
              <dt className="text-white/80">confidence (0.40)</dt>
              <dd className="text-white/50 mt-1">
                How far the prediction is from the decision boundary. A prediction sitting on the
                boundary is unusable however well explained.
              </dd>
            </div>
            <div>
              <dt className="text-white/80">model agreement (0.30)</dt>
              <dd className="text-white/50 mt-1">
                Do the deep network and the classical model say the same thing? One disagreeing
                means at least one is wrong and you cannot tell which.
              </dd>
            </div>
            <div>
              <dt className="text-white/80">explanation stability (0.30)</dt>
              <dd className="text-white/50 mt-1">
                Do SHAP and LIME name the same drivers? If two established methods disagree about
                <em> why</em>, the explanation is not trustworthy even when the prediction is right.
              </dd>
            </div>
          </dl>
          <div className="mt-4 grid sm:grid-cols-2 lg:grid-cols-4 gap-2 text-xs">
            {Object.entries(trust[0].bands).map(([band, meaning]) => (
              <div key={band} className="rounded border border-white/10 p-2">
                <span className={`badge ${BAND_CLASS[band] ?? ""}`}>{band}</span>
                <p className="text-white/50 mt-1">{meaning}</p>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
