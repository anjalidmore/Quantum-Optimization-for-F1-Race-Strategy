import { artifactUrl, XaiShap, XaiTrust } from "@/lib/api";
import { ArtifactImage } from "@/components/ArtifactImage";
import { TARGET_LABEL } from "@/lib/format";

const BAND_CLASS: Record<string, string> = {
  HIGH: "badge-success",
  MODERATE: "badge-warning",
  LOW: "badge-warning",
  "DO NOT ACT": "badge-danger",
};

const TRUST_TEXT: Record<string, string> = {
  confidence:
    "Distance of the prediction from the model's tuned decision threshold. Not used for lap time, which has no decision point; the other weights are renormalised.",
  model_agreement: "Do the Task 7 DNN and Task 6's selected model say the same thing?",
  explanation_stability: "Do SHAP and LIME name the same top-3 drivers of this prediction?",
  input_validity:
    "Share of inputs inside the training data's 1st–99th percentile — is the model extrapolating?",
};

/** Mean |SHAP| per feature across every test lap, and the trust score's definition. */
export function GlobalShapAndTrust({
  targets,
  shap,
  trust,
}: {
  targets: string[];
  shap: (XaiShap | null)[];
  trust: (XaiTrust | null)[];
}) {
  const weights = trust.find((t) => t) ?? null;

  return (
    <>
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
                  {s.deep_network.exact ? " (exact)" : " (sampled)"} · classical: {s.classical.explainer ?? "—"} on{" "}
                  {s.classical.model}
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
                      src={artifactUrl(s.figure)}
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

      {weights && (
        <section className="card">
          <h2 className="font-semibold text-white mb-2">How the trust score works</h2>
          <code className="text-sm text-sky-300">{weights.formula}</code>
          {weights.note && <p className="text-xs text-amber-400/70 mt-2">{weights.note}</p>}
          <dl className="grid md:grid-cols-4 gap-4 mt-4 text-sm">
            {Object.entries(weights.weights).map(([k, w]) => (
              <div key={k}>
                <dt className="text-white/80">
                  {k.replace(/_/g, " ")} ({w.toFixed(2)})
                </dt>
                <dd className="text-white/50 mt-1">{TRUST_TEXT[k] ?? ""}</dd>
              </div>
            ))}
          </dl>
          <div className="mt-4 grid sm:grid-cols-2 lg:grid-cols-4 gap-2 text-xs">
            {Object.entries(weights.bands).map(([band, meaning]) => (
              <div key={band} className="rounded border border-white/10 p-2">
                <span className={`badge ${BAND_CLASS[band] ?? ""}`}>{band}</span>
                <p className="text-white/50 mt-1">{meaning}</p>
              </div>
            ))}
          </div>
        </section>
      )}
    </>
  );
}
