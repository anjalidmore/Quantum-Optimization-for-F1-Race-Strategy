import { artifactUrl, XaiFairness } from "@/lib/api";
import { ArtifactImage } from "@/components/ArtifactImage";
import { pct, TARGET_LABEL } from "@/lib/format";

/**
 * Identity's share of attribution against what an even spread would give.
 * A model leaning on driver/team dummies has learned "this driver laps like this"
 * rather than "a tyre this old on a track this hot laps like this".
 */
export function FairnessCards({ fairness }: { fairness: XaiFairness }) {
  return (
    <section>
      <h2 className="text-lg font-semibold text-white mb-1">Fairness — race state, or who is driving?</h2>
      <p className="text-sm text-white/50 mb-3 max-w-3xl">
        Task 5 kept one-hot driver and team dummies. A model leaning on them cannot generalise to an unseen
        driver, and would give two cars in an identical race state different calls purely because of the name on
        the car.
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
                  <ArtifactImage src={artifactUrl(f.figure)} alt={`${target} fairness`} className="w-full rounded" />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
