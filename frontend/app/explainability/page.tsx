import { api, ApiError, artifactUrl } from "@/lib/api";
import { DatasetBadge } from "@/components/DatasetBadge";
import { ArtifactImage } from "@/components/ArtifactImage";
import LapInspector from "@/components/xai/LapInspector";
import { StratificationTable } from "@/components/xai/StratificationTable";
import { FairnessCards } from "@/components/xai/FairnessCards";
import { ExplainedPredictions } from "@/components/xai/ExplainedPredictions";
import { GlobalShapAndTrust } from "@/components/xai/GlobalShapAndTrust";

/** Task 8: everything on this page is read from artifacts/xai/, except the lap
 *  inspector's counterfactual, which the saved network computes on request. */
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
      <div className="card border-accent">
        <div className="badge badge-warning">No explanations available</div>
        <p className="text-sm text-paper-700 mt-2">{error ?? "Run the explainability stage to generate results."}</p>
        <p className="text-sm text-paper-500 mt-1">
          <code className="text-paper-700">python scripts/build_all.py --force</code>
        </p>
      </div>
    );
  }

  const targets = Object.keys(summary.targets);
  const [explanations, shap, trust, strat] = await Promise.all([
    Promise.all(targets.map((t) => api.xaiExplanation(t).catch(() => null))),
    Promise.all(targets.map((t) => api.xaiShap(t).catch(() => null))),
    Promise.all(targets.map((t) => api.xaiTrust(t).catch(() => null))),
    api.xaiStratification().catch(() => null),
  ]);

  return (
    <div className="space-y-8">
      <section>
        <div className="flex items-center gap-3 flex-wrap">
          <h1 className="text-2xl font-bold text-paper-900">Explainability</h1>
          <DatasetBadge source={summary.dataset_source} />
          <span className="badge">Task 8</span>
        </div>
        <p className="text-paper-500 mt-1 max-w-3xl">
          Explains the trained <strong className="text-paper-700">Task 7 DNN</strong> on the chronological test
          laps: which race-state factors moved each prediction (SHAP, LIME), what change would flip it
          (counterfactual), how far to trust it, and whether it performs evenly across drivers, teams and
          compounds. These describe the model&rsquo;s behaviour — not causes, and not strategy instructions.
        </p>
      </section>

      <section>
        <h2 className="text-lg font-semibold text-paper-900 mb-1">Inspect a race scenario</h2>
        <p className="text-sm text-paper-500 mb-3">
          Pick any test lap. SHAP and trust come from the committed Task 8 run; the tyre-age counterfactual is
          computed live by the saved network.
        </p>
        <LapInspector />
      </section>

      {summary.feature_importance_figure && (
        <section className="card">
          <h2 className="font-semibold text-paper-900 mb-1">Global feature importance</h2>
          <p className="text-sm text-paper-500 mb-3">
            Permutation importance of the Task 7 DNN on the test laps: how much performance is lost when one
            feature is shuffled. This is a <em>global</em> summary; the cards below are <em>local</em>{" "}
            explanations of single predictions.
          </p>
          <ArtifactImage
            src={artifactUrl(summary.feature_importance_figure)}
            alt="Global feature importance"
            className="w-full rounded"
          />
        </section>
      )}

      {strat && <StratificationTable strat={strat} />}

      <FairnessCards fairness={fairness} />

      <ExplainedPredictions targets={targets} explanations={explanations} trust={trust} />

      <GlobalShapAndTrust targets={targets} shap={shap} trust={trust} />
    </div>
  );
}
