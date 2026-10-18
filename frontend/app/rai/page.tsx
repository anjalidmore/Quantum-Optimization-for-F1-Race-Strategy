import { api, ApiError } from "@/lib/api";
import { DatasetBadge } from "@/components/DatasetBadge";
import { PageHero } from "@/components/PageHero";
import { RaiView, type RaiTargetData } from "@/components/rai/RaiView";
import { buildModelCards } from "@/lib/modelCards";
import { buildComparison, TARGETS, type Target } from "@/lib/modelComparison";

/**
 * Responsible AI: bias and fairness, confidence and uncertainty, and model
 * cards for the Simple, ML and DNN models.
 *
 * Fairness and confidence come from /api/rai/summary, which runs the saved
 * models over the shared held-out laps (inference only). Model cards are
 * assembled from the registry and metrics endpoints the rest of the
 * dashboard already reads. Every source is optional; a missing one shows
 * "Not available" or "Insufficient Data" in place, never a stand-in number.
 */
export default async function RaiPage() {
  const [dl, ml, registry, dlModels, history, dlMetrics, manifest] = await Promise.all([
    api.dlComparison().catch(() => null),
    api.comparison().catch(() => null),
    api.models().catch(() => null),
    api.dlModels().catch(() => null),
    api.dlHistory().catch(() => null),
    api.dlMetrics().catch(() => null),
    api.artifacts().catch(() => null),
  ]);
  const inputs = { dl, ml, registry, dlModels: dlModels?.models ?? null, history };

  const data: Partial<Record<Target, RaiTargetData>> = {};
  await Promise.all(
    TARGETS.map(async (t) => {
      let rai = null;
      let raiError: string | null = null;
      try {
        rai = await api.raiSummary(t.key);
      } catch (e) {
        raiError = e instanceof ApiError ? `The RAI endpoint returned ${e.status}: ${e.message}` : "Could not reach the backend API.";
      }
      const comparison = buildComparison(t.key, inputs);
      const cards = comparison
        ? buildModelCards(comparison, registry, dlModels?.models ?? null, dlMetrics, manifest?.dataset_source ?? null)
        : null;
      if (rai || comparison) data[t.key] = { rai, raiError, comparison, cards };
    }),
  );

  return (
    <div className="space-y-10 pb-10">
      <PageHero
        photo="/images/photo-pitlane.jpg"
        focus="50% 60%"
        eyebrow="Responsible AI"
        title="Responsible AI (RAI)"
        blurb="Who each model serves well, how sure it is, and what it is for. The Simple, ML and DNN models checked for performance gaps across teams, tyres and drivers, for how far their confidence can be trusted, and documented in model cards built from the project's own records."
      />
      {manifest && (
        <div className="flex flex-wrap items-center gap-3">
          <DatasetBadge source={manifest.dataset_source} />
          <span className="t-micro">
            Read live from /api/rai/summary (inference over saved models, no retraining) and the Task 6/7 metric endpoints.
          </span>
        </div>
      )}
      {Object.keys(data).length ? (
        <RaiView data={data} />
      ) : (
        <div className="card border-accent">
          <span className="badge badge-warning">No trained models available</span>
          <p className="t-body mt-2 text-[13px]">Run <code>python scripts/build_all.py</code> to train them.</p>
        </div>
      )}
    </div>
  );
}
