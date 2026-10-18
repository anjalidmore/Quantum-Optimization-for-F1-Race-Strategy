import { api } from "@/lib/api";
import { DatasetBadge } from "@/components/DatasetBadge";
import { PageHero } from "@/components/PageHero";
import { ModelComparison } from "@/components/comparison/ModelComparison";
import { buildComparison, TARGETS, type Target, type TargetComparison } from "@/lib/modelComparison";

/**
 * Model Comparison: Simple vs ML vs DNN on the same held-out laps.
 *
 * Reads only what the training pipelines already wrote (see
 * lib/modelComparison.ts for exactly which endpoint supplies which figure).
 * Each source is optional: a missing one becomes "N/A" in the view rather
 * than a failed page, and nothing is retrained to fill a gap.
 */
export default async function ModelComparisonPage() {
  const [dl, ml, registry, dlModels, history, manifest] = await Promise.all([
    api.dlComparison().catch(() => null),
    api.comparison().catch(() => null),
    api.models().catch(() => null),
    api.dlModels().catch(() => null),
    api.dlHistory().catch(() => null),
    api.artifacts().catch(() => null),
  ]);

  const inputs = { dl, ml, registry, dlModels: dlModels?.models ?? null, history };
  const data: Partial<Record<Target, TargetComparison>> = {};
  for (const t of TARGETS) {
    const c = buildComparison(t.key, inputs);
    if (c) data[t.key] = c;
  }

  return (
    <div className="space-y-10 pb-10">
      <PageHero
        photo="/images/photo-car-silhouette.jpg"
        focus="70% 45%"
        eyebrow="Tasks 6 & 7 · head to head"
        title="Model Comparison"
        blurb="Three ways to predict the same race: a Simple linear baseline, the machine-learning model Task 6 selected, and Task 7's deep neural network. Same features, same training laps, same untouched test laps, so any difference comes from the model."
      />
      {manifest && (
        <div className="flex flex-wrap items-center gap-3">
          <DatasetBadge source={manifest.dataset_source} />
          <span className="t-micro">
            Metrics read live from /api/dl/comparison, /api/ml/comparison, /api/ml/models and /api/dl/models.
          </span>
        </div>
      )}
      <ModelComparison data={data} />
    </div>
  );
}
