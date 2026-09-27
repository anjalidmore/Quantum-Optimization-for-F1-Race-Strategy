import { api, ApiError } from "@/lib/api";
import { DatasetBadge } from "@/components/DatasetBadge";
import { ClassicalRegression } from "@/components/models/ClassicalRegression";
import { ClassicalClassification } from "@/components/models/ClassicalClassification";
import { DeepNetworks } from "@/components/models/DeepNetworks";
import { DeepEvaluation } from "@/components/models/DeepEvaluation";
import { QuantumSection } from "@/components/models/QuantumSection";

/**
 * Task 6 and Task 7 on one page: the same two prediction problems, solved by
 * classical models and then by neural networks on the same folds and the same
 * holdout. Keeping them together is the point — the comparison is the result.
 */
export default async function ModelsPage() {
  let comparison = null;
  let manifest = null;
  let registry = null;
  let mlMetrics: Record<string, any> | null = null;
  let error: string | null = null;

  try {
    [comparison, manifest, registry, mlMetrics] = await Promise.all([
      api.comparison(),
      api.artifacts(),
      api.models(),
      api.metrics().catch(() => null),
    ]);
  } catch (e) {
    error = e instanceof ApiError ? e.message : "Could not reach the backend API.";
  }

  if (error || !comparison || !manifest || !registry) {
    return (
      <div className="card border-red-500/30 bg-red-500/5">
        <div className="badge badge-warning">No trained model available</div>
        <p className="text-sm text-white/70 mt-2">{error ?? "Run the training pipeline to generate results."}</p>
        <p className="text-sm text-white/50 mt-1">
          <code className="text-white/80">python scripts/build_all.py --force</code>
        </p>
      </div>
    );
  }

  // Tasks 7 and the quantum stage are optional: the page still works without them.
  const [dlModels, dlComparison, dlHistory, dlArtifacts, dlMetrics, qml] = await Promise.all([
    api.dlModels().catch(() => null),
    api.dlComparison().catch(() => null),
    api.dlHistory().catch(() => null),
    api.dlArtifacts().catch(() => null),
    api.dlMetrics().catch(() => null),
    api.qmlSummary().catch(() => null),
  ]);

  const isReal = manifest.dataset_source.source === "real_fastf1";
  const anyUndefined = comparison.classification.some((r) => r.test_undefined_reason);
  const clfBest = mlMetrics?.classification?.models?.[mlMetrics?.classification?.best_model];

  return (
    <div className="space-y-10">
      <section>
        <div className="flex items-center gap-3 flex-wrap">
          <h1 className="text-2xl font-bold text-white">Models</h1>
          <DatasetBadge source={manifest.dataset_source} />
          <span className="badge">Tasks 6 &amp; 7</span>
        </div>
        <p className="text-white/60 mt-1 max-w-3xl">
          Lap-time regression and pit-decision classification, trained twice: with classical models (Task 6) and
          with Keras neural networks (Task 7). Both use the same Task 5 feature contract, the same
          expanding-window folds and the same untouched chronological holdout, so any difference is the model
          rather than the harness.
        </p>
        <p className="text-xs text-white/40 mt-2 max-w-3xl">
          {isReal ? (
            <>
              This is a single real Grand Prix session — a genuine result, not a placeholder, but it reflects one
              race and should not be generalised beyond it.
              {anyUndefined &&
                " The chronological holdout contains zero pit events for at least one model, so its ROC-AUC/PR-AUC are undefined there."}
            </>
          ) : (
            <>
              ⚠ The synthetic pit schedule is close to deterministic, so classification metrics are inflated
              relative to real telemetry. These are not real-world F1 performance figures.
            </>
          )}
        </p>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mt-5">
          <div className="card">
            <div className="stat-label">Models trained</div>
            <div className="stat-value">{registry.models.length}</div>
          </div>
          <div className="card">
            <div className="stat-label">Best regression</div>
            <div className="text-lg font-semibold text-white">{manifest.best_regression_model ?? "—"}</div>
          </div>
          <div className="card">
            <div className="stat-label">Best classification</div>
            <div className="text-lg font-semibold text-white">{manifest.best_classification_model ?? "—"}</div>
          </div>
          <div className="card">
            <div className="stat-label">Last trained</div>
            <div className="text-sm font-medium text-white">
              {new Date(manifest.generated_at).toLocaleString()}
            </div>
          </div>
          <div className="card">
            <div className="stat-label">Data source</div>
            <div className="text-lg font-semibold text-white">{isReal ? "Real FastF1" : "Synthetic"}</div>
          </div>
        </div>
      </section>

      <h2 className="text-sm uppercase tracking-wider text-white/40 border-b border-white/10 pb-2">
        Task 6 — classical models
      </h2>

      <ClassicalRegression
        comparison={comparison}
        manifest={manifest}
        features={registry.models.find((m) => m.target === "target_laptime")?.features ?? []}
      />

      <ClassicalClassification
        comparison={comparison}
        manifest={manifest}
        features={registry.models.find((m) => m.target === "target_pit_next_lap")?.features ?? []}
        holdout={{
          positives: clfBest?.test_metrics?.n_positive,
          laps: clfBest?.test_metrics?.n,
          oofPositives: clfBest?.threshold?.n_positive,
          oofSamples: clfBest?.threshold?.n_samples,
        }}
      />

      <h2 className="text-sm uppercase tracking-wider text-white/40 border-b border-white/10 pb-2">
        Task 7 — deep learning
      </h2>

      {dlModels && dlComparison && dlHistory && dlArtifacts ? (
        <>
          <DeepNetworks
            models={dlModels.models}
            comparison={dlComparison}
            history={dlHistory}
            artifacts={dlArtifacts}
            metrics={dlMetrics}
          />
          <DeepEvaluation comparison={dlComparison} artifacts={dlArtifacts} metrics={dlMetrics} />
        </>
      ) : (
        <div className="card text-sm text-white/50">
          No deep model available yet. Run <code className="text-white/80">python scripts/build_all.py</code>.
        </div>
      )}

      {qml && (
        <>
          <h2 className="text-sm uppercase tracking-wider text-white/40 border-b border-white/10 pb-2">
            Quantum — simulated circuits on the same split
          </h2>
          <QuantumSection summary={qml} />
        </>
      )}
    </div>
  );
}
