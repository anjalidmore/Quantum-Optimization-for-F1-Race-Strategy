import { api, ApiError, artifactUrl, ComparisonRow } from "@/lib/api";
import { LaptimePredictPanel, PitPredictPanel } from "@/components/PredictPanel";
import { DatasetBadge } from "@/components/DatasetBadge";
import { ArtifactImage } from "@/components/ArtifactImage";

function fmt(x: number | null | undefined, digits = 3): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return x.toFixed(digits);
}

function Figure({ src, alt }: { src: string; alt: string }) {
  return (
    <div className="card">
      <ArtifactImage src={src} alt={alt} className="w-full rounded" />
    </div>
  );
}

export default async function MachineLearningPage() {
  let comparison = null;
  let manifest = null;
  let registry = null;
  let importance = null;
  let fetchError: string | null = null;

  try {
    [comparison, manifest, registry, importance] = await Promise.all([
      api.comparison(),
      api.artifacts(),
      api.models(),
      api.featureImportance(),
    ]);
  } catch (e) {
    fetchError = e instanceof ApiError ? e.message : "Could not reach the backend API.";
  }

  if (fetchError || !comparison || !manifest || !registry) {
    return (
      <div className="card border-red-500/30 bg-red-500/5">
        <div className="badge badge-warning">No trained model available</div>
        <p className="text-sm text-white/70 mt-2">
          {fetchError ?? "Run the training pipeline to generate results."}
        </p>
        <p className="text-sm text-white/50 mt-1">
          <code className="text-white/80">python scripts/build_all.py --force</code>
        </p>
      </div>
    );
  }

  const bestReg = comparison.regression.find((r: ComparisonRow) => r.selected);
  const bestClf = comparison.classification.find((r: ComparisonRow) => r.selected);
  const findFigure = (name: string) => manifest.figures.find((f) => f.endsWith(name));

  const regFeatures = registry.models.find((m) => m.target === "target_laptime")?.features ?? [];
  const clfFeatures = registry.models.find((m) => m.target === "target_pit_next_lap")?.features ?? [];
  const isReal = manifest.dataset_source.source === "real_fastf1";
  const anyUndefined = comparison.classification.some((r: ComparisonRow) => r.test_undefined_reason);

  return (
    <div className="space-y-10">
      <section>
        <div className="flex items-center gap-3 flex-wrap">
          <h1 className="text-2xl font-bold text-white">Machine Learning Intelligence</h1>
          <DatasetBadge source={manifest.dataset_source} />
        </div>
        <p className="text-white/60 mt-1">Classical prediction models for F1 race strategy (Task 6).</p>
        <p className="text-xs text-white/40 mt-2 max-w-3xl">
          {isReal ? (
            <>
              This is a single real Grand Prix session — a genuine result, not a placeholder, but it
              reflects one race and should not be generalised beyond it.
              {anyUndefined &&
                " The chronological holdout test set contains zero pit events for at least one model, so its ROC-AUC/PR-AUC are undefined there."}
            </>
          ) : (
            <>
              ⚠ The synthetic pit schedule is close to deterministic. Classification metrics — especially
              ROC-AUC/PR-AUC — are inflated relative to what real-world F1 telemetry would produce, and the
              chronological holdout test set here falls entirely after the last synthetic pit event. See the
              model-selection report for the full discussion. These are not real-world F1 performance figures.
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
            <div className="text-sm font-medium text-white">{new Date(manifest.generated_at).toLocaleString()}</div>
          </div>
          <div className="card">
            <div className="stat-label">Data source</div>
            <div className="text-lg font-semibold text-white">{isReal ? "Real FastF1" : "Synthetic"}</div>
          </div>
        </div>
      </section>

      {/* Regression */}
      <section>
        <h2 className="text-lg font-semibold text-white mb-3">Lap-Time Regression</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
          <div className="card">
            <div className="stat-label">Best model</div>
            <div className="text-lg font-semibold text-white">{bestReg?.model ?? "—"}</div>
          </div>
          <div className="card">
            <div className="stat-label">CV MAE</div>
            <div className="stat-value">{fmt(bestReg?.cv_mae)}s</div>
          </div>
          <div className="card">
            <div className="stat-label">Test MAE</div>
            <div className="stat-value">{fmt(bestReg?.test_mae)}s</div>
          </div>
          <div className="card">
            <div className="stat-label">Test R²</div>
            <div className="stat-value">{fmt(bestReg?.test_r2)}</div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          {findFigure("regression_model_comparison.png") && (
            <Figure src={artifactUrl(findFigure("regression_model_comparison.png")!)} alt="Regression model comparison" />
          )}
          {findFigure("prediction_vs_actual.png") && (
            <Figure src={artifactUrl(findFigure("prediction_vs_actual.png")!)} alt="Predicted vs actual lap time" />
          )}
          {findFigure("residuals.png") && (
            <Figure src={artifactUrl(findFigure("residuals.png")!)} alt="Residual distribution" />
          )}
          {findFigure("feature_importance.png") && (
            <Figure src={artifactUrl(findFigure("feature_importance.png")!)} alt="Feature importance" />
          )}
        </div>

        <div className="card overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Status</th>
                <th>CV MAE</th>
                <th>CV RMSE</th>
                <th>CV R²</th>
                <th>Test MAE</th>
                <th>Test R²</th>
                <th>Selected</th>
              </tr>
            </thead>
            <tbody>
              {comparison.regression.map((r: ComparisonRow) => (
                <tr key={r.model}>
                  <td className="text-white">{r.model}</td>
                  <td className="text-white/50">{r.status === "skipped" ? `skipped — ${r.reason}` : "trained"}</td>
                  <td>{fmt(r.cv_mae)}</td>
                  <td>{fmt(r.cv_rmse)}</td>
                  <td>{fmt(r.cv_r2)}</td>
                  <td>{fmt(r.test_mae)}</td>
                  <td>{fmt(r.test_r2)}</td>
                  <td>{r.selected ? "✅" : ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {regFeatures.length > 0 && (
          <div className="mt-4">
            <LaptimePredictPanel features={regFeatures} />
          </div>
        )}
      </section>

      {/* Classification */}
      <section>
        <h2 className="text-lg font-semibold text-white mb-3">Pit-Decision Classification</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
          <div className="card">
            <div className="stat-label">Best model</div>
            <div className="text-lg font-semibold text-white">{bestClf?.model ?? "—"}</div>
          </div>
          <div className="card">
            <div className="stat-label">CV ROC-AUC</div>
            <div className="stat-value">{fmt(bestClf?.cv_roc_auc)}</div>
          </div>
          <div className="card">
            <div className="stat-label">CV PR-AUC</div>
            <div className="stat-value">{fmt(bestClf?.cv_pr_auc)}</div>
          </div>
          <div className="card">
            <div className="stat-label">CV F1</div>
            <div className="stat-value">{fmt(bestClf?.cv_f1)}</div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          {findFigure("classification_model_comparison.png") && (
            <Figure
              src={artifactUrl(findFigure("classification_model_comparison.png")!)}
              alt="Classification model comparison"
            />
          )}
          {findFigure("roc_curves.png") && <Figure src={artifactUrl(findFigure("roc_curves.png")!)} alt="ROC curves" />}
          {findFigure("precision_recall_curves.png") && (
            <Figure src={artifactUrl(findFigure("precision_recall_curves.png")!)} alt="Precision-recall curves" />
          )}
          {findFigure("confusion_matrix.png") && (
            <Figure src={artifactUrl(findFigure("confusion_matrix.png")!)} alt="Confusion matrix" />
          )}
          {findFigure("classification_feature_importance.png") && (
            <Figure
              src={artifactUrl(findFigure("classification_feature_importance.png")!)}
              alt="Classification feature importance"
            />
          )}
          {findFigure("probability_distribution.png") && (
            <Figure src={artifactUrl(findFigure("probability_distribution.png")!)} alt="Predicted probability distribution" />
          )}
        </div>

        <div className="card overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Status</th>
                <th title="Primary selection metric — ROC-AUC is misleading at a 4.8% positive rate">
                  CV PR-AUC*
                </th>
                <th>CV ROC-AUC</th>
                <th>CV F1</th>
                <th title="Tuned on out-of-fold predictions, not left at sklearn's default 0.5">
                  Threshold
                </th>
                <th>Test precision</th>
                <th>Test recall</th>
                <th>Test F1</th>
                <th>Test ROC-AUC</th>
                <th>Selected</th>
              </tr>
            </thead>
            <tbody>
              {comparison.classification.map((r: ComparisonRow) => (
                <tr key={r.model}>
                  <td className="text-white">{r.model}</td>
                  <td className="text-white/50">{r.status === "skipped" ? `skipped — ${r.reason}` : "trained"}</td>
                  <td className="text-white">{fmt(r.cv_pr_auc)}</td>
                  <td>{fmt(r.cv_roc_auc)}</td>
                  <td>{fmt(r.cv_f1)}</td>
                  <td>{r.decision_threshold === undefined || r.decision_threshold === null ? "—" : fmt(r.decision_threshold)}</td>
                  <td>{fmt(r.test_precision)}</td>
                  <td>{fmt(r.test_recall)}</td>
                  <td>{fmt(r.test_f1)}</td>
                  <td>{r.test_roc_auc === null ? "undefined†" : fmt(r.test_roc_auc)}</td>
                  <td>{r.selected ? "✅" : ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="text-xs text-white/40 mt-2 space-y-1">
            <p>
              *Models are selected on <strong>CV PR-AUC</strong>, not ROC-AUC. Pit events are 4.8% of laps,
              and at that prevalence ROC-AUC stays high for a model that never fires — it measures ranking,
              not usefulness. PR-AUC asks how many of the flagged laps are real pit windows.
            </p>
            <p>
              Decision thresholds are tuned on pooled out-of-fold CV predictions rather than left at
              sklearn&rsquo;s default 0.5, which is only optimal for balanced classes with equal error costs.
              Neither holds here.
            </p>
            <p className="text-amber-400/70">
              ⚠ The holdout contains <strong>1 pit event in 180 laps</strong>. Test precision/recall/F1 on a
              single positive example carry almost no information — read the CV columns, which cover 36
              positives across the folds. Broadening evaluation to multiple races is tracked in the backlog.
            </p>
            <p>†undefined = the holdout contains only one class, so the metric cannot be computed there.</p>
          </div>
        </div>

        {clfFeatures.length > 0 && (
          <div className="mt-4">
            <PitPredictPanel features={clfFeatures} />
          </div>
        )}
      </section>
    </div>
  );
}
