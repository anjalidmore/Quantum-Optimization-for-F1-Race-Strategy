import { artifactUrl, Comparison, Manifest } from "@/lib/api";
import { ArtifactImage } from "@/components/ArtifactImage";
import { PitPredictPanel } from "@/components/PredictPanel";
import { fmt } from "@/lib/format";

const FIGURES = [
  ["classification_model_comparison.png", "Classification model comparison"],
  ["roc_curves.png", "ROC curves"],
  ["precision_recall_curves.png", "Precision-recall curves"],
  ["confusion_matrix.png", "Confusion matrix"],
  ["classification_feature_importance.png", "Classification feature importance"],
  ["probability_distribution.png", "Predicted probability distribution"],
];

/** Task 6 pit-decision classification, including why PR-AUC rather than ROC-AUC decides. */
export function ClassicalClassification({
  comparison,
  manifest,
  features,
  holdout,
}: {
  comparison: Comparison;
  manifest: Manifest;
  features: string[];
  /** Positive counts read from the committed metrics, never hard-coded here. */
  holdout: { positives?: number; laps?: number; oofPositives?: number; oofSamples?: number };
}) {
  const best = comparison.classification.find((r) => r.selected);
  const figure = (name: string) => manifest.figures.find((f) => f.endsWith(name));

  return (
    <section>
      <h2 className="text-lg font-semibold text-white mb-3">Pit-Decision Classification</h2>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div className="card">
          <div className="stat-label">Best model</div>
          <div className="text-lg font-semibold text-white">{best?.model ?? "—"}</div>
        </div>
        <div className="card">
          <div className="stat-label">CV PR-AUC</div>
          <div className="stat-value">{fmt(best?.cv_pr_auc)}</div>
        </div>
        <div className="card">
          <div className="stat-label">CV ROC-AUC</div>
          <div className="stat-value">{fmt(best?.cv_roc_auc)}</div>
        </div>
        <div className="card">
          <div className="stat-label">CV F1</div>
          <div className="stat-value">{fmt(best?.cv_f1)}</div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        {FIGURES.map(([name, alt]) =>
          figure(name) ? (
            <div key={name} className="card">
              <ArtifactImage src={artifactUrl(figure(name)!)} alt={alt} className="w-full rounded" />
            </div>
          ) : null,
        )}
      </div>

      <div className="card overflow-x-auto">
        <table className="data-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Status</th>
              <th title="Primary selection metric — ROC-AUC is misleading at this positive rate">CV PR-AUC*</th>
              <th>CV ROC-AUC</th>
              <th>CV F1</th>
              <th title="Tuned on out-of-fold predictions, not left at sklearn's default 0.5">Threshold</th>
              <th>Test precision</th>
              <th>Test recall</th>
              <th>Test F1</th>
              <th>Test ROC-AUC</th>
              <th>Selected</th>
            </tr>
          </thead>
          <tbody>
            {comparison.classification.map((r) => (
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
                <td>{r.selected ? <span className="marker">selected</span> : null}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="text-xs text-white/40 mt-2 space-y-1">
          <p>
            *Models are selected on <strong>CV PR-AUC</strong>, not ROC-AUC. Pit events are a small minority of
            laps, and at that prevalence ROC-AUC stays high for a model that never fires — it measures ranking,
            not usefulness. PR-AUC asks how many of the flagged laps are real pit windows.
          </p>
          <p>
            Decision thresholds are tuned on pooled out-of-fold CV predictions rather than left at sklearn&rsquo;s
            default 0.5, which is only optimal for balanced classes with equal error costs. Neither holds here.
          </p>
          {holdout.positives !== undefined && holdout.laps !== undefined && (
            <p className="text-amber-400/70">
              The chronological holdout contains{" "}
              <strong>
                {holdout.positives} pit event(s) in {holdout.laps} laps
              </strong>
              {holdout.positives < 5 && ". Test precision/recall/F1 on so few positives carry little information"}
              {holdout.oofPositives !== undefined &&
                holdout.oofSamples !== undefined &&
                ` — the CV columns cover ${holdout.oofPositives} pit laps across ${holdout.oofSamples} out-of-fold predictions.`}
            </p>
          )}
          <p>†undefined = the holdout contains only one class, so the metric cannot be computed there.</p>
        </div>
      </div>

      {features.length > 0 && (
        <div className="mt-4">
          <PitPredictPanel features={features} />
        </div>
      )}
    </section>
  );
}
