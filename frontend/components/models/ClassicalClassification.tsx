import { Comparison, Manifest } from "@/lib/api";
import { Chart } from "@/components/charts/Chart";
import { PitPredictPanel } from "@/components/PredictPanel";
import { fmt } from "@/lib/format";

const CHARTS = [
  "classification_model_comparison",
  "roc_curves",
  "precision_recall_curves",
  "confusion_matrix",
  "classification_feature_importance",
  "probability_distribution",
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

  return (
    <section aria-labelledby="clf-h">
      <h2 id="clf-h" className="t-title">
        Pit-decision classification
      </h2>

      <dl className="mt-4 grid grid-cols-2 gap-x-8 gap-y-5 border-t border-track-300 pt-5 sm:grid-cols-4">
        <div>
          <dt className="stat-label">Best model</dt>
          <dd className="stat-value text-[20px]">{best?.model ?? "—"}</dd>
        </div>
        <div>
          <dt className="stat-label">CV PR-AUC</dt>
          <dd className="stat-value">{fmt(best?.cv_pr_auc)}</dd>
        </div>
        <div>
          <dt className="stat-label">CV ROC-AUC</dt>
          <dd className="stat-value">{fmt(best?.cv_roc_auc)}</dd>
        </div>
        <div>
          <dt className="stat-label">CV F1</dt>
          <dd className="stat-value">{fmt(best?.cv_f1)}</dd>
        </div>
      </dl>

      <div className="mt-6 grid gap-5 lg:grid-cols-2">
        {CHARTS.map((name) => (
          <Chart key={name} name={name} />
        ))}
      </div>

      <ModelTable rows={comparison.classification} />

      <div className="mt-4 space-y-2 border-l-2 border-track-300 pl-3.5 text-[13px] text-paper-500">
        <p className="max-w-measure">
          Models are selected on <strong className="text-paper-700">CV PR-AUC</strong>, not ROC-AUC. Pit events
          are a small minority of laps, and at that prevalence ROC-AUC stays high for a model that never fires —
          it measures ranking, not usefulness. PR-AUC asks how many of the flagged laps are real pit windows.
        </p>
        <p className="max-w-measure">
          Decision thresholds are tuned on pooled out-of-fold CV predictions rather than left at sklearn&rsquo;s
          default 0.5, which is only optimal for balanced classes with equal error costs. Neither holds here.
        </p>
        {holdout.positives !== undefined && holdout.laps !== undefined && (
          <p className="max-w-measure">
            The chronological holdout contains{" "}
            <strong className="tabular-nums text-paper-700">
              {holdout.positives} pit event(s) in {holdout.laps} laps
            </strong>
            {holdout.positives < 5 && ". Test precision, recall and F1 on so few positives carry little information"}
            {holdout.oofPositives !== undefined &&
              holdout.oofSamples !== undefined &&
              ` — the CV columns cover ${holdout.oofPositives} pit laps across ${holdout.oofSamples} out-of-fold predictions.`}
          </p>
        )}
        <p>An undefined test ROC-AUC means the holdout holds one class, so the metric does not exist there.</p>
      </div>

      {features.length > 0 && (
        <div className="mt-6">
          <PitPredictPanel features={features} />
        </div>
      )}
    </section>
  );
}

/** Eleven numeric columns cannot fit a phone, so below `sm` each model is a block. */
function ModelTable({ rows }: { rows: Comparison["classification"] }) {
  const cell = (v: number | null | undefined) => (v === null ? "undefined" : fmt(v));
  return (
    <div className="mt-6">
      <h3 className="t-title text-[15px]">Every model trained</h3>

      <table className="t-table mt-3 hidden lg:table">
        <thead>
          <tr>
            <th scope="col">Model</th>
            <th scope="col">Status</th>
            <th scope="col" className="num">CV PR-AUC</th>
            <th scope="col" className="num">CV ROC-AUC</th>
            <th scope="col" className="num">CV F1</th>
            <th scope="col" className="num">Threshold</th>
            <th scope="col" className="num">Test F1</th>
            <th scope="col" className="num">Test ROC-AUC</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.model} className={r.selected ? "is-marked" : ""}>
              <td>
                {r.model}
                {r.selected && <span className="marker ml-2">selected</span>}
              </td>
              <td className="text-paper-500">
                {r.status === "skipped" ? `skipped — ${r.reason}` : "trained"}
              </td>
              <td className="num">{fmt(r.cv_pr_auc)}</td>
              <td className="num">{fmt(r.cv_roc_auc)}</td>
              <td className="num">{fmt(r.cv_f1)}</td>
              <td className="num">{r.decision_threshold == null ? "—" : fmt(r.decision_threshold)}</td>
              <td className="num">{fmt(r.test_f1)}</td>
              <td className="num">{cell(r.test_roc_auc)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <ul className="mt-3 lg:hidden">
        {rows.map((r) => (
          <li key={r.model} className="border-b border-track-300 py-3">
            <p className="flex items-baseline gap-2">
              <span className="font-display text-[15px] font-semibold text-paper-900">{r.model}</span>
              {r.selected && <span className="marker">selected</span>}
            </p>
            {r.status === "skipped" ? (
              <p className="t-micro mt-1">skipped — {r.reason}</p>
            ) : (
              <dl className="mt-1.5 grid grid-cols-2 gap-x-5 gap-y-1 text-[13px]">
                {([["CV PR-AUC", fmt(r.cv_pr_auc)], ["CV F1", fmt(r.cv_f1)],
                   ["Threshold", r.decision_threshold == null ? "—" : fmt(r.decision_threshold)],
                   ["Test F1", fmt(r.test_f1)]] as const).map(([k, v]) => (
                  <div key={k} className="flex justify-between gap-3">
                    <dt className="text-paper-500">{k}</dt>
                    <dd className="tabular-nums text-paper-900">{v}</dd>
                  </div>
                ))}
              </dl>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
