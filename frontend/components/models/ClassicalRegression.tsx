import { Comparison, Manifest } from "@/lib/api";
import { Chart } from "@/components/charts/Chart";
import { LaptimePredictPanel } from "@/components/PredictPanel";
import { fmt } from "@/lib/format";

/* Charts by name, not by filename. Each one is a JSON the pipeline wrote from
   the same arrays that drew the PNG, so the page and the report agree.

   feature_importance is deliberately absent: the selected model is an SVR,
   which has no feature importance to report. The page used to ask for the
   figure anyway and render "Artifact unavailable" - a missing chart is now
   simply not requested, and the permutation importances on the Explainability
   page are where that question is answered. */
const CHARTS = [
  "regression_model_comparison",
  "prediction_vs_actual",
  "residuals",
  "residuals_vs_predictions",
];

/** Task 6 lap-time regression: the ten-model comparison and a live prediction form. */
export function ClassicalRegression({
  comparison,
  manifest,
  features,
}: {
  comparison: Comparison;
  manifest: Manifest;
  features: string[];
}) {
  const best = comparison.regression.find((r) => r.selected);

  return (
    <section aria-labelledby="reg-h">
      <h2 id="reg-h" className="t-title">
        Lap-time regression
      </h2>

      <dl className="mt-4 grid grid-cols-2 gap-x-8 gap-y-5 border-t border-track-300 pt-5 sm:grid-cols-4">
        <div>
          <dt className="stat-label">Best model</dt>
          <dd className="stat-value text-[20px]">{best?.model ?? "—"}</dd>
        </div>
        <div>
          <dt className="stat-label">CV MAE</dt>
          <dd className="stat-value">
            {fmt(best?.cv_mae)}
            <span className="unit"> s</span>
          </dd>
        </div>
        <div>
          <dt className="stat-label">Test MAE</dt>
          <dd className="stat-value">
            {fmt(best?.test_mae)}
            <span className="unit"> s</span>
          </dd>
        </div>
        <div>
          <dt className="stat-label">Test R²</dt>
          <dd className="stat-value">{fmt(best?.test_r2)}</dd>
        </div>
      </dl>

      <div className="mt-6 grid gap-5 lg:grid-cols-2">
        {CHARTS.map((name) => (
          <Chart key={name} name={name} />
        ))}
      </div>

      <ModelTable rows={comparison.regression} />

      {features.length > 0 && (
        <div className="mt-6">
          <LaptimePredictPanel features={features} />
        </div>
      )}
    </section>
  );
}

/**
 * The per-model numbers.
 *
 * Below `sm` the same rows become stacked blocks rather than a table forced
 * into a horizontal scroll: seven numeric columns cannot fit 390px, and a
 * table you have to drag sideways is a table nobody reads.
 */
function ModelTable({ rows }: { rows: Comparison["regression"] }) {
  return (
    <div className="mt-6">
      <h3 className="t-title text-[15px]">Every model trained</h3>

      <table className="t-table mt-3 hidden sm:table">
        <thead>
          <tr>
            <th scope="col">Model</th>
            <th scope="col">Status</th>
            <th scope="col" className="num">CV MAE</th>
            <th scope="col" className="num">CV RMSE</th>
            <th scope="col" className="num">CV R²</th>
            <th scope="col" className="num">Test MAE</th>
            <th scope="col" className="num">Test R²</th>
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
              <td className="num">{fmt(r.cv_mae)}</td>
              <td className="num">{fmt(r.cv_rmse)}</td>
              <td className="num">{fmt(r.cv_r2)}</td>
              <td className="num">{fmt(r.test_mae)}</td>
              <td className="num">{fmt(r.test_r2)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <ul className="mt-3 sm:hidden">
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
                {([["CV MAE", r.cv_mae], ["Test MAE", r.test_mae],
                   ["CV R²", r.cv_r2], ["Test R²", r.test_r2]] as const).map(([k, v]) => (
                  <div key={k} className="flex justify-between gap-3">
                    <dt className="text-paper-500">{k}</dt>
                    <dd className="tabular-nums text-paper-900">{fmt(v)}</dd>
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
