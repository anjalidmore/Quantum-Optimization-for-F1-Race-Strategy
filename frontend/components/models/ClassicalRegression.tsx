import { artifactUrl, Comparison, Manifest } from "@/lib/api";
import { ArtifactImage } from "@/components/ArtifactImage";
import { LaptimePredictPanel } from "@/components/PredictPanel";
import { fmt } from "@/lib/format";

const FIGURES = [
  ["regression_model_comparison.png", "Regression model comparison"],
  ["prediction_vs_actual.png", "Predicted vs actual lap time"],
  ["residuals.png", "Residual distribution"],
  ["feature_importance.png", "Feature importance"],
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
  const figure = (name: string) => manifest.figures.find((f) => f.endsWith(name));

  return (
    <section>
      <h2 className="text-lg font-semibold text-paper-900 mb-3">Lap-Time Regression</h2>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div className="card">
          <div className="stat-label">Best model</div>
          <div className="text-lg font-semibold text-paper-900">{best?.model ?? "—"}</div>
        </div>
        <div className="card">
          <div className="stat-label">CV MAE</div>
          <div className="stat-value">{fmt(best?.cv_mae)}s</div>
        </div>
        <div className="card">
          <div className="stat-label">Test MAE</div>
          <div className="stat-value">{fmt(best?.test_mae)}s</div>
        </div>
        <div className="card">
          <div className="stat-label">Test R²</div>
          <div className="stat-value">{fmt(best?.test_r2)}</div>
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
              <th>CV MAE</th>
              <th>CV RMSE</th>
              <th>CV R²</th>
              <th>Test MAE</th>
              <th>Test R²</th>
              <th>Selected</th>
            </tr>
          </thead>
          <tbody>
            {comparison.regression.map((r) => (
              <tr key={r.model}>
                <td className="text-paper-900">{r.model}</td>
                <td className="text-paper-500">{r.status === "skipped" ? `skipped — ${r.reason}` : "trained"}</td>
                <td>{fmt(r.cv_mae)}</td>
                <td>{fmt(r.cv_rmse)}</td>
                <td>{fmt(r.cv_r2)}</td>
                <td>{fmt(r.test_mae)}</td>
                <td>{fmt(r.test_r2)}</td>
                <td>{r.selected ? <span className="marker">selected</span> : null}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {features.length > 0 && (
        <div className="mt-4">
          <LaptimePredictPanel features={features} />
        </div>
      )}
    </section>
  );
}
