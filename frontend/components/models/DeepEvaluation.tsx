import { artifactUrl, DlArtifacts, DlComparison } from "@/lib/api";
import { Chart } from "@/components/charts/Chart";
import { fmt, TARGET_LABEL } from "@/lib/format";

/* Chart names follow the artifact layout: dl_<laptime|pit_decision>_<figure>. */
const DIR: Record<string, string> = { target_laptime: "laptime", target_pit_next_lap: "pit_decision" };

/** Task 7 results: train/validation/test per target, the comparison against Task 6, and the saved files. */
export function DeepEvaluation({
  comparison,
  artifacts,
  metrics,
}: {
  comparison: DlComparison;
  artifacts: DlArtifacts;
  metrics: Record<string, any> | null;
}) {
  const m = (target: string) => metrics?.models?.[target] ?? {};

  return (
    <>
      <section>
        <h2 className="text-lg font-semibold text-paper-900 mb-1">Evaluation — train / validation / test</h2>
        <p className="text-sm text-paper-500 mb-3">
          Test = the chronological holdout (the last laps of the race), used once, after hyperparameters,
          threshold and early-stopping epoch were fixed on earlier laps.
        </p>
        <div className="grid gap-4 md:grid-cols-2 [&>*]:min-w-0">
          {Object.keys(TARGET_LABEL).map((target) => {
            const e = m(target);
            if (!e.test_metrics) return null;
            const reg = e.task === "regression";
            const keys = reg
              ? ["mae", "rmse", "r2"]
              : ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"];
            const splits: [string, Record<string, any>][] = [
              ["Train", e.train_metrics],
              ["Validation", e.validation_metrics],
              ["Test", e.test_metrics],
            ];
            return (
              <div key={target} className="card space-y-3">
                <h3 className="t-title text-[15px]">{TARGET_LABEL[target]}</h3>
                {/* Up to six metric columns will not fit a phone, so the table
                    is transposed below `sm`: metrics down, splits across. That
                    is three columns, which does fit, and nothing scrolls. */}
                <table className="t-table">
                  <thead>
                    <tr>
                      <th scope="col">Metric</th>
                      {splits.map(([name]) => (
                        <th key={name} scope="col" className="num">{name}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {!reg && (
                      <tr>
                        <th scope="row" className="text-left font-normal">pit laps</th>
                        {splits.map(([name, s]) => (
                          <td key={name} className="num">{s?.n_positive}/{s?.n}</td>
                        ))}
                      </tr>
                    )}
                    {keys.map((k) => (
                      <tr key={k}>
                        <th scope="row" className="text-left font-normal">{k.toUpperCase()}</th>
                        {splits.map(([name, s]) => (
                          <td key={name} className="num">{fmt(s?.[k], 4)}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
                {!reg && e.threshold && (
                  <p className="text-xs text-paper-500">
                    Decision threshold <strong className="text-paper-700">{fmt(e.threshold.threshold, 4)}</strong>,
                    tuned on {e.threshold.n_samples} out-of-fold predictions ({e.threshold.n_positive} pit laps):
                    F1 {fmt(e.threshold["at_default_0.5"]?.f1, 4)} at 0.5, {fmt(e.threshold.at_threshold?.f1, 4)}{" "}
                    at the tuned value.
                  </p>
                )}
                {!reg && e.test_metrics.n_positive < 5 && (
                  <p className="text-xs text-paper-500">
                    The test laps contain{" "}
                    <strong>
                      {e.test_metrics.n_positive} pit event(s) in {e.test_metrics.n} laps
                    </strong>
                    . Precision, recall, F1 and PR-AUC on so few positives are dominated by chance; the
                    cross-validated figures rest on {e.threshold?.n_positive ?? "more"} pit laps and are the
                    better guide.
                  </p>
                )}
                <div className="grid gap-4 sm:grid-cols-2">
                  {(reg ? ["prediction_vs_actual"] : ["confusion_matrix", "roc_curve"]).map((n) => (
                    <Chart key={n} name={`dl_${DIR[target]}_${n}`} height={230} />
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      <section>
        <h2 className="text-lg font-semibold text-paper-900 mb-1">Deep network vs Task 6 classical models</h2>
        <p className="text-sm text-paper-500 mb-3">{comparison.note}</p>
        <div className="grid gap-4">
          {Object.entries(comparison.targets).map(([target, t]) => {
            const keys =
              t.task === "regression"
                ? ["mae", "rmse", "r2", "mape"]
                : ["pr_auc", "roc_auc", "precision", "recall", "f1", "accuracy"];
            return (
              <div key={target} className="card min-w-0">
                <h3 className="t-title mb-3 text-[15px]">{TARGET_LABEL[target] ?? target}</h3>
                {/* Six metric columns plus a model name do not fit 390px. Rather
                    than force a sideways drag, the low-value columns drop away
                    below `sm` and the ranking metric stays. */}
                <table className="t-table">
                  <thead>
                    <tr>
                      <th scope="col">Model</th>
                      {keys.map((k, i) => (
                        <th key={k} scope="col" className={`num ${i > 1 ? "hidden sm:table-cell" : ""}`}>
                          {k.toUpperCase()}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {t.comparison.map((row) => (
                      <tr key={row.model} className={row.family === "deep" ? "is-marked" : ""}>
                        <td>
                          {row.model}
                          {row.family === "deep" && <span className="marker ml-2">deep</span>}
                          {row.model === t.task6_best_model && <span className="marker ml-2">Task 6 best</span>}
                        </td>
                        {keys.map((k, i) => (
                          <td key={k} className={`num ${i > 1 ? "hidden sm:table-cell" : ""}`}>
                            {row.metrics?.[k] === null || row.metrics?.[k] === undefined ? (
                              <span className="text-paper-400">undefined</span>
                            ) : (
                              fmt(row.metrics[k], 4)
                            )}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="text-sm text-paper-700 mt-3">{t.verdict.replace(/\*\*/g, "")}</p>
                <div className="mt-4">
                  <Chart name={`dl_${DIR[target]}_model_comparison`} height={260} />
                </div>
              </div>
            );
          })}
        </div>
      </section>

      <section className="card">
        <h2 className="font-semibold text-paper-900 mb-2">Saved models</h2>
        <p className="text-sm text-paper-500">
          Saved as HDF5 (<code>{artifacts.model_format}</code>) with their fitted scalers. Each file is reloaded
          after saving and must reproduce the trained network&rsquo;s predictions exactly. Weights are kept in the
          private <code>models/</code> tree and are not served over HTTP. Full reports:{" "}
          {artifacts.reports
            .filter((r) => r.endsWith(".md") || r.endsWith(".csv"))
            .map((r, i) => (
              <span key={r}>
                {i > 0 && ", "}
                <a className="text-paper-900 underline decoration-edge underline-offset-[3px] hover:decoration-paper-900" href={artifactUrl(r)} target="_blank" rel="noreferrer">
                  {r.split("/").pop()}
                </a>
              </span>
            ))}
          .
        </p>
        <ul className="text-sm text-paper-500 mt-2 space-y-1">
          {artifacts.models.map((path) => (
            <li key={path}>
              <code className="text-paper-700">{path}</code>
            </li>
          ))}
        </ul>
      </section>
    </>
  );
}
