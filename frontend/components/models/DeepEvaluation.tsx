import { artifactUrl, DlArtifacts, DlComparison } from "@/lib/api";
import { ArtifactImage } from "@/components/ArtifactImage";
import { fmt, TARGET_LABEL } from "@/lib/format";

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
  const dir: Record<string, string> = { target_laptime: "laptime", target_pit_next_lap: "pit_decision" };
  const figure = (target: string, name: string) =>
    artifacts.figures.find((f) => f.endsWith(`${dir[target]}/${name}`));
  const m = (target: string) => metrics?.models?.[target] ?? {};

  return (
    <>
      <section>
        <h2 className="text-lg font-semibold text-white mb-1">Evaluation — train / validation / test</h2>
        <p className="text-sm text-white/50 mb-3">
          Test = the chronological holdout (the last laps of the race), used once, after hyperparameters,
          threshold and early-stopping epoch were fixed on earlier laps.
        </p>
        <div className="grid gap-4 md:grid-cols-2">
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
              <div key={target} className="card overflow-x-auto space-y-3">
                <h3 className="font-semibold text-white">{TARGET_LABEL[target]}</h3>
                <table className="w-full text-sm">
                  <thead className="text-white/50">
                    <tr>
                      <th className="text-left font-normal py-1">Split</th>
                      {!reg && <th className="text-right font-normal">pit laps</th>}
                      {keys.map((k) => (
                        <th key={k} className="text-right font-normal">
                          {k.toUpperCase()}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="text-white/80">
                    {splits.map(([name, s]) => (
                      <tr key={name} className={`border-t border-white/5 ${name === "Test" ? "bg-white/5" : ""}`}>
                        <td className="py-1.5">{name}</td>
                        {!reg && (
                          <td className="text-right tabular-nums">
                            {s?.n_positive}/{s?.n}
                          </td>
                        )}
                        {keys.map((k) => (
                          <td key={k} className="text-right tabular-nums">
                            {fmt(s?.[k], 4)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
                {!reg && e.threshold && (
                  <p className="text-xs text-white/50">
                    Decision threshold <strong className="text-white/80">{fmt(e.threshold.threshold, 4)}</strong>,
                    tuned on {e.threshold.n_samples} out-of-fold predictions ({e.threshold.n_positive} pit laps):
                    F1 {fmt(e.threshold["at_default_0.5"]?.f1, 4)} at 0.5, {fmt(e.threshold.at_threshold?.f1, 4)}{" "}
                    at the tuned value.
                  </p>
                )}
                {!reg && e.test_metrics.n_positive < 5 && (
                  <p className="text-xs text-amber-400/70">
                    The test laps contain{" "}
                    <strong>
                      {e.test_metrics.n_positive} pit event(s) in {e.test_metrics.n} laps
                    </strong>
                    . Precision, recall, F1 and PR-AUC on so few positives are dominated by chance; the
                    cross-validated figures rest on {e.threshold?.n_positive ?? "more"} pit laps and are the
                    better guide.
                  </p>
                )}
                <div className="grid gap-3 sm:grid-cols-2">
                  {(reg ? ["prediction_vs_actual.png"] : ["confusion_matrix.png", "roc_curve.png"]).map((n) =>
                    figure(target, n) ? (
                      <ArtifactImage
                        key={n}
                        src={artifactUrl(figure(target, n)!)}
                        alt={`${target} ${n}`}
                        className="w-full rounded"
                      />
                    ) : null,
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      <section>
        <h2 className="text-lg font-semibold text-white mb-1">Deep network vs Task 6 classical models</h2>
        <p className="text-sm text-white/50 mb-3">{comparison.note}</p>
        <div className="grid gap-4">
          {Object.entries(comparison.targets).map(([target, t]) => {
            const keys =
              t.task === "regression"
                ? ["mae", "rmse", "r2", "mape"]
                : ["pr_auc", "roc_auc", "precision", "recall", "f1", "accuracy"];
            return (
              <div key={target} className="card overflow-x-auto">
                <h3 className="font-semibold text-white mb-3">{TARGET_LABEL[target] ?? target}</h3>
                <table className="w-full text-sm min-w-[36rem]">
                  <thead className="text-white/50">
                    <tr>
                      <th className="text-left font-normal py-1">Model</th>
                      {keys.map((k) => (
                        <th key={k} className="text-right font-normal">
                          {k.toUpperCase()}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="text-white/80">
                    {t.comparison.map((row) => (
                      <tr
                        key={row.model}
                        className={`border-t border-white/5 ${row.family === "deep" ? "bg-white/5" : ""}`}
                      >
                        <td className="py-1.5">
                          {row.model}
                          {row.family === "deep" && <span className="badge ml-2">deep</span>}
                          {row.model === t.task6_best_model && <span className="badge ml-2">Task 6 best</span>}
                        </td>
                        {keys.map((k) => (
                          <td key={k} className="text-right tabular-nums">
                            {row.metrics?.[k] === null || row.metrics?.[k] === undefined ? (
                              <span className="text-white/30" title="mathematically undefined on this split">
                                undefined
                              </span>
                            ) : (
                              fmt(row.metrics[k], 4)
                            )}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="text-sm text-white/70 mt-3">{t.verdict.replace(/\*\*/g, "")}</p>
                {figure(target, "model_comparison.png") && (
                  <div className="mt-4">
                    <ArtifactImage
                      src={artifactUrl(figure(target, "model_comparison.png")!)}
                      alt={`${target} model comparison`}
                      className="w-full rounded"
                    />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </section>

      <section className="card">
        <h2 className="font-semibold text-white mb-2">Saved models</h2>
        <p className="text-sm text-white/60">
          Saved as HDF5 (<code>{artifacts.model_format}</code>) with their fitted scalers. Each file is reloaded
          after saving and must reproduce the trained network&rsquo;s predictions exactly. Weights are kept in the
          private <code>models/</code> tree and are not served over HTTP. Full reports:{" "}
          {artifacts.reports
            .filter((r) => r.endsWith(".md") || r.endsWith(".csv"))
            .map((r, i) => (
              <span key={r}>
                {i > 0 && ", "}
                <a className="text-sky-400 hover:underline" href={artifactUrl(r)} target="_blank" rel="noreferrer">
                  {r.split("/").pop()}
                </a>
              </span>
            ))}
          .
        </p>
        <ul className="text-sm text-white/50 mt-2 space-y-1">
          {artifacts.models.map((path) => (
            <li key={path}>
              <code className="text-white/70">{path}</code>
            </li>
          ))}
        </ul>
      </section>
    </>
  );
}
