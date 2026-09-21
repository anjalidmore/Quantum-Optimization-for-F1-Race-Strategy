import { api, ApiError, artifactUrl } from "@/lib/api";
import { DatasetBadge } from "@/components/DatasetBadge";
import { ArtifactImage } from "@/components/ArtifactImage";

function fmt(x: number | null | undefined, digits = 4): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return x.toFixed(digits);
}

const TARGET_LABEL: Record<string, string> = {
  target_laptime: "Lap-time regression",
  target_pit_next_lap: "Pit-decision classification",
};

export default async function DeepLearningPage() {
  let models = null;
  let comparison = null;
  let history = null;
  let artifacts = null;
  let metrics: Record<string, any> | null = null;
  let error: string | null = null;

  try {
    [models, comparison, history, artifacts, metrics] = await Promise.all([
      api.dlModels(),
      api.dlComparison(),
      api.dlHistory(),
      api.dlArtifacts(),
      api.dlMetrics(),
    ]);
  } catch (e) {
    error = e instanceof ApiError ? e.message : "Could not reach the backend API.";
  }

  if (error || !models || !comparison || !history || !artifacts) {
    return (
      <div className="card border-red-500/30 bg-red-500/5">
        <div className="badge badge-warning">No deep model available</div>
        <p className="text-sm text-white/70 mt-2">
          {error ?? "Run the deep-learning stage to generate results."}
        </p>
        <p className="text-sm text-white/50 mt-1">
          <code className="text-white/80">python scripts/build_all.py --force</code>
        </p>
      </div>
    );
  }

  const DIR: Record<string, string> = { target_laptime: "laptime", target_pit_next_lap: "pit_decision" };
  // Figures live at deep_learning/<laptime|pit_decision>/<name>.png
  const figure = (target: string, name: string) =>
    artifacts.figures.find((f) => f.endsWith(`${DIR[target]}/${name}`));
  const m = (target: string) => metrics?.models?.[target] ?? {};

  return (
    <div className="space-y-8">
      <section>
        <div className="flex items-center gap-3 flex-wrap">
          <h1 className="text-2xl font-bold text-white">Deep Learning</h1>
          <DatasetBadge source={comparison.dataset_source} />
          <span className="badge">Task 7</span>
        </div>
        <p className="text-white/60 mt-1 max-w-3xl">
          Keras neural networks trained on the same Task 5 feature contract, the same folds
          and the same untouched chronological holdout as the classical models — so any
          difference below is attributable to the model, not to the harness.
        </p>
      </section>

      {/* --- headline verdicts ------------------------------------------- */}
      <section className="grid gap-4 md:grid-cols-2">
        {Object.entries(comparison.targets).map(([target, t]) => {
          const dnn = t.comparison.find((c) => c.family === "deep");
          const metric = t.selection_metric;
          const deepWins = t.verdict.includes("deep network wins");
          return (
            <div key={target} className="card">
              <div className="flex items-center justify-between gap-2">
                <h2 className="font-semibold text-white">{TARGET_LABEL[target] ?? target}</h2>
                <span className={`badge ${deepWins ? "badge-success" : "badge-warning"}`}>
                  {deepWins ? "DNN wins" : "classical wins"}
                </span>
              </div>
              <div className="mt-3 text-3xl font-bold text-white">
                {fmt(dnn?.metrics?.[metric])}
                <span className="text-sm font-normal text-white/50 ml-2">
                  test {metric.toUpperCase()}
                </span>
              </div>
              <p
                className="text-sm text-white/60 mt-3"
                dangerouslySetInnerHTML={{
                  __html: t.verdict.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>"),
                }}
              />
            </div>
          );
        })}
      </section>

      {/* --- architectures ------------------------------------------------ */}
      <section>
        <h2 className="text-lg font-semibold text-white mb-3">Network architectures</h2>
        <div className="grid gap-4 md:grid-cols-2">
          {models.models.map((m) => {
            const ratio = m.architecture.total_parameters / Math.max(m.training_rows, 1);
            return (
              <div key={m.target} className="card">
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold text-white">{TARGET_LABEL[m.target] ?? m.target}</h3>
                  <span className="badge">{m.model_format}</span>
                </div>
                <table className="w-full text-sm mt-3">
                  <thead className="text-white/50">
                    <tr>
                      <th className="text-left font-normal py-1">Layer</th>
                      <th className="text-left font-normal">Type</th>
                      <th className="text-right font-normal">Detail</th>
                    </tr>
                  </thead>
                  <tbody className="text-white/80">
                    {m.architecture.layers.map((l: any) => (
                      <tr key={l.name} className="border-t border-white/5">
                        <td className="py-1">{l.name}</td>
                        <td>{l.type}</td>
                        <td className="text-right text-white/60">
                          {l.units !== undefined ? `${l.units} units, ${l.activation}` : ""}
                          {l.rate !== undefined ? `rate ${l.rate}` : ""}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="mt-3 text-sm text-white/60 space-y-1">
                  <div>
                    <span className="text-white/80">{m.architecture.total_parameters.toLocaleString()}</span>{" "}
                    parameters · <span className="text-white/80">{m.training_rows}</span> training rows ·
                    ratio <span className={ratio > 1 ? "text-amber-400" : "text-white/80"}>{ratio.toFixed(2)}</span>
                  </div>
                  {ratio > 1 && (
                    <p className="text-amber-400/80">
                      More parameters than training examples — the expected small-data regime here,
                      and why dropout, L2 and early stopping are all applied together.
                    </p>
                  )}
                  <div>
                    optimizer {m.architecture.optimizer} · loss <code>{m.architecture.loss}</code>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* --- training curves --------------------------------------------- */}
      <section>
        <h2 className="text-lg font-semibold text-white mb-1">Training curves</h2>
        <p className="text-sm text-white/50 mb-3">
          The final network trains on earlier laps and early-stops on the block of laps just before the
          test laps. The dashed line marks the epoch whose weights were restored and saved.
        </p>
        <div className="grid gap-4 md:grid-cols-2">
          {Object.entries(history).map(([target, h]) => {
            const extra = target === "target_laptime" ? "mae_curve.png" : "accuracy_curve.png";
            const o = m(target).overfitting ?? {};
            return (
              <div key={target} className="card space-y-3">
                <h3 className="font-semibold text-white">{TARGET_LABEL[target] ?? target}</h3>
                {figure(target, "loss_curve.png") && (
                  <ArtifactImage src={artifactUrl(figure(target, "loss_curve.png")!)} alt={`${target} loss curve`} className="w-full rounded" />
                )}
                {figure(target, extra) && (
                  <ArtifactImage src={artifactUrl(figure(target, extra)!)} alt={`${target} ${extra}`} className="w-full rounded" />
                )}
                <p className="text-sm text-white/60">
                  Ran {h.epochs_run} of a maximum {h.max_epochs} epochs; early stopping (patience{" "}
                  {h.early_stopping_patience}) restored epoch {h.best_epoch}.{" "}
                  {o.verdict && (
                    <>
                      Diagnosis from the curves: <strong className="text-white">{o.verdict}</strong>
                      {o.overfitting_emerged_after_best_epoch && " — overfitting emerged after the restored epoch and was cut off by early stopping"}
                      .
                    </>
                  )}
                </p>
              </div>
            );
          })}
        </div>
      </section>

      {/* --- evaluation ---------------------------------------------------- */}
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
            const keys = reg ? ["mae", "rmse", "r2"] : ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"];
            const splits: [string, Record<string, any>][] = [
              ["Train", e.train_metrics], ["Validation", e.validation_metrics], ["Test", e.test_metrics],
            ];
            return (
              <div key={target} className="card overflow-x-auto space-y-3">
                <h3 className="font-semibold text-white">{TARGET_LABEL[target]}</h3>
                <table className="w-full text-sm">
                  <thead className="text-white/50">
                    <tr>
                      <th className="text-left font-normal py-1">Split</th>
                      {!reg && <th className="text-right font-normal">pit laps</th>}
                      {keys.map((k) => <th key={k} className="text-right font-normal">{k.toUpperCase()}</th>)}
                    </tr>
                  </thead>
                  <tbody className="text-white/80">
                    {splits.map(([name, s]) => (
                      <tr key={name} className={`border-t border-white/5 ${name === "Test" ? "bg-white/5" : ""}`}>
                        <td className="py-1.5">{name}</td>
                        {!reg && <td className="text-right tabular-nums">{s?.n_positive}/{s?.n}</td>}
                        {keys.map((k) => (
                          <td key={k} className="text-right tabular-nums">{fmt(s?.[k])}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
                {!reg && e.threshold && (
                  <p className="text-xs text-white/50">
                    Decision threshold <strong className="text-white/80">{fmt(e.threshold.threshold)}</strong>, tuned on{" "}
                    {e.threshold.n_samples} out-of-fold predictions ({e.threshold.n_positive} pit laps): F1{" "}
                    {fmt(e.threshold["at_default_0.5"]?.f1)} at 0.5 → {fmt(e.threshold.at_threshold?.f1)} at the tuned value.
                  </p>
                )}
                {!reg && e.test_metrics.n_positive < 5 && (
                  <p className="text-xs text-amber-400/70">
                    ⚠ The test laps contain <strong>{e.test_metrics.n_positive} pit event(s) in {e.test_metrics.n} laps</strong>.
                    Precision, recall, F1 and PR-AUC on so few positives are dominated by chance; the cross-validated
                    figures rest on {e.threshold?.n_positive ?? "more"} pit laps and are the better guide.
                  </p>
                )}
                <div className="grid gap-3 sm:grid-cols-2">
                  {(reg ? ["prediction_vs_actual.png"] : ["confusion_matrix.png", "roc_curve.png"]).map((n) =>
                    figure(target, n) ? (
                      <ArtifactImage key={n} src={artifactUrl(figure(target, n)!)} alt={`${target} ${n}`} className="w-full rounded" />
                    ) : null,
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* --- full comparison ----------------------------------------------- */}
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
                          {row.model === t.task6_best_model && (
                            <span className="badge ml-2">Task 6 best</span>
                          )}
                        </td>
                        {keys.map((k) => (
                          <td key={k} className="text-right tabular-nums">
                            {row.metrics?.[k] === null || row.metrics?.[k] === undefined ? (
                              <span className="text-white/30" title="mathematically undefined on this split">
                                undefined
                              </span>
                            ) : (
                              fmt(row.metrics[k])
                            )}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="text-sm text-white/70 mt-3">{t.verdict.replace(/\*\*/g, "")}</p>
                {t.task === "classification" && (
                  <p className="text-xs text-white/50 mt-2">
                    Ranked by <strong>PR-AUC</strong>, not ROC-AUC, which stays high at this prevalence for a
                    model that never fires. Task 6&rsquo;s rows are its own committed results on the same test laps.
                  </p>
                )}
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
          {artifacts.reports.filter((r) => r.endsWith(".md") || r.endsWith(".csv")).map((r, i) => (
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
          {artifacts.models.map((m) => (
            <li key={m}>
              <code className="text-white/70">{m}</code>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
