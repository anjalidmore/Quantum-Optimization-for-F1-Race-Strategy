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
  let error: string | null = null;

  try {
    [models, comparison, history, artifacts] = await Promise.all([
      api.dlModels(),
      api.dlComparison(),
      api.dlHistory(),
      api.dlArtifacts(),
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

  const figure = (name: string) => artifacts.figures.find((f) => f.endsWith(name));

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

      {/* --- training history --------------------------------------------- */}
      <section>
        <h2 className="text-lg font-semibold text-white mb-1">Training history</h2>
        <p className="text-sm text-white/50 mb-3">
          These curves are the diagnostic: they separate &ldquo;still learning&rdquo; from
          &ldquo;plateaued&rdquo; from &ldquo;overfitting&rdquo;. The marked epoch is the one early
          stopping restored.
        </p>
        <div className="grid gap-4 md:grid-cols-2">
          {Object.entries(history).map(([target, h]) => {
            const src = figure(`dl_${target}_training_history.png`);
            const discarded = h.epochs_run - h.best_epoch;
            return (
              <div key={target} className="card">
                <h3 className="font-semibold text-white mb-2">{TARGET_LABEL[target] ?? target}</h3>
                {src && <ArtifactImage src={artifactUrl(src)} alt={`${target} training history`} className="w-full rounded" />}
                <p className="text-sm text-white/60 mt-3">
                  Ran {h.epochs_run} of a maximum {h.max_epochs} epochs; early stopping (patience{" "}
                  {h.early_stopping_patience}) restored epoch {h.best_epoch}
                  {discarded > 0
                    ? `, discarding ${discarded} epochs of validation-loss deterioration.`
                    : " — validation loss was still improving at the cap."}
                </p>
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
                : ["roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy"];
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
                {figure(`dl_${target}_model_comparison.png`) && (
                  <div className="mt-4">
                    <ArtifactImage
                      src={artifactUrl(figure(`dl_${target}_model_comparison.png`)!)}
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
        <h2 className="font-semibold text-white mb-2">Model format</h2>
        <p className="text-sm text-white/60">{artifacts.format_note}</p>
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
