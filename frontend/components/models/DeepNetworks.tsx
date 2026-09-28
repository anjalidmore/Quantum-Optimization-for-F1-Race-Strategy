import { artifactUrl, DlArtifacts, DlComparison, DlHistory, DlModel } from "@/lib/api";
import { ArtifactImage } from "@/components/ArtifactImage";
import { fmt, TARGET_LABEL } from "@/lib/format";

/** Task 7: the headline verdict per target, each network's shape, and its training curves. */
export function DeepNetworks({
  models,
  comparison,
  history,
  artifacts,
  metrics,
}: {
  models: DlModel[];
  comparison: DlComparison;
  history: DlHistory;
  artifacts: DlArtifacts;
  metrics: Record<string, any> | null;
}) {
  const dir: Record<string, string> = { target_laptime: "laptime", target_pit_next_lap: "pit_decision" };
  // Figures live at deep_learning/<laptime|pit_decision>/<name>.png
  const figure = (target: string, name: string) =>
    artifacts.figures.find((f) => f.endsWith(`${dir[target]}/${name}`));

  return (
    <>
      <section className="grid gap-4 md:grid-cols-2">
        {Object.entries(comparison.targets).map(([target, t]) => {
          const dnn = t.comparison.find((c) => c.family === "deep");
          const deepWins = t.verdict.includes("deep network wins");
          return (
            <div key={target} className="card">
              <div className="flex items-center justify-between gap-2">
                <h3 className="font-semibold text-paper-900">{TARGET_LABEL[target] ?? target}</h3>
                <span className={`badge ${deepWins ? "badge-success" : "badge-warning"}`}>
                  {deepWins ? "DNN wins" : "classical wins"}
                </span>
              </div>
              <div className="mt-3 text-3xl font-bold text-paper-900">
                {fmt(dnn?.metrics?.[t.selection_metric], 4)}
                <span className="text-sm font-normal text-paper-500 ml-2">
                  test {t.selection_metric.toUpperCase()}
                </span>
              </div>
              <p className="text-sm text-paper-500 mt-3">{t.verdict.replace(/\*\*/g, "")}</p>
            </div>
          );
        })}
      </section>

      <section>
        <h2 className="text-lg font-semibold text-paper-900 mb-3">Network architectures</h2>
        <div className="grid gap-4 md:grid-cols-2">
          {models.map((m) => {
            const ratio = m.architecture.total_parameters / Math.max(m.training_rows, 1);
            return (
              <div key={m.target} className="card">
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold text-paper-900">{TARGET_LABEL[m.target] ?? m.target}</h3>
                  <span className="badge">{m.model_format}</span>
                </div>
                <table className="w-full text-sm mt-3">
                  <thead className="text-paper-500">
                    <tr>
                      <th className="text-left font-normal py-1">Layer</th>
                      <th className="text-left font-normal">Type</th>
                      <th className="text-right font-normal">Detail</th>
                    </tr>
                  </thead>
                  <tbody className="text-paper-700">
                    {m.architecture.layers.map((l: any) => (
                      <tr key={l.name} className="border-t border-track-300">
                        <td className="py-1">{l.name}</td>
                        <td>{l.type}</td>
                        <td className="text-right text-paper-500">
                          {l.units !== undefined ? `${l.units} units, ${l.activation}` : ""}
                          {l.rate !== undefined ? `rate ${l.rate}` : ""}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="mt-3 text-sm text-paper-500 space-y-1">
                  <div>
                    <span className="text-paper-700">{m.architecture.total_parameters.toLocaleString()}</span>{" "}
                    parameters · <span className="text-paper-700">{m.training_rows}</span> training rows · ratio{" "}
                    <span className={ratio > 1 ? "text-accent" : "text-paper-700"}>{ratio.toFixed(2)}</span>
                  </div>
                  {ratio > 1 && (
                    <p className="text-paper-500">
                      More parameters than training examples — the expected small-data regime here, and why
                      dropout, L2 and early stopping are all applied together.
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

      <section>
        <h2 className="text-lg font-semibold text-paper-900 mb-1">Training curves</h2>
        <p className="text-sm text-paper-500 mb-3">
          The final network trains on earlier laps and early-stops on the block of laps just before the test
          laps. The dashed line marks the epoch whose weights were restored and saved.
        </p>
        <div className="grid gap-4 md:grid-cols-2">
          {Object.entries(history).map(([target, h]) => {
            const extra = target === "target_laptime" ? "mae_curve.png" : "accuracy_curve.png";
            const o = metrics?.models?.[target]?.overfitting ?? {};
            return (
              <div key={target} className="card space-y-3">
                <h3 className="font-semibold text-paper-900">{TARGET_LABEL[target] ?? target}</h3>
                {figure(target, "loss_curve.png") && (
                  <ArtifactImage
                    src={artifactUrl(figure(target, "loss_curve.png")!)}
                    alt={`${target} loss curve`}
                    className="w-full rounded"
                  />
                )}
                {figure(target, extra) && (
                  <ArtifactImage
                    src={artifactUrl(figure(target, extra)!)}
                    alt={`${target} ${extra}`}
                    className="w-full rounded"
                  />
                )}
                <p className="text-sm text-paper-500">
                  Ran {h.epochs_run} of a maximum {h.max_epochs} epochs; early stopping (patience{" "}
                  {h.early_stopping_patience}) restored epoch {h.best_epoch}.{" "}
                  {o.verdict && (
                    <>
                      Diagnosis from the curves: <strong className="text-paper-900">{o.verdict}</strong>
                      {o.overfitting_emerged_after_best_epoch &&
                        " — overfitting emerged after the restored epoch and was cut off by early stopping"}
                      .
                    </>
                  )}
                </p>
              </div>
            );
          })}
        </div>
      </section>
    </>
  );
}
