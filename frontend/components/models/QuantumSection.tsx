import { artifactUrl, QmlSummary, QmlTarget } from "@/lib/api";
import { ArtifactImage } from "@/components/ArtifactImage";
import { fmt, TARGET_LABEL } from "@/lib/format";

/**
 * Quantum ML, read only from artifacts/metrics/qml_metrics.json.
 *
 * The honesty note is rendered first and comes from the API, not from this
 * file: a page that shows quantum results next to classical ones has to say
 * plainly that it is a noiseless simulation before it shows a single number.
 */
export function QuantumSection({ summary }: { summary: QmlSummary }) {
  return (
    <>
      <section>
        <div className="flex items-center gap-3 flex-wrap">
          <h2 className="text-lg font-semibold text-white">Quantum machine learning</h2>
          <span className="badge">PennyLane {summary.framework.pennylane}</span>
          <span className="badge badge-warning">simulated</span>
        </div>
        <p className="text-white/60 mt-2 max-w-3xl text-sm">{summary.honesty_note}</p>
        <p className="text-xs text-white/40 mt-2 max-w-3xl">
          {summary.simulator}. Same laps, same chronological split and the same metric code as Tasks 6 and 7.
          Hyperparameters (layers ∈ {summary.search_space.n_layers.join(", ")}, learning rate ∈{" "}
          {summary.search_space.learning_rate.join(", ")}) were chosen on the cross-validation folds only. Whole
          run: {summary.wall_seconds.toFixed(1)} s, seed {summary.seed}.
        </p>
      </section>

      {Object.entries(summary.targets).map(([target, t]) => (
        <QuantumTargetTable key={target} target={target} t={t} />
      ))}

      <section className="grid gap-4 md:grid-cols-2">
        {[
          ["circuit", "The variational circuit, drawn by PennyLane from the training code"],
          ["loss_curves", "Training loss per epoch (final fit)"],
          ["metric_comparison", "Every model on the same test laps"],
          ["training_time", "Training cost — simulated circuits vs equivalent classical models"],
          ["roc_pr", "ROC and precision-recall, pit decision"],
          ["predicted_vs_actual", "Predicted vs actual lap time"],
        ].map(([key, caption]) =>
          summary.figures[key] ? (
            <div key={key} className="card">
              <ArtifactImage src={artifactUrl(summary.figures[key])} alt={caption} className="w-full rounded" />
              <p className="text-xs text-white/50 mt-2">{caption}</p>
            </div>
          ) : null,
        )}
      </section>

      <section className="card">
        <p className="text-sm text-white/60">
          Full write-up, including what this experiment cannot tell you:{" "}
          <a
            className="text-sky-400 hover:underline"
            href={artifactUrl(summary.report)}
            target="_blank"
            rel="noreferrer"
          >
            classical_vs_quantum_report.md
          </a>
        </p>
      </section>
    </>
  );
}

function QuantumTargetTable({ target, t }: { target: string; t: QmlTarget }) {
  const isClf = t.task === "classification";
  const metric = t.selection_metric.toUpperCase().replace("_", "-");
  const testKeys = isClf ? ["pr_auc", "roc_auc", "f1", "precision", "recall"] : ["mae", "rmse", "r2"];

  const rows = [...t.models];
  if (t.task6_reference) {
    rows.push({
      model: `${t.task6_reference.model} — Task 6, all features`,
      family: "classical (full)",
      description: "",
      n_parameters: 0,
      train_seconds: null,
      cv_mean: (t.task6_reference.cv_summary?.[t.selection_metric] ?? {}).mean ?? null,
      cv_std: (t.task6_reference.cv_summary?.[t.selection_metric] ?? {}).std ?? null,
      test_metrics: t.task6_reference.test_metrics,
    });
  }

  return (
    <section>
      <h3 className="font-semibold text-white mb-1">{TARGET_LABEL[target] ?? target}</h3>
      <p className="text-xs text-white/50 mb-3">
        {t.encoding.method}
        {t.encoding.explained_variance_ratio !== null &&
          `, retaining ${(t.encoding.explained_variance_ratio * 100).toFixed(1)}% of the variance`}{" "}
        · {t.encoding.n_qubits} qubits · {t.n_dev} development laps, {t.n_test} test laps
        {isClf && t.n_test_positive !== undefined && ` (${t.n_test_positive} pit event)`} ·{" "}
        {t.encoding.fitted_on}
      </p>
      <div className="card overflow-x-auto">
        <table className="w-full text-sm min-w-[42rem]">
          <thead className="text-white/50">
            <tr>
              <th className="text-left font-normal py-1">Model</th>
              <th className="text-left font-normal">Family</th>
              <th className="text-right font-normal">Params</th>
              <th className="text-right font-normal">Train (s)</th>
              <th className="text-right font-normal">CV {metric} (mean ± sd)</th>
              {testKeys.map((k) => (
                <th key={k} className="text-right font-normal">
                  test {k.toUpperCase().replace("_", "-")}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="text-white/80">
            {rows.map((m) => (
              <tr
                key={m.model}
                className={`border-t border-white/5 ${m.family === "quantum" ? "bg-violet-500/10" : ""}`}
              >
                <td className="py-1.5">
                  {m.model}
                  {m.family === "quantum" && <span className="badge ml-2">quantum</span>}
                </td>
                <td className="text-white/50">{m.family}</td>
                <td className="text-right tabular-nums">{m.n_parameters || "—"}</td>
                <td className="text-right tabular-nums">
                  {m.train_seconds === null ? "—" : m.train_seconds.toFixed(2)}
                </td>
                <td className="text-right tabular-nums">
                  {m.cv_mean === null ? "undefined" : `${fmt(m.cv_mean, 4)} ± ${fmt(m.cv_std, 4)}`}
                </td>
                {testKeys.map((k) => (
                  <td key={k} className="text-right tabular-nums">
                    {fmt(m.test_metrics?.[k], 4)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {isClf && (
          <p className="text-xs text-amber-400/70 mt-2">
            The test laps hold {t.n_test_positive} pit event, so every test column here is decided by one lap
            — a test PR-AUC of 1.0 only means that lap got the top score. The CV column is the one to read.
          </p>
        )}
        {t.notes.map((n) => (
          <p key={n} className="text-xs text-white/40 mt-1">
            {n}
          </p>
        ))}
      </div>
    </section>
  );
}
