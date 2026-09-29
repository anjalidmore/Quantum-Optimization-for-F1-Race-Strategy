import { artifactUrl, QmlSummary, QmlTarget } from "@/lib/api";
import { Chart } from "@/components/charts/Chart";
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
          <h2 className="text-lg font-semibold text-paper-900">Quantum machine learning</h2>
          <span className="badge">PennyLane {summary.framework.pennylane}</span>
          <span className="badge badge-warning">simulated</span>
        </div>
        <p className="text-paper-500 mt-2 max-w-3xl text-sm">{summary.honesty_note}</p>
        <p className="text-xs text-paper-400 mt-2 max-w-3xl">
          {summary.simulator}. Same laps, same chronological split and the same metric code as Tasks 6 and 7.
          Hyperparameters (layers ∈ {summary.search_space.n_layers.join(", ")}, learning rate ∈{" "}
          {summary.search_space.learning_rate.join(", ")}) were chosen on the cross-validation folds only. Whole
          run: {summary.wall_seconds.toFixed(1)} s, seed {summary.seed}.
        </p>
      </section>

      {Object.entries(summary.targets).map(([target, t]) => (
        <QuantumTargetTable key={target} target={target} t={t} />
      ))}

      {/* The circuit is wide by nature, so it gets a full-width row of its own
          and is the only element permitted to scroll sideways - inside its own
          box, never the page. */}
      <Chart name="qml_circuit" height={300} />

      <section className="grid gap-5 md:grid-cols-2">
        {["qml_training_loss", "qml_metrics_target_laptime", "qml_metrics_target_pit_next_lap",
          "qml_training_time", "qml_roc_vs_classical", "qml_pr_vs_classical",
          "qml_predicted_vs_actual"].map((name) => (
          <Chart key={name} name={name} />
        ))}
      </section>

      <section className="card">
        <p className="text-sm text-paper-500">
          Full write-up, including what this experiment cannot tell you:{" "}
          <a
            className="text-paper-900 underline decoration-edge underline-offset-[3px] hover:decoration-paper-900"
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
      <h3 className="font-semibold text-paper-900 mb-1">{TARGET_LABEL[target] ?? target}</h3>
      <p className="text-xs text-paper-500 mb-3">
        {t.encoding.method}
        {t.encoding.explained_variance_ratio !== null &&
          `, retaining ${(t.encoding.explained_variance_ratio * 100).toFixed(1)}% of the variance`}{" "}
        · {t.encoding.n_qubits} qubits · {t.n_dev} development laps, {t.n_test} test laps
        {isClf && t.n_test_positive !== undefined && ` (${t.n_test_positive} pit event)`} ·{" "}
        {t.encoding.fitted_on}
      </p>
      <div className="card min-w-0">
        {/* The CV column is the one that carries information here, so it is the
            column that survives at phone width. Family, parameter count and the
            test columns drop away rather than forcing the page sideways. */}
        <table className="t-table">
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col" className="hidden md:table-cell">Family</th>
              <th scope="col" className="num hidden md:table-cell">Params</th>
              <th scope="col" className="num hidden sm:table-cell">Train (s)</th>
              <th scope="col" className="num">CV {metric} (mean ± sd)</th>
              {testKeys.map((k) => (
                <th key={k} scope="col" className="num hidden lg:table-cell">
                  test {k.toUpperCase().replace("_", "-")}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((m) => (
              <tr key={m.model} className={m.family === "quantum" ? "is-marked" : ""}>
                <td>
                  {m.model}
                  {m.family === "quantum" && <span className="marker ml-2">quantum</span>}
                </td>
                <td className="hidden text-paper-500 md:table-cell">{m.family}</td>
                <td className="num hidden md:table-cell">{m.n_parameters || "—"}</td>
                <td className="num hidden sm:table-cell">
                  {m.train_seconds === null ? "—" : m.train_seconds.toFixed(2)}
                </td>
                <td className="num">
                  {m.cv_mean === null ? "undefined" : `${fmt(m.cv_mean, 4)} ± ${fmt(m.cv_std, 4)}`}
                </td>
                {testKeys.map((k) => (
                  <td key={k} className="num hidden lg:table-cell">
                    {fmt(m.test_metrics?.[k], 4)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {isClf && (
          <p className="text-xs text-paper-500 mt-2">
            The test laps hold {t.n_test_positive} pit event, so every test column here is decided by one lap
            — a test PR-AUC of 1.0 only means that lap got the top score. The CV column is the one to read.
          </p>
        )}
        {t.notes.map((n) => (
          <p key={n} className="text-xs text-paper-400 mt-1">
            {n}
          </p>
        ))}
      </div>
    </section>
  );
}
