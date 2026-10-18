/**
 * Model cards for the RAI tab, assembled from what the project already
 * records. No field is written free-hand:
 *
 *   metrics, architecture, capacity  lib/modelComparison.ts (same holdout)
 *   training rows, CV folds, laps,
 *   hyperparameters, versions        /api/ml/models (Task 6 registry)
 *   network config, epochs,
 *   overfitting diagnosis, split     /api/dl/models, /api/dl/metrics
 *   preprocessing                    app/intelligence/ml/preprocessing.py and
 *                                    the "preprocessing" line Task 7 writes to
 *                                    artifacts/deep_learning/model_metadata.json
 *   intended use / not-for           app/services/strategy_report.py's human
 *                                    oversight line, and how
 *                                    strategy_service.py actually uses each model
 *
 * Strengths and limitations are derived from the measured results (who leads
 * which metric, negative R², overfitting verdict, holdout pit count), not
 * asserted per algorithm.
 */
import type { DatasetSource, DlModel, Registry } from "@/lib/api";
import { MODEL_KEYS, MODEL_ROLE, type ModelKey, type TargetComparison } from "@/lib/modelComparison";

export type CardFact = { label: string; value: string };
export type ModelCard = {
  key: ModelKey;
  type: string;
  purpose: string;
  intendedUse: string;
  servedBy: string;
  data: CardFact[];
  preprocessing: string;
  evaluation: CardFact[];
  cv: CardFact[];
  strengths: string[];
  limitations: string[];
  risks: string[];
  config: CardFact[];
  version: CardFact[];
  interpretability: { level: "Direct" | "Indirect" | "Post-hoc only"; how: string };
};

const OVERSIGHT = "Decision support for engineers, not an autonomous strategy system, and not for betting.";

const PURPOSE: Record<string, string> = {
  target_laptime: "Predict the next lap time, in seconds, from the current race state.",
  target_pit_next_lap: "Estimate the probability that the driver pits on the next lap.",
};

const SERVED_BY: Record<ModelKey, string> = {
  simple: "Evaluation baseline. Not used by the Strategy simulator by default; reachable through the strategy API only when named explicitly.",
  ml: "The Strategy simulator's default classical model (Task 6's selection), and the fallback when the ML and DNN pit calls disagree.",
  dnn: "The Strategy simulator's deep-learning stage, cross-checked against the ML model before a pit call is recommended.",
};

const SCALED = new Set(["linear_regression", "logistic_regression", "svr", "svm"]);

function classicalPreprocessing(model: string | null): string {
  return model && SCALED.has(model)
    ? "Defensive imputation (median for continuous, most-frequent for 0/1 columns; the data has no gaps), then StandardScaler on continuous columns, fitted on training laps only. 0/1 indicator columns are not scaled."
    : "Defensive imputation only (median / most-frequent; the data has no gaps). Trees split on raw values, so no scaling is applied.";
}

const DNN_PREPROCESSING =
  "StandardScaler fitted on the training laps only; 0/1 indicator columns pass through unscaled. For lap time, the target is also standardised on training laps and converted back to seconds before every metric.";

const INTERPRETABILITY: Record<string, ModelCard["interpretability"]> = {
  linear_regression: { level: "Direct", how: "Each input has one weight; the prediction is their sum, readable without extra tools." },
  logistic_regression: { level: "Direct", how: "Each input has one weight on the log-odds of pitting, readable without extra tools." },
  random_forest: { level: "Indirect", how: "Global feature importances from the trees; per-lap reasons need SHAP (Task 8)." },
  decision_tree: { level: "Direct", how: "A single tree of yes/no splits that can be read top to bottom." },
  xgboost: { level: "Indirect", how: "Global feature importances; per-lap reasons need SHAP." },
  svr: { level: "Post-hoc only", how: "An RBF kernel has no readable weights; permutation importance and SHAP are the only window in." },
  svm: { level: "Post-hoc only", how: "An RBF kernel has no readable weights; explanations need post-hoc tools." },
};

const fmt = (v: unknown, d = 3) => (typeof v === "number" && Number.isFinite(v) ? v.toFixed(d) : "N/A");
const date = (iso?: string | null) => (iso ? new Date(iso).toISOString().slice(0, 10) : "N/A");
const lapRange = (laps?: number[]) => (laps?.length ? `laps ${laps[0]}–${laps[laps.length - 1]}` : "N/A");

function datasetLine(ds: DatasetSource | null | undefined): string {
  if (!ds) return "N/A";
  return ds.source === "real_fastf1"
    ? `Real FastF1 telemetry: ${ds.year} ${ds.event} GP (${ds.session === "R" ? "race" : ds.session}), ${ds.n_laps} laps, ${ds.n_drivers} drivers`
    : "Synthetic data: not real-world F1 performance";
}

export function buildModelCards(
  c: TargetComparison,
  registry: Registry | null,
  dlModels: DlModel[] | null,
  dlMetrics: Record<string, any> | null,
  dataset: DatasetSource | null,
): Record<ModelKey, ModelCard> {
  const out = {} as Record<ModelKey, ModelCard>;
  const primary = c.rankings.find((r) => r.metric.key === c.selectionMetric);
  const leads = (k: ModelKey) =>
    c.rankings.filter((r) => r.order.length > 1 && Math.abs(r.order[0].value - (r.order.find((o) => o.key === k)?.value ?? NaN)) < 1e-12);
  const last = (k: ModelKey) =>
    c.rankings.filter((r) => r.order.length > 1 && r.order[r.order.length - 1].key === k
      && Math.abs(r.order[r.order.length - 1].value - r.order[0].value) > 1e-12);

  for (const k of MODEL_KEYS) {
    const m = c.models[k];
    const isDnn = k === "dnn";
    const reg = !isDnn ? registry?.models.find((r) => r.target === c.target && r.model_name === m.model) : undefined;
    const dl = isDnn ? dlModels?.find((d) => d.target === c.target) : undefined;
    const dm = isDnn ? dlMetrics?.models?.[c.target] : undefined;
    const holdout = (reg?.validation as any)?.holdout;

    // ---- strengths / limitations, from results ----
    const strengths: string[] = [];
    const limitations: string[] = [];
    const won = leads(k);
    if (won.length) strengths.push(`Best (or tied best) of the three on ${won.map((r) => r.metric.label).join(", ")} on the held-out laps.`);
    const lost = last(k);
    if (lost.length) limitations.push(`Last of the three on ${lost.map((r) => r.metric.label).join(", ")}.`);
    const interp: ModelCard["interpretability"] = isDnn
      ? { level: "Post-hoc only",
          how: `${m.parameters !== null ? m.parameters.toLocaleString() : "Its"} weights spread across hidden layers with non-linear activations; explained per lap with SHAP and LIME in Task 8.` }
      : INTERPRETABILITY[m.model ?? ""] ?? { level: "Indirect", how: "See Task 8 for explanations." };
    if (interp.level === "Direct") strengths.push(`Transparent: ${interp.how.charAt(0).toLowerCase()}${interp.how.slice(1)}`);
    if (m.fitSeconds !== null && MODEL_KEYS.every((o) => o === k || c.models[o].fitSeconds === null || c.models[o].fitSeconds! >= m.fitSeconds!))
      strengths.push(`Fastest recorded fit: ${(m.fitSeconds * 1000).toFixed(1)} ms.`);
    if (c.task === "regression" && (m.metrics.r2 ?? 0) < 0) limitations.push("Test R² below zero: worse than always predicting the average lap.");
    if (c.task === "classification") {
      if (m.metrics.recall === 0) limitations.push("Caught none of the held-out pit laps (recall 0) at its decision threshold.");
      if ((m.metrics.precision ?? 1) < 0.05 && (m.metrics.recall ?? 0) > 0)
        limitations.push(`Many false alarms: precision ${fmt(m.metrics.precision)} at its threshold.`);
    }
    if (isDnn && dm?.overfitting?.verdict) limitations.push(`Training diagnosis: ${dm.overfitting.verdict.toLowerCase()}; early stopping restored epoch ${dm.best_epoch}.`);
    if (interp.level === "Post-hoc only") limitations.push(`Not directly interpretable: ${interp.how.charAt(0).toLowerCase()}${interp.how.slice(1)}`);
    if (k === "ml" && reg && primary) {
      const cvKey = c.selectionMetric;
      const cvv = (reg.metrics?.cv as any)?.[cvKey]?.mean;
      const tv = m.metrics[cvKey];
      if (typeof cvv === "number" && tv !== null && Math.abs(cvv - tv) / Math.max(Math.abs(cvv), 1e-9) > 0.25)
        limitations.push(`Cross-validation ${primary.metric.label} ${fmt(cvv)} vs held-out ${fmt(tv)}: results move with the evaluation window.`);
    }

    // ---- risks, from documented data limits ----
    const risks = [
      `Trained on one race: ${datasetLine(dataset)}. Unsuitable for other circuits, seasons or weather without retraining.`,
      "Unsuitable as an autonomous strategy call or for betting; outputs need an engineer's review.",
    ];
    if (c.task === "classification" && c.nPositive !== null)
      risks.push(`The held-out laps contain ${c.nPositive} real pit ${c.nPositive === 1 ? "stop" : "stops"}, so its test precision, recall and F1 can change completely on a single lap.`);
    if (c.task === "regression")
      risks.push("Pit in- and out-laps are recorded at Task 4's outlier cap rather than their real time; the drivers with the largest held-out errors are the ones whose window includes such a lap.");

    // ---- facts ----
    const data: CardFact[] = isDnn
      ? [
          { label: "Dataset", value: datasetLine(dataset) },
          { label: "Train / validation / test", value: dm ? `${dm.n_train} / ${dm.n_validation} / ${dm.n_test} laps` : "N/A" },
          { label: "Lap windows", value: dm?.final_fit_split ? `train ${lapRange(dm.final_fit_split.train_laps)}, early-stop ${lapRange(dm.final_fit_split.validation_laps)}, test ${lapRange(dm.final_fit_split.test_laps)}` : "N/A" },
          { label: "Inputs", value: m.nFeatures !== null ? `${m.nFeatures} features from the Task 5 contract` : "N/A" },
        ]
      : [
          { label: "Dataset", value: datasetLine(dataset) },
          { label: "Train / test", value: reg ? `${reg.training_rows} / ${reg.test_rows} laps` : "N/A" },
          { label: "Lap windows", value: holdout ? `train ${lapRange(holdout.dev_laps)}, test ${lapRange(holdout.test_laps)}` : "N/A" },
          { label: "Inputs", value: m.nFeatures !== null ? `${m.nFeatures} features from the Task 5 contract` : "N/A" },
        ];

    const evaluation: CardFact[] = c.metrics.map((metric) => ({
      label: metric.label,
      value: m.metrics[metric.key] === null ? "N/A" : `${fmt(m.metrics[metric.key], metric.digits)}${metric.unit === "s" ? " s" : metric.unit === "%" ? "%" : ""}`,
    }));
    const cv: CardFact[] = reg
      ? c.metrics
          .filter((metric) => (reg.metrics?.cv as any)?.[metric.key])
          .map((metric) => {
            const e = (reg.metrics.cv as any)[metric.key];
            return { label: metric.label, value: `${fmt(e.mean, metric.digits)} ± ${fmt(e.std, metric.digits)}` };
          })
      : [];

    const config: CardFact[] = isDnn
      ? [
          ...Object.entries(dl?.hyperparameters ?? {}).map(([kk, v]) => ({ label: kk.replace(/_/g, " "), value: Array.isArray(v) ? v.join(" → ") : String(v) })),
          { label: "epochs", value: dm ? `${dm.epochs_run} run, best ${dm.best_epoch} restored` : "N/A" },
          ...(m.threshold !== null ? [{ label: "decision threshold", value: m.threshold.toFixed(4) }] : []),
        ]
      : [
          ...(Object.keys(reg?.hyperparameters ?? {}).length
            ? Object.entries(reg!.hyperparameters).map(([kk, v]) => ({ label: kk.replace(/^estimator__/, ""), value: String(v) }))
            : [{ label: "hyperparameters", value: "none tuned (closed-form fit)" }]),
          { label: "validation", value: `${reg?.cv_folds ?? "N/A"} expanding-window lap-forward folds` },
          ...(m.threshold !== null ? [{ label: "decision threshold", value: m.threshold.toFixed(4) }] : []),
          { label: "random state", value: String((reg as any)?.random_state ?? "N/A") },
        ];

    const versions = (reg as any)?.software_versions as Record<string, string> | undefined;
    const version: CardFact[] = isDnn
      ? [
          { label: "artifact", value: dl?.artifact ?? "N/A" },
          { label: "framework", value: dl?.framework ? `${dl.framework} (${dl.model_format})` : "N/A" },
          { label: "metrics written", value: date(dlMetrics?.generated_at) },
        ]
      : [
          { label: "artifact", value: reg?.artifact ?? "N/A" },
          { label: "software", value: versions ? Object.entries(versions).map(([a, b]) => `${a.replace(/_/g, "-")} ${b}`).join(", ") : "N/A" },
          { label: "trained", value: date(reg?.trained_at) },
        ];

    out[k] = {
      key: k,
      type: `${m.algorithm}${isDnn ? " · deep learning (Task 7)" : k === "ml" ? " · classical ML, Task 6's selection" : " · linear baseline (Task 6)"}`,
      purpose: PURPOSE[c.target],
      intendedUse: OVERSIGHT,
      servedBy: SERVED_BY[k],
      data,
      preprocessing: isDnn ? DNN_PREPROCESSING : classicalPreprocessing(m.model),
      evaluation,
      cv,
      strengths: strengths.length ? strengths : ["No metric where it leads on this holdout."],
      limitations,
      risks,
      config,
      version,
      interpretability: interp,
    };
  }
  return out;
}

export { MODEL_ROLE };
