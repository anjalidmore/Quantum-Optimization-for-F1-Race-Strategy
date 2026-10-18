/**
 * Model Comparison — Simple vs ML vs DNN, built from artifacts the pipeline
 * already wrote. Nothing here trains, predicts or invents a number: every
 * figure comes from one of these, and anything they do not contain is null
 * and rendered as "N/A".
 *
 *   /api/dl/comparison  every model's test metrics on the identical
 *                       chronological holdout (Task 7 wrote the classical rows
 *                       from Task 6's own committed metrics)
 *   /api/ml/comparison  Task 6's final-fit wall time (fit_seconds)
 *   /api/ml/models      Task 6's registry: feature lists, hyperparameters
 *   /api/dl/models      the Keras architecture (layers, parameter count)
 *   /api/dl/history     epochs run / restored by early stopping
 *
 * Which model plays which role:
 *   Simple — the linear model in Task 6's line-up (linear regression for lap
 *            time, logistic regression for pit decisions): the smallest model
 *            the project trains, so the natural baseline.
 *   ML     — Task 6's *selected* model for that target, whatever it is.
 *   DNN    — Task 7's Keras multilayer perceptron.
 */
import type { Comparison, DlComparison, DlHistory, DlModel, Registry } from "@/lib/api";

export type ModelKey = "simple" | "ml" | "dnn";
export const MODEL_KEYS: ModelKey[] = ["simple", "ml", "dnn"];

export const MODEL_ROLE: Record<ModelKey, { label: string; short: string; color: string }> = {
  simple: { label: "Simple Model", short: "Simple", color: "var(--model-simple)" },
  ml: { label: "ML Model", short: "ML", color: "var(--model-ml)" },
  dnn: { label: "DNN Model", short: "DNN", color: "var(--model-dnn)" },
};

/** The linear model in each Task 6 line-up (first spec in regression.py / classification.py). */
const SIMPLE_MODEL: Record<string, string> = {
  target_laptime: "linear_regression",
  target_pit_next_lap: "logistic_regression",
};

export type Target = "target_laptime" | "target_pit_next_lap";
export const TARGETS: { key: Target; label: string; question: string }[] = [
  { key: "target_laptime", label: "Lap time", question: "How fast will the next lap be?" },
  { key: "target_pit_next_lap", label: "Pit decision", question: "Will the driver pit next lap?" },
];

export type MetricDef = {
  key: string;
  label: string;
  better: "lower" | "higher";
  unit: string | null;
  digits: number;
  /** One sentence for a reader without an ML background. */
  plain: string;
};

const REGRESSION_METRICS: MetricDef[] = [
  { key: "mae", label: "MAE", better: "lower", unit: "s", digits: 3,
    plain: "Average miss per lap, in seconds." },
  { key: "rmse", label: "RMSE", better: "lower", unit: "s", digits: 3,
    plain: "Like MAE, but big misses count extra." },
  { key: "r2", label: "R²", better: "higher", unit: null, digits: 3,
    plain: "Share of lap-to-lap variation explained. Below 0 is worse than always guessing the average lap." },
  { key: "mape", label: "MAPE", better: "lower", unit: "%", digits: 3,
    plain: "Average miss as a percentage of the lap time." },
];

const CLASSIFICATION_METRICS: MetricDef[] = [
  { key: "pr_auc", label: "PR-AUC", better: "higher", unit: null, digits: 3,
    plain: "How well pit laps are ranked above non-pit laps, focusing on the rare pit laps." },
  { key: "roc_auc", label: "ROC-AUC", better: "higher", unit: null, digits: 3,
    plain: "Chance a random pit lap gets a higher score than a random non-pit lap." },
  { key: "f1", label: "F1", better: "higher", unit: null, digits: 3,
    plain: "Balance of precision and recall at the model's decision threshold." },
  { key: "precision", label: "Precision", better: "higher", unit: null, digits: 3,
    plain: "When it says “pit”, how often that is right." },
  { key: "recall", label: "Recall", better: "higher", unit: null, digits: 3,
    plain: "Of the real pit laps, how many it caught." },
  { key: "accuracy", label: "Accuracy", better: "higher", unit: null, digits: 3,
    plain: "Share of all laps classified correctly. Misleading when pit laps are rare." },
];

/** Plain-language approach per algorithm, matching how app/intelligence/ml builds it. */
const ALGORITHM: Record<string, { name: string; approach: string; scaled: boolean }> = {
  linear_regression: { name: "Linear regression", scaled: true,
    approach: "One weight per input, added up into a lap time: a straight-line rule and the yardstick every other model has to beat." },
  logistic_regression: { name: "Logistic regression", scaled: true,
    approach: "One weight per input, summed and squashed into a 0–1 pit probability. Class-balanced, so the rare pit laps still count." },
  svr: { name: "Support vector regression", scaled: true,
    approach: "Fits a smooth curved (RBF-kernel) function that keeps most laps inside a narrow error tube, ignoring misses smaller than ε." },
  svm: { name: "Support vector machine", scaled: true,
    approach: "Draws a curved (RBF-kernel) boundary between pit and non-pit laps with the widest possible margin." },
  random_forest: { name: "Random forest", scaled: false,
    approach: "Hundreds of decision trees, each trained on a resample of the laps; their answers are averaged into one prediction." },
  decision_tree: { name: "Decision tree", scaled: false,
    approach: "A single flowchart of yes/no questions about the race state, ending in a prediction." },
  xgboost: { name: "XGBoost", scaled: false,
    approach: "Gradient-boosted trees: each small tree corrects the errors of the ones before it." },
  dnn_mlp: { name: "Multilayer perceptron (Keras)", scaled: true,
    approach: "Stacked dense layers with ReLU, dropout and L2 weight decay, trained with RMSprop and stopped early when validation loss stops improving." },
};

export function algorithmName(model: string | null): string {
  if (!model) return "N/A";
  return ALGORITHM[model]?.name ?? model.replace(/_/g, " ");
}

export type ArchStage = { title: string; detail?: string; kind: "input" | "prep" | "core" | "layer" | "dropout" | "output" };

export type ModelEntry = {
  key: ModelKey;
  model: string | null;
  algorithm: string;
  approach: string;
  available: boolean;
  metrics: Record<string, number | null>;
  fitSeconds: number | null;
  /** Explicit when timing is missing, so N/A is explained rather than silent. */
  timingNote: string | null;
  nFeatures: number | null;
  /** Trainable parameter count, only where it is an exact count. */
  parameters: number | null;
  capacity: string;
  threshold: number | null;
  stages: ArchStage[];
};

export type Ranking = { metric: MetricDef; order: { key: ModelKey; value: number }[] };

export type Insight = { title: string; body: string; tone: "win" | "info" | "warn"; model?: ModelKey };

export type TargetComparison = {
  target: Target;
  task: "regression" | "classification";
  selectionMetric: string;
  metrics: MetricDef[];
  models: Record<ModelKey, ModelEntry>;
  rankings: Ranking[];
  wins: Record<ModelKey, number>;
  insights: Insight[];
  nTest: number | null;
  nPositive: number | null;
  verdict: string | null;
};

export type ComparisonInputs = {
  dl: DlComparison | null;
  ml: Comparison | null;
  registry: Registry | null;
  dlModels: DlModel[] | null;
  history: DlHistory | null;
};

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

function fmtHyper(params: Record<string, unknown> | undefined): string {
  if (!params || !Object.keys(params).length) return "No tuned hyperparameters";
  return Object.entries(params)
    .map(([k, v]) => `${k.replace(/^estimator__/, "")}=${v === null ? "None" : String(v)}`)
    .join(" · ");
}

function classicalCapacity(model: string, nFeatures: number | null, params: Record<string, unknown> | undefined): string {
  if ((model === "linear_regression" || model === "logistic_regression") && nFeatures !== null) {
    return `${nFeatures + 1} weights (${nFeatures} inputs + intercept)`;
  }
  if (model === "random_forest" && params) {
    return `${params.n_estimators ?? "?"} trees · max depth ${params.max_depth ?? "unlimited"}`;
  }
  return fmtHyper(params);
}

function classicalStages(model: string, task: string, nFeatures: number | null, params: Record<string, unknown> | undefined, threshold: number | null): ArchStage[] {
  const algo = ALGORITHM[model];
  const stages: ArchStage[] = [
    { title: `${nFeatures ?? "?"} features`, detail: "Task 5 feature contract", kind: "input" },
    algo?.scaled
      ? { title: "StandardScaler", detail: "continuous columns only", kind: "prep" }
      : { title: "No scaling", detail: "trees split on raw values", kind: "prep" },
  ];
  const hp = params ?? {};
  switch (model) {
    case "linear_regression":
      stages.push({ title: "Weighted sum", detail: "Σ wᵢxᵢ + b", kind: "core" });
      break;
    case "logistic_regression":
      stages.push({ title: "Weighted sum", detail: `Σ wᵢxᵢ + b · C=${hp.C ?? "?"}`, kind: "core" });
      stages.push({ title: "Sigmoid", detail: "squash to 0–1", kind: "core" });
      break;
    case "svr":
      stages.push({ title: "RBF kernel", detail: `γ=${hp.gamma ?? "?"}`, kind: "core" });
      stages.push({ title: "Support vectors", detail: `C=${hp.C ?? "?"} · ε=${hp.epsilon ?? "?"}`, kind: "core" });
      break;
    case "random_forest":
      stages.push({ title: `${hp.n_estimators ?? "?"} trees`, detail: `max depth ${hp.max_depth ?? "∞"}`, kind: "core" });
      stages.push({ title: task === "classification" ? "Average votes" : "Average outputs", kind: "core" });
      break;
    default:
      stages.push({ title: algorithmName(model), detail: fmtHyper(params), kind: "core" });
  }
  stages.push(outputStage(task, threshold));
  return stages;
}

function outputStage(task: string, threshold: number | null): ArchStage {
  return task === "classification"
    ? { title: "P(pit next lap)", detail: threshold !== null ? `pit if ≥ ${threshold.toFixed(2)}` : undefined, kind: "output" }
    : { title: "Lap time", detail: "seconds", kind: "output" };
}

function dnnStages(m: DlModel, nFeatures: number | null, task: string, threshold: number | null): ArchStage[] {
  const stages: ArchStage[] = [
    { title: `${nFeatures ?? "?"} features`, detail: "same contract", kind: "input" },
    { title: "StandardScaler", detail: "train laps only", kind: "prep" },
  ];
  const layers = m.architecture.layers ?? [];
  layers.forEach((l: any, i: number) => {
    const last = i === layers.length - 1;
    if (l.type === "Dropout") stages.push({ title: "Dropout", detail: `${l.rate}`, kind: "dropout" });
    else if (!last) stages.push({ title: `Dense ${l.units}`, detail: l.activation, kind: "layer" });
  });
  stages.push(outputStage(task, threshold));
  return stages;
}

/** Normalised 0–1 score for the radar: 1 = best of the three. */
export function radarScore(metric: MetricDef, value: number | null, all: (number | null)[]): number | null {
  if (value === null) return null;
  if (metric.better === "lower") {
    const min = Math.min(...all.filter((v): v is number => v !== null && v > 0));
    return value > 0 && Number.isFinite(min) ? min / value : null;
  }
  // Higher-is-better metrics here are already on a 0–1 scale; R² can go
  // negative, and "worse than the mean" plots at the centre.
  return Math.max(0, Math.min(1, value));
}

export function buildComparison(target: Target, inputs: ComparisonInputs): TargetComparison | null {
  const t = inputs.dl?.targets?.[target];
  const mlTask = target === "target_laptime" ? "regression" : "classification";
  const task: "regression" | "classification" = (t?.task as any) ?? mlTask;
  const mlRows = inputs.ml?.[mlTask] ?? [];
  const selectedMl = t?.task6_best_model ?? mlRows.find((r) => r.selected)?.model ?? null;
  if (!t && !mlRows.length) return null;

  const metrics = task === "regression" ? REGRESSION_METRICS : CLASSIFICATION_METRICS;
  const rowFor = (model: string | null, family?: string) =>
    t?.comparison.find((c) => (family ? c.family === family : c.model === model && c.family !== "deep"));

  const regFor = (model: string | null) =>
    inputs.registry?.models.find((m) => m.target === target && m.model_name === model);

  const dlModel = inputs.dlModels?.find((m) => m.target === target) ?? null;
  const hist = inputs.history?.[target];

  const classical = (key: ModelKey, model: string | null): ModelEntry => {
    const row = rowFor(model);
    const reg = regFor(model);
    const mlRow = mlRows.find((r) => r.model === model);
    const nFeatures = reg?.features?.length ?? null;
    const threshold = num(row?.metrics?.decision_threshold);
    return {
      key,
      model,
      algorithm: algorithmName(model),
      approach: (model && ALGORITHM[model]?.approach) ?? "Not available.",
      available: !!row,
      metrics: Object.fromEntries(metrics.map((m) => [m.key, num(row?.metrics?.[m.key])])),
      fitSeconds: num(mlRow?.fit_seconds),
      timingNote: num(mlRow?.fit_seconds) === null ? "Fit time not recorded" : "Final fit on the training laps",
      nFeatures,
      parameters:
        (model === "linear_regression" || model === "logistic_regression") && nFeatures !== null ? nFeatures + 1 : null,
      capacity: model ? classicalCapacity(model, nFeatures, reg?.hyperparameters) : "N/A",
      threshold,
      stages: model ? classicalStages(model, task, nFeatures, reg?.hyperparameters, threshold) : [],
    };
  };

  const dnnRow = rowFor(null, "deep");
  const dnnThreshold = num(dnnRow?.metrics?.decision_threshold);
  const dnnFeatures: number | null = dlModel?.features?.length ?? null;

  const models: Record<ModelKey, ModelEntry> = {
    simple: classical("simple", SIMPLE_MODEL[target]),
    ml: classical("ml", selectedMl),
    dnn: {
      key: "dnn",
      model: dnnRow?.model ?? dlModel?.model_name ?? null,
      algorithm: algorithmName("dnn_mlp"),
      approach: ALGORITHM.dnn_mlp.approach,
      available: !!dnnRow,
      metrics: Object.fromEntries(metrics.map((m) => [m.key, num(dnnRow?.metrics?.[m.key])])),
      fitSeconds: null,
      timingNote: hist
        ? `Wall time not recorded · ${hist.epochs_run} of max ${hist.max_epochs} epochs, best epoch ${hist.best_epoch}`
        : "Training time not recorded",
      nFeatures: dnnFeatures,
      parameters: dlModel ? dlModel.architecture.total_parameters : null,
      capacity: dlModel
        ? `${dlModel.architecture.total_parameters.toLocaleString()} parameters · ${dlModel.architecture.optimizer} · ${dlModel.architecture.loss} loss`
        : "N/A",
      threshold: dnnThreshold,
      stages: dlModel ? dnnStages(dlModel, dnnFeatures, task, dnnThreshold) : [],
    },
  };

  const rankings: Ranking[] = metrics.map((metric) => {
    const order = MODEL_KEYS.map((key) => ({ key, value: models[key].metrics[metric.key] }))
      .filter((o): o is { key: ModelKey; value: number } => o.value !== null)
      .sort((a, b) => (metric.better === "lower" ? a.value - b.value : b.value - a.value));
    return { metric, order };
  });

  // A tie for first counts as a win for every model sharing it.
  const wins: Record<ModelKey, number> = { simple: 0, ml: 0, dnn: 0 };
  for (const r of rankings) {
    if (!r.order.length) continue;
    const best = r.order[0].value;
    r.order.filter((o) => Math.abs(o.value - best) < 1e-12).forEach((o) => (wins[o.key] += 1));
  }

  const nTest = num(dnnRow?.metrics?.n) ?? num(rowFor(selectedMl)?.metrics?.n);
  const nPositive = num(dnnRow?.metrics?.n_positive) ?? num(rowFor(selectedMl)?.metrics?.n_positive);

  const result: TargetComparison = {
    target, task, selectionMetric: t?.selection_metric ?? metrics[0].key, metrics, models, rankings, wins,
    insights: [], nTest, nPositive, verdict: t?.verdict?.replace(/\*\*/g, "").replace(/`/g, "") ?? null,
  };
  result.insights = buildInsights(result);
  return result;
}

function fmtVal(m: MetricDef, v: number): string {
  return `${v.toFixed(m.digits)}${m.unit === "s" ? " s" : m.unit === "%" ? "%" : ""}`;
}

function buildInsights(c: TargetComparison): Insight[] {
  const out: Insight[] = [];
  const name = (k: ModelKey) => MODEL_ROLE[k].short;
  const primary = c.rankings.find((r) => r.metric.key === c.selectionMetric);

  // 1. The metric the pipeline itself selects models on.
  if (primary && primary.order.length >= 2) {
    const [first, ...rest] = primary.order;
    const tied = rest.filter((o) => Math.abs(o.value - first.value) < 1e-12);
    const simple = primary.order.find((o) => o.key === "simple");
    let body = `${name(first.key)} scores ${fmtVal(primary.metric, first.value)} on ${primary.metric.label}, the metric this project selects ${c.task} models on.`;
    if (tied.length) body = `${[first, ...tied].map((o) => name(o.key)).join(" and ")} tie on ${primary.metric.label} at ${fmtVal(primary.metric, first.value)}, the metric this project selects ${c.task} models on.`;
    else if (simple && simple.key !== first.key && simple.value !== 0) {
      const change = primary.metric.better === "lower"
        ? (1 - first.value / simple.value) * 100
        : (first.value / simple.value - 1) * 100;
      body += ` That is ${Math.abs(change).toFixed(0)}% ${primary.metric.better === "lower" ? "lower" : "higher"} than the Simple baseline (${fmtVal(primary.metric, simple.value)}).`;
    }
    out.push({ title: `Best on ${primary.metric.label}`, body, tone: "win", model: first.key });
  }

  // 2. Overall: who leads on the most metrics.
  const scored = c.rankings.filter((r) => r.order.length >= 2);
  if (scored.length) {
    const ranked = MODEL_KEYS.slice().sort((a, b) => c.wins[b] - c.wins[a]);
    const top = ranked[0];
    const co = ranked.filter((k) => k !== top && c.wins[k] === c.wins[top]);
    out.push({
      title: "Most metrics led",
      body: co.length
        ? `${[top, ...co].map(name).join(" and ")} share the lead, best on ${c.wins[top]} of ${scored.length} metrics each.`
        : `${name(top)} is best (or tied best) on ${c.wins[top]} of ${scored.length} metrics: ${scored
            .filter((r) => Math.abs(r.order[0].value - (r.order.find((o) => o.key === top)?.value ?? NaN)) < 1e-12)
            .map((r) => r.metric.label).join(", ")}.`,
      tone: "info",
      model: co.length ? undefined : top,
    });
  }

  // 3. Speed, only where it was actually measured.
  const timed = MODEL_KEYS.filter((k) => c.models[k].fitSeconds !== null);
  if (timed.length >= 2) {
    const sorted = timed.slice().sort((a, b) => c.models[a].fitSeconds! - c.models[b].fitSeconds!);
    const fast = sorted[0], slow = sorted[sorted.length - 1];
    const missing = MODEL_KEYS.filter((k) => c.models[k].fitSeconds === null);
    out.push({
      title: "Fastest to train",
      body: `${name(fast)} fits in ${(c.models[fast].fitSeconds! * 1000).toFixed(1)} ms against ${name(slow)}'s ${(c.models[slow].fitSeconds! * 1000).toFixed(1)} ms (final fit only).${
        missing.length ? ` ${missing.map(name).join(", ")} training wall time was not recorded by its pipeline, so it is left out rather than guessed.` : ""}`,
      tone: "info",
      model: fast,
    });
  }

  // 4. Cost vs benefit: parameter counts are exact only for Simple and DNN.
  const s = c.models.simple, d = c.models.dnn;
  if (s.parameters && d.parameters && primary) {
    const sv = s.metrics[primary.metric.key], dv = d.metrics[primary.metric.key];
    if (sv !== null && dv !== null) {
      const dnnBetter = primary.metric.better === "lower" ? dv < sv : dv > sv;
      const ratio = d.parameters / s.parameters;
      out.push({
        title: "Size vs payoff",
        body: `The DNN has ${d.parameters.toLocaleString()} trainable parameters, ${ratio.toFixed(0)}× the Simple model's ${s.parameters}. ${
          dnnBetter
            ? `Here the extra capacity pays off: ${primary.metric.label} ${fmtVal(primary.metric, dv)} vs ${fmtVal(primary.metric, sv)}.`
            : `Here it does not pay off: ${primary.metric.label} ${fmtVal(primary.metric, dv)} vs the Simple model's ${fmtVal(primary.metric, sv)}. One race session is a small dataset for a network.`}`,
        tone: dnnBetter ? "info" : "warn",
      });
    }
  }

  // 5. Data caveats that change how the numbers should be read.
  if (c.task === "classification" && c.nPositive !== null && c.nTest !== null && c.nPositive < 10) {
    const acc = c.rankings.find((r) => r.metric.key === "accuracy");
    const blind = MODEL_KEYS.filter((k) => c.models[k].metrics.recall === 0);
    out.push({
      title: "Read with care",
      body: `The test laps contain ${c.nPositive} real pit ${c.nPositive === 1 ? "stop" : "stops"} in ${c.nTest} laps, so precision, recall and F1 can swing on a single lap.${
        blind.length && acc?.order[0] && blind.includes(acc.order[0].key)
          ? ` ${blind.map(name).join(", ")} ${blind.length > 1 ? "have" : "has"} the highest accuracy while catching none of the pit laps: never predicting a pit is ${(acc.order[0].value * 100).toFixed(1)}% “accurate” here.`
          : ""}`,
      tone: "warn",
    });
  }
  if (c.task === "regression") {
    const neg = MODEL_KEYS.filter((k) => (c.models[k].metrics.r2 ?? 0) < 0);
    if (neg.length) {
      out.push({
        title: "Worse than average",
        body: `${neg.map(name).join(", ")} ${neg.length > 1 ? "score" : "scores"} R² below zero on the test laps: worse than always predicting the mean lap time.`,
        tone: "warn",
      });
    }
  }
  return out;
}
