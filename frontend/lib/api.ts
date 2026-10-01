export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export function artifactUrl(relativePath: string): string {
  // relativePath looks like "artifacts/figures/xyz.png" — the backend
  // mounts the whole artifacts/ tree at /artifacts.
  const trimmed = relativePath.replace(/^artifacts\//, "");
  return `${API_BASE}/artifacts/${trimmed}`;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) {
    const body = await res.text();
    throw new ApiError(res.status, body || res.statusText);
  }
  return res.json() as Promise<T>;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new ApiError(res.status, typeof data?.detail === "string" ? data.detail : JSON.stringify(data?.detail ?? data));
  }
  return data as T;
}

export type HealthResponse = {
  status: string;
  models_trained: boolean;
  model_count: number;
  xgboost_available: boolean;
  xgboost_status: string | null;
  models?: {
    ml: Record<string, { trained: boolean; selected_model: string | null }>;
    dl: Record<string, { trained: boolean }>;
    xai_available: boolean;
  };
};

export type RegistryModel = {
  model_name: string;
  task: string;
  target: string;
  features: string[];
  validation: Record<string, unknown>;
  metrics: { cv: Record<string, any>; test: Record<string, any> };
  artifact: string;
  hyperparameters: Record<string, unknown>;
  training_rows: number;
  test_rows: number;
  cv_folds: number;
  is_selected_best: boolean;
  synthetic_data_warning: boolean;
  xgboost_status: string | null;
  trained_at: string;
};

export type Registry = { generated_at: string; task: string; models: RegistryModel[] };

// Rows carry per-task fields (cv_mae / cv_pr_auc / decision_threshold / …), so
// this stays permissive rather than two near-duplicate interfaces.
export type ComparisonRow = Record<string, any>;
export type Comparison = { regression: ComparisonRow[]; classification: ComparisonRow[] };

export type DatasetSource =
  | { source: "synthetic" }
  | { source: "real_fastf1"; year: number; event: string; session: string; fetched_at: string; n_laps: number; n_drivers: number };

export type Manifest = {
  generated_at: string;
  dataset: string;
  dataset_source: DatasetSource;
  synthetic_data_warning: boolean;
  best_regression_model: string | null;
  best_classification_model: string | null;
  models: string[];
  metrics: string[];
  figures: string[];
  reports: string[];
};

export type DataOptions = {
  drivers: string[];
  teams: string[];
  compounds: string[];
  total_laps_hint: number;
  track_temperature_range: { min: number; mean: number; max: number };
  dataset_source: DatasetSource;
};

export type FeatureDescriptor = {
  feature: string;
  display_name: string;
  description: string;
  unit: string | null;
  min?: number;
  median?: number;
  max?: number;
};

export type TopFeaturesResponse = {
  target: string;
  model: string | null;
  ranking_method: string;
  top_features: FeatureDescriptor[];
  remaining_features: FeatureDescriptor[];
};

export type TaskEvidence = {
  id: string;
  number: number;
  label: string;
  purpose: string;
  status: "completed" | "in_progress" | "upcoming";
  reports: string[];
  figures: string[];
  other_artifacts: string[];
  documents_complete?: number;
  documents_total?: number;
};

export type TaskEvidenceResponse = {
  tasks: TaskEvidence[];
  completed_count: number;
  total_count: number;
};

export type RaceState = {
  driver: string;
  team: string;
  current_lap: number;
  total_laps: number;
  tyre_compound: string;
  tyre_age: number;
  track_temperature: number;
  weather: string;
  fuel_kg: number;
  track_status: string;
  current_position: number;
  laptime_model?: string | null;
  pit_model?: string | null;
  explain?: boolean;
};

export type StrategyResponse = {
  race_state: RaceState;
  validation?: {
    passed: boolean;
    current_lap: number;
    total_laps: number;
    laps_remaining: number;
    tyre_compound: string;
    note: string;
  };
  prediction: {
    predicted_lap_time_seconds: number | null;
    laptime_model: string | null;
    probability_pit: number | null;
    pit_model: string | null;
    feature_rows: Record<string, Record<string, number>>;
    approximated_features: Record<string, string[]>;
    out_of_range: Record<string, { feature: string; value: number; training_min: number; training_max: number }[]>;
    context_only: Record<string, Record<string, boolean>>;
    errors: string[];
  };
  dl_prediction?: {
    predicted_lap_time_seconds: number | null;
    probability_pit: number | null;
    predicted_class: number | null;
    threshold: number | null;
    errors: string[];
  };
  recommendation?: {
    action: string;
    confidence: "high" | "moderate" | "low" | "none" | string;
    source: string;
    disagreement: boolean;
    ml_pit_probability: number | null;
    dl_pit_probability: number | null;
    reason: string;
  };
  recommended_action: string | null;
  expected_cost_seconds: number | null;
  optimal_search_strategy: {
    algorithm: string;
    found: boolean;
    expected_cost_seconds: number | null;
    plan: { type: string; compound: string | null }[];
    next_action: { type: string; compound: string | null } | null;
    note: string;
  };
  triggered_expert_rules: { rule_id: string; name: string; matched_conditions: string[]; asserted: Record<string, unknown> }[];
  evidence: Record<string, unknown>;
  xai_explanation?: Record<string, any>;
  data_source: string;
};

// --- Task 7 (Deep Learning) -------------------------------------------------
export type DlModel = {
  model_name: string;
  family: string;
  target: string;
  task: string;
  architecture: { name: string; total_parameters: number; layers: any[]; optimizer: string; loss: string };
  hyperparameters: Record<string, any>;
  metrics: { test: Record<string, any> };
  model_format: string;
  format_note: string;
  training_rows: number;
  test_rows: number;
};
export type DlComparison = {
  generated_at: string;
  note: string;
  dataset_source: DatasetSource;
  targets: Record<
    string,
    {
      task: string;
      selection_metric: string;
      task6_best_model: string | null;
      comparison: { model: string; family: string; metrics: Record<string, any> }[];
      verdict: string;
    }
  >;
};
export type DlHistory = Record<
  string,
  {
    epochs_run: number;
    best_epoch: number;
    early_stopping_patience: number;
    max_epochs: number;
    hyperparameters: Record<string, any>;
    history: Record<string, number[]>;
  }
>;
export type DlArtifacts = {
  figures: string[];
  models: string[];
  reports: string[];
  model_format: string;
  models_served_over_http: boolean;
};

// --- Task 8: stratification and per-lap inspection ---------------------------
export type XaiStratRow = Record<string, any> & { group_type: string; group: string; sample_note: string };
export type XaiStratification = Record<string, { task: string; rows: XaiStratRow[]; figure?: string }>;
export type XaiLapRow = Record<string, any> & {
  row_index: number; driver?: string; team?: string; compound?: string; lap: number;
  dnn_prediction: number; trust_score: number; trust_band: string;
};
export type XaiLaps = { target: string; task: string; laps: XaiLapRow[] };
export type XaiLap = {
  target: string; task: string; lap: XaiLapRow; race_state: Record<string, number>;
  shap_base_value: number; shap_factors: { feature: string; value: number; shap_value: number }[];
  counterfactual: Record<string, any>; decision_threshold: number | null;
};

// --- Task 8 (Explainable AI) ------------------------------------------------
export type TrustBand = { label: string; meaning: string };
export type XaiSummary = {
  feature_importance_figure?: string;
  trust_weights?: Record<string, number>;
  generated_at: string;
  dataset_source: DatasetSource;
  targets: Record<
    string,
    {
      task: string;
      classical_model_explained: string;
      n_features: number;
      n_identity_features: number;
      identity_attribution_share: number;
      concentration_ratio: number | null;
      trust: { n: number; mean: number | null; min: number | null; max: number | null; bands: Record<string, number> };
      explained_rows: string[];
    }
  >;
};
export type ShapAttribution = { feature: string; shap_value: number; direction: string };
export type XaiExplanation = {
  target: string;
  classical_model_explained: string;
  rows: Record<
    string,
    {
      row_index: number;
      lap: number;
      prediction: number;
      classical_prediction: number;
      narrative: string;
      counterfactual_sentence: string;
      trust_score: number;
      trust_band: TrustBand;
      top_factors: ShapAttribution[];
      trust_components?: Record<string, number>;
      driver?: string;
      team?: string;
      compound?: string;
      actual?: number;
      race_state?: Record<string, number>;
      shap_top3?: string[];
      lime_top3?: string[];
      lime?: { contributions?: { condition: string; weight: number }[]; local_r2?: number; local_prediction?: number };
      counterfactual?: Record<string, any>;
      figures?: Record<string, string>;
    }
  >;
  model_explained?: Record<string, any>;
};
export type XaiFairness = Record<
  string,
  {
    n_features: number;
    n_identity_features: number;
    identity_features: string[];
    identity_attribution_share: number;
    expected_share_if_uniform: number;
    concentration_ratio: number | null;
    race_state_attribution_share: number;
    top_race_state_features: string[];
    reading: string;
    figure?: string;
  }
>;
export type XaiShap = {
  target: string;
  deep_network: { ranking: { feature: string; mean_abs_shap: number }[]; note: string; explainer: string; exact: boolean };
  classical: { ranking: { feature: string; mean_abs_shap: number }[]; note: string; explainer: string | null; model: string };
  figure?: string;
  available_rows: string[];
};
export type XaiTrust = {
  target: string;
  formula: string;
  weights: Record<string, number>;
  bands: Record<string, string>;
  note?: string;
  all_test_laps?: { n: number; mean: number; bands: Record<string, number> };
  summary: { n: number; mean: number | null; bands: Record<string, number> };
  rows: Record<
    string,
    {
      row_index: number;
      lap: number;
      trust_score: number;
      components: Record<string, number>;
      band: TrustBand;
      narrative: string;
    }
  >;
};


// Quantum ML (PennyLane, simulated)
export type QmlModelRow = {
  model: string;
  family: string;
  description: string;
  n_parameters: number;
  train_seconds: number | null;
  cv_mean: number | null;
  cv_std: number | null;
  test_metrics: Record<string, any>;
};

export type QmlTarget = {
  task: string;
  selection_metric: string;
  encoding: { method: string; n_qubits: number; explained_variance_ratio: number | null; values_clipped_to_range: number; fitted_on: string };
  n_dev: number;
  n_test: number;
  n_test_positive?: number | null;
  n_folds: number;
  notes: string[];
  models: QmlModelRow[];
  task6_reference: { model: string; test_metrics: Record<string, any>; cv_summary: Record<string, any> } | null;
};

export type QmlSummary = {
  generated_at: string;
  simulator: string;
  framework: { pennylane: string };
  seed: number;
  search_space: { n_layers: number[]; learning_rate: number[]; selected_on: string };
  dataset_source: DatasetSource;
  wall_seconds: number;
  figures: Record<string, string>;
  report: string;
  targets: Record<string, QmlTarget>;
  honesty_note: string;
};


// Tasks 1-3 — symbolic reasoning
export type Unavailable = { available: false; reason: string };

export type KnowledgeSummary = {
  available: true;
  n_entities: number;
  n_relationships: number;
  n_attributes: number;
  entities_by_category: Record<string, number>;
  categories: string[];
  relationships: string[];
  files: Record<string, boolean>;
};

export type ExpertSystemSummary = {
  available: true;
  n_rules: number;
  rules_by_category: Record<string, number>;
  salience_levels: number[];
  rules: { rule_id: string; name: string; category: string; salience: number; description: string; n_conditions: number; actions: string[] }[];
};

export type SearchSummary = {
  available: true;
  problem: Record<string, any>;
  algorithms: Record<string, any>[];
  summary: Record<string, any>;
  pit_stops: Record<string, any>[];
  n_plan_steps: number;
  final_cost_seconds: number | null;
};

export const api = {
  health: () => getJson<HealthResponse>("/api/health"),
  models: () => getJson<Registry>("/api/ml/models"),
  comparison: () => getJson<Comparison>("/api/ml/comparison"),
  artifacts: () => getJson<Manifest>("/api/ml/artifacts"),
  metrics: () => getJson<Record<string, any>>("/api/ml/metrics"),
  featureImportance: () => getJson<Record<string, any>>("/api/ml/feature-importance"),
  dataOptions: () => getJson<DataOptions>("/api/data/options"),
  topFeatures: (target: string, n = 8) => getJson<TopFeaturesResponse>(`/api/ml/top-features?target=${target}&n=${n}`),
  taskEvidence: () => getJson<TaskEvidenceResponse>("/api/tasks/evidence"),
  predictLaptime: (features: Record<string, number>) =>
    postJson<{ model: string; prediction: number; unit: string; data_source: string }>(
      "/api/ml/predict/laptime",
      features
    ),
  predictPit: (features: Record<string, number>) =>
    postJson<{ model: string; probability_pit: number; predicted_class: number; data_source: string }>(
      "/api/ml/predict/pit",
      features
    ),
  strategyPredict: (raceState: RaceState) => postJson<StrategyResponse>("/api/strategy/predict", raceState),

  // Task 7 — Deep Learning
  dlModels: () => getJson<{ models: DlModel[] }>("/api/dl/models"),
  dlMetrics: () => getJson<Record<string, any>>("/api/dl/metrics"),
  dlComparison: () => getJson<DlComparison>("/api/dl/comparison"),
  dlHistory: () => getJson<DlHistory>("/api/dl/history"),
  dlArtifacts: () => getJson<DlArtifacts>("/api/dl/artifacts"),
  dlPredictLaptime: (features: Record<string, number>) =>
    postJson<{ model: string; target: string; prediction: number; model_format: string; data_source: string }>(
      "/api/dl/predict/laptime",
      features
    ),

  // Task 8 — Explainable AI
  xaiSummary: () => getJson<XaiSummary>("/api/xai/summary"),
  xaiExplanation: (target: string) => getJson<XaiExplanation>(`/api/xai/explanation?target=${target}`),
  xaiShap: (target: string) => getJson<XaiShap>(`/api/xai/shap?target=${target}`),
  xaiTrust: (target: string) => getJson<XaiTrust>(`/api/xai/trust-score?target=${target}`),
  xaiFairness: () => getJson<XaiFairness>("/api/xai/fairness"),
  xaiFeatureImportance: (target: string) =>
    getJson<Record<string, any>>(`/api/xai/feature-importance?target=${target}`),
  xaiCounterfactual: (target: string) => getJson<Record<string, any>>(`/api/xai/counterfactual?target=${target}`),
  xaiStratification: () => getJson<XaiStratification>("/api/xai/stratification"),
  xaiLaps: (target: string) => getJson<XaiLaps>(`/api/xai/laps?target=${target}`),
  xaiLap: (target: string, rowIndex: number) =>
    getJson<XaiLap>(`/api/xai/lap?target=${target}&row_index=${rowIndex}`),

  // Quantum ML
  qmlSummary: () => getJson<QmlSummary>("/api/qml/summary"),

  // Tasks 1-3 — symbolic reasoning
  knowledge: () => getJson<KnowledgeSummary | Unavailable>("/api/reasoning/knowledge"),
  expertSystem: () => getJson<ExpertSystemSummary | Unavailable>("/api/reasoning/expert-system"),
  searchComparison: () => getJson<SearchSummary | Unavailable>("/api/reasoning/search"),

  // Task 9 — the race-strategy report generator. Returns the rendered text
  // (Markdown or HTML, per `format`), not JSON.
  strategyReport: async (
    raceState: RaceState,
    format: "markdown" | "html" = "markdown",
  ): Promise<{ filename: string; content: string; mimeType: string }> => {
    const res = await fetch(`${API_BASE}/api/strategy/report?format=${format}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(raceState),
      cache: "no-store",
    });
    if (!res.ok) throw new ApiError(res.status, `Report generation failed (${res.status})`);
    const disposition = res.headers.get("content-disposition") ?? "";
    const match = disposition.match(/filename="?([^"]+)"?/);
    const fallback = format === "html" ? "strategy_report.html" : "strategy_report.md";
    return {
      filename: match?.[1] ?? fallback,
      content: await res.text(),
      mimeType: format === "html" ? "text/html" : "text/markdown",
    };
  },
};
