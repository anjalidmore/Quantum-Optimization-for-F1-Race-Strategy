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
};

export type StrategyResponse = {
  race_state: RaceState;
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
  data_source: string;
};

export const api = {
  health: () => getJson<HealthResponse>("/api/health"),
  models: () => getJson<Registry>("/api/ml/models"),
  comparison: () => getJson<Comparison>("/api/ml/comparison"),
  artifacts: () => getJson<Manifest>("/api/ml/artifacts"),
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
};
