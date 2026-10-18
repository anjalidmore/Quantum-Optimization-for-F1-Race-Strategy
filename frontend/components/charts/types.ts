/** The shape every chart JSON uses. Written by app/intelligence/charts.py. */
export type Axis = { label: string; unit: string | null; kind: "number" | "category" };
export type Point = { x: number | string; y: number | null };
export type Series = { name: string; points: Point[]; role?: string };

export type ChartData = {
  name: string;
  kind: "bar" | "bar-horizontal" | "line" | "scatter" | "histogram" | "confusion" | "circuit";
  title: string;
  caption: string | null;
  axes: { x: Axis; y: Axis };
  series: Series[];
  source: string | null;
  data_source: string;
  /* kind-specific extras */
  identity_line?: [number, number];
  marker_x?: number | null;
  marker_y?: number | null;
  labels?: string[];
  matrix?: number[][];
  threshold?: number;
  text?: string;
  n_qubits?: number;
  n_layers?: number;
  error?: number[];
  families?: string[];
  highlight?: string[];
  lower_is_better?: boolean;
  log_x?: boolean;
  log_y?: boolean;
};

/**
 * Series colours are CSS variables, not hex.
 *
 * SVG presentation attributes resolve var() the same way CSS does, so a theme
 * change repaints every chart with no JavaScript, no re-fetch and no re-mount.
 */
export const SERIES_COLORS = [
  "var(--chart-1)", "var(--chart-2)", "var(--chart-3)",
  "var(--chart-4)", "var(--chart-5)", "var(--chart-6)",
];

const COMPOUND_ROLE: Record<string, string> = {
  SOFT: "var(--compound-soft)", MEDIUM: "var(--compound-medium)", HARD: "var(--compound-hard)",
  INTERMEDIATE: "var(--compound-inter)", WET: "var(--compound-wet)",
};

/** A tyre series keeps its compound's colour; everything else takes the ramp. */
export function seriesColor(s: Series, i: number): string {
  const role = s.role?.toUpperCase();
  return (role && COMPOUND_ROLE[role]) || SERIES_COLORS[i % SERIES_COLORS.length];
}
