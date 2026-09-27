/**
 * Small formatting helpers, shared by every page.
 *
 * These were copy-pasted into four pages before the 2026-09-27 cleanup, which
 * meant "—" for a missing number was spelled three different ways.
 */

/** A number for display, or an em dash when the value is missing or undefined. */
export function fmt(x: number | null | undefined, digits = 3): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return x.toFixed(digits);
}

/** A 0-1 share as a percentage. */
export function pct(x: number | null | undefined, digits = 1): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return `${(x * 100).toFixed(digits)}%`;
}

/** Last path segment of an artifact path. */
export function basename(path: string): string {
  return path.split("/").pop() ?? path;
}

/** "regression_model_comparison.png" -> "Regression Model Comparison" */
export function prettify(filename: string): string {
  return filename
    .replace(/\.[a-z0-9]+$/i, "")
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/** The two prediction targets, as a reader should see them. */
export const TARGET_LABEL: Record<string, string> = {
  target_laptime: "Lap-time regression",
  target_pit_next_lap: "Pit-decision classification",
};
