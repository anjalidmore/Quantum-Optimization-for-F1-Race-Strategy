"use client";

import {
  Bar, BarChart, CartesianGrid, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { AXIS, ChartTooltip, GRID, LEGEND } from "@/components/charts/Plot";
import { MODEL_KEYS, MODEL_ROLE, type ModelKey } from "@/lib/modelComparison";
import type { RaiHistBin } from "@/lib/api";

/* Browser-only, like every Recharts chart in the app. Animation is off to
   match components/charts/Plot.tsx. */

/** One bar per model per group; null values are left out, never drawn as 0. */
export function GroupBars({
  rows, unit, overall,
}: {
  rows: { group: string; values: Partial<Record<ModelKey, number | null>> }[];
  unit: string;
  overall?: Partial<Record<ModelKey, number | null>>;
}) {
  const data = rows.map((r) => {
    const row: Record<string, string | number | null> = { x: r.group };
    for (const k of MODEL_KEYS) row[MODEL_ROLE[k].short] = r.values[k] ?? null;
    return row;
  });
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ top: 10, right: 8, bottom: 4, left: 4 }} barGap={2} barCategoryGap="20%">
        <CartesianGrid {...GRID} vertical={false} />
        <XAxis type="category" dataKey="x" {...AXIS} interval={0} angle={rows.length > 6 ? -30 : 0}
               textAnchor={rows.length > 6 ? "end" : "middle"} height={rows.length > 6 ? 54 : 30} />
        <YAxis type="number" {...AXIS} width={46}
               label={{ value: unit, angle: -90, position: "insideLeft", ...AXIS.label }} />
        <Tooltip content={<ChartTooltip />} cursor={{ fill: "var(--track-200)" }} />
        <Legend {...LEGEND} itemSorter={null} />
        {MODEL_KEYS.map((k) => (
          <Bar key={k} dataKey={MODEL_ROLE[k].short} fill={MODEL_ROLE[k].color} radius={[3, 3, 0, 0]}
               maxBarSize={28} isAnimationActive={false} />
        ))}
        {overall && MODEL_KEYS.map((k) =>
          overall[k] != null ? (
            <ReferenceLine key={`o-${k}`} y={overall[k]!} stroke={MODEL_ROLE[k].color} strokeDasharray="4 4" strokeOpacity={0.8} />
          ) : null,
        )}
      </BarChart>
    </ResponsiveContainer>
  );
}

/** A model's score histogram, split by whether the call at its own threshold was right. */
export function ScoreHistogram({ bins, threshold, model }: { bins: RaiHistBin[]; threshold: number | null; model: ModelKey }) {
  const data = bins.map((b) => ({
    x: `${b.lo.toFixed(1)}–${b.hi.toFixed(1)}`, mid: (b.lo + b.hi) / 2, Correct: b.correct ?? 0, Wrong: b.incorrect ?? 0,
  }));
  const tIdx = threshold === null ? null : Math.min(data.length - 1, Math.floor(threshold * data.length));
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ top: 14, right: 6, bottom: 4, left: 0 }} barCategoryGap={1}>
        <CartesianGrid {...GRID} vertical={false} />
        <XAxis dataKey="x" {...AXIS} interval={1} tick={{ ...AXIS.tick, fontSize: 10 }} />
        <YAxis {...AXIS} width={34} allowDecimals={false} />
        <Tooltip content={<ChartTooltip />} cursor={{ fill: "var(--track-200)" }} />
        <Legend {...LEGEND} />
        <Bar dataKey="Correct" stackId="s" fill={MODEL_ROLE[model].color} isAnimationActive={false} />
        <Bar dataKey="Wrong" stackId="s" fill="var(--paper-400)" isAnimationActive={false} radius={[3, 3, 0, 0]} />
        {tIdx !== null && (
          <ReferenceLine x={data[tIdx].x} stroke="var(--paper-900)" strokeDasharray="3 3"
                         label={{ value: `threshold ${threshold!.toFixed(2)}`, position: "top", fill: "var(--paper-700)", fontSize: 10 }} />
        )}
      </BarChart>
    </ResponsiveContainer>
  );
}

/** A plain count histogram in one model's colour (errors, uncertainty). */
export function CountHistogram({ bins, model, unit }: { bins: RaiHistBin[]; model: ModelKey; unit: string }) {
  const data = bins.map((b) => ({ x: `${b.lo.toFixed(2)}`, Laps: b.count }));
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ top: 6, right: 6, bottom: 4, left: 0 }} barCategoryGap={1}>
        <CartesianGrid {...GRID} vertical={false} />
        <XAxis dataKey="x" {...AXIS} interval={1} tick={{ ...AXIS.tick, fontSize: 10 }}
               label={{ value: unit, position: "insideBottomRight", offset: -2, ...AXIS.label }} />
        <YAxis {...AXIS} width={34} allowDecimals={false} />
        <Tooltip content={<ChartTooltip />} cursor={{ fill: "var(--track-200)" }} />
        <Bar dataKey="Laps" fill={MODEL_ROLE[model].color} isAnimationActive={false} radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Mean actual error for laps grouped by how uncertain the model said it was. */
export function TercileBars({ rows, model }: { rows: { label: string; mae: number; mean_std: number }[]; model: ModelKey }) {
  const data = rows.map((r) => ({ x: r.label, "Actual MAE": r.mae, "Predicted spread": r.mean_std }));
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ top: 10, right: 8, bottom: 4, left: 4 }} barGap={2}>
        <CartesianGrid {...GRID} vertical={false} />
        <XAxis dataKey="x" {...AXIS} />
        <YAxis {...AXIS} width={40} label={{ value: "seconds", angle: -90, position: "insideLeft", ...AXIS.label }} />
        <Tooltip content={<ChartTooltip />} cursor={{ fill: "var(--track-200)" }} />
        <Legend {...LEGEND} itemSorter={null} />
        <Bar dataKey="Predicted spread" fill="var(--paper-400)" isAnimationActive={false} radius={[3, 3, 0, 0]} maxBarSize={36} />
        <Bar dataKey="Actual MAE" fill={MODEL_ROLE[model].color} isAnimationActive={false} radius={[3, 3, 0, 0]} maxBarSize={36} />
      </BarChart>
    </ResponsiveContainer>
  );
}
