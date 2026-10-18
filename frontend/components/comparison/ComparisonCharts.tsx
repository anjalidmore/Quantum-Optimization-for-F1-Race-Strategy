"use client";

import {
  Bar, BarChart, CartesianGrid, LabelList, Legend, PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar,
  RadarChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { AXIS, ChartTooltip, GRID, LEGEND } from "@/components/charts/Plot";
import { MODEL_KEYS, MODEL_ROLE, radarScore, type TargetComparison } from "@/lib/modelComparison";

/**
 * The two comparison charts. Loaded browser-only by the page, the same way
 * components/charts/Chart.tsx loads Recharts, and styled with the same axis,
 * grid, tooltip and legend constants so they read as part of one system.
 */

/** Metrics that share one axis: seconds for regression, 0–1 for classification. */
function barMetrics(c: TargetComparison) {
  return c.task === "regression" ? c.metrics.filter((m) => m.unit === "s") : c.metrics;
}

export function GroupedBars({ c }: { c: TargetComparison }) {
  const metrics = barMetrics(c);
  const rows = metrics.map((m) => {
    const row: Record<string, string | number | null> = { x: m.label };
    for (const k of MODEL_KEYS) row[MODEL_ROLE[k].short] = c.models[k].metrics[m.key];
    return row;
  });
  const showLabels = metrics.length <= 3;
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={rows} margin={{ top: 22, right: 8, bottom: 4, left: 4 }} barGap={2} barCategoryGap="22%">
        <CartesianGrid {...GRID} vertical={false} />
        <XAxis type="category" dataKey="x" {...AXIS} interval={0} />
        <YAxis
          type="number"
          {...AXIS}
          domain={c.task === "classification" ? [0, 1] : [0, "auto"]}
          width={44}
          label={{ value: c.task === "regression" ? "seconds" : "score (0–1)", angle: -90, position: "insideLeft", ...AXIS.label }}
        />
        <Tooltip content={<ChartTooltip />} cursor={{ fill: "var(--track-200)" }} />
        <Legend {...LEGEND} itemSorter={null} />
        {MODEL_KEYS.map((k) => (
          <Bar
            key={k}
            dataKey={MODEL_ROLE[k].short}
            fill={MODEL_ROLE[k].color}
            radius={[4, 4, 0, 0]}
            maxBarSize={44}
            isAnimationActive={false}
          >
            {showLabels && (
              <LabelList
                dataKey={MODEL_ROLE[k].short}
                position="top"
                formatter={(v: any) => (typeof v === "number" ? v.toFixed(3) : "")}
                style={{ fill: "var(--paper-700)", fontSize: 11, fontVariantNumeric: "tabular-nums" }}
              />
            )}
          </Bar>
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

/** The radar's tooltip shows the real value, not the normalised radius. */
function RadarTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <div className="rounded-sm border border-edge bg-track-100 px-2.5 py-2 text-[12px]">
      <p className="font-medium text-paper-900">{label ?? row.x}</p>
      {payload.map((p: any) => (
        <p key={p.name} className="tabular-nums text-paper-700">
          <span style={{ color: p.color }}>■</span> {p.name}: {row[`${p.name}__raw`] ?? "N/A"}
        </p>
      ))}
    </div>
  );
}

export function MetricRadar({ c }: { c: TargetComparison }) {
  const rows = c.metrics.map((m) => {
    const all = MODEL_KEYS.map((k) => c.models[k].metrics[m.key]);
    const row: Record<string, string | number | null> = { x: m.label };
    for (const k of MODEL_KEYS) {
      const v = c.models[k].metrics[m.key];
      row[MODEL_ROLE[k].short] = radarScore(m, v, all);
      row[`${MODEL_ROLE[k].short}__raw`] = v === null ? null : `${v.toFixed(m.digits)}${m.unit === "s" ? " s" : m.unit === "%" ? "%" : ""}`;
    }
    return row;
  });
  return (
    <ResponsiveContainer width="100%" height="100%">
      <RadarChart data={rows} outerRadius="72%" margin={{ top: 8, right: 24, bottom: 8, left: 24 }}>
        <PolarGrid stroke="var(--grid)" />
        <PolarAngleAxis dataKey="x" tick={{ fill: "var(--paper-500)", fontSize: 11 }} />
        <PolarRadiusAxis domain={[0, 1]} tick={false} axisLine={false} />
        <Tooltip content={<RadarTooltip />} />
        <Legend {...LEGEND} itemSorter={null} />
        {MODEL_KEYS.map((k) => (
          <Radar
            key={k}
            name={MODEL_ROLE[k].short}
            dataKey={MODEL_ROLE[k].short}
            stroke={MODEL_ROLE[k].color}
            fill={MODEL_ROLE[k].color}
            fillOpacity={0.1}
            strokeWidth={2}
            dot={{ r: 3, fill: MODEL_ROLE[k].color, strokeWidth: 0 }}
            isAnimationActive={false}
          />
        ))}
      </RadarChart>
    </ResponsiveContainer>
  );
}
