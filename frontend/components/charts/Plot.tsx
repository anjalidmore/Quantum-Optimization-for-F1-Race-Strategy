"use client";

import {
  Bar, BarChart, CartesianGrid, ErrorBar, Legend, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from "recharts";
import type { ChartData, Point } from "./types";
import { seriesColor } from "./types";

/* One place for the look of every axis, grid and tooltip, so a chart cannot
   drift from the design system by being written later than the others.
   Colours are var() strings: the theme repaints them with no JS. */
const AXIS = {
  stroke: "var(--edge)",
  tick: { fill: "var(--paper-500)", fontSize: 11, fontVariantNumeric: "tabular-nums" as const },
  label: { fill: "var(--paper-500)", fontSize: 11 },
};
const GRID = { stroke: "var(--grid)", strokeDasharray: "2 4" } as const;
const MARGIN = { top: 6, right: 12, bottom: 22, left: 4 };

function axisLabel(text: string, unit: string | null) {
  return unit ? `${text} (${unit})` : text;
}

/** Recharts' default tooltip is a white box. This one is a card. */
function ChartTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-sm border border-edge bg-track-100 px-2.5 py-2 text-[12px] shadow-none">
      {label !== undefined && <p className="font-medium text-paper-900">{String(label)}</p>}
      {payload.map((p: any) => (
        <p key={p.name} className="tabular-nums text-paper-700">
          <span style={{ color: p.color }}>■</span> {p.name}:{" "}
          {typeof p.value === "number" ? p.value.toPrecision(5) : String(p.value)}
        </p>
      ))}
    </div>
  );
}

const LEGEND = { wrapperStyle: { fontSize: 11, color: "var(--paper-500)" } };

/** Series shaped as rows keyed by x, which is what Recharts wants.
    A per-point error (the QML fold spread) rides along on its own row key. */
function rows(data: ChartData) {
  const byX = new Map<string | number, Record<string, number | string | null>>();
  for (const s of data.series) {
    s.points.forEach((p, i) => {
      const row = byX.get(p.x) ?? { x: p.x };
      row[s.name] = p.y;
      if (data.error && data.series.length === 1) row._error = data.error[i] ?? 0;
      byX.set(p.x, row);
    });
  }
  return [...byX.values()];
}

export function BarPlot({ data, horizontal = false }: { data: ChartData; horizontal?: boolean }) {
  const d = rows(data);
  const errors = data.error;
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={d} layout={horizontal ? "vertical" : "horizontal"} margin={MARGIN}>
        <CartesianGrid {...GRID} vertical={horizontal} horizontal={!horizontal} />
        {horizontal ? (
          <>
            <XAxis type="number" {...AXIS} label={{ value: axisLabel(data.axes.x.label, data.axes.x.unit), position: "insideBottom", offset: -14, ...AXIS.label }} />
            <YAxis type="category" dataKey="x" width={104} {...AXIS} />
          </>
        ) : (
          <>
            <XAxis type="category" dataKey="x" {...AXIS} interval={0} angle={-18} textAnchor="end" height={54} />
            <YAxis type="number" {...AXIS} label={{ value: axisLabel(data.axes.y.label, data.axes.y.unit), angle: -90, position: "insideLeft", ...AXIS.label }} />
          </>
        )}
        <Tooltip content={<ChartTooltip />} cursor={{ fill: "var(--track-200)" }} />
        {data.series.length > 1 && <Legend {...LEGEND} />}
        {data.series.map((s, i) => (
          <Bar key={s.name} dataKey={s.name} fill={seriesColor(s, i)} maxBarSize={horizontal ? 18 : 34}>
            {errors && data.series.length === 1 && (
              <ErrorBar dataKey="_error" width={4} strokeWidth={1} stroke="var(--paper-500)"
                        direction={horizontal ? "x" : "y"} />
            )}
          </Bar>
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

export function LinePlot({ data }: { data: ChartData }) {
  const d = rows(data).sort((a, b) => Number(a.x) - Number(b.x));
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={d} margin={MARGIN}>
        <CartesianGrid {...GRID} />
        <XAxis dataKey="x" type="number" {...AXIS}
               label={{ value: axisLabel(data.axes.x.label, data.axes.x.unit), position: "insideBottom", offset: -14, ...AXIS.label }} />
        <YAxis {...AXIS} scale={data.log_y ? "log" : "auto"} domain={data.log_y ? ["auto", "auto"] : undefined}
               label={{ value: axisLabel(data.axes.y.label, data.axes.y.unit), angle: -90, position: "insideLeft", ...AXIS.label }} />
        <Tooltip content={<ChartTooltip />} />
        {data.series.length > 1 && <Legend {...LEGEND} />}
        {data.identity_line && (
          <ReferenceLine segment={[{ x: data.identity_line[0], y: data.identity_line[0] },
                                   { x: data.identity_line[1], y: data.identity_line[1] }]}
                         stroke="var(--paper-400)" strokeDasharray="3 3" />
        )}
        {data.marker_x != null && <ReferenceLine x={data.marker_x} stroke="var(--accent)" strokeDasharray="3 3" />}
        {data.series.map((s, i) => (
          <Line key={s.name} type="monotone" dataKey={s.name} stroke={seriesColor(s, i)}
                dot={false} strokeWidth={1.8} isAnimationActive={false} connectNulls />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function ScatterPlot({ data }: { data: ChartData }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <ScatterChart margin={MARGIN}>
        <CartesianGrid {...GRID} />
        <XAxis type="number" dataKey="x" {...AXIS} domain={["auto", "auto"]}
               label={{ value: axisLabel(data.axes.x.label, data.axes.x.unit), position: "insideBottom", offset: -14, ...AXIS.label }} />
        <YAxis type="number" dataKey="y" {...AXIS} domain={["auto", "auto"]}
               label={{ value: axisLabel(data.axes.y.label, data.axes.y.unit), angle: -90, position: "insideLeft", ...AXIS.label }} />
        <Tooltip content={<ChartTooltip />} cursor={{ strokeDasharray: "2 4", stroke: "var(--edge)" }} />
        {data.series.length > 1 && <Legend {...LEGEND} />}
        {data.identity_line && (
          <ReferenceLine segment={[{ x: data.identity_line[0], y: data.identity_line[0] },
                                   { x: data.identity_line[1], y: data.identity_line[1] }]}
                         stroke="var(--paper-400)" strokeDasharray="3 3" />
        )}
        {data.marker_y != null && <ReferenceLine y={data.marker_y} stroke="var(--accent)" strokeDasharray="3 3" />}
        {data.series.map((s, i) => (
          <Scatter key={s.name} name={s.name} data={s.points as Point[]} fill={seriesColor(s, i)}
                   fillOpacity={0.55} isAnimationActive={false} />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  );
}

/** Pre-binned counts from the pipeline, drawn as touching bars. */
export function HistogramPlot({ data }: { data: ChartData }) {
  const d = rows(data);
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={d} margin={MARGIN} barGap={0} barCategoryGap={1}>
        <CartesianGrid {...GRID} vertical={false} />
        <XAxis dataKey="x" type="number" {...AXIS} domain={["auto", "auto"]}
               tickFormatter={(v) => (typeof v === "number" ? v.toPrecision(3) : v)}
               label={{ value: axisLabel(data.axes.x.label, data.axes.x.unit), position: "insideBottom", offset: -14, ...AXIS.label }} />
        <YAxis {...AXIS} label={{ value: data.axes.y.label, angle: -90, position: "insideLeft", ...AXIS.label }} />
        <Tooltip content={<ChartTooltip />} cursor={{ fill: "var(--track-200)" }} />
        {data.series.length > 1 && <Legend {...LEGEND} />}
        {data.marker_x != null && <ReferenceLine x={data.marker_x} stroke="var(--accent)" strokeDasharray="3 3" />}
        {data.series.map((s, i) => (
          <Bar key={s.name} dataKey={s.name} fill={seriesColor(s, i)} fillOpacity={0.75} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}
