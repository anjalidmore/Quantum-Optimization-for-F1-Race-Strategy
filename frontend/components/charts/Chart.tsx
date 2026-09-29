"use client";

import dynamic from "next/dynamic";
import { ChartFrame } from "./ChartFrame";
import { ConfusionMatrix, CircuitText } from "./Svg";
import { useChart } from "./useChart";

/* Recharts is loaded in the browser only. It is the single heaviest dependency
   in the app and no page needs it to render its text, so it must not sit in
   the server bundle or block hydration. */
const Plots = {
  BarPlot: dynamic(() => import("./Plot").then((m) => m.BarPlot), { ssr: false }),
  LinePlot: dynamic(() => import("./Plot").then((m) => m.LinePlot), { ssr: false }),
  ScatterPlot: dynamic(() => import("./Plot").then((m) => m.ScatterPlot), { ssr: false }),
  HistogramPlot: dynamic(() => import("./Plot").then((m) => m.HistogramPlot), { ssr: false }),
};

/**
 * One chart, by name. Fetches its own numbers, picks a renderer from the
 * `kind` the pipeline recorded, and hands the rest to ChartFrame.
 */
export function Chart({ name, height }: { name: string; height?: number }) {
  const { data, loading, error } = useChart(name);

  return (
    <ChartFrame data={data} loading={loading} error={error} name={name} height={height}>
      {data && renderer(data)}
    </ChartFrame>
  );
}

function renderer(data: NonNullable<ReturnType<typeof useChart>["data"]>) {
  switch (data.kind) {
    case "bar":
      return <Plots.BarPlot data={data} />;
    case "bar-horizontal":
      return <Plots.BarPlot data={data} horizontal />;
    case "line":
      return <Plots.LinePlot data={data} />;
    case "scatter":
      return <Plots.ScatterPlot data={data} />;
    case "histogram":
      return <Plots.HistogramPlot data={data} />;
    case "confusion":
      return <ConfusionMatrix data={data} />;
    case "circuit":
      return <CircuitText data={data} />;
    default:
      return <p className="t-body text-[13px] text-paper-500">No renderer for “{data.kind}”.</p>;
  }
}
