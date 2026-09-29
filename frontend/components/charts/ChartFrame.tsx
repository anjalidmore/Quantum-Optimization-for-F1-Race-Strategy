"use client";

import { useState } from "react";
import type { ChartData } from "./types";

/**
 * Everything around a chart: title, caption, the data behind it, and the two
 * states that are not a chart at all.
 *
 * The "View data" table is the accessibility contract. A chart is a picture of
 * numbers, and a picture is not readable by everyone, so the numbers stay one
 * click away on every single chart, in source order.
 */
export function ChartFrame({
  data,
  loading,
  error,
  name,
  height = 260,
  children,
}: {
  data: ChartData | null;
  loading: boolean;
  error: string | null;
  name: string;
  height?: number;
  children: React.ReactNode;
}) {
  const [showData, setShowData] = useState(false);

  if (loading) {
    return (
      <figure className="card" aria-busy="true">
        <div className="skeleton" style={{ height }} />
        <span className="sr-only">Loading {name}</span>
      </figure>
    );
  }

  if (error || !data) {
    return (
      <figure className="card">
        <figcaption className="stat-label">{name.replace(/_/g, " ")}</figcaption>
        <p className="t-body mt-2 text-[13px] text-paper-500">{error ?? "Not generated yet"}</p>
        <p className="t-micro mt-1">
          Run <span className="text-paper-500">python scripts/build_all.py</span> to produce it.
        </p>
      </figure>
    );
  }

  return (
    <figure className="card">
      <figcaption>
        <h3 className="t-title text-[15px]">{data.title}</h3>
        {data.caption && <p className="t-micro mt-1 max-w-measure">{data.caption}</p>}
      </figcaption>

      <div className="mt-4" style={{ height }}>
        {children}
      </div>

      <div className="mt-3 flex flex-wrap items-center justify-between gap-2 border-t border-track-300 pt-2.5">
        <button
          type="button"
          onClick={() => setShowData((v) => !v)}
          aria-expanded={showData}
          className="text-[12px] font-medium text-paper-700 underline decoration-edge underline-offset-[3px] hover:text-paper-900 hover:decoration-paper-900"
        >
          {showData ? "Hide data" : "View data"}
        </button>
        {data.source && <span className="t-micro">{data.source}</span>}
      </div>

      {showData && <DataTable data={data} />}
    </figure>
  );
}

/** The chart's numbers as a table, in the order the series were written. */
function DataTable({ data }: { data: ChartData }) {
  if (data.matrix && data.labels) {
    return (
      <table className="t-table mt-3">
        <caption className="sr-only">{data.title}, counts</caption>
        <thead>
          <tr>
            <th scope="col">Actual \ predicted</th>
            {data.labels.map((l) => (
              <th key={l} scope="col" className="num">{l}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.matrix.map((row, i) => (
            <tr key={i}>
              <th scope="row" className="text-left font-normal">{data.labels![i]}</th>
              {row.map((v, j) => (
                <td key={j} className="num">{v}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    );
  }

  return (
    <div className="mt-3 max-h-[320px] overflow-y-auto">
      <table className="t-table">
        <caption className="sr-only">{data.title}, underlying values</caption>
        <thead>
          <tr>
            <th scope="col">Series</th>
            <th scope="col" className="num">{data.axes.x.label || "x"}</th>
            <th scope="col" className="num">{data.axes.y.label || "y"}</th>
          </tr>
        </thead>
        <tbody>
          {data.series.flatMap((s) =>
            s.points.map((p, i) => (
              <tr key={`${s.name}-${i}`}>
                <td>{s.name}</td>
                <td className="num">{typeof p.x === "number" ? p.x.toPrecision(5) : p.x}</td>
                <td className="num">{p.y === null ? "—" : p.y.toPrecision(5)}</td>
              </tr>
            )),
          )}
        </tbody>
      </table>
    </div>
  );
}
