"use client";

import type { ChartData } from "./types";

/**
 * The charts a library would make harder, not easier.
 *
 * A confusion matrix is four numbers in a grid; a circuit is text. Reaching for
 * a charting library for either would add indirection and no clarity.
 */

/** Four counts, tinted by share of the row. Colour is never the only signal. */
export function ConfusionMatrix({ data }: { data: ChartData }) {
  const m = data.matrix ?? [];
  const labels = data.labels ?? ["0", "1"];
  const max = Math.max(...m.flat(), 1);
  return (
    <div className="flex h-full items-center justify-center">
      <table className="border-collapse">
        <caption className="sr-only">{data.title}</caption>
        <thead>
          <tr>
            <th className="p-1.5 text-[11px] font-medium text-paper-500" />
            {labels.map((l) => (
              <th key={l} scope="col" className="p-1.5 text-[11px] font-medium text-paper-500">
                {l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {m.map((row, i) => (
            <tr key={i}>
              <th scope="row" className="p-1.5 text-right text-[11px] font-medium text-paper-500">
                {labels[i]}
              </th>
              {row.map((v, j) => (
                <td key={j} className="p-0.5">
                  <div
                    className="grid h-[68px] w-[92px] place-items-center rounded-sm border border-track-300 tabular-nums"
                    style={{
                      // A tint of the accent, so the strong cells read first.
                      backgroundColor: `color-mix(in srgb, var(--accent) ${(v / max) * 26}%, var(--track-200))`,
                    }}
                  >
                    <span className="font-display text-[20px] font-semibold text-paper-900">{v}</span>
                    <span className="t-micro">{i === j ? "correct" : "error"}</span>
                  </div>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * The circuit, as PennyLane drew it.
 *
 * This is the one element allowed to scroll sideways, inside its own box —
 * a circuit is wide by nature and wrapping it would destroy the wires.
 */
export function CircuitText({ data }: { data: ChartData }) {
  return (
    <div className="h-full overflow-auto rounded-sm border border-track-300 bg-track-050 p-3">
      <pre className="t-code whitespace-pre text-[11px] leading-[1.7] text-paper-700">{data.text}</pre>
    </div>
  );
}
