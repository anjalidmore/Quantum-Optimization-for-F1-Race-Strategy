"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { ArchitectureFlow } from "./ArchitectureFlow";
import {
  MODEL_KEYS, MODEL_ROLE, TARGETS, radarScore,
  type Insight, type MetricDef, type ModelEntry, type ModelKey, type Target, type TargetComparison,
} from "@/lib/modelComparison";

/* Recharts stays out of the server bundle, as on every other page. */
const GroupedBars = dynamic(() => import("./ComparisonCharts").then((m) => m.GroupedBars), {
  ssr: false,
  loading: () => <div className="skeleton h-full" />,
});
const MetricRadar = dynamic(() => import("./ComparisonCharts").then((m) => m.MetricRadar), {
  ssr: false,
  loading: () => <div className="skeleton h-full" />,
});

const fmtMetric = (m: MetricDef, v: number | null) =>
  v === null ? "N/A" : `${v.toFixed(m.digits)}${m.unit === "%" ? "%" : ""}`;

const fmtSeconds = (s: number | null) =>
  s === null ? "N/A" : s < 1 ? `${(s * 1000).toFixed(1)} ms` : `${s.toFixed(2)} s`;

export function ModelComparison({ data }: { data: Partial<Record<Target, TargetComparison>> }) {
  const available = TARGETS.filter((t) => data[t.key]);
  const [target, setTarget] = useState<Target>(available[0]?.key ?? "target_laptime");
  const c = data[target];

  return (
    <div className="space-y-12">
      {/* ---- target switch ------------------------------------------------ */}
      <div className="flex flex-wrap items-end justify-between gap-4 border-b border-track-300 pb-5">
        <div>
          <p className="stat-label">Prediction task</p>
          <div role="tablist" aria-label="Prediction task" className="mt-2 inline-flex gap-1 rounded-md border border-track-300 bg-track-100 p-1">
            {/* Only tasks with results get a tab; nothing is shown greyed out. */}
            {available.map((t) => (
              <button
                key={t.key}
                type="button"
                role="tab"
                aria-selected={target === t.key}
                aria-controls="mc-panel"
                onClick={() => setTarget(t.key)}
                className="mc-tab"
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>
        {c && (
          <p className="t-body max-w-[52ch] text-[13px] text-paper-500">
            <span className="text-paper-900">{TARGETS.find((t) => t.key === target)?.question}</span>{" "}
            All three models were scored on the same{" "}
            <span className="tabular-nums text-paper-900">{c.nTest ?? "N/A"}</span> held-out laps: the closing laps
            of the race, never seen during training.
          </p>
        )}
      </div>

      {!c ? (
        <div className="card border-accent">
          <span className="badge badge-warning">No comparison available</span>
          <p className="t-body mt-2 text-[13px]">
            Run <code className="text-paper-900">python scripts/build_all.py</code> to train the models and write their
            metrics.
          </p>
        </div>
      ) : (
        <div id="mc-panel" role="tabpanel" className="space-y-12">
          {/* ---- A. model cards ------------------------------------------- */}
          <section aria-labelledby="mc-cards-h">
            <h2 id="mc-cards-h" className="sr-only">The three models</h2>
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              {MODEL_KEYS.map((k) => (
                <ModelCard key={k} c={c} m={c.models[k]} />
              ))}
            </div>
          </section>

          {/* ---- B. charts ------------------------------------------------ */}
          <section aria-labelledby="mc-charts-h">
            <SectionHead id="mc-charts-h" kicker="Performance" title="Head to head" />
            <div className="mt-5 grid gap-4 lg:grid-cols-[1.35fr_1fr] [&>*]:min-w-0">
              <figure className="card">
                <figcaption>
                  <h3 className="t-title text-[15px]">
                    {c.task === "regression" ? "Prediction error, seconds per lap" : "Classification scores"}
                  </h3>
                  <p className="t-micro mt-1 max-w-measure">
                    {c.task === "regression"
                      ? "Lower is better. R² and MAPE use different units, so they appear in the radar and table instead of on this axis."
                      : "Higher is better on every metric. All six share the same 0–1 scale."}
                  </p>
                </figcaption>
                <div className="mt-4 h-[300px]">
                  <GroupedBars c={c} />
                </div>
              </figure>
              <figure className="card">
                <figcaption>
                  <h3 className="t-title text-[15px]">All metrics at once</h3>
                  <p className="t-micro mt-1 max-w-measure">
                    Further out is better. Error metrics are scaled so the best of the three touches the edge; the
                    others are plotted on their own 0–1 scale (negative R² sits at the centre). Hover for real values.
                  </p>
                </figcaption>
                <div className="mt-4 h-[300px]">
                  <MetricRadar c={c} />
                </div>
              </figure>
            </div>
          </section>

          {/* ---- C. leaderboard + insights -------------------------------- */}
          <div className="grid gap-x-10 gap-y-12 lg:grid-cols-[1.2fr_1fr] [&>*]:min-w-0">
            <section aria-labelledby="mc-board-h">
              <SectionHead id="mc-board-h" kicker="Leaderboard" title="Grid positions per metric" />
              <Leaderboard c={c} />
            </section>
            <section aria-labelledby="mc-insight-h">
              <SectionHead id="mc-insight-h" kicker="Generated from the results" title="Key insights" />
              <ul className="mt-5 space-y-3">
                {c.insights.map((i) => (
                  <InsightCard key={i.title} insight={i} />
                ))}
              </ul>
            </section>
          </div>

          {/* ---- D. architecture ----------------------------------------- */}
          <section aria-labelledby="mc-arch-h">
            <SectionHead id="mc-arch-h" kicker="Under the hood" title="How each model makes its call" />
            <p className="t-body mt-2 max-w-measure text-[13px] text-paper-500">
              Same inputs, three very different machines. Each block is read from the trained model&rsquo;s saved
              configuration: Task 6&rsquo;s registry, and the layer list Keras wrote for Task 7.
            </p>
            <div className="mt-5">
              <ArchitectureFlow c={c} />
            </div>
          </section>

          {/* ---- E. exact numbers ----------------------------------------- */}
          <section aria-labelledby="mc-table-h">
            <SectionHead id="mc-table-h" kicker="Exact values" title="Comparison table" />
            <ExactTable c={c} />
          </section>

          {c.verdict && (
            <p className="t-micro max-w-measure border-l-2 border-[var(--papaya-fill)] pl-3">
              Pipeline verdict (artifacts/deep_learning/model_comparison.json): {c.verdict}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function SectionHead({ id, kicker, title }: { id: string; kicker: string; title: string }) {
  return (
    <div>
      <p className="font-display text-[12px] font-bold uppercase tracking-[0.12em] text-[var(--papaya)]">{kicker}</p>
      <h2 id={id} className="mt-0.5 font-display text-[26px] font-bold leading-tight text-paper-900">
        {title}
      </h2>
    </div>
  );
}

/* ------------------------------------------------------------------------ */

export function ModelIcon({ k, model }: { k: ModelKey; model: string | null }) {
  const common = { width: 20, height: 20, viewBox: "0 0 20 20", fill: "none", stroke: "currentColor", strokeWidth: 1.6, strokeLinecap: "round" as const, "aria-hidden": true };
  if (k === "dnn") {
    return (
      <svg {...common}>
        {[4, 10, 16].map((y) => <circle key={`a${y}`} cx="3" cy={y} r="1.5" />)}
        {[7, 13].map((y) => <circle key={`b${y}`} cx="10" cy={y} r="1.5" />)}
        <circle cx="17" cy="10" r="1.5" />
        <path d="M4.5 4 8.5 7M4.5 10l4-3M4.5 10l4 3M4.5 16l4-3M11.5 7l4 3M11.5 13l4-3" strokeWidth="1" />
      </svg>
    );
  }
  if (model === "random_forest" || model === "decision_tree" || model === "xgboost") {
    return (
      <svg {...common}>
        <path d="M10 3v4M10 7 5 11M10 7l5 4M5 11l-2 5M5 11l2 5M15 11l-2 5M15 11l2 5" />
      </svg>
    );
  }
  if (k === "ml") {
    return (
      <svg {...common}>
        <path d="M2 15c3 0 4-9 8-9s5 6 8 6" />
        <path d="M2 12c3 0 4-9 8-9s5 6 8 6" strokeDasharray="1.5 2.5" strokeWidth="1" />
      </svg>
    );
  }
  return (
    <svg {...common}>
      <path d="M3 16 17 4" />
      {[[5, 12], [9, 11], [11, 7], [14, 7]].map(([x, y]) => <circle key={`${x}${y}`} cx={x} cy={y} r="1.1" fill="currentColor" />)}
    </svg>
  );
}

function ModelCard({ c, m }: { c: TargetComparison; m: ModelEntry }) {
  const role = MODEL_ROLE[m.key];
  const primary = c.rankings.find((r) => r.metric.key === c.selectionMetric);
  // Tie-aware, matching the leaderboard: equal values share a position.
  const own = primary?.order.find((o) => o.key === m.key);
  const pos = primary && own ? primary.order.findIndex((o) => Math.abs(o.value - own.value) < 1e-12) + 1 : 0;
  const strengths = c.rankings.filter(
    (r) => r.order.length > 1 && Math.abs(r.order[0].value - (r.order.find((o) => o.key === m.key)?.value ?? NaN)) < 1e-12,
  );

  return (
    <article className="mc-card flex flex-col" style={{ ["--m" as string]: role.color }}>
      <div className="mc-card__stripe" aria-hidden />
      <div className="flex flex-1 flex-col p-5">
        <header className="flex items-start gap-3">
          <span className="mc-icon">
            <ModelIcon k={m.key} model={m.model} />
          </span>
          <div className="min-w-0 flex-1">
            <h3 className="font-display text-[20px] font-bold uppercase leading-tight tracking-[0.03em] text-paper-900">
              {role.label}
            </h3>
            <p className="t-micro mt-0.5">
              {m.algorithm}
              {m.model && <code className="ml-1.5 text-paper-500">{m.model}</code>}
            </p>
          </div>
          {pos > 0 && primary && (
            <span className="mc-pos text-[20px]" data-p={pos} title={`Position ${pos} of 3 on ${primary.metric.label}`}>
              P{pos}
            </span>
          )}
        </header>

        <p className="t-body mt-4 text-[13px]">{m.approach}</p>

        {!m.available ? (
          <p className="mt-4 rounded-sm border border-dashed border-edge p-3 text-[13px] text-paper-500">
            N/A: no test metrics found for this model. Run <code>python scripts/build_all.py</code>.
          </p>
        ) : (
          <dl className="mt-5 space-y-3">
            {c.metrics.map((metric) => {
              const v = m.metrics[metric.key];
              const all = MODEL_KEYS.map((k) => c.models[k].metrics[metric.key]);
              const score = radarScore(metric, v, all);
              const isBest = strengths.some((s) => s.metric.key === metric.key);
              return (
                <div key={metric.key}>
                  <div className="flex items-baseline justify-between gap-3">
                    <dt className="text-[12px] text-paper-500" title={metric.plain}>
                      {metric.label}
                      {metric.key === c.selectionMetric && <span className="ml-1.5 text-[10px] uppercase tracking-wider text-[var(--papaya)]">key metric</span>}
                    </dt>
                    <dd className="t-data text-paper-900">
                      {isBest && <span className="marker mr-1.5">best</span>}
                      {fmtMetric(metric, v)}
                      {metric.unit === "s" && v !== null && <span className="unit"> s</span>}
                    </dd>
                  </div>
                  <div className="mc-meter mt-1" aria-hidden>
                    <span style={{ width: `${Math.round((score ?? 0) * 100)}%` }} />
                  </div>
                </div>
              );
            })}
          </dl>
        )}

        <dl className="mt-auto grid grid-cols-2 gap-x-4 gap-y-3 border-t border-track-300 pt-4 text-[12px]" style={{ marginTop: 20 }}>
          <div>
            <dt className="text-paper-500">Training time</dt>
            <dd className="t-data mt-0.5 text-paper-900">{fmtSeconds(m.fitSeconds)}</dd>
            {m.timingNote && <dd className="t-micro mt-0.5">{m.timingNote}</dd>}
          </div>
          <div>
            <dt className="text-paper-500">Model size</dt>
            <dd className="t-data mt-0.5 text-paper-900">
              {m.parameters !== null ? `${m.parameters.toLocaleString()} params` : "N/A"}
            </dd>
            <dd className="t-micro mt-0.5">{m.capacity}</dd>
          </div>
        </dl>

        {strengths.length > 0 && (
          <div className="mt-4 flex flex-wrap gap-1.5">
            {strengths.map((s) => (
              <span key={s.metric.key} className="badge" style={{ borderColor: role.color }}>
                <span className="mc-swatch !h-[6px] !w-[6px]" aria-hidden />
                Leads {s.metric.label}
              </span>
            ))}
          </div>
        )}
      </div>
    </article>
  );
}

/* ------------------------------------------------------------------------ */

function Leaderboard({ c }: { c: TargetComparison }) {
  return (
    <div className="mt-5 overflow-hidden rounded-md border border-track-300">
      {c.rankings.map((r) => (
        <div
          key={r.metric.key}
          className="grid grid-cols-[minmax(84px,auto)_1fr] items-center gap-3 border-b border-track-300 bg-track-100 px-3 py-2.5 last:border-b-0 sm:grid-cols-[120px_1fr]"
        >
          <div>
            <p className="text-[13px] font-semibold text-paper-900" title={r.metric.plain}>
              {r.metric.label}
            </p>
            <p className="t-micro">{r.metric.better === "lower" ? "lower wins" : "higher wins"}</p>
          </div>
          {r.order.length ? (
            <ol className="flex flex-wrap gap-1.5">
              {r.order.map((o, i) => {
                // Shared values share a position, as a timing screen would show a dead heat.
                const p = r.order.findIndex((x) => Math.abs(x.value - o.value) < 1e-12) + 1;
                return (
                  <li
                    key={o.key}
                    className="flex items-center gap-2 rounded-sm border px-2 py-1"
                    style={{
                      ["--m" as string]: MODEL_ROLE[o.key].color,
                      borderColor: i === 0 || p === 1 ? MODEL_ROLE[o.key].color : "var(--track-300)",
                    }}
                  >
                    <span className="mc-pos text-[12px]" data-p={p}>P{p}</span>
                    <span className="mc-swatch" aria-hidden />
                    <span className="text-[12px] text-paper-900">{MODEL_ROLE[o.key].short}</span>
                    <span className="t-data text-[12px] text-paper-500">{fmtMetric(r.metric, o.value)}</span>
                  </li>
                );
              })}
            </ol>
          ) : (
            <p className="t-micro">N/A</p>
          )}
        </div>
      ))}
    </div>
  );
}

function InsightCard({ insight }: { insight: Insight }) {
  const color = insight.model
    ? MODEL_ROLE[insight.model].color
    : insight.tone === "warn" ? "var(--papaya-fill)" : "var(--edge)";
  return (
    <li className="rounded-md border border-track-300 bg-track-100 p-4" style={{ borderLeft: `3px solid ${color}` }}>
      <p className="flex items-center gap-2 font-display text-[14px] font-bold uppercase tracking-[0.05em] text-paper-900">
        <span aria-hidden className="text-[var(--papaya)]">
          {insight.tone === "win" ? "◆" : insight.tone === "warn" ? "!" : "›"}
        </span>
        {insight.title}
      </p>
      <p className="t-body mt-1.5 text-[13px]">{insight.body}</p>
    </li>
  );
}

/* ------------------------------------------------------------------------ */

function ExactTable({ c }: { c: TargetComparison }) {
  const bestOf = (key: string) => c.rankings.find((r) => r.metric.key === key)?.order[0]?.value;
  const extra: { label: string; get: (m: ModelEntry) => string }[] = [
    { label: "Training time", get: (m) => fmtSeconds(m.fitSeconds) },
    { label: "Trainable parameters", get: (m) => (m.parameters !== null ? m.parameters.toLocaleString() : "N/A") },
    { label: "Input features", get: (m) => (m.nFeatures !== null ? String(m.nFeatures) : "N/A") },
    ...(c.task === "classification"
      ? [{ label: "Decision threshold", get: (m: ModelEntry) => (m.threshold !== null ? m.threshold.toFixed(3) : "N/A") }]
      : []),
  ];
  return (
    <div className="mt-5 overflow-x-auto">
      <table className="t-table min-w-[560px]">
        <caption className="sr-only">Exact test metrics for the three models</caption>
        <thead>
          <tr>
            <th scope="col">Metric</th>
            {MODEL_KEYS.map((k) => (
              <th key={k} scope="col" className="num">
                <span className="inline-flex items-center gap-1.5" style={{ ["--m" as string]: MODEL_ROLE[k].color }}>
                  <span className="mc-swatch" aria-hidden />
                  {MODEL_ROLE[k].short}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {c.metrics.map((metric) => (
            <tr key={metric.key}>
              <th scope="row" className="min-w-[150px] whitespace-normal text-left font-normal">
                <span className="text-paper-900">{metric.label}</span>
                {metric.unit && <span className="unit"> {metric.unit}</span>}
                <span className="t-micro block">{metric.plain}</span>
              </th>
              {MODEL_KEYS.map((k) => {
                const v = c.models[k].metrics[metric.key];
                const best = v !== null && bestOf(metric.key) !== undefined && Math.abs(v - bestOf(metric.key)!) < 1e-12;
                return (
                  <td key={k} className="num">
                    {best && <span className="marker mr-1.5">best</span>}
                    <span className={best ? "font-semibold text-paper-900" : "text-paper-700"}>{fmtMetric(metric, v)}</span>
                  </td>
                );
              })}
            </tr>
          ))}
          {extra.map((row) => (
            <tr key={row.label}>
              <th scope="row" className="text-left font-normal text-paper-700">{row.label}</th>
              {MODEL_KEYS.map((k) => (
                <td key={k} className="num">{row.get(c.models[k])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
