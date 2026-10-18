"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import type { RaiFlag, RaiSummary } from "@/lib/api";
import { MODEL_KEYS, MODEL_ROLE, type ModelKey } from "@/lib/modelComparison";
import { Explainer, InsufficientData, RaiHead, Seg } from "./ui";

const GroupBars = dynamic(() => import("./RaiCharts").then((m) => m.GroupBars), {
  ssr: false, loading: () => <div className="skeleton h-full" />,
});

type MetricOpt = { key: string; label: string; unit: string; kind: "error" | "rate" | "signed"; plain: string };

const REG_METRICS: MetricOpt[] = [
  { key: "mae", label: "MAE", unit: "s", kind: "error", plain: "Average miss per lap for this group, in seconds." },
  { key: "mean_error_bias", label: "Bias", unit: "s", kind: "signed",
    plain: "Average signed error (predicted − actual). Positive: the model predicts this group slower than it ran." },
  { key: "rmse", label: "RMSE", unit: "s", kind: "error", plain: "Like MAE, but large misses count extra." },
];
const CLF_METRICS: MetricOpt[] = [
  { key: "predicted_pit_rate", label: "Predicted pit rate", unit: "share", kind: "rate",
    plain: "Share of this group's laps the model calls a pit. Compare with the actual pit rate column." },
  { key: "false_positive_rate", label: "False-alarm rate", unit: "share", kind: "rate",
    plain: "Of the laps with no pit, the share the model wrongly flagged as a pit." },
  { key: "recall", label: "Recall", unit: "share", kind: "rate",
    plain: "Of the real pit laps in this group, the share the model caught. Undefined with no pit laps." },
  { key: "accuracy", label: "Accuracy", unit: "share", kind: "rate", plain: "Share of this group's laps classified correctly." },
];

const fmt = (v: number | null | undefined, kind: MetricOpt["kind"]) =>
  v === null || v === undefined ? null : kind === "rate" ? `${(v * 100).toFixed(0)}%` : `${kind === "signed" && v > 0 ? "+" : ""}${v.toFixed(3)}`;

export function FairnessSection({ s }: { s: RaiSummary }) {
  const f = s.fairness;
  const metrics = s.task === "regression" ? REG_METRICS : CLF_METRICS;
  // Open on the first grouping where a between-group gap is measurable (2+
  // groups pass the sample rule), so the page does not start on a wall of
  // "Insufficient Data". Falls back to the first grouping.
  const measurable = f.groups.find((g) => MODEL_KEYS.some((k) => (f.summary[k]?.[g]?.n_adequate_groups ?? 0) >= 2));
  const [groupType, setGroupType] = useState(measurable ?? f.groups[0]);
  const [metricKey, setMetricKey] = useState(metrics[0].key);
  const metric = metrics.find((m) => m.key === metricKey) ?? metrics[0];

  const rows = f.rows.filter((r) => r.group_type === groupType);
  const overallRows = f.rows.filter((r) => r.group_type === "overall");
  const groups = [...new Set(rows.map((r) => r.group))].sort((a, b) => {
    const na = rows.find((r) => r.group === a)?.n_laps ?? 0;
    const nb = rows.find((r) => r.group === b)?.n_laps ?? 0;
    return nb - na || String(a).localeCompare(String(b));
  });
  const cell = (k: ModelKey, g: string) => rows.find((r) => r.model === k && r.group === g);
  const overall = Object.fromEntries(MODEL_KEYS.map((k) => [k, overallRows.find((r) => r.model === k)?.[metric.key] ?? null]));
  const flagFor = (k: ModelKey, g: string): RaiFlag | undefined =>
    f.flags.find((fl) => fl.model === k && fl.group_type === groupType && fl.group === g && fl.metric === metric.key);
  const absMax = Math.max(1e-9, ...rows.map((r) => Math.abs(r[metric.key] ?? 0)));
  const flags = f.flags.filter((fl) => fl.group_type === groupType);
  const [showAll, setShowAll] = useState(false);
  const driverNs = f.rows.filter((r) => r.group_type === "Driver" && r.model === "dnn").map((r) => r.n_laps);
  const driverRange = driverNs.length ? [Math.min(...driverNs), Math.max(...driverNs)] : null;

  return (
    <div className="space-y-10">
      {/* ---- what can and cannot be measured ------------------------------ */}
      <div className="grid gap-4 lg:grid-cols-[1fr_1.3fr]">
        <div className="rounded-md border border-dashed border-edge bg-track-100 p-4">
          <p className="stat-label">Demographic fairness</p>
          <InsufficientData className="mt-1" />
          <p className="t-body mt-2 text-[13px]">
            The dataset contains no protected attributes (no age, gender or nationality), so protected-class
            fairness cannot be measured, and none is estimated or inferred from names.
          </p>
        </div>
        <div className="rounded-md border border-track-300 bg-track-100 p-4">
          <p className="stat-label">What is measured instead</p>
          <p className="t-body mt-1.5 text-[13px]">
            Performance across <strong className="text-paper-900">teams</strong>, <strong className="text-paper-900">tyre compounds</strong> and{" "}
            <strong className="text-paper-900">drivers</strong>: operational groups a race engineer cares about.
            A model that is systematically worse for one team is less useful to that team. Every lap is from one
            circuit, so circuit and weather groups do not exist.
          </p>
          <p className="t-micro mt-2">
            Sample rules (Task 8&rsquo;s stratification): a group needs at least {f.rules.min_rows_per_group} laps
            {s.task === "classification" && <> and {f.rules.min_pit_laps_per_group} real pit laps</>} before a gap
            between groups is reported. Smaller groups are shown, marked &ldquo;small n&rdquo;, and are descriptive only.
          </p>
        </div>
      </div>

      {/* ---- controls -------------------------------------------------------- */}
      <div className="flex flex-wrap items-end gap-x-8 gap-y-4">
        <Seg label="Group by" value={groupType} onChange={setGroupType}
             options={f.groups.map((g) => ({ value: g, label: g }))} />
        <Seg label="Metric" value={metricKey} onChange={setMetricKey}
             options={metrics.map((m) => ({ value: m.key, label: m.label }))} />
      </div>
      <p className="t-micro -mt-6">{metric.plain}</p>

      {/* ---- heatmap + bars ------------------------------------------------- */}
      <div className="grid gap-4 xl:grid-cols-[1fr_1.1fr] [&>*]:min-w-0">
        <section aria-labelledby="heat-h" className="card">
          <h3 id="heat-h" className="t-title text-[15px]">{metric.label} by {groupType.toLowerCase()}: heatmap</h3>
          <p className="t-micro mt-1">
            {metric.kind === "error"
              ? "Shading grows with how far a group's error exceeds that model's overall error."
              : metric.kind === "signed"
                ? "Red: predicted slower than actual. Blue: predicted faster."
                : "Shading grows with the rate."}{" "}
            ⚑ marks a highlighted disparity.
          </p>
          <div className="mt-4 max-h-[460px] overflow-auto">
            <table className="w-full border-separate border-spacing-[3px] text-[12px]">
              <caption className="sr-only">{metric.label} per {groupType} and model</caption>
              <thead className="sticky top-0 bg-track-100">
                <tr>
                  <th scope="col" className="px-2 py-1 text-left font-medium text-paper-500">{groupType}</th>
                  {s.task === "classification" && <th scope="col" className="px-2 py-1 text-right font-medium text-paper-500">Actual</th>}
                  {MODEL_KEYS.map((k) => (
                    <th key={k} scope="col" className="px-2 py-1 text-center font-medium text-paper-700" style={{ ["--m" as string]: MODEL_ROLE[k].color }}>
                      <span className="inline-flex items-center gap-1.5"><span className="mc-swatch" aria-hidden />{MODEL_ROLE[k].short}</span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th scope="row" className="px-2 py-1.5 text-left font-semibold text-paper-900">All laps</th>
                  {s.task === "classification" && (
                    <td className="px-2 text-right tabular-nums text-paper-500">{fmt(overallRows[0]?.actual_pit_rate, "rate")}</td>
                  )}
                  {MODEL_KEYS.map((k) => (
                    <td key={k} className="rounded-sm border border-track-300 px-2 py-1.5 text-center tabular-nums font-semibold text-paper-900">
                      {fmt(overall[k], metric.kind) ?? <span className="font-normal text-paper-500">N/A</span>}
                    </td>
                  ))}
                </tr>
                {groups.map((g) => {
                  const any = cell("dnn", g) ?? cell("ml", g) ?? cell("simple", g);
                  const small = (any?.n_laps ?? 0) < f.rules.min_rows_per_group;
                  return (
                    <tr key={g}>
                      <th scope="row" className="whitespace-nowrap px-2 py-1 text-left font-normal text-paper-700">
                        {g} <span className="t-micro">n={any?.n_laps}</span>
                        {small && <span className="ml-1 text-[10px] uppercase tracking-wide text-[var(--papaya)]">small n</span>}
                      </th>
                      {s.task === "classification" && (
                        <td className="px-2 text-right tabular-nums text-paper-500">{fmt(any?.actual_pit_rate, "rate")}</td>
                      )}
                      {MODEL_KEYS.map((k) => {
                        const r = cell(k, g);
                        const v = r?.[metric.key] ?? null;
                        const flag = flagFor(k, g);
                        return (
                          <HeatCell key={k} value={v} text={fmt(v, metric.kind)} kind={metric.kind}
                                    overall={overall[k]} absMax={absMax} small={small} flag={flag} />
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>

        <section aria-labelledby="bars-h" className="card">
          <h3 id="bars-h" className="t-title text-[15px]">{metric.label} by {groupType.toLowerCase()}: three models</h3>
          <p className="t-micro mt-1">Dashed lines: each model&rsquo;s overall value. Groups with no defined value have no bar.</p>
          <div className="mt-4 overflow-x-auto">
            <div className="h-[340px]" style={{ minWidth: Math.max(480, groups.length * 52) }}>
              <GroupBars
                unit={metric.unit === "share" ? "share of laps" : "seconds"}
                rows={groups.map((g) => ({ group: g, values: Object.fromEntries(MODEL_KEYS.map((k) => [k, cell(k, g)?.[metric.key] ?? null])) }))}
                overall={overall}
              />
            </div>
          </div>
        </section>
      </div>

      {/* ---- fairness metrics ----------------------------------------------- */}
      <section aria-labelledby="fm-h">
        <RaiHead id="fm-h" kicker="Fairness metrics" title={`Gaps between ${groupType.toLowerCase()} groups`} />
        <div className="mt-5 grid gap-4 md:grid-cols-3">
          {MODEL_KEYS.map((k) => (
            <FairnessMetricCard key={k} model={k} task={s.task} entry={f.summary[k]?.[groupType]} rules={f.rules} />
          ))}
        </div>
        <div className="mt-4 grid gap-3 md:grid-cols-2">
          {s.task === "classification" ? (
            <>
              <Explainer title="Demographic parity difference">
                The largest gap in predicted pit rate between groups. Zero means every group is called a pit equally often.
                Limitation: it ignores the real pit rate. Teams run different strategies, so equal call rates are not the goal
                here; read it beside the actual pit rate.
              </Explainer>
              <Explainer title="Equal opportunity difference">
                The largest gap in recall (pit laps caught) between groups. It needs at least{" "}
                {f.rules.min_pit_laps_per_group} real pit laps in each group compared. This holdout has{" "}
                {s.n_positive ?? "N/A"} in total, so wherever groups fall short it is shown as Insufficient Data rather
                than computed from one or two laps.
              </Explainer>
              <Explainer title="False-alarm rate gap (predictive equality)">
                The largest gap in false-positive rate between groups: whether one team gets more unnecessary pit calls.
                It only needs non-pit laps, so it is the most measurable fairness signal on this data.
              </Explainer>
            </>
          ) : (
            <>
              <Explainer title="MAE gap between groups">
                The difference between the worst- and best-served group&rsquo;s average error. For lap time, accuracy gaps
                matter more than outcome parity, because there is no &ldquo;favourable&rdquo; prediction to share out.
                Limitation: it only uses groups that pass the sample rule.
              </Explainer>
              <Explainer title="Bias gap">
                The difference in average signed error between groups. A model can have similar MAE for two groups while
                over-predicting one and under-predicting the other.
              </Explainer>
            </>
          )}
          <Explainer title="Why small groups are not compared">
            {driverRange
              ? <>Each driver has {driverRange[0]}–{driverRange[1]} held-out laps. </>
              : null}
            One pit lap in a window that small can double a group&rsquo;s error, so a gap between two small groups says
            more about which laps fell in the window than about the model.
          </Explainer>
        </div>
      </section>

      {/* ---- disparity flags ------------------------------------------------ */}
      <section aria-labelledby="flags-h">
        <RaiHead id="flags-h" kicker="For investigation" title="Highlighted disparities" />
        <p className="t-micro mt-2 max-w-measure">
          Rule: {s.task === "regression"
            ? `a group's error is at least ${f.rules.disparity_ratio}× the model's overall error`
            : `a group's false-alarm rate is at least ${Math.round(f.rules.disparity_abs_rate * 100)} points above the model's overall rate`}.
          Flags on groups that pass the sample rule are listed first.
        </p>
        {flags.length === 0 ? (
          <p className="t-body mt-4 text-[13px] text-paper-500">No group meets the rule for {groupType.toLowerCase()}.</p>
        ) : (
          <>
            <ul className="mt-4 grid gap-2 md:grid-cols-2">
              {(showAll ? flags : flags.slice(0, 6)).map((fl, i) => (
                <li key={i} className="flex items-start gap-3 rounded-md border border-track-300 bg-track-100 p-3"
                    style={{ ["--m" as string]: MODEL_ROLE[fl.model].color, borderLeft: `3px solid ${MODEL_ROLE[fl.model].color}` }}>
                  <span className={`badge shrink-0 ${fl.adequate_sample ? "badge-danger" : ""}`}>
                    {fl.adequate_sample ? "Investigate" : "Small n"}
                  </span>
                  <p className="text-[13px] text-paper-700">
                    <span className="font-semibold text-paper-900">{MODEL_ROLE[fl.model].short}</span> on{" "}
                    <span className="font-semibold text-paper-900">{fl.group}</span>:{" "}
                    {fl.metric === "mae"
                      ? <>MAE {fl.value.toFixed(3)} s, {fl.ratio}× its overall {fl.overall.toFixed(3)} s</>
                      : <>false-alarm rate {(fl.value * 100).toFixed(0)}% vs {(fl.overall * 100).toFixed(0)}% overall</>}
                    <span className="t-micro"> · {fl.n_laps} laps</span>
                  </p>
                </li>
              ))}
            </ul>
            {flags.length > 6 && (
              <button type="button" onClick={() => setShowAll((v) => !v)}
                      className="mt-3 text-[12px] font-medium text-paper-700 underline decoration-edge underline-offset-[3px] hover:text-paper-900">
                {showAll ? "Show fewer" : `Show all ${flags.length}`}
              </button>
            )}
          </>
        )}
      </section>
    </div>
  );
}

function HeatCell({ value, text, kind, overall, absMax, small, flag }: {
  value: number | null; text: string | null; kind: MetricOpt["kind"]; overall: number | null;
  absMax: number; small: boolean; flag?: RaiFlag;
}) {
  if (value === null) {
    return (
      <td className="rounded-sm border border-dashed border-track-300 px-2 py-1 text-center text-[10px] uppercase tracking-wide text-paper-400"
          title="Undefined for this group (for example, recall with no pit laps)">
        Insufficient data
      </td>
    );
  }
  let bg = "transparent";
  if (kind === "error" && overall) {
    const t = Math.max(0, Math.min(1, (value / overall - 1) / 2));
    bg = `color-mix(in srgb, var(--ferrari) ${Math.round(t * 70)}%, transparent)`;
  } else if (kind === "signed") {
    const t = Math.min(1, Math.abs(value) / absMax);
    bg = `color-mix(in srgb, ${value > 0 ? "var(--data-pos)" : "var(--data-neg)"} ${Math.round(t * 55)}%, transparent)`;
  } else if (kind === "rate") {
    bg = `color-mix(in srgb, var(--papaya-fill) ${Math.round(Math.min(1, value) * 65)}%, transparent)`;
  }
  return (
    <td className={`rounded-sm px-2 py-1 text-center tabular-nums text-paper-900 ${small ? "border border-dashed border-track-300" : "border border-track-300"}`}
        style={{ backgroundColor: bg, ...(flag ? { outline: "2px solid var(--ferrari)", outlineOffset: -2 } : {}) }}
        title={flag ? "Highlighted disparity" : undefined}>
      {flag && <span aria-label="flagged" className="mr-1">⚑</span>}
      {text}
    </td>
  );
}

function FairnessMetricCard({ model, task, entry, rules }: {
  model: ModelKey; task: string; entry?: Record<string, any>; rules: RaiSummary["fairness"]["rules"];
}) {
  const role = MODEL_ROLE[model];
  const items: { label: string; gap: any; why: string; pct?: boolean }[] =
    task === "regression"
      ? [
          { label: "MAE gap", gap: entry?.mae_gap, why: `needs 2+ groups with ≥${rules.min_rows_per_group} laps; ${entry?.n_adequate_groups ?? 0} qualify` },
          { label: "Bias gap", gap: entry?.bias_gap, why: `needs 2+ groups with ≥${rules.min_rows_per_group} laps` },
          { label: "MAE gap, all groups (descriptive)", gap: entry?.mae_gap_all_groups, why: "fewer than two groups" },
        ]
      : [
          { label: "Demographic parity diff.", gap: entry?.demographic_parity_difference, pct: true, why: `needs 2+ groups with ≥${rules.min_rows_per_group} laps; ${entry?.n_adequate_groups ?? 0} qualify` },
          { label: "False-alarm rate gap", gap: entry?.false_positive_rate_gap, pct: true, why: `needs 2+ groups with ≥${rules.min_rows_per_group} laps` },
          { label: "Equal opportunity diff.", gap: entry?.equal_opportunity_difference, pct: true, why: `needs 2+ groups with ≥${rules.min_pit_laps_per_group} pit laps; ${entry?.groups_with_enough_pit_laps ?? 0} qualify` },
        ];
  return (
    <div className="mc-card" style={{ ["--m" as string]: role.color }}>
      <div className="mc-card__stripe" aria-hidden />
      <div className="p-4">
        <p className="font-display text-[15px] font-bold uppercase tracking-[0.04em] text-paper-900">{role.label}</p>
        <dl className="mt-3 space-y-3">
          {items.map((it) => (
            <div key={it.label}>
              <dt className="text-[12px] text-paper-500">{it.label}</dt>
              {it.gap ? (
                <dd className="mt-0.5">
                  <span className="t-data text-[18px] text-paper-900">
                    {it.pct ? `${(it.gap.value * 100).toFixed(1)} pts` : `${it.gap.value.toFixed(3)} s`}
                  </span>
                  <span className="t-micro block">
                    {it.gap.max_group} {it.pct ? `${(it.gap.max * 100).toFixed(0)}%` : it.gap.max.toFixed(3)} vs{" "}
                    {it.gap.min_group} {it.pct ? `${(it.gap.min * 100).toFixed(0)}%` : it.gap.min.toFixed(3)}
                  </span>
                </dd>
              ) : (
                <dd className="mt-0.5">
                  <InsufficientData />
                  <span className="t-micro block">{it.why}</span>
                </dd>
              )}
            </div>
          ))}
        </dl>
      </div>
    </div>
  );
}
