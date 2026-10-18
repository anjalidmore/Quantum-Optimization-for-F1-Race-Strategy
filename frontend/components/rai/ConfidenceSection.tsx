"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import type { RaiConfidenceModel, RaiSummary } from "@/lib/api";
import { MODEL_KEYS, MODEL_ROLE, type ModelKey } from "@/lib/modelComparison";
import { Explainer, InsufficientData, RaiHead } from "./ui";

const load = () => <div className="skeleton h-full" />;
const ScoreHistogram = dynamic(() => import("./RaiCharts").then((m) => m.ScoreHistogram), { ssr: false, loading: load });
const CountHistogram = dynamic(() => import("./RaiCharts").then((m) => m.CountHistogram), { ssr: false, loading: load });
const TercileBars = dynamic(() => import("./RaiCharts").then((m) => m.TercileBars), { ssr: false, loading: load });

const pct = (v: number | null | undefined, d = 0) => (v === null || v === undefined ? "N/A" : `${(v * 100).toFixed(d)}%`);

export function ConfidenceSection({ s }: { s: RaiSummary }) {
  const c = s.confidence;
  return (
    <div className="space-y-10">
      {/* ---- confidence is not reliability -------------------------------- */}
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-md border border-track-300 bg-track-100 p-4" style={{ borderLeft: "3px solid var(--papaya-fill)" }}>
          <p className="font-display text-[14px] font-bold uppercase tracking-[0.05em] text-paper-900">Confidence ≠ reliability</p>
          <p className="t-body mt-1.5 text-[13px]">
            <strong className="text-paper-900">Confidence</strong> is what a model says about itself: a probability,
            or a spread around its estimate. <strong className="text-paper-900">Reliability</strong> is whether that
            statement holds up against what actually happened on the held-out laps. A model can be confidently wrong,
            so every confidence view below sits next to the real outcome.
          </p>
        </div>
        <MechanismTable s={s} />
      </div>

      {c.kind === "probability" ? <ProbabilityView s={s} /> : <IntervalView s={s} />}

      <ReviewQueue s={s} />
    </div>
  );
}

function MechanismTable({ s }: { s: RaiSummary }) {
  return (
    <div className="rounded-md border border-track-300 bg-track-100 p-4">
      <p className="stat-label">How each model expresses confidence</p>
      <ul className="mt-2 space-y-2">
        {MODEL_KEYS.map((k) => {
          const m = s.confidence.models[k];
          return (
            <li key={k} className="flex items-start gap-2.5 text-[13px]" style={{ ["--m" as string]: MODEL_ROLE[k].color }}>
              <span className="mc-swatch mt-1" aria-hidden />
              <span className="min-w-[56px] font-semibold text-paper-900">{MODEL_ROLE[k].short}</span>
              <span className={m?.available ? "text-paper-700" : "text-paper-500"}>
                {m?.available ? mechanismLabel(m.mechanism!) : "Not Available"}
              </span>
            </li>
          );
        })}
      </ul>
      <p className="t-micro mt-2">
        {s.task === "classification"
          ? "Each score is judged against its own model's decision threshold. The three thresholds differ, so the scores are shown side by side, never on one shared scale."
          : "Only the DNN has a justified uncertainty method here. Point predictions are not given a made-up spread."}
      </p>
    </div>
  );
}

function mechanismLabel(m: string) {
  if (m === "predict_proba") return "Native class probability (scikit-learn predict_proba)";
  if (m === "sigmoid output") return "Sigmoid output of the network's last layer";
  return m;
}

/* ------------------------------------------------------------------------ */

function ProbabilityView({ s }: { s: RaiSummary }) {
  const c = s.confidence;
  return (
    <>
      <section aria-labelledby="dist-h">
        <RaiHead id="dist-h" kicker="Distribution" title="P(pit next lap), and whether each call was right" />
        <p className="t-micro mt-2 max-w-measure">
          Each held-out lap lands in the bin of the score the model gave it. Coloured: the model&rsquo;s call at its
          threshold was correct. Grey: it was wrong. Dashed line: that model&rsquo;s decision threshold.
        </p>
        <div className="mt-5 grid gap-4 lg:grid-cols-3">
          {MODEL_KEYS.map((k) => {
            const m = c.models[k];
            return (
              <div key={k} className="mc-card" style={{ ["--m" as string]: MODEL_ROLE[k].color }}>
                <div className="mc-card__stripe" aria-hidden />
                <div className="p-4">
                  <p className="font-display text-[15px] font-bold uppercase tracking-[0.04em] text-paper-900">{MODEL_ROLE[k].label}</p>
                  {!m?.available ? (
                    <NotAvailable reason={m?.reason} />
                  ) : (
                    <>
                      <div className="mt-3 h-[200px]">
                        <ScoreHistogram bins={m.histogram} threshold={m.threshold} model={k} />
                      </div>
                      <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 text-[12px]">
                        <Stat label="Pit calls" value={`${m.n_predicted_pit} of ${c.n_test}`} />
                        <Stat label={`Within ±${c.near_threshold} of threshold`} value={`${m.n_near_threshold} laps`} />
                        <Stat label="Accuracy near threshold" value={pct(m.accuracy_near_threshold)} />
                        <Stat label="Accuracy elsewhere" value={pct(m.accuracy_far_from_threshold)} />
                        <Stat label="Brier score" value={m.brier_score.toFixed(3)} hint="mean squared gap between probability and outcome; lower is better" />
                      </dl>
                    </>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      <section aria-labelledby="cal-h">
        <RaiHead id="cal-h" kicker="Reliability" title="Calibration" />
        <div className="mt-5 grid gap-4 lg:grid-cols-[1fr_1.2fr]">
          {(() => {
            const first = MODEL_KEYS.map((k) => c.models[k]).find((m) => m?.available);
            if (first?.calibration) {
              return <CalibrationTable s={s} />;
            }
            return (
              <div className="rounded-md border border-dashed border-edge bg-track-100 p-4">
                <p className="stat-label">Calibration curve</p>
                <InsufficientData className="mt-1" />
                <p className="t-body mt-2 text-[13px]">{first?.calibration_reason ?? "Not available."}</p>
              </div>
            );
          })()}
          <div className="space-y-3">
            <Explainer title="What calibration means">
              A model is calibrated if, among the laps it gives a 30% pit probability, about 30% really end in a pit.
              It is checked by binning predictions and comparing each bin&rsquo;s average score with its observed rate.
            </Explainer>
            <Explainer title="Why the Brier score is shown anyway, and its limit">
              The Brier score needs no binning, so it is computable here. But with almost every lap a non-pit lap, it
              mostly rewards low scores everywhere: a model that never predicts a pit scores well. Read it beside the
              pit-call count and recall, not alone.
            </Explainer>
            <Explainer title="Scores are not interchangeable">
              Task 6 trains logistic regression with class balancing, which deliberately pushes pit probabilities up.
              {s.models.ml.name === "random_forest"
                ? " The random forest is class-balanced too, and its probability is the average of its trees' leaf class fractions."
                : ` The ML model (${s.models.ml.name ?? "N/A"}) has its own probability mechanism.`}{" "}
              The network&rsquo;s sigmoid was trained without class weights. The same number means different things for each model, which is why each is judged
              against its own threshold.
            </Explainer>
          </div>
        </div>
      </section>
    </>
  );
}

function CalibrationTable({ s }: { s: RaiSummary }) {
  return (
    <div className="card overflow-x-auto">
      <table className="t-table">
        <thead><tr><th>Model</th><th className="num">Mean predicted</th><th className="num">Observed</th><th className="num">Laps</th></tr></thead>
        <tbody>
          {MODEL_KEYS.flatMap((k) => (s.confidence.models[k]?.calibration ?? []).map((b: any, i: number) => (
            <tr key={`${k}-${i}`}><td>{MODEL_ROLE[k].short}</td><td className="num">{b.mean_predicted.toFixed(2)}</td>
              <td className="num">{b.observed_rate.toFixed(2)}</td><td className="num">{b.count}</td></tr>
          )))}
        </tbody>
      </table>
    </div>
  );
}

/* ------------------------------------------------------------------------ */

function IntervalView({ s }: { s: RaiSummary }) {
  const c = s.confidence;
  const withUnc = MODEL_KEYS.filter((k) => c.models[k]?.available);
  return (
    <>
      <section aria-labelledby="unc-h">
        <RaiHead id="unc-h" kicker="Uncertainty" title="How sure is each lap-time prediction?" />
        <div className="mt-5 grid gap-4 lg:grid-cols-3">
          {MODEL_KEYS.map((k) => {
            const m = c.models[k];
            return (
              <div key={k} className="mc-card" style={{ ["--m" as string]: MODEL_ROLE[k].color }}>
                <div className="mc-card__stripe" aria-hidden />
                <div className="p-4">
                  <p className="font-display text-[15px] font-bold uppercase tracking-[0.04em] text-paper-900">{MODEL_ROLE[k].label}</p>
                  {!m?.available ? (
                    <NotAvailable reason={m?.reason} />
                  ) : (
                    <UncertaintyCard m={m} k={k} passes={c.mc_passes ?? null} />
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {withUnc.map((k) => {
        const m = c.models[k];
        return (
          <section key={k} aria-labelledby={`rel-${k}`}>
            <RaiHead id={`rel-${k}`} kicker="Reliability" title={`Does ${MODEL_ROLE[k].short}'s uncertainty track its real error?`} />
            <div className="mt-5 grid gap-4 lg:grid-cols-[1.2fr_1fr] [&>*]:min-w-0">
              <figure className="card">
                <figcaption>
                  <h3 className="t-title text-[15px]">Actual error by uncertainty third</h3>
                  <p className="t-micro mt-1">
                    Laps sorted by the model&rsquo;s own spread and cut into thirds. If uncertainty were informative,
                    actual error would rise from left to right.
                  </p>
                </figcaption>
                <div className="mt-4 h-[240px]"><TercileBars rows={m.error_by_uncertainty_tercile} model={k} /></div>
              </figure>
              <div className="space-y-3">
                <CoverageBar label="Actual lap within ±1σ" value={m.coverage_1sigma} target={0.683} k={k} />
                <CoverageBar label="Actual lap within ±2σ" value={m.coverage_2sigma} target={0.954} k={k} />
                <div className="rounded-md border border-track-300 bg-track-100 p-3">
                  <p className="text-[12px] text-paper-500">Rank correlation, spread vs actual error</p>
                  <p className="t-data mt-0.5 text-[20px] text-paper-900">{m.spearman_std_vs_error.toFixed(2)}</p>
                  <p className="t-micro">
                    {Math.abs(m.spearman_std_vs_error) < 0.1
                      ? "Close to zero: on these laps, a high-uncertainty lap is no more likely to be badly predicted than a low-uncertainty one. Treat the spread as a coverage band, not as a per-lap warning."
                      : m.spearman_std_vs_error > 0
                        ? "Positive: laps the model is less sure about do tend to have larger errors."
                        : "Negative: higher stated uncertainty goes with smaller errors here, the opposite of what it should."}
                  </p>
                </div>
                <Explainer title="Why MC dropout is justified for this network">
                  The network was trained with dropout layers. Keeping dropout switched on at prediction time and
                  repeating the forward pass {c.mc_passes ?? "N/A"} times samples many slightly different networks; the
                  spread of their answers approximates the model&rsquo;s uncertainty (Gal &amp; Ghahramani, 2016). It
                  captures model uncertainty, not noise in the lap times themselves, which is why coverage is checked
                  against real laps rather than assumed.
                </Explainer>
              </div>
            </div>
          </section>
        );
      })}

      <section aria-labelledby="err-h">
        <RaiHead id="err-h" kicker="Ground truth" title="Actual error distribution, all three models" />
        <p className="t-micro mt-2 max-w-measure">
          What reliability looks like without any self-reported confidence: how far each model&rsquo;s prediction
          landed from the real lap time on the held-out laps. Same bins for all three.
        </p>
        <div className="mt-5 grid gap-4 lg:grid-cols-3">
          {MODEL_KEYS.map((k) => (
            <div key={k} className="card" style={{ ["--m" as string]: MODEL_ROLE[k].color }}>
              <p className="flex items-center gap-2 text-[13px] font-semibold text-paper-900">
                <span className="mc-swatch" aria-hidden />{MODEL_ROLE[k].short}
                <span className="t-micro ml-auto">MAE {c.models[k]?.mae?.toFixed(3) ?? "N/A"} s</span>
              </p>
              <div className="mt-3 h-[170px]">
                {c.models[k]?.error_histogram
                  ? <CountHistogram bins={c.models[k].error_histogram} model={k} unit="abs. error, s" />
                  : <InsufficientData />}
              </div>
            </div>
          ))}
        </div>
      </section>
    </>
  );
}

function UncertaintyCard({ m, k, passes }: { m: RaiConfidenceModel; k: ModelKey; passes: number | null }) {
  return (
    <>
      <p className="t-micro mt-1">{m.mechanism}</p>
      <div className="mt-3 h-[160px]"><CountHistogram bins={m.std_histogram} model={k} unit="spread (σ), s" /></div>
      <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 text-[12px]">
        <Stat label="Mean spread (σ)" value={`${m.mean_std.toFixed(3)} s`} />
        <Stat label="Flagged (top 10% σ)" value={`${m.n_flagged} laps`} hint={`σ ≥ ${m.flag_cutoff_std.toFixed(3)} s`} />
        {passes && <Stat label="Passes per lap" value={String(passes)} />}
      </dl>
    </>
  );
}

function CoverageBar({ label, value, target, k }: { label: string; value: number; target: number; k: ModelKey }) {
  return (
    <div className="rounded-md border border-track-300 bg-track-100 p-3" style={{ ["--m" as string]: MODEL_ROLE[k].color }}>
      <div className="flex items-baseline justify-between text-[12px]">
        <span className="text-paper-500">{label}</span>
        <span className="t-data text-paper-900">{pct(value, 1)} <span className="t-micro">expected {pct(target, 0)}</span></span>
      </div>
      <div className="relative mt-2 h-2 rounded-full bg-track-300">
        <span className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${value * 100}%`, backgroundColor: "var(--m)" }} />
        <span className="absolute -top-1 h-4 w-[2px] bg-paper-900" style={{ left: `${target * 100}%` }} aria-hidden />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------------ */

function ReviewQueue({ s }: { s: RaiSummary }) {
  const c = s.confidence;
  const [n, setN] = useState(10);
  const reason = (r: string) =>
    r === "models_disagree" ? "models disagree"
      : r.endsWith("_near_threshold") ? `${MODEL_ROLE[r.split("_")[0] as ModelKey]?.short} near threshold`
        : r.endsWith("_high_uncertainty") ? `${MODEL_ROLE[r.split("_")[0] as ModelKey]?.short} high uncertainty` : r;
  return (
    <section aria-labelledby="rev-h">
      <RaiHead id="rev-h" kicker="Human review" title={`Laps flagged for review: ${c.n_review} of ${c.n_test}`} />
      <p className="t-micro mt-2 max-w-measure">
        {c.kind === "probability"
          ? `Flagged when a model's score is within ${c.near_threshold} of its own threshold, or when the three models make different pit calls. Laps with more reasons come first.`
          : "Flagged when the DNN's Monte Carlo spread is in its top 10%. Sorted by spread."}
      </p>
      {c.review.length === 0 ? (
        <p className="t-body mt-4 text-[13px] text-paper-500">No laps meet the review rule.</p>
      ) : (
        <div className="mt-4 overflow-x-auto">
          <table className="t-table min-w-[640px]">
            <thead>
              <tr>
                <th>Lap</th><th>Driver</th><th>Tyre</th>
                <th className="num">Actual</th>
                {MODEL_KEYS.map((k) => <th key={k} className="num">{MODEL_ROLE[k].short}</th>)}
                {c.kind === "interval" && <th className="num">σ</th>}
                <th>Why</th>
              </tr>
            </thead>
            <tbody>
              {c.review.slice(0, n).map((r) => (
                <tr key={r.row} className={r.actual === 1 && c.kind === "probability" ? "is-marked" : undefined}>
                  <td className="tabular-nums">{r.lap ?? "N/A"}</td>
                  <td>{r.driver ?? "N/A"}</td>
                  <td>{r.compound ?? "N/A"}</td>
                  <td className="num">{c.kind === "probability" ? (r.actual === 1 ? "pit" : "no pit") : r.actual?.toFixed(3)}</td>
                  {MODEL_KEYS.map((k) => (
                    <td key={k} className="num">
                      {r.scores[k] == null ? "N/A" : c.kind === "probability"
                        ? <>{r.scores[k]!.toFixed(2)}{r.calls?.[k] === 1 && <span className="marker ml-1">pit</span>}</>
                        : r.scores[k]!.toFixed(3)}
                    </td>
                  ))}
                  {c.kind === "interval" && <td className="num">{r.std?.toFixed(3)}</td>}
                  <td className="text-[12px] text-paper-500">{r.reasons.map(reason).join(" · ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {c.review.length > n && (
            <button type="button" onClick={() => setN((v) => v + 20)}
                    className="mt-3 text-[12px] font-medium text-paper-700 underline decoration-edge underline-offset-[3px] hover:text-paper-900">
              Show {Math.min(20, c.review.length - n)} more ({c.review.length - n} hidden)
            </button>
          )}
        </div>
      )}
    </section>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div>
      <dt className="text-paper-500" title={hint}>{label}</dt>
      <dd className="t-data mt-0.5 text-paper-900">{value}</dd>
      {hint && <dd className="t-micro">{hint}</dd>}
    </div>
  );
}

function NotAvailable({ reason }: { reason?: string }) {
  return (
    <div className="mt-3 rounded-sm border border-dashed border-edge p-3">
      <InsufficientData label="Not Available" />
      <p className="t-body mt-1.5 text-[12px]">{reason ?? "No confidence mechanism."}</p>
    </div>
  );
}
