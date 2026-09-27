import { api, ApiError } from "@/lib/api";
import { DatasetBadge } from "@/components/DatasetBadge";
import { TyreBadge } from "@/components/TyreBadge";
import { TaskLedger } from "@/components/overview/TaskLedger";
import { DriverErrorTable } from "@/components/overview/DriverErrorTable";
import { fmt } from "@/lib/format";

/**
 * Overview — one focal readout, a task ledger beside it, the driver table below.
 *
 * The focal figure is the A* optimal strategy cost, because that is the one
 * number this whole platform exists to produce. Everything else on the page is
 * supporting evidence, set at a third of its size.
 */
export default async function OverviewPage() {
  let health = null;
  let manifest = null;
  let evidence = null;
  let error: string | null = null;

  try {
    [health, manifest, evidence] = await Promise.all([
      api.health(),
      api.artifacts().catch(() => null),
      api.taskEvidence(),
    ]);
  } catch (e) {
    error = e instanceof ApiError ? e.message : "Could not reach the backend API.";
  }

  // Optional sections: a missing artifact renders "Not generated yet", never a zero.
  const [search, expert, dlMetrics, strat] = await Promise.all([
    api.searchComparison().catch(() => null),
    api.expertSystem().catch(() => null),
    api.dlMetrics().catch(() => null),
    api.xaiStratification().catch(() => null),
  ]);

  if (error || !evidence) {
    return (
      <div className="card mt-14 border-accent">
        <span className="badge badge-danger">Backend unreachable</span>
        <p className="t-body mt-3">{error}</p>
        <p className="t-micro mt-1">
          Start it with <code className="text-paper-500">./run.sh</code> or{" "}
          <code className="text-paper-500">uvicorn app.api.main:app --reload</code>
        </p>
      </div>
    );
  }

  const plan = search?.available ? search : null;
  const stop = plan?.pit_stops?.[0] ?? null;
  const laptimeDnn = dlMetrics?.models?.target_laptime?.test_metrics?.mae;

  return (
    <div>
      {/* ---- focal band: dominant readout left, ledger right ---------------- */}
      <div className="grid gap-x-14 gap-y-10 border-b border-track-300 pb-10 pt-14 lg:grid-cols-[1.55fr_1fr]">
        <div>
          {manifest && <DatasetBadge source={manifest.dataset_source} />}

          {plan ? (
            <>
              <p className="stat-label mt-5">Optimal strategy cost, remaining race — A* search</p>
              <p className="t-display-hero mt-1.5">
                {fmt(plan.final_cost_seconds, 2)}
                <span className="ml-2 font-display text-[22px] font-semibold not-italic text-accent">s</span>
              </p>
              <p className="t-body mt-3.5 max-w-[46ch] text-[13px] text-paper-500">
                {stop ? (
                  <>
                    One stop, lap {stop.lap}, onto <TyreBadge compound={stop.compound} />.{" "}
                  </>
                ) : (
                  <>No stop on this instance. </>
                )}
                A* and uniform-cost search agree to four decimals, which is the admissibility check — A* reached it
                expanding{" "}
                <span className="tabular-nums text-paper-700">
                  {plan.algorithms.find((a) => a.algorithm === "A*")?.nodes_expanded}
                </span>{" "}
                nodes against UCS&rsquo;s{" "}
                <span className="tabular-nums text-paper-700">
                  {plan.algorithms.find((a) => a.algorithm === "UCS")?.nodes_expanded}
                </span>
                .
              </p>
            </>
          ) : (
            <>
              <p className="stat-label mt-5">Optimal strategy cost, remaining race</p>
              <p className="t-display-md mt-1.5 text-paper-500">Not generated yet</p>
              <p className="t-micro mt-2">
                Run <span className="text-paper-500">python scripts/run_search.py</span> to produce it.
              </p>
            </>
          )}

          {/* One tile pattern: label, figure, dimmed unit. No boxes. */}
          <dl className="mt-8 grid grid-cols-2 gap-x-8 gap-y-6 border-t border-track-300 pt-6 sm:grid-cols-4">
            <div>
              <dt className="stat-label">Lap-time model error</dt>
              <dd className="stat-value">
                {laptimeDnn !== undefined ? fmt(laptimeDnn, 3) : "—"}
                <span className="unit"> s MAE</span>
              </dd>
            </div>
            <div>
              <dt className="stat-label">Expert rules</dt>
              <dd className="stat-value">
                {expert?.available ? expert.n_rules : "—"}
                <span className="unit"> available</span>
              </dd>
            </div>
            <div>
              <dt className="stat-label">Trained models</dt>
              <dd className="stat-value">
                {health?.model_count ?? "—"}
                <span className="unit"> registered</span>
              </dd>
            </div>
            <div>
              <dt className="stat-label">Tasks built</dt>
              <dd className="stat-value">
                {evidence.completed_count}
                <span className="unit"> of {evidence.total_count}</span>
              </dd>
            </div>
          </dl>
        </div>

        <TaskLedger evidence={evidence} />
      </div>

      {/* ---- supporting evidence ------------------------------------------- */}
      <div className="grid gap-x-14 gap-y-12 pt-10 lg:grid-cols-[1.55fr_1fr]">
        {strat ? (
          <DriverErrorTable strat={strat} />
        ) : (
          <section>
            <h2 className="t-title">Lap-time error by driver</h2>
            <p className="t-body mt-2 text-paper-500">Not generated yet.</p>
            <p className="t-micro mt-1">
              Run <span className="text-paper-500">python scripts/build_all.py</span> to produce it.
            </p>
          </section>
        )}

        <section aria-labelledby="what-h">
          <h2 id="what-h" className="t-title">
            What this is
          </h2>
          <p className="t-body mt-3 max-w-measure">
            One Grand Prix, answered three ways: an ontology and a rule base that reason symbolically, models that
            learn from its laps, and explainability tools that say
            why each call was made — and where it stops being trustworthy.
          </p>
          <p className="t-body mt-3 max-w-measure text-paper-500">
            Every figure on every page is read from a generated artifact at request time. Where a figure would
            mislead, the interface says so in words instead of dressing it up.
          </p>
        </section>
      </div>
    </div>
  );
}
