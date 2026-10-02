import Image from "next/image";
import { api, ApiError } from "@/lib/api";
import { DatasetBadge } from "@/components/DatasetBadge";
import { TyreBadge } from "@/components/TyreBadge";
import { TaskLedger } from "@/components/overview/TaskLedger";
import { PipelineHealth } from "@/components/overview/PipelineHealth";
import { DriverErrorTable } from "@/components/overview/DriverErrorTable";
import { fmt } from "@/lib/format";

/**
 * Overview — a photographic hero stage carrying the one focal readout, then
 * the task ledger and pipeline health as glass cards over the same image,
 * then the dense evidence (driver error table) below on a plain surface.
 *
 * The focal figure is still the A* optimal strategy cost — that has not
 * changed, only its stage has. The hero photo and the cutout car are real,
 * licensed photography (credited in the footer), not decoration standing in
 * for data: every number layered over them is still read live from the API,
 * exactly as before.
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
      {/* ---- hero stage: real photography, the focal readout, glass cards --- */}
      <section className="hero-stage mt-8 min-h-[560px]" style={{ ["--hero-focus" as string]: "32% 42%" }}>
        <Image
          src="/images/photo-car-silhouette.jpg"
          alt=""
          fill
          priority
          sizes="100vw"
          className="hero-stage__photo"
        />
        <div className="hero-stage__ember" aria-hidden />
        <div className="hero-stage__veil" aria-hidden />

        <div className="hero-stage__content flex min-h-[560px] flex-col justify-between p-7 md:p-12">
          <div className="flex flex-wrap items-start justify-between gap-6">
            {manifest && <DatasetBadge source={manifest.dataset_source} />}
            <Image
              src="/images/sticker-car-redbull.png"
              alt="A Formula 1 car mid-corner, photographed at Circuit Paul Ricard"
              width={520}
              height={375}
              className="sticker hidden w-[clamp(260px,32vw,520px)] md:block"
              priority
            />
          </div>

          <div className="max-w-2xl">
            {plan ? (
              <>
                <p className="stat-label text-paper-400">Optimal strategy cost, remaining race — A* search</p>
                <p className="t-display-hero mt-1.5 text-paper-900">
                  {fmt(plan.final_cost_seconds, 2)}
                  <span className="ml-2 font-display text-[22px] font-semibold not-italic text-accent">s</span>
                </p>
                <p className="t-body mt-3.5 max-w-[48ch] text-[13px] text-paper-700">
                  {stop ? (
                    <>
                      One stop, lap {stop.lap}, onto <TyreBadge compound={stop.compound} />.{" "}
                    </>
                  ) : (
                    <>No stop on this instance. </>
                  )}
                  A* and uniform-cost search agree to four decimals, which is the admissibility check — A* reached
                  it expanding{" "}
                  <span className="tabular-nums text-paper-900">
                    {plan.algorithms.find((a) => a.algorithm === "A*")?.nodes_expanded}
                  </span>{" "}
                  nodes against UCS&rsquo;s{" "}
                  <span className="tabular-nums text-paper-900">
                    {plan.algorithms.find((a) => a.algorithm === "UCS")?.nodes_expanded}
                  </span>
                  .
                </p>
              </>
            ) : (
              <>
                <p className="stat-label text-paper-400">Optimal strategy cost, remaining race</p>
                <p className="t-display-md mt-1.5 text-paper-500">Not generated yet</p>
                <p className="t-micro mt-2">
                  Run <span className="text-paper-400">python scripts/run_search.py</span> to produce it.
                </p>
              </>
            )}

            <div className="mt-7 flex flex-wrap gap-3">
              <div className="glass-card readout px-4 py-3">
                <p className="readout__value">{laptimeDnn !== undefined ? fmt(laptimeDnn, 3) : "—"}</p>
                <p className="readout__label">Lap error, s</p>
              </div>
              <div className="glass-card readout px-4 py-3">
                <p className="readout__value">{expert?.available ? expert.n_rules : "—"}</p>
                <p className="readout__label">Expert rules</p>
              </div>
              <div className="glass-card readout px-4 py-3">
                <p className="readout__value">{health?.model_count ?? "—"}</p>
                <p className="readout__label">Models trained</p>
              </div>
              <div className="glass-card readout px-4 py-3">
                <p className="readout__value">
                  {evidence.completed_count}
                  <span className="text-[16px] text-paper-500">/{evidence.total_count}</span>
                </p>
                <p className="readout__label">Tasks built</p>
              </div>
            </div>
          </div>
        </div>
      </section>
      <p className="t-micro mt-2 text-right">
        Photo: Circuit Paul Ricard, via Unsplash — licensed stock, not this project&rsquo;s own session.
      </p>

      {/* ---- ledger + pipeline health, now beside the hero on wide screens -- */}
      <div className="mt-10 grid gap-x-14 gap-y-10 border-b border-track-300 pb-10 lg:grid-cols-2">
        <TaskLedger evidence={evidence} />
        {health && <PipelineHealth health={health} />}
      </div>

      {/* ---- supporting evidence ------------------------------------------- */}
      <div className="grid gap-x-14 gap-y-12 pt-10 lg:grid-cols-[1.55fr_1fr] [&>*]:min-w-0">
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

        <section aria-labelledby="what-h" className="lg:sticky lg:top-[76px] lg:self-start">
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
