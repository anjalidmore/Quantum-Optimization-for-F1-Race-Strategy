import Link from "next/link";
import { api, ApiError } from "@/lib/api";
import { DatasetBadge } from "@/components/DatasetBadge";
import { TyreBadge } from "@/components/TyreBadge";
import { DemoPath } from "@/components/overview/DemoPath";
import { TaskLedger } from "@/components/overview/TaskLedger";
import { PipelineHealth } from "@/components/overview/PipelineHealth";
import { fmt } from "@/lib/format";

/**
 * Overview: what the project is, its headline results, a route through the
 * dashboard for a demonstration, and which tasks and models are live. Every
 * figure is read from the API; a missing artifact shows "—", never a zero.
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

  const [search, expert, dlMetrics] = await Promise.all([
    api.searchComparison().catch(() => null),
    api.expertSystem().catch(() => null),
    api.dlMetrics().catch(() => null),
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

  const stats = [
    { label: "Optimal strategy cost (A*)", value: plan ? `${fmt(plan.final_cost_seconds, 2)} s` : "—" },
    { label: "Lap-time error (DNN, MAE)", value: laptimeDnn !== undefined ? `${fmt(laptimeDnn, 3)} s` : "—" },
    { label: "Expert rules", value: expert?.available ? String(expert.n_rules) : "—" },
    { label: "Models trained", value: health?.model_count !== undefined ? String(health.model_count) : "—" },
    { label: "Tasks built", value: `${evidence.completed_count} / ${evidence.total_count}` },
  ];

  return (
    <div>
      <section className="pt-10">
        {manifest && <DatasetBadge source={manifest.dataset_source} />}
        <h1 className="t-page-title mt-4 !text-[clamp(28px,3.4vw,40px)]">When should a Formula 1 driver pit, and what will the next lap cost?</h1>
        <p className="t-body mt-3 max-w-measure">
          One real Grand Prix, answered three ways: an ontology, a rule base and a search that reason symbolically,
          models that learn from its laps, and explainability tools that say why each call was made and where it stops
          being trustworthy. Every figure on every page is read from a generated artifact.
        </p>
        <div className="mt-5">
          <Link href="/strategy" className="btn-primary inline-block">
            Try the strategy simulator
          </Link>
        </div>
      </section>

      <dl className="mt-8 grid grid-cols-2 gap-3 md:grid-cols-5">
        {stats.map((s) => (
          <div key={s.label} className="card">
            <dt className="stat-label">{s.label}</dt>
            <dd className="stat-value mt-1">{s.value}</dd>
          </div>
        ))}
      </dl>

      {plan && (
        <p className="t-body mt-4 max-w-measure text-paper-500">
          {stop ? (
            <>
              Best plan: one stop, lap {stop.lap}, onto <TyreBadge compound={stop.compound} />.{" "}
            </>
          ) : (
            <>Best plan: no stop on this instance. </>
          )}
          A* and uniform-cost search agree, which is the admissibility check; A* got there expanding{" "}
          <span className="tabular-nums">{plan.algorithms.find((a) => a.algorithm === "A*")?.nodes_expanded}</span> nodes
          against UCS&rsquo;s{" "}
          <span className="tabular-nums">{plan.algorithms.find((a) => a.algorithm === "UCS")?.nodes_expanded}</span>.
        </p>
      )}

      <DemoPath />

      <div className="mt-10 grid gap-x-14 gap-y-10 lg:grid-cols-2">
        <TaskLedger evidence={evidence} />
        {health && <PipelineHealth health={health} />}
      </div>
    </div>
  );
}
