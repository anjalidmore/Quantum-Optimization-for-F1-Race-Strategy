import Link from "next/link";
import { api, ApiError } from "@/lib/api";
import { DatasetBadge } from "@/components/DatasetBadge";
import { DemoPath } from "@/components/overview/DemoPath";
import { RaceStrip } from "@/components/overview/RaceStrip";
import { ChequeredFlag, F1Car, Icons } from "@/components/overview/F1Art";
import { fmt } from "@/lib/format";

/**
 * Overview: the question, the headline results and a way into each page, with
 * the best plan drawn on the track below. Every figure is read from the API; a
 * missing artifact shows "—", never a zero.
 */
export default async function OverviewPage() {
  let health = null;
  let manifest = null;
  let error: string | null = null;

  try {
    [health, manifest] = await Promise.all([api.health(), api.artifacts().catch(() => null)]);
  } catch (e) {
    error = e instanceof ApiError ? e.message : "Could not reach the backend API.";
  }

  const [search, expert, dlMetrics] = await Promise.all([
    api.searchComparison().catch(() => null),
    api.expertSystem().catch(() => null),
    api.dlMetrics().catch(() => null),
  ]);

  if (error || !health) {
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
  const laptimeDnn = dlMetrics?.models?.target_laptime?.test_metrics?.mae;

  const stats = [
    { label: "Optimal plan (A*)", value: plan ? fmt(plan.final_cost_seconds, 2) : "—", unit: plan ? "s" : null, icon: Icons.flag },
    { label: "Lap-time MAE (DNN)", value: laptimeDnn !== undefined ? fmt(laptimeDnn, 3) : "—", unit: laptimeDnn !== undefined ? "s" : null, icon: Icons.stopwatch },
    { label: "Expert rules", value: expert?.available ? String(expert.n_rules) : "—", unit: null, icon: Icons.rules },
    { label: "Models trained", value: health.model_count !== undefined ? String(health.model_count) : "—", unit: null, icon: Icons.chip },
  ];

  return (
    <div className="home-page">
      <section className="home-hero">
        <div className="relative z-10">
          {manifest && <DatasetBadge source={manifest.dataset_source} />}
          <h1 className="home-title">
            When to <br />
            pit?
          </h1>
          <Link href="/strategy" className="btn-primary mt-6 inline-block">
            Strategy simulator
          </Link>
        </div>
        <div className="home-hero__art">
          <F1Car className="home-hero__car" />
          <ChequeredFlag className="home-hero__flag" />
        </div>
      </section>

      <dl className="home-stats">
        {stats.map((s) => (
          <div key={s.label} className="home-stat">
            <span className="text-[var(--papaya)]">{s.icon("h-8 w-8")}</span>
            <div>
              <dd className="stat-value !mt-0">
                {s.value}
                {s.unit && <span className="unit"> {s.unit}</span>}
              </dd>
              <dt className="stat-label mt-0.5">{s.label}</dt>
            </div>
          </div>
        ))}
      </dl>

      <DemoPath />

      <RaceStrip plan={plan} />
    </div>
  );
}
