"use client";

import { useEffect, useState } from "react";
import { api, ApiError, DataOptions, Registry } from "@/lib/api";
import { FullScenarioTab } from "@/components/strategy/FullScenarioTab";
import { TopFeaturesTab } from "@/components/strategy/TopFeaturesTab";
import { PageHero } from "@/components/PageHero";
import { PageTabs } from "@/components/PageTabs";

export default function StrategyPage() {
  const [options, setOptions] = useState<DataOptions | null>(null);
  const [registry, setRegistry] = useState<Registry | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([api.dataOptions(), api.models()])
      .then(([opts, reg]) => {
        setOptions(opts);
        setRegistry(reg);
      })
      .catch((e) => setError(e instanceof ApiError ? e.message : "Could not reach the backend API."))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div>
      <PageHero
        eyebrow="Tasks 2, 3, 6, 7 & 8 — live"
        title="Race strategy"
        blurb="Describe a race situation. The trained Task 6 and Task 7 models predict lap time and pit probability, the Task 2 rules fire over the same state, and the Task 3 search plans the rest of the race — combined into one recommendation."
      />

      {loading && (
        <div className="mt-10" aria-busy="true" aria-live="polite">
          <span className="sr-only">Loading drivers, teams and models</span>
          <div className="skeleton h-[52px] w-full" />
          <div className="mt-6 skeleton h-[260px] w-full" />
        </div>
      )}

      {error && (
        <div className="card mt-10 border-accent">
          <span className="badge badge-danger">Backend unreachable</span>
          <p className="t-body mt-3">{error}</p>
          <p className="t-micro mt-1">
            Start it with <code className="text-paper-500">./run.sh</code> or{" "}
            <code className="text-paper-500">uvicorn app.api.main:app --reload</code>
          </p>
        </div>
      )}

      {options && registry && (
        <PageTabs
          label="Simulator mode"
          tabs={[
            { id: "full", label: "Full race scenario", content: <FullScenarioTab options={options} registry={registry} /> },
            { id: "top", label: "Top features", content: <TopFeaturesTab registry={registry} /> },
          ]}
        />
      )}
    </div>
  );
}
