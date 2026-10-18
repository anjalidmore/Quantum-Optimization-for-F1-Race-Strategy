"use client";

import { useEffect, useState } from "react";
import { api, ApiError, DataOptions, Registry } from "@/lib/api";
import { FullScenarioTab } from "@/components/strategy/FullScenarioTab";
import { TopFeaturesTab } from "@/components/strategy/TopFeaturesTab";
import { PageHero } from "@/components/PageHero";

export default function StrategyPage() {
  const [tab, setTab] = useState<"full" | "top">("full");
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
        photo="/images/photo-pitlane.jpg"
        focus="55% 65%"
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
        <>
          <div role="tablist" aria-label="Simulator mode" className="mt-10 flex gap-1 border-b border-track-300">
            {([["full", "Full race scenario"], ["top", "Top features"]] as const).map(([id, label]) => (
              <button
                key={id}
                role="tab"
                id={`tab-${id}`}
                aria-selected={tab === id}
                aria-controls={`panel-${id}`}
                onClick={() => setTab(id)}
                className={`-mb-px border-b-2 px-4 py-2.5 text-sm font-medium transition-colors duration-[120ms] ${
                  tab === id ? "border-accent text-paper-900" : "border-transparent text-paper-500 hover:text-paper-900"
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          <div role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`} className="mt-8">
            {tab === "full" ? <FullScenarioTab options={options} registry={registry} /> : <TopFeaturesTab registry={registry} />}
          </div>
        </>
      )}
    </div>
  );
}
