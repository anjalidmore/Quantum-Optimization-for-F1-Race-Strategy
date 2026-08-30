"use client";

import { useEffect, useState } from "react";
import { api, ApiError, DataOptions, Registry } from "@/lib/api";
import { FullScenarioTab } from "@/components/strategy/FullScenarioTab";
import { TopFeaturesTab } from "@/components/strategy/TopFeaturesTab";

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
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Race Strategy Simulator</h1>
        <p className="text-white/60 mt-1 max-w-2xl">
          Enter a race situation and use the trained Task 6 models to predict race behaviour, combined with
          the Task 2 expert system and Task 3 search for a strategy recommendation.
        </p>
      </div>

      {loading && <div className="card text-white/50 text-sm">Loading real driver/team/model data…</div>}

      {error && (
        <div className="card border-red-500/30 bg-red-500/5">
          <div className="badge badge-warning">Backend unreachable</div>
          <p className="text-sm text-white/70 mt-2">{error}</p>
          <p className="text-sm text-white/50 mt-1">
            Start it with <code className="text-white/80">./run.sh</code> or{" "}
            <code className="text-white/80">uvicorn app.api.main:app --reload</code>.
          </p>
        </div>
      )}

      {options && registry && (
        <>
          <div className="flex gap-1 border-b border-white/10">
            <button
              onClick={() => setTab("full")}
              className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px transition ${
                tab === "full" ? "border-f1red text-white" : "border-transparent text-white/50 hover:text-white"
              }`}
            >
              Full Race Scenario
            </button>
            <button
              onClick={() => setTab("top")}
              className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px transition ${
                tab === "top" ? "border-f1red text-white" : "border-transparent text-white/50 hover:text-white"
              }`}
            >
              Top Features
            </button>
          </div>

          {tab === "full" ? <FullScenarioTab options={options} registry={registry} /> : <TopFeaturesTab registry={registry} />}
        </>
      )}
    </div>
  );
}
