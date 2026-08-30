import { api, ApiError } from "@/lib/api";
import { DatasetBadge } from "@/components/DatasetBadge";
import { TaskCard } from "@/components/TaskEvidence";

const WORKFLOW = ["F1 Data", "Data Preparation", "Feature Engineering", "Prediction", "Strategy Analysis", "Optimised Decision"];

export default async function DashboardPage() {
  let health = null;
  let manifest = null;
  let evidence = null;
  let error: string | null = null;

  try {
    [health, manifest, evidence] = await Promise.all([api.health(), api.artifacts().catch(() => null), api.taskEvidence()]);
  } catch (e) {
    error = e instanceof ApiError ? e.message : "Could not reach the backend API.";
  }

  const currentTask = evidence?.tasks.find((t) => t.status !== "completed");

  return (
    <div className="space-y-10">
      <section>
        <h1 className="text-2xl font-bold text-white">F1 Race Strategy Intelligence</h1>
        <p className="text-white/60 mt-1 max-w-2xl">
          Computational Intelligence for race prediction, decision support and strategy optimisation —
          knowledge representation, rule-based reasoning, state-space search, and classical machine
          learning, all working over one shared data and artifact contract.
        </p>
      </section>

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

      {/* Simple workflow */}
      <section>
        <div className="flex flex-wrap items-center gap-2 text-sm">
          {WORKFLOW.map((step, i) => (
            <div key={step} className="flex items-center gap-2">
              <div className="px-3 py-1.5 rounded-lg bg-panel border border-white/10 text-white/80">{step}</div>
              {i < WORKFLOW.length - 1 && <span className="text-white/30">→</span>}
            </div>
          ))}
        </div>
      </section>

      {evidence && (
        <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="card">
            <div className="stat-label">Tasks completed</div>
            <div className="stat-value">
              {evidence.completed_count} / {evidence.total_count}
            </div>
          </div>
          <div className="card">
            <div className="stat-label">Current focus</div>
            <div className="text-lg font-semibold text-white">
              {currentTask ? `Task ${currentTask.number} — ${currentTask.label}` : "All tasks complete"}
            </div>
          </div>
          <div className="card">
            <div className="stat-label">Trained models</div>
            <div className="stat-value">{health?.model_count ?? "—"}</div>
          </div>
          <div className="card">
            <div className="stat-label">Dataset</div>
            {manifest ? (
              <DatasetBadge source={manifest.dataset_source} />
            ) : (
              <div className="text-lg font-semibold text-white">—</div>
            )}
          </div>
        </section>
      )}

      {evidence && (
        <section>
          <h2 className="text-sm uppercase tracking-wider text-white/50 mb-3">
            Computational Intelligence Workflow — Tasks 1–10
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {evidence.tasks.map((t) => (
              <TaskCard key={t.id} task={t} />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
