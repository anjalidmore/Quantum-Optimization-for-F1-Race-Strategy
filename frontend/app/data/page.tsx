import { api, ApiError, artifactUrl } from "@/lib/api";
import { DatasetBadge } from "@/components/DatasetBadge";
import { ArtifactImage } from "@/components/ArtifactImage";
import { TaskArtifactList } from "@/components/TaskEvidence";
import { basename, prettify } from "@/lib/format";

const STATUS_LABEL: Record<string, string> = {
  completed: "Completed",
  in_progress: "In Progress",
  upcoming: "Upcoming",
};

const STATUS_CLASS: Record<string, string> = {
  completed: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  in_progress: "bg-amber-500/15 text-amber-300 border-amber-500/30",
  upcoming: "bg-white/10 text-white/50 border-white/20",
};

/**
 * Task 4's cleaning and EDA output, followed by the evidence table for every
 * task. Both answer the same question — "what has this project actually
 * generated?" — so they live on one page, read live from artifacts/.
 */
export default async function DataPage() {
  let evidence = null;
  let manifest = null;
  let error: string | null = null;

  try {
    [evidence, manifest] = await Promise.all([api.taskEvidence(), api.artifacts().catch(() => null)]);
  } catch (e) {
    error = e instanceof ApiError ? e.message : "Could not reach the backend API.";
  }

  if (error || !evidence) {
    return (
      <div className="card border-red-500/30 bg-red-500/5">
        <div className="badge badge-warning">Backend unreachable</div>
        <p className="text-sm text-white/70 mt-2">{error}</p>
      </div>
    );
  }

  const task4 = evidence.tasks.find((t) => t.id === "task4")!;
  const hasArtifacts = task4.figures.length + task4.reports.length > 0;

  return (
    <div className="space-y-10">
      <section>
        <div className="flex items-center gap-3 flex-wrap">
          <h1 className="text-2xl font-bold text-white">Data &amp; Evidence</h1>
          {manifest && <DatasetBadge source={manifest.dataset_source} />}
        </div>
        <p className="text-white/60 mt-1 max-w-2xl">{task4.purpose}</p>
      </section>

      {!hasArtifacts && (
        <div className="card text-white/50 text-sm">
          Artifact not generated yet. Run <code className="text-white/80">python scripts/build_all.py</code>.
        </div>
      )}

      {task4.figures.length > 0 && (
        <section>
          <h2 className="text-sm uppercase tracking-wider text-white/50 mb-3">Generated Figures</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {task4.figures.map((f) => (
              <div key={f} className="card">
                <ArtifactImage src={artifactUrl(f)} alt={prettify(basename(f))} className="w-full rounded" />
                <div className="text-sm text-white/60 mt-2">{prettify(basename(f))}</div>
              </div>
            ))}
          </div>
        </section>
      )}

      {task4.reports.length > 0 && (
        <section>
          <h2 className="text-sm uppercase tracking-wider text-white/50 mb-3">Generated Reports</h2>
          <div className="card">
            <ul className="space-y-2">
              {task4.reports.map((r) => (
                <li key={r}>
                  <a
                    href={artifactUrl(r)}
                    target="_blank"
                    rel="noreferrer"
                    className="text-sky-400 hover:underline text-sm"
                  >
                    {prettify(basename(r))} ↗
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </section>
      )}

      <section>
        <h2 className="text-lg font-semibold text-white">Project evidence</h2>
        <p className="text-white/60 mt-1 mb-4 max-w-2xl text-sm">
          Every task in the laboratory workflow, mapped to what has actually been built here —{" "}
          {evidence.completed_count} of {evidence.total_count} complete. Nothing below is a placeholder:
          artifacts are read live from <code className="text-white/80">artifacts/</code>.
        </p>
        <div className="space-y-4">
          {evidence.tasks.map((task) => (
            <details key={task.id} className="card" open={task.status !== "upcoming"}>
              <summary className="cursor-pointer flex items-center justify-between">
                <div>
                  <span className="text-[10px] text-white/40 uppercase tracking-wider mr-2">
                    Task {task.number}
                  </span>
                  <span className="font-semibold text-white">{task.label}</span>
                </div>
                <span className={`badge border ${STATUS_CLASS[task.status]}`}>{STATUS_LABEL[task.status]}</span>
              </summary>
              <p className="text-sm text-white/60 mt-3 mb-4">{task.purpose}</p>
              <TaskArtifactList task={task} />
            </details>
          ))}
        </div>
      </section>
    </div>
  );
}
