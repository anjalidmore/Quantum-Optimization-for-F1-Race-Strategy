import { api, ApiError } from "@/lib/api";
import { TaskArtifactList } from "@/components/TaskEvidence";

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

export default async function ProjectEvidencePage() {
  let evidence = null;
  let error: string | null = null;

  try {
    evidence = await api.taskEvidence();
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

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-2xl font-bold text-white">Project Evidence</h1>
        <p className="text-white/60 mt-1 max-w-2xl">
          Every computational-intelligence task in the laboratory workflow, mapped to what has actually been
          built and generated in this repository — {evidence.completed_count} of {evidence.total_count}{" "}
          complete. Nothing below is a placeholder: artifacts are read live from{" "}
          <code className="text-white/80">artifacts/</code>.
        </p>
      </section>

      <div className="space-y-4">
        {evidence.tasks.map((task) => (
          <details key={task.id} className="card" open={task.status !== "upcoming"}>
            <summary className="cursor-pointer flex items-center justify-between">
              <div>
                <span className="text-[10px] text-white/40 uppercase tracking-wider mr-2">Task {task.number}</span>
                <span className="font-semibold text-white">{task.label}</span>
              </div>
              <span className={`badge border ${STATUS_CLASS[task.status]}`}>{STATUS_LABEL[task.status]}</span>
            </summary>
            <p className="text-sm text-white/60 mt-3 mb-4">{task.purpose}</p>
            <TaskArtifactList task={task} />
          </details>
        ))}
      </div>
    </div>
  );
}
