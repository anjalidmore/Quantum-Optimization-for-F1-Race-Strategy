import { api, ApiError, artifactUrl } from "@/lib/api";
import { DatasetBadge } from "@/components/DatasetBadge";
import { ArtifactImage } from "@/components/ArtifactImage";
import { PageHero } from "@/components/PageHero";
import { PageTabs } from "@/components/PageTabs";
import { TaskArtifactList } from "@/components/TaskEvidence";
import { basename, prettify } from "@/lib/format";

const STATUS_LABEL: Record<string, string> = {
  completed: "Completed",
  in_progress: "In Progress",
  upcoming: "Upcoming",
};

const STATUS_CLASS: Record<string, string> = {
  completed: "badge-success",
  in_progress: "badge-warning",
  upcoming: "badge-info",
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
      <div className="card border-accent">
        <div className="badge badge-warning">Backend unreachable</div>
        <p className="text-sm text-paper-700 mt-2">{error}</p>
      </div>
    );
  }

  const task4 = evidence.tasks.find((t) => t.id === "task4")!;
  const hasArtifacts = task4.figures.length + task4.reports.length > 0;

  const tabs = [
    ...(task4.figures.length > 0
      ? [
          {
            id: "figures",
            label: "Analysis figures",
            content: (
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                {task4.figures.map((f) => (
                  <div key={f} className="card">
                    <ArtifactImage src={artifactUrl(f)} alt={prettify(basename(f))} className="w-full rounded" />
                    <div className="mt-2 text-sm text-paper-500">{prettify(basename(f))}</div>
                  </div>
                ))}
              </div>
            ),
          },
        ]
      : []),
    ...(task4.reports.length > 0
      ? [
          {
            id: "reports",
            label: "Reports",
            content: (
              <div className="card">
                <ul className="space-y-2">
                  {task4.reports.map((r) => (
                    <li key={r}>
                      <a
                        href={artifactUrl(r)}
                        target="_blank"
                        rel="noreferrer"
                        className="text-sm text-paper-900 underline decoration-edge underline-offset-[3px] hover:decoration-paper-900"
                      >
                        {prettify(basename(r))}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            ),
          },
        ]
      : []),
    {
      id: "evidence",
      label: "All task evidence",
      content: (
        <section>
          <p className="mb-4 max-w-2xl text-sm text-paper-500">
            Every task in the laboratory workflow, mapped to what has actually been built here —{" "}
            {evidence.completed_count} of {evidence.total_count} complete. Open a task to see its artifacts, read live
            from <code className="text-paper-700">artifacts/</code>.
          </p>
          <div className="space-y-3">
            {evidence.tasks.map((task) => (
              <details key={task.id} className="card">
                <summary className="flex cursor-pointer items-center justify-between">
                  <div>
                    <span className="mr-2 text-[10px] uppercase tracking-wider text-paper-400">Task {task.number}</span>
                    <span className="font-semibold text-paper-900">{task.label}</span>
                  </div>
                  <span className={`badge ${STATUS_CLASS[task.status]}`}>{STATUS_LABEL[task.status]}</span>
                </summary>
                <p className="mb-4 mt-3 text-sm text-paper-500">{task.purpose}</p>
                <TaskArtifactList task={task} />
              </details>
            ))}
          </div>
        </section>
      ),
    },
  ];

  return (
    <div className="pb-10">
      <PageHero eyebrow="Task 4" title="Data & Evidence" blurb={task4.purpose} />
      {manifest && (
        <div className="mt-6">
          <DatasetBadge source={manifest.dataset_source} />
        </div>
      )}

      {!hasArtifacts && (
        <div className="card mt-6 text-sm text-paper-500">
          Artifact not generated yet. Run <code className="text-paper-700">python scripts/build_all.py</code>.
        </div>
      )}

      <PageTabs label="Data sections" tabs={tabs} />
    </div>
  );
}
