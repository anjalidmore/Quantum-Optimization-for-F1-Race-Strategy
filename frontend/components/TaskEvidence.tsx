import { TaskEvidence, artifactUrl } from "@/lib/api";
import { basename, prettify } from "@/lib/format";
import { ArtifactImage } from "@/components/ArtifactImage";

export function TaskArtifactList({ task }: { task: TaskEvidence }) {
  const hasArtifacts = task.reports.length + task.figures.length + task.other_artifacts.length > 0;

  if (!hasArtifacts) {
    return <p className="text-sm text-paper-400">Artifact not generated yet.</p>;
  }

  return (
    <div className="space-y-4">
      {task.figures.length > 0 && (
        <div>
          <div className="stat-label mb-2">Figures</div>
          <div className="grid grid-cols-2 gap-2">
            {task.figures.map((f) => (
              <ArtifactImage key={f} src={artifactUrl(f)} alt={prettify(basename(f))} className="rounded border border-track-300" />
            ))}
          </div>
        </div>
      )}
      {task.reports.length > 0 && (
        <div>
          <div className="stat-label mb-2">Reports</div>
          <ul className="space-y-1">
            {task.reports.map((r) => (
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
      )}
      {task.other_artifacts.length > 0 && (
        <div>
          <div className="stat-label mb-2">Other generated artifacts</div>
          <ul className="space-y-1">
            {task.other_artifacts.map((a) => (
              <li key={a} className="text-sm text-paper-500 font-mono text-xs">
                {a}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
