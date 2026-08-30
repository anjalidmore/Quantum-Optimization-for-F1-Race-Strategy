import { api, ApiError, artifactUrl } from "@/lib/api";
import { DatasetBadge } from "@/components/DatasetBadge";
import { ArtifactImage } from "@/components/ArtifactImage";

function basename(path: string): string {
  return path.split("/").pop() ?? path;
}

function prettify(filename: string): string {
  return filename
    .replace(/\.[a-z0-9]+$/i, "")
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export default async function DataAnalysisPage() {
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
    <div className="space-y-8">
      <section>
        <div className="flex items-center gap-3 flex-wrap">
          <h1 className="text-2xl font-bold text-white">Data & Analysis</h1>
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
                  <a href={artifactUrl(r)} target="_blank" rel="noreferrer" className="text-sky-400 hover:underline text-sm">
                    {prettify(basename(r))} ↗
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </section>
      )}
    </div>
  );
}
