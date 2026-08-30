"use client";

import { useState } from "react";
import { TaskEvidence, artifactUrl } from "@/lib/api";
import { Modal } from "@/components/Modal";
import { ArtifactImage } from "@/components/ArtifactImage";

function StatusBadge({ status }: { status: TaskEvidence["status"] }) {
  const map: Record<TaskEvidence["status"], { label: string; cls: string }> = {
    completed: { label: "Completed", cls: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30" },
    in_progress: { label: "In Progress", cls: "bg-amber-500/15 text-amber-300 border-amber-500/30" },
    upcoming: { label: "Upcoming", cls: "bg-white/10 text-white/50 border-white/20" },
  };
  const { label, cls } = map[status];
  return <span className={`badge border ${cls}`}>{label}</span>;
}

function basename(path: string): string {
  return path.split("/").pop() ?? path;
}

function prettify(filename: string): string {
  return filename
    .replace(/\.[a-z0-9]+$/i, "")
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export function TaskCard({ task }: { task: TaskEvidence }) {
  const [open, setOpen] = useState(false);
  const hasArtifacts = task.reports.length + task.figures.length + task.other_artifacts.length > 0;

  return (
    <>
      <button
        onClick={() => hasArtifacts && setOpen(true)}
        className={`card text-left w-full transition ${hasArtifacts ? "hover:border-white/25 cursor-pointer" : "cursor-default opacity-70"}`}
      >
        <div className="flex items-start justify-between">
          <div className="text-[10px] text-white/40 uppercase tracking-wider">Task {task.number}</div>
          <StatusBadge status={task.status} />
        </div>
        <div className="font-semibold text-white text-sm mt-1">{task.label}</div>
        <div className="text-xs text-white/50 mt-1.5 line-clamp-3">{task.purpose}</div>
        {hasArtifacts && <div className="text-xs text-sky-400 mt-2">View evidence →</div>}
      </button>

      {open && (
        <Modal title={`Task ${task.number} — ${task.label}`} onClose={() => setOpen(false)}>
          <p className="text-sm text-white/70 mb-4">{task.purpose}</p>
          <TaskArtifactList task={task} />
        </Modal>
      )}
    </>
  );
}

export function TaskArtifactList({ task }: { task: TaskEvidence }) {
  const hasArtifacts = task.reports.length + task.figures.length + task.other_artifacts.length > 0;

  if (!hasArtifacts) {
    return <p className="text-sm text-white/40">Artifact not generated yet.</p>;
  }

  return (
    <div className="space-y-4">
      {task.figures.length > 0 && (
        <div>
          <div className="stat-label mb-2">Figures</div>
          <div className="grid grid-cols-2 gap-2">
            {task.figures.map((f) => (
              <ArtifactImage key={f} src={artifactUrl(f)} alt={prettify(basename(f))} className="rounded border border-white/10" />
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
                  className="text-sm text-sky-400 hover:underline"
                >
                  {prettify(basename(r))} ↗
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
              <li key={a} className="text-sm text-white/60 font-mono text-xs">
                {a}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
