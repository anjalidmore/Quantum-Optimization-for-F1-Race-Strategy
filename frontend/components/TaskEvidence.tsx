"use client";

import { useState } from "react";
import { TaskEvidence, artifactUrl } from "@/lib/api";
import { basename, prettify } from "@/lib/format";
import { Modal } from "@/components/Modal";
import { ArtifactImage } from "@/components/ArtifactImage";

function StatusBadge({ status }: { status: TaskEvidence["status"] }) {
  const map: Record<TaskEvidence["status"], { label: string; cls: string; mark: string }> = {
    // Status reads from the marker shape and the word, not from a hue: the
    // detector flags emerald as the reflex accent, and it carried no meaning
    // a filled square does not.
    completed: { label: "Built", cls: "badge-success", mark: "full" },
    in_progress: { label: "In progress", cls: "badge-warning", mark: "half" },
    upcoming: { label: "Not built", cls: "badge-info", mark: "empty" },
  };
  const { label, cls, mark } = map[status];
  return (
    <span className={`badge ${cls}`}>
      <span
        aria-hidden
        className={`block h-[7px] w-[7px] ${
          mark === "full" ? "bg-paper-900" : mark === "half" ? "bg-accent" : "border border-edge"
        }`}
      />
      {label}
    </span>
  );
}

export function TaskCard({ task }: { task: TaskEvidence }) {
  const [open, setOpen] = useState(false);
  const hasArtifacts = task.reports.length + task.figures.length + task.other_artifacts.length > 0;

  return (
    <>
      <button
        onClick={() => hasArtifacts && setOpen(true)}
        className={`card w-full text-left transition-colors duration-[120ms] ${hasArtifacts ? "cursor-pointer hover:border-edge" : "cursor-default opacity-70"}`}
      >
        <div className="flex items-start justify-between">
          <div className="font-display text-[13px] font-semibold text-paper-400">Task {task.number}</div>
          <StatusBadge status={task.status} />
        </div>
        <div className="mt-1 text-sm font-semibold text-paper-900">{task.label}</div>
        <div className="mt-1.5 line-clamp-3 text-xs text-paper-500">{task.purpose}</div>
        {hasArtifacts && <div className="mt-2 text-[11px] text-paper-500 underline decoration-track-300">View evidence</div>}
      </button>

      {open && (
        <Modal title={`Task ${task.number} — ${task.label}`} onClose={() => setOpen(false)}>
          <p className="mb-4 text-sm text-paper-700">{task.purpose}</p>
          <TaskArtifactList task={task} />
        </Modal>
      )}
    </>
  );
}

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
