import Link from "next/link";
import { TaskEvidenceResponse } from "@/lib/api";

/** The page where each task's work can be seen. */
const TASK_PAGE: Record<string, { href: string; label: string }> = {
  task1: { href: "/reasoning", label: "Reasoning" },
  task2: { href: "/reasoning", label: "Reasoning" },
  task3: { href: "/reasoning", label: "Reasoning" },
  task4: { href: "/data", label: "Data" },
  task5: { href: "/data", label: "Data" },
  task6: { href: "/models", label: "Models" },
  task7: { href: "/models", label: "Models" },
  task8: { href: "/explainability", label: "Explainability" },
  task9: { href: "/strategy", label: "Strategy" },
  task10: { href: "/rai", label: "Responsible AI" },
};

const STATUS = { completed: "Built", in_progress: "In progress", upcoming: "Not built" };

/** The ten lab tasks, with status from a live scan of artifacts/ and docs/. */
export function TaskLedger({ evidence }: { evidence: TaskEvidenceResponse }) {
  return (
    <section aria-labelledby="ledger-h">
      <h2 id="ledger-h" className="t-title mb-3">
        Lab tasks
      </h2>
      <div className="border-t border-track-300">
        {evidence.tasks.map((task) => {
          const page = TASK_PAGE[task.id];
          return (
            <div
              key={task.id}
              className="grid grid-cols-[24px_1fr_auto_72px] items-center gap-3 border-b border-track-300 py-2 text-[13px]"
            >
              <span className="text-right font-display text-[13px] font-semibold text-paper-400">{task.number}</span>
              <span className="text-paper-700">{task.label}</span>
              {page ? (
                <Link href={page.href} className="text-[11px] text-paper-500 underline decoration-track-300 underline-offset-[3px] hover:text-paper-900">
                  {page.label}
                </Link>
              ) : (
                <span />
              )}
              <span className={`text-right text-[11px] ${task.status === "completed" ? "text-paper-700" : "text-accent"}`}>
                {STATUS[task.status]}
              </span>
            </div>
          );
        })}
      </div>
      <p className="t-micro mt-3">
        Read live from <span className="text-paper-500">/api/tasks/evidence</span> — {evidence.completed_count} of{" "}
        {evidence.total_count} built
      </p>
    </section>
  );
}
