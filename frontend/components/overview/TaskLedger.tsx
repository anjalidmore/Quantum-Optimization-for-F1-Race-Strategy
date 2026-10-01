import { TaskEvidenceResponse } from "@/lib/api";

/**
 * The ten tasks as a ledger of dense rows.
 *
 * It replaces a 5x2 grid of ten identical cards: the same information at a
 * third of the height, scannable down a single column, with each task's own
 * headline count on the right instead of truncated prose.
 */
const COUNT: Record<string, (t: TaskEvidenceResponse["tasks"][number]) => string> = {
  task1: () => "61 entities",
  task2: () => "32 rules",
  task3: () => "5 algorithms",
  task4: () => "1055 laps",
  task5: () => "45 / 8 features",
  task6: () => "10 models",
  task7: () => "2 networks",
  task8: () => "360 trust rows",
  task9: () => "6 pages",
  // Documents remaining is read from the API's live scan of docs/ and
  // artifacts/, not typed here — it used to say "6 documents left"
  // unconditionally, which would have gone stale the moment any got written.
  task10: (t) =>
    t.documents_total !== undefined ? `${t.documents_complete}/${t.documents_total} documents` : "documents",
};

export function TaskLedger({ evidence }: { evidence: TaskEvidenceResponse }) {
  return (
    <section aria-labelledby="ledger-h">
      <h2 id="ledger-h" className="t-title mb-3">
        Task state
      </h2>
      <div className="border-t border-track-300">
        {evidence.tasks.map((task) => {
          const built = task.status === "completed";
          const inProgress = task.status === "in_progress";
          return (
            <div
              key={task.id}
              className="grid grid-cols-[24px_1fr_auto] items-center gap-3 border-b border-track-300 py-2 text-[13px]"
            >
              <span className="text-right font-display text-[13px] font-semibold text-paper-400">
                {task.number}
              </span>
              <span className={built ? "text-paper-700" : "text-paper-400"}>{task.label}</span>
              <span className="flex items-center gap-2 text-[11px] text-paper-500">
                <span
                  aria-hidden
                  className={`block h-[7px] w-[7px] ${
                    built ? "bg-paper-900" : inProgress ? "border border-paper-900 bg-track-100" : "border border-edge"
                  }`}
                />
                <span className="tabular-nums">{COUNT[task.id]?.(task) ?? (built ? "built" : "not built")}</span>
                <span className="sr-only">{built ? "built" : inProgress ? "in progress" : "not built"}</span>
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
