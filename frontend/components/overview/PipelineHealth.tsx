import { HealthResponse } from "@/lib/api";

const TARGET_LABEL: Record<string, string> = {
  target_laptime: "Lap time",
  target_pit_next_lap: "Pit decision",
};

/**
 * Task 9's pipeline-health panel — which stages of a strategy prediction are
 * actually trained and loadable right now, read from GET /api/health.
 *
 * This is a different question from the Task Ledger beside it: the ledger
 * asks "did Task N ever run", this asks "if I hit Predict right now, which
 * of ML / DL / XAI would actually answer". A model can be built once and
 * later go missing (a retrain interrupted, an artifact moved) without any
 * task ever un-completing, which is exactly the gap this panel closes.
 */
export function PipelineHealth({ health }: { health: HealthResponse }) {
  const models = health.models;
  const rows: { label: string; ok: boolean; detail: string }[] = models
    ? [
        ...Object.entries(models.ml).map(([target, m]) => ({
          label: `ML — ${TARGET_LABEL[target] ?? target}`,
          ok: m.trained,
          detail: m.selected_model ?? "not trained",
        })),
        ...Object.entries(models.dl).map(([target, m]) => ({
          label: `DL — ${TARGET_LABEL[target] ?? target}`,
          ok: m.trained,
          detail: m.trained ? "dnn_mlp" : "not trained",
        })),
        { label: "XAI (Task 8)", ok: models.xai_available, detail: models.xai_available ? "available" : "not generated" },
      ]
    : [];

  return (
    <section aria-labelledby="pipeline-health-h">
      <h2 id="pipeline-health-h" className="t-title mb-3">
        Pipeline health
      </h2>
      {!models ? (
        <p className="t-body text-paper-500">
          Per-model status not reported by this backend version.
        </p>
      ) : (
        <div className="border-t border-track-300">
          {rows.map((r) => (
            <div
              key={r.label}
              className="grid grid-cols-[1fr_auto] items-center gap-3 border-b border-track-300 py-2 text-[13px]"
            >
              <span className="text-paper-700">{r.label}</span>
              <span className="flex items-center gap-2 text-[11px] text-paper-500">
                <span aria-hidden className={`block h-[7px] w-[7px] ${r.ok ? "bg-paper-900" : "border border-edge"}`} />
                <span className="tabular-nums">{r.detail}</span>
                <span className="sr-only">{r.ok ? "available" : "unavailable"}</span>
              </span>
            </div>
          ))}
        </div>
      )}
      <p className="t-micro mt-3">
        Read live from <span className="text-paper-500">/api/health</span> — what{" "}
        <span className="text-paper-500">POST /api/strategy/predict</span> can actually serve right now.
      </p>
    </section>
  );
}
