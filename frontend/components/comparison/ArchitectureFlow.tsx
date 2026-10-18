import { Fragment } from "react";
import { MODEL_KEYS, MODEL_ROLE, type ArchStage, type ModelKey, type TargetComparison } from "@/lib/modelComparison";

/**
 * How the three models turn the same inputs into a prediction, one lane each.
 * Every block comes from the model's real configuration: Task 6's registry
 * hyperparameters, and the layer list Keras wrote to the Task 7 metadata.
 * A dense layer's block height grows with its unit count, so the network's
 * funnel shape is visible at a glance.
 */
export function ArchitectureFlow({ c, only }: { c: TargetComparison; only?: ModelKey }) {
  // `only` draws a single model's flow without its lane label, for a model card.
  if (only) {
    const m = c.models[only];
    return (
      <ol className="flex flex-wrap items-center gap-x-1.5 gap-y-2" style={{ ["--m" as string]: MODEL_ROLE[only].color }}
          aria-label={`${MODEL_ROLE[only].label} pipeline`}>
        {m.stages.map((s, i) => (
          <Fragment key={`${s.title}-${i}`}>
            {i > 0 && <Arrow />}
            <li><Block stage={s} /></li>
          </Fragment>
        ))}
      </ol>
    );
  }
  return (
    <div className="space-y-3">
      {MODEL_KEYS.map((k) => {
        const m = c.models[k];
        const role = MODEL_ROLE[k];
        return (
          <div
            key={k}
            className="grid gap-3 rounded-md border border-track-300 bg-track-100 p-3 md:grid-cols-[150px_1fr] md:items-center"
            style={{ ["--m" as string]: role.color }}
          >
            <div className="flex items-center gap-2.5 md:flex-col md:items-start md:gap-1">
              <span className="mc-swatch" aria-hidden />
              <p className="font-display text-[15px] font-bold uppercase tracking-[0.04em] text-paper-900">{role.label}</p>
              <p className="t-micro">{m.algorithm}</p>
            </div>
            {m.stages.length ? (
              <ol className="flex flex-wrap items-center gap-x-1.5 gap-y-2" aria-label={`${role.label} pipeline`}>
                {m.stages.map((s, i) => (
                  <Fragment key={`${s.title}-${i}`}>
                    {i > 0 && <Arrow />}
                    <li>
                      <Block stage={s} />
                    </li>
                  </Fragment>
                ))}
              </ol>
            ) : (
              <p className="t-body text-[13px] text-paper-500">N/A: model configuration not found.</p>
            )}
          </div>
        );
      })}
    </div>
  );
}

function Block({ stage }: { stage: ArchStage }) {
  const units = stage.kind === "layer" ? Number(stage.title.replace(/\D/g, "")) : 0;
  const height = units ? Math.round(34 + 7 * Math.log2(units)) : undefined;
  return (
    <div className="mc-block" data-kind={stage.kind} style={height ? { minHeight: height } : undefined}>
      <span className="text-[12px] font-semibold leading-tight text-paper-900">{stage.title}</span>
      {stage.detail && <span className="t-micro mt-0.5 leading-tight">{stage.detail}</span>}
    </div>
  );
}

function Arrow() {
  return (
    <svg className="mc-arrow" width="16" height="10" viewBox="0 0 16 10" aria-hidden focusable="false">
      <path d="M0 5h13M9 1l4 4-4 4" stroke="currentColor" strokeWidth="1.4" fill="none" strokeLinecap="round" />
    </svg>
  );
}
