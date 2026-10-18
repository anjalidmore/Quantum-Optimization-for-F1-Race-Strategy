"use client";

import { ArchitectureFlow } from "@/components/comparison/ArchitectureFlow";
import { ModelIcon } from "@/components/comparison/ModelComparison";
import type { CardFact, ModelCard } from "@/lib/modelCards";
import { MODEL_KEYS, MODEL_ROLE, type ModelKey, type TargetComparison } from "@/lib/modelComparison";
import { RaiHead } from "./ui";

const INTERP_STEP: Record<ModelCard["interpretability"]["level"], number> = { Direct: 3, Indirect: 2, "Post-hoc only": 1 };

export function ModelCardsSection({ c, cards }: { c: TargetComparison; cards: Record<ModelKey, ModelCard> }) {
  const maxParams = Math.max(...MODEL_KEYS.map((k) => c.models[k].parameters ?? 0));
  return (
    <div className="space-y-10">
      {/* ---- at-a-glance comparison ----------------------------------------- */}
      <section aria-labelledby="cmp-h">
        <RaiHead id="cmp-h" kicker="At a glance" title="How the three models differ" />
        <div className="mt-5 overflow-x-auto">
          <table className="w-full min-w-[720px] border-separate border-spacing-0 text-[13px]">
            <caption className="sr-only">Model comparison: complexity, interpretability, use and main limitation</caption>
            <thead>
              <tr>
                <th scope="col" className="w-[150px]" />
                {MODEL_KEYS.map((k) => (
                  <th key={k} scope="col" className="px-3 pb-3 text-left" style={{ ["--m" as string]: MODEL_ROLE[k].color }}>
                    <span className="flex items-center gap-2">
                      <span className="mc-icon !h-8 !w-8"><ModelIcon k={k} model={c.models[k].model} /></span>
                      <span>
                        <span className="block font-display text-[15px] font-bold uppercase tracking-[0.04em] text-paper-900">{MODEL_ROLE[k].label}</span>
                        <span className="t-micro block font-normal">{c.models[k].algorithm}</span>
                      </span>
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="[&>tr>*]:border-t [&>tr>*]:border-track-300 [&>tr>*]:px-3 [&>tr>*]:py-3 [&>tr>*]:align-top">
              <tr>
                <th scope="row" className="text-left font-medium text-paper-500">Complexity</th>
                {MODEL_KEYS.map((k) => {
                  const p = c.models[k].parameters;
                  return (
                    <td key={k} style={{ ["--m" as string]: MODEL_ROLE[k].color }}>
                      <p className="t-data text-paper-900">{p !== null ? `${p.toLocaleString()} parameters` : "Parameter count N/A"}</p>
                      {p !== null && maxParams > 0 && (
                        <div className="mc-meter mt-1.5" title="log scale">
                          <span style={{ width: `${Math.max(4, (Math.log10(p + 1) / Math.log10(maxParams + 1)) * 100)}%` }} />
                        </div>
                      )}
                      <p className="t-micro mt-1">{c.models[k].capacity}</p>
                    </td>
                  );
                })}
              </tr>
              <tr>
                <th scope="row" className="text-left font-medium text-paper-500">Interpretability</th>
                {MODEL_KEYS.map((k) => {
                  const it = cards[k].interpretability;
                  return (
                    <td key={k} style={{ ["--m" as string]: MODEL_ROLE[k].color }}>
                      <p className="flex items-center gap-2 font-semibold text-paper-900">
                        <span className="flex gap-0.5" aria-hidden>
                          {[1, 2, 3].map((i) => (
                            <span key={i} className="h-3 w-1.5 rounded-[1px]"
                                  style={{ backgroundColor: i <= INTERP_STEP[it.level] ? "var(--m)" : "var(--track-300)" }} />
                          ))}
                        </span>
                        {it.level}
                      </p>
                      <p className="t-micro mt-1">{it.how}</p>
                    </td>
                  );
                })}
              </tr>
              <tr>
                <th scope="row" className="text-left font-medium text-paper-500">Role in the system</th>
                {MODEL_KEYS.map((k) => <td key={k} className="text-paper-700">{cards[k].servedBy}</td>)}
              </tr>
              <tr>
                <th scope="row" className="text-left font-medium text-paper-500">Main limitation</th>
                {MODEL_KEYS.map((k) => <td key={k} className="text-paper-700">{cards[k].limitations[0] ?? "None measured on this holdout."}</td>)}
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* ---- the three cards ------------------------------------------------ */}
      <section aria-labelledby="cards-h">
        <RaiHead id="cards-h" kicker="Model cards" title="Full documentation, one card per model" />
        <div className="mt-5 grid gap-4 xl:grid-cols-3">
          {MODEL_KEYS.map((k) => <Card key={k} c={c} card={cards[k]} />)}
        </div>
      </section>
    </div>
  );
}

function Card({ c, card }: { c: TargetComparison; card: ModelCard }) {
  const k = card.key;
  const role = MODEL_ROLE[k];
  const m = c.models[k];
  return (
    <article className="mc-card flex flex-col" style={{ ["--m" as string]: role.color }}>
      <div className="mc-card__stripe" aria-hidden />
      <div className="flex flex-1 flex-col p-5">
        <header className="flex items-start gap-3">
          <span className="mc-icon"><ModelIcon k={k} model={m.model} /></span>
          <div className="min-w-0">
            <h3 className="font-display text-[20px] font-bold uppercase leading-tight tracking-[0.03em] text-paper-900">{role.label}</h3>
            <p className="t-micro mt-0.5">{card.type}{m.model && <code className="ml-1.5 text-paper-500">{m.model}</code>}</p>
          </div>
        </header>

        <p className="t-body mt-4 text-[13px]"><span className="font-semibold text-paper-900">Purpose. </span>{card.purpose}</p>
        <p className="t-body mt-2 text-[13px]"><span className="font-semibold text-paper-900">Intended use. </span>{card.intendedUse}</p>

        <div className="mt-4 divide-y divide-track-300 border-y border-track-300">
          <Fold title="Architecture / algorithm" icon="◇" open>
            <p className="t-body text-[13px]">{m.approach}</p>
            <div className="mt-3">
              <ArchitectureFlow c={c} only={k} />
            </div>
          </Fold>
          <Fold title="Training data & preprocessing" icon="▤">
            <Facts facts={card.data} />
            <p className="t-body mt-2 text-[12px]">{card.preprocessing}</p>
          </Fold>
          <Fold title="Evaluation" icon="◎">
            <p className="t-micro mb-1.5">Held-out test laps (identical for all three models)</p>
            <Facts facts={card.evaluation} cols />
            {card.cv.length > 0 && (
              <>
                <p className="t-micro mb-1.5 mt-3">Cross-validation, mean ± std over folds</p>
                <Facts facts={card.cv} cols />
              </>
            )}
            {k === "dnn" && <p className="t-micro mt-2">The network was tuned on folds but no per-fold test metrics are recorded for it.</p>}
          </Fold>
          <Fold title="Strengths & limitations" icon="±">
            <List items={card.strengths} mark="+" tone="good" />
            <List items={card.limitations} mark="−" tone="bad" />
          </Fold>
          <Fold title="Risks & unsuitable uses" icon="!">
            <List items={card.risks} mark="!" tone="bad" />
          </Fold>
          <Fold title="Training configuration & version" icon="⚙">
            <Facts facts={card.config} />
            <div className="mt-2 border-t border-track-300 pt-2"><Facts facts={card.version} /></div>
          </Fold>
        </div>
      </div>
    </article>
  );
}

function Fold({ title, icon, open, children }: { title: string; icon: string; open?: boolean; children: React.ReactNode }) {
  return (
    <details className="group" open={open}>
      <summary className="flex cursor-pointer list-none items-center gap-2.5 py-2.5 text-[13px] font-semibold text-paper-900 hover:text-[var(--papaya)]">
        <span aria-hidden className="grid h-5 w-5 place-items-center rounded-sm bg-track-200 text-[11px] text-paper-700">{icon}</span>
        {title}
        <span aria-hidden className="ml-auto text-paper-500 transition-transform duration-[180ms] group-open:rotate-90">›</span>
      </summary>
      <div className="pb-3">{children}</div>
    </details>
  );
}

function Facts({ facts, cols }: { facts: CardFact[]; cols?: boolean }) {
  return (
    <dl className={cols ? "grid grid-cols-2 gap-x-4 gap-y-1.5" : "space-y-1.5"}>
      {facts.map((f) => (
        <div key={f.label} className={cols ? "" : "grid grid-cols-[110px_1fr] gap-2"}>
          <dt className="text-[12px] capitalize text-paper-500">{f.label}</dt>
          <dd className="t-data min-w-0 text-[12px] text-paper-900 [overflow-wrap:anywhere]">{f.value}</dd>
        </div>
      ))}
    </dl>
  );
}

function List({ items, mark, tone }: { items: string[]; mark: string; tone: "good" | "bad" }) {
  if (!items.length) return null;
  return (
    <ul className="mt-1 space-y-1.5">
      {items.map((t) => (
        <li key={t} className="flex gap-2 text-[12px] text-paper-700">
          <span aria-hidden className={`mt-px w-3 shrink-0 font-bold ${tone === "good" ? "text-[var(--m)]" : "text-[var(--papaya)]"}`}>{mark}</span>
          {t}
        </li>
      ))}
    </ul>
  );
}
