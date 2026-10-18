"use client";

import { useEffect, useState } from "react";
import type { RaiSummary } from "@/lib/api";
import type { ModelCard } from "@/lib/modelCards";
import { MODEL_KEYS, MODEL_ROLE, TARGETS, type ModelKey, type Target, type TargetComparison } from "@/lib/modelComparison";
import { ConfidenceSection } from "./ConfidenceSection";
import { FairnessSection } from "./FairnessSection";
import { ModelCardsSection } from "./ModelCardsSection";

export type RaiTargetData = {
  rai: RaiSummary | null;
  raiError: string | null;
  comparison: TargetComparison | null;
  cards: Record<ModelKey, ModelCard> | null;
};

const SECTIONS = [
  { id: "fairness", label: "Bias & Fairness", icon: "M2 13h12M4 13V8M8 13V4M12 13V6" },
  { id: "confidence", label: "Confidence & Uncertainty", icon: "M1 12c2 0 3-8 7-8s5 8 7 8M8 4v9" },
  { id: "cards", label: "Transparency & Model Cards", icon: "M3 2h7l3 3v9H3zM10 2v3h3M5.5 8h5M5.5 10.5h5" },
] as const;
type SectionId = (typeof SECTIONS)[number]["id"];

export function RaiView({ data }: { data: Partial<Record<Target, RaiTargetData>> }) {
  const targets = TARGETS.filter((t) => data[t.key]);
  const [target, setTarget] = useState<Target>(targets[0]?.key ?? "target_laptime");
  const [section, setSection] = useState<SectionId>("fairness");

  // The section lives in the URL hash so a reload or a shared link lands on it.
  useEffect(() => {
    const h = window.location.hash.slice(1) as SectionId;
    if (SECTIONS.some((s) => s.id === h)) setSection(h);
  }, []);
  const choose = (id: SectionId) => {
    setSection(id);
    history.replaceState(null, "", `#${id}`);
  };

  const d = data[target];
  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="stat-label">Prediction task</p>
          <div role="tablist" aria-label="Prediction task" className="mt-2 inline-flex gap-1 rounded-md border border-track-300 bg-track-100 p-1">
            {targets.map((t) => (
              <button key={t.key} type="button" role="tab" aria-selected={target === t.key}
                      onClick={() => setTarget(t.key)} className="mc-tab">
                {t.label}
              </button>
            ))}
          </div>
        </div>
        <ul className="flex flex-wrap gap-x-4 gap-y-1 text-[12px] text-paper-500" aria-label="Model colours">
          {MODEL_KEYS.map((k) => (
            <li key={k} className="flex items-center gap-1.5" style={{ ["--m" as string]: MODEL_ROLE[k].color }}>
              <span className="mc-swatch" aria-hidden />{MODEL_ROLE[k].label}
              {d?.comparison && <span className="t-micro">({d.comparison.models[k].algorithm})</span>}
            </li>
          ))}
        </ul>
      </div>

      {/* ---- section subtabs ------------------------------------------------ */}
      <nav aria-label="Responsible AI sections" className="grid gap-2 sm:grid-cols-3">
        {SECTIONS.map((s, i) => {
          const active = section === s.id;
          return (
            <button key={s.id} type="button" onClick={() => choose(s.id)} aria-current={active ? "page" : undefined}
                    className={`group flex items-center gap-3 rounded-md border px-4 py-3 text-left transition-colors duration-[180ms] ${
                      active ? "border-[var(--ferrari)] bg-track-100" : "border-track-300 hover:border-edge"}`}>
              <span className={`grid h-9 w-9 shrink-0 place-items-center rounded-md ${active ? "bg-[var(--ferrari)] text-[var(--ferrari-ink)]" : "bg-track-200 text-paper-700"}`}>
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                  <path d={s.icon} />
                </svg>
              </span>
              <span>
                <span className="block font-display text-[11px] font-bold uppercase tracking-[0.1em] text-[var(--papaya)]">Part {i + 1}</span>
                <span className={`block font-display text-[16px] font-bold leading-tight ${active ? "text-paper-900" : "text-paper-700 group-hover:text-paper-900"}`}>{s.label}</span>
              </span>
            </button>
          );
        })}
      </nav>

      <div className="border-t border-track-300 pt-8">
        {!d ? null : section === "cards" ? (
          d.comparison && d.cards
            ? <ModelCardsSection c={d.comparison} cards={d.cards} />
            : <Unavailable msg="Model metrics are not available. Run python scripts/build_all.py." />
        ) : d.rai ? (
          // Keyed by target: each task has its own metrics, so selections must reset.
          section === "fairness" ? <FairnessSection key={target} s={d.rai} /> : <ConfidenceSection key={target} s={d.rai} />
        ) : (
          <Unavailable msg={d.raiError ?? "The RAI analysis is not available."} />
        )}
      </div>
    </div>
  );
}

function Unavailable({ msg }: { msg: string }) {
  return (
    <div className="card border-accent">
      <span className="badge badge-warning">Not available</span>
      <p className="t-body mt-2 text-[13px]">{msg}</p>
    </div>
  );
}
