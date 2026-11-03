"use client";

import { ReactNode, useEffect, useState } from "react";

export type PageTab = { id: string; label: string; content: ReactNode };

/**
 * Splits a long page into sections. A section mounts when first opened and then
 * stays mounted but hidden, so charts lay out at their real width and anything
 * entered in a section survives switching away. The open tab is kept in the URL
 * hash, so a link or a reload lands on the same section.
 */
export function PageTabs({ tabs, label }: { tabs: PageTab[]; label: string }) {
  const [active, setActive] = useState(tabs[0].id);
  const [visited, setVisited] = useState<Set<string>>(() => new Set([tabs[0].id]));

  const open = (id: string) => {
    setActive(id);
    setVisited((v) => new Set(v).add(id));
  };

  useEffect(() => {
    const fromHash = window.location.hash.slice(1);
    if (tabs.some((t) => t.id === fromHash)) open(fromHash);
    // Only on arrival; tab clicks update the hash themselves.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <>
      <div role="tablist" aria-label={label} className="mt-6 flex flex-wrap gap-1 border-b border-track-300">
        {tabs.map((t) => (
          <button
            key={t.id}
            role="tab"
            id={`tab-${t.id}`}
            aria-selected={active === t.id}
            aria-controls={`panel-${t.id}`}
            onClick={() => {
              open(t.id);
              window.history.replaceState(null, "", `#${t.id}`);
            }}
            className={`-mb-px border-b-2 px-4 py-2.5 text-sm font-medium transition-colors duration-[120ms] ${
              active === t.id ? "border-accent text-paper-900" : "border-transparent text-paper-500 hover:text-paper-900"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>
      {tabs.map((t) =>
        visited.has(t.id) ? (
          <div
            key={t.id}
            role="tabpanel"
            id={`panel-${t.id}`}
            aria-labelledby={`tab-${t.id}`}
            hidden={active !== t.id}
            className="mt-8 space-y-10"
          >
            {t.content}
          </div>
        ) : null
      )}
    </>
  );
}
