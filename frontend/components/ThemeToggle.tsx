"use client";

import { useEffect, useState } from "react";

export type Theme = "light" | "dark";
export const THEME_KEY = "pitwall-theme";

/**
 * Light/dark toggle.
 *
 * With no stored choice the app follows the OS, which is why the button starts
 * from the resolved theme rather than a hard-coded default. The choice is
 * written to <html data-theme>, which every token block keys off, and to
 * localStorage so the inline script in layout.tsx can apply it before paint.
 */
export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme | null>(null);

  useEffect(() => {
    const el = document.documentElement;
    const explicit = el.getAttribute("data-theme") as Theme | null;
    setTheme(explicit ?? (window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark"));
  }, []);

  function choose(next: Theme) {
    document.documentElement.setAttribute("data-theme", next);
    try {
      localStorage.setItem(THEME_KEY, next);
    } catch {
      // Private browsing can refuse storage. The theme still applies for this
      // page; only remembering it across loads is lost, which is survivable.
    }
    setTheme(next);
  }

  // Render a same-size placeholder until the theme is known, so the button
  // never flashes the wrong state on hydration and the nav does not shift.
  if (!theme) return <span className="inline-block h-[28px] w-[58px]" aria-hidden />;

  const next: Theme = theme === "dark" ? "light" : "dark";
  return (
    <button
      type="button"
      role="switch"
      aria-checked={theme === "light"}
      onClick={() => choose(next)}
      aria-label="Light theme"
      title={`Switch to ${next} theme`}
      data-mode={theme}
      className="theme-switch"
    >
      <span className="theme-switch__knob" aria-hidden />
      <svg className="theme-switch__moon" width="12" height="12" viewBox="0 0 12 12" aria-hidden focusable="false">
        <path d="M8.6 7.9A4.2 4.2 0 0 1 4.1 3.4a4.2 4.2 0 0 1 .7-2.3A4.6 4.6 0 1 0 10.9 7.2a4.2 4.2 0 0 1-2.3.7Z" fill="currentColor" />
      </svg>
      <svg className="theme-switch__sun" width="12" height="12" viewBox="0 0 12 12" aria-hidden focusable="false">
        <circle cx="6" cy="6" r="2.4" fill="currentColor" />
        <path
          d="M6 .6v1.5M6 9.9v1.5M.6 6h1.5M9.9 6h1.5M2.2 2.2l1 1M8.8 8.8l1 1M2.2 9.8l1-1M8.8 3.2l1-1"
          stroke="currentColor"
          strokeWidth="1.1"
          strokeLinecap="round"
        />
      </svg>
    </button>
  );
}
