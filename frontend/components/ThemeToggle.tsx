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

  // Render nothing until the theme is known, so the button never flashes the
  // wrong label on hydration.
  if (!theme) return <span className="h-[30px] w-[62px]" aria-hidden />;

  const next: Theme = theme === "dark" ? "light" : "dark";
  return (
    <button
      type="button"
      onClick={() => choose(next)}
      aria-label={`Switch to ${next} theme`}
      title={`Switch to ${next} theme`}
      className="rounded-sm border border-track-300 px-2.5 py-1.5 text-[12px] font-medium text-paper-700 transition-colors duration-[120ms] hover:border-edge hover:text-paper-900"
    >
      {theme === "dark" ? "Light" : "Dark"}
    </button>
  );
}
