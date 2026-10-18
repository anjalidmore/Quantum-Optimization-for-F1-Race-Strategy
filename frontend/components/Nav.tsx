"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ThemeToggle } from "@/components/ThemeToggle";

// `short` is shown between lg and xl, where the full label would not fit the bar.
const LINKS: { href: string; label: string; short?: string }[] = [
  { href: "/", label: "Overview" },
  { href: "/strategy", label: "Strategy" },
  { href: "/reasoning", label: "Reasoning" },
  { href: "/models", label: "Models" },
  { href: "/explainability", label: "Explainability" },
  { href: "/data", label: "Data" },
  { href: "/model-comparison", label: "Model Comparison" },
  { href: "/rai", label: "Responsible AI (RAI)", short: "RAI" },
];

export default function Nav() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);


  return (
    /* The one glass surface in the app (DESIGN.md, Constraints): translucent
       over the page ground with a modest blur, so content reads as passing
       underneath rather than vanishing at a hard edge. */
    <header className="nav-glass sticky top-0 z-20">
      <div className="f1-topline" aria-hidden />
      <div className="mx-auto flex h-[52px] max-w-[1320px] items-center gap-7 px-5 sm:px-8">
        <Link href="/" className="flex items-center gap-2.5 whitespace-nowrap">
          <span aria-hidden className="f1-logo-mark">
            <span className="bg-[var(--ferrari)]" />
            <span className="bg-[var(--papaya-fill)]" />
          </span>
          <span className="font-display text-[19px] font-black italic leading-none tracking-[0.02em] text-paper-900">
            PIT WALL
          </span>
        </Link>

        {/* Idle links are paper-700, not paper-500: over the translucent surface
            with page content behind it, paper-500 fails AA. */}
        <nav aria-label="Main" className="ml-auto hidden h-[52px] items-stretch lg:flex">
          {LINKS.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              aria-current={pathname === link.href ? "page" : undefined}
              className="f1-navlink flex items-center whitespace-nowrap"
              title={link.short ? link.label : undefined}
            >
              {link.short ? (
                <>
                  <span className="xl:hidden" aria-hidden>{link.short}</span>
                  <span className="sr-only xl:not-sr-only">{link.label}</span>
                </>
              ) : (
                link.label
              )}
            </Link>
          ))}
        </nav>

        {/* The toggle sits at the end in both layouts; below lg it shares the
            row with the menu button, which is why they share one flex box
            rather than each claiming ml-auto. */}
        <div className="ml-auto flex items-center gap-2">
          <ThemeToggle />

          {/* Below lg the site had no navigation at all. */}
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            aria-expanded={open}
            aria-controls="mobile-nav"
            className="btn-secondary flex items-center gap-2 px-3 py-1.5 text-[13px] lg:hidden"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden focusable="false">
              {open ? (
                <path d="M2 2 L12 12 M12 2 L2 12" stroke="currentColor" strokeWidth="1.5" fill="none" />
              ) : (
                <path d="M1 3h12M1 7h12M1 11h12" stroke="currentColor" strokeWidth="1.5" fill="none" />
              )}
            </svg>
            {open ? "Close" : "Menu"}
          </button>
        </div>
      </div>

      {open && (
        <nav
          id="mobile-nav"
          aria-label="Main"
          /* Solid, not glass: a full-height menu has nothing meaningfully
             passing beneath it, and the blur only ghosted the page behind
             the labels. The sticky bar above keeps the glass. */
          className="border-t border-track-300 bg-track-050 px-5 pb-3 lg:hidden"
        >
          {LINKS.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              onClick={() => setOpen(false)}
              aria-current={pathname === link.href ? "page" : undefined}
              className={`block border-b border-track-300 py-3 font-display text-[15px] font-bold uppercase tracking-[0.05em] ${
                pathname === link.href ? "border-l-[3px] border-l-[var(--ferrari)] pl-3 text-paper-900" : "text-paper-700"
              }`}
            >
              {link.label}
            </Link>
          ))}
        </nav>
      )}
    </header>
  );
}
