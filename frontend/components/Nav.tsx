"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS: { href: string; label: string }[] = [
  { href: "/", label: "Dashboard" },
  { href: "/strategy", label: "Race Strategy" },
  { href: "/machine-learning", label: "Machine Learning" },
  { href: "/deep-learning", label: "Deep Learning" },
  { href: "/explainability", label: "Explainability" },
  { href: "/data-analysis", label: "Data & Analysis" },
  { href: "/evidence", label: "Project Evidence" },
];

export default function Nav() {
  const pathname = usePathname();

  return (
    <nav className="border-b border-white/10 bg-carbon/95 sticky top-0 z-20 backdrop-blur">
      <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-2.5 h-8 bg-f1red rounded-sm" />
          <div>
            <div className="font-bold text-white leading-tight">F1 Race Strategy Intelligence</div>
            <div className="text-[10px] text-white/40 tracking-wider">
              Computational Intelligence for race prediction & strategy
            </div>
          </div>
        </div>
        <div className="hidden lg:flex items-center gap-1 text-sm">
          {LINKS.map((link) => (
            <Link
              key={link.label}
              href={link.href}
              className={`px-3 py-1.5 rounded-md transition ${
                pathname === link.href
                  ? "bg-white/10 text-white"
                  : "text-white/60 hover:text-white hover:bg-white/5"
              }`}
            >
              {link.label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
}
