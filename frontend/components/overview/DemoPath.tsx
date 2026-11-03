import Link from "next/link";
import { Icons } from "@/components/overview/F1Art";

/** A route through the dashboard, in the order that tells the project's story. */
const STEPS: { href: string; page: string; icon: keyof typeof Icons }[] = [
  { href: "/strategy", page: "Strategy", icon: "wheel" },
  { href: "/data", page: "Data", icon: "telemetry" },
  { href: "/reasoning", page: "Reasoning", icon: "tree" },
  { href: "/models", page: "Models", icon: "chip" },
  { href: "/model-comparison", page: "Model Comparison", icon: "podium" },
  { href: "/explainability", page: "Explainability", icon: "lens" },
  { href: "/rai", page: "Responsible AI", icon: "shield" },
];

export function DemoPath() {
  return (
    <nav aria-label="Pages" className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-7">
      {STEPS.map((s) => (
        <Link key={s.href} href={s.href} className="home-tile first:col-span-2 sm:first:col-span-1">
          {Icons[s.icon]("h-7 w-7")}
          <span className="home-tile__label">{s.page}</span>
        </Link>
      ))}
    </nav>
  );
}
