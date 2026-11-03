import Link from "next/link";

/** A route through the dashboard, in the order that tells the project's story. */
const STEPS: { href: string; page: string; show: string }[] = [
  { href: "/strategy", page: "Strategy", show: "Pick a scenario and press Predict: models, rules, search and SHAP combine into one call." },
  { href: "/data", page: "Data", show: "The real race session, how it was cleaned, and what the analysis found." },
  { href: "/reasoning", page: "Reasoning", show: "The ontology, the rule base and five search algorithms on one strategy problem." },
  { href: "/models", page: "Models", show: "Classical, deep and simulated quantum models on the same untouched holdout laps." },
  { href: "/model-comparison", page: "Model Comparison", show: "A simple baseline, the best ML model and the DNN, side by side." },
  { href: "/explainability", page: "Explainability", show: "Inspect one lap: what moved the prediction, what would flip it, how far to trust it." },
  { href: "/rai", page: "Responsible AI", show: "Performance gaps between groups, confidence checks and model cards." },
];

export function DemoPath() {
  return (
    <section aria-labelledby="demo-h" className="mt-10">
      <h2 id="demo-h" className="t-title mb-3">
        Demo path
      </h2>
      <ol className="border-t border-track-300">
        {STEPS.map((s, i) => (
          <li key={s.href} className="border-b border-track-300">
            <Link
              href={s.href}
              className="grid gap-x-3 py-2.5 text-[13px] transition-colors duration-[120ms] hover:bg-track-100 sm:grid-cols-[24px_170px_1fr]"
            >
              <span className="hidden text-right font-display font-semibold text-paper-400 sm:block">{i + 1}</span>
              <span className="font-semibold text-paper-900">{s.page}</span>
              <span className="text-paper-500">{s.show}</span>
            </Link>
          </li>
        ))}
      </ol>
    </section>
  );
}
