import { artifactUrl, ExpertSystemSummary, Unavailable } from "@/lib/api";
import { NotGenerated } from "@/components/reasoning/NotGenerated";

/** Task 2 — the rule base, read from artifacts/expert_system/rules/rule_base.json. */
export function ExpertSystemCard({ data }: { data: ExpertSystemSummary | Unavailable }) {
  if (!data.available) return <NotGenerated title="Task 2 — Expert system" reason={data.reason} />;

  return (
    <section>
      <h2 className="text-lg font-semibold text-paper-900 mb-1">Task 2 — Expert system</h2>
      <p className="text-sm text-paper-500 mb-3">
        Forward-chaining rules over a race state. Every firing records which conditions matched and what it
        asserted, so a recommendation can always be traced back to named rules — these are the rules that appear
        in the strategy report.
      </p>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-4">
        <div className="card">
          <div className="stat-label">Rules</div>
          <div className="stat-value">{data.n_rules}</div>
        </div>
        <div className="card">
          <div className="stat-label">Categories</div>
          <div className="stat-value">{Object.keys(data.rules_by_category).length}</div>
        </div>
        <div className="card">
          <div className="stat-label">Salience levels</div>
          <div className="text-lg font-semibold text-paper-900">{data.salience_levels.join(" · ")}</div>
        </div>
      </div>

      <div className="card overflow-x-auto">
        <table className="w-full text-sm min-w-[38rem]">
          <thead className="text-paper-500">
            <tr>
              <th className="text-left font-normal py-1">Rule</th>
              <th className="text-left font-normal">Name</th>
              <th className="text-left font-normal">Category</th>
              <th className="text-right font-normal">Salience</th>
              <th className="text-right font-normal">Conditions</th>
              <th className="text-left font-normal pl-3">Asserts</th>
            </tr>
          </thead>
          <tbody className="text-paper-700">
            {data.rules.map((r) => (
              <tr key={r.rule_id} className="border-t border-track-300">
                <td className="py-1"><code className="text-paper-700">{r.rule_id}</code></td>
                <td>{r.name}</td>
                <td className="text-paper-500">{r.category}</td>
                <td className="text-right tabular-nums">{r.salience}</td>
                <td className="text-right tabular-nums">{r.n_conditions}</td>
                <td className="pl-3 text-xs text-paper-500">{r.actions.join(", ")}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="text-xs text-paper-400 mt-2">
          Worked inference traces:{" "}
          <a
            className="text-paper-900 underline decoration-edge underline-offset-[3px] hover:decoration-paper-900"
            href={artifactUrl("expert_system/reports/rule_catalogue.md")}
            target="_blank"
            rel="noreferrer"
          >
            rule_catalogue.md
          </a>
        </p>
      </div>
    </section>
  );
}
