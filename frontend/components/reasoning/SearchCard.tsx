import { artifactUrl, SearchSummary, Unavailable } from "@/lib/api";
import { ArtifactImage } from "@/components/ArtifactImage";
import { NotGenerated } from "@/components/reasoning/NotGenerated";
import { fmt } from "@/lib/format";

const COLUMNS: [string, string][] = [
  ["solution_cost", "Cost (s)"],
  ["cost_gap_pct", "Gap %"],
  ["nodes_expanded", "Expanded"],
  ["nodes_generated", "Generated"],
  ["max_frontier_size", "Max frontier"],
  ["elapsed_ms", "Time (ms)"],
  ["solution_depth", "Depth"],
  ["n_pit_stops", "Stops"],
];

/** Task 3 — five algorithms on one race-strategy problem, from comparison.json. */
export function SearchCard({ data }: { data: SearchSummary | Unavailable }) {
  if (!data.available) return <NotGenerated title="Task 3 — State-space search" reason={data.reason} />;

  const p = data.problem;
  return (
    <section>
      <h2 className="text-lg font-semibold text-white mb-1">Task 3 — State-space search</h2>
      <p className="text-sm text-white/50 mb-3">
        Pit strategy as a search problem: each state is a lap, a compound and a tyre age; each action is run or
        pit. Five algorithms solve the same instance, so the cost of an uninformed search is visible next to an
        informed one.
      </p>

      <p className="text-xs text-white/40 mb-3">
        Instance: {p.total_laps} laps, starting on {p.start_compound}, pit loss {p.pit_loss_seconds} s, track{" "}
        {p.track_temperature_c} °C, at most {p.max_stops} stops, compounds {p.allowed_compounds.join(" / ")}.
      </p>

      <div className="card overflow-x-auto">
        <table className="w-full text-sm min-w-[40rem]">
          <thead className="text-white/50">
            <tr>
              <th className="text-left font-normal py-1">Algorithm</th>
              {COLUMNS.map(([, label]) => (
                <th key={label} className="text-right font-normal">{label}</th>
              ))}
              <th className="text-center font-normal">Optimal</th>
            </tr>
          </thead>
          <tbody className="text-white/80">
            {data.algorithms.map((a) => (
              <tr key={a.algorithm} className={`border-t border-track-300 ${a.is_optimal ? "is-marked" : ""}`}>
                <td className="py-1.5">{a.algorithm}</td>
                {COLUMNS.map(([key]) => (
                  <td key={key} className="text-right tabular-nums">
                    {typeof a[key] === "number" ? fmt(a[key], key === "solution_cost" ? 2 : 1) : "—"}
                  </td>
                ))}
                <td className="text-center">{a.is_optimal ? <span className="marker">optimal</span> : null}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="text-xs text-white/50 mt-2">
          Optimal: {(data.summary.optimal_algorithms ?? []).join(" and ")} — both reach{" "}
          {fmt(data.final_cost_seconds, 2)} s, which is the correctness check for the A\* heuristic. Fewest nodes
          expanded: {data.summary.fewest_nodes_expanded}.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 mt-4">
        <div className="card">
          <h3 className="font-semibold text-white mb-2 text-sm">Optimal plan — {data.n_plan_steps} steps</h3>
          {data.pit_stops.length === 0 ? (
            <p className="text-sm text-white/50">The optimal plan makes no stop on this instance.</p>
          ) : (
            <table className="w-full text-sm">
              <thead className="text-white/50">
                <tr>
                  <th className="text-left font-normal py-1">Lap</th>
                  <th className="text-left font-normal">Action</th>
                  <th className="text-left font-normal">Fit</th>
                  <th className="text-right font-normal">Cost so far (s)</th>
                </tr>
              </thead>
              <tbody className="text-white/80">
                {data.pit_stops.map((s) => (
                  <tr key={`${s.lap}-${s.type}`} className="border-t border-white/5">
                    <td className="py-1">{s.lap}</td>
                    <td>{s.type}</td>
                    <td>{s.compound ?? "—"}</td>
                    <td className="text-right tabular-nums">{fmt(s.cumulative_cost_seconds, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <p className="text-xs text-white/40 mt-3">
            Only the stops are listed; the remaining steps are plain running laps.
          </p>
        </div>
        <div className="card">
          <ArtifactImage
            src={artifactUrl("search/diagrams/optimal_strategy_path.png")}
            alt="A* optimal strategy path"
            className="w-full rounded"
          />
          <p className="text-xs text-white/50 mt-2">The A\* plan over the race.</p>
        </div>
      </div>
    </section>
  );
}
