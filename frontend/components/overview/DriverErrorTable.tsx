import { XaiStratification } from "@/lib/api";
import { fmt } from "@/lib/format";

/**
 * Lap-time error per driver, from the Task 8 stratification artifact.
 *
 * A leaderboard in the sense that matters here: it ranks where the model is
 * weakest, not who is fastest. The bar is scaled to the worst row so the two
 * outliers are visible without a chart library, and the two worst rows carry a
 * text marker as well as the accent, because colour is never the only signal.
 */
export function DriverErrorTable({ strat }: { strat: XaiStratification }) {
  const laptime = strat["target_laptime"];
  if (!laptime) {
    return (
      <section>
        <h2 className="t-title">Lap-time error by driver</h2>
        <p className="t-body mt-2 text-paper-500">Not generated yet.</p>
        <p className="t-micro mt-1">
          Run <span className="text-paper-500">python scripts/build_all.py</span> to produce it.
        </p>
      </section>
    );
  }

  const rows = laptime.rows
    .filter((r) => r.group_type === "Driver" && typeof r.mae === "number")
    .sort((a, b) => a.mae - b.mae);
  const worst = Math.max(...rows.map((r) => r.mae), 0.0001);

  return (
    <section aria-labelledby="driver-h">
      <div className="flex flex-wrap items-baseline gap-3">
        <h2 id="driver-h" className="t-title">
          Lap-time error by driver
        </h2>
        <span className="t-micro tabular-nums">{rows.length} drivers, 9–11 laps each</span>
      </div>
      <p className="t-micro mt-1 mb-4">
        artifacts/xai/fairness_assessment.csv — samples this small are descriptive, not rates
      </p>

      <table className="t-table">
          <thead>
            <tr>
              <th scope="col">Driver</th>
              <th scope="col" className="num">Laps</th>
              <th scope="col" className="num">MAE s</th>
              <th scope="col" className="num hidden sm:table-cell">Bias s</th>
              <th scope="col" className="hidden w-[140px] sm:table-cell">
                <span className="sr-only">Relative error</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => {
              const isWorst = i === rows.length - 1;
              return (
                <tr key={r.group} className={i === 0 || isWorst ? "is-marked" : ""}>
                  <td>
                    <span className="t-code text-[13px]">{r.group}</span>
                    {i === 0 && <span className="marker ml-2">best</span>}
                    {isWorst && <span className="marker ml-2">out-lap</span>}
                  </td>
                  <td className="num">{r.n_laps}</td>
                  <td className="num">{fmt(r.mae, 3)}</td>
                  <td className="num hidden sm:table-cell">
                    {r.mean_error_bias > 0 ? "+" : "−"}
                    {fmt(Math.abs(r.mean_error_bias), 3)}
                  </td>
                  <td className="hidden sm:table-cell">
                    <span
                      aria-hidden
                      className={`block h-[5px] ${isWorst ? "bg-accent" : "bg-paper-400"}`}
                      style={{ width: `${Math.max((r.mae / worst) * 100, 3)}%` }}
                    />
                  </td>
                </tr>
              );
            })}
          </tbody>
      </table>
      <p className="t-body mt-4 border-l-2 border-track-300 pl-3.5 text-[13px] text-paper-500">
        The two worst rows are the drivers whose laps include a pit stop, whose recorded time sits at Task 4&rsquo;s
        outlier cap. The spread is about pit laps, not drivers.
      </p>
    </section>
  );
}
