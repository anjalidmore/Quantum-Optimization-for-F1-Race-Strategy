import { artifactUrl, XaiStratification } from "@/lib/api";
import { ArtifactImage } from "@/components/ArtifactImage";
import { TARGET_LABEL } from "@/lib/format";

/**
 * Does the model perform differently by driver, team or compound? Per-driver rows
 * are left to the CSV and the figure: at ~9 laps each they are too small to read
 * as rates, and showing them as a table invites exactly that.
 */
export function StratificationTable({ strat }: { strat: XaiStratification }) {
  return (
    <section>
      <h2 className="text-lg font-semibold text-paper-900 mb-1">F1-specific performance stratification</h2>
      <p className="text-sm text-paper-500 mb-3 max-w-3xl">
        This is <strong>not</strong> a protected-attribute fairness analysis — the data holds no demographic
        attributes. All laps come from one race, so there is no circuit stratum. Rows marked small are
        descriptive only.
      </p>
      <div className="grid gap-4">
        {Object.entries(strat).map(([target, s]) => {
          const reg = s.task === "regression";
          const cols = reg
            ? ["mae", "rmse", "mean_error_bias"]
            : ["n_pit_laps", "precision", "recall", "f1", "false_positive_rate"];
          const rows = s.rows.filter((r) => r.group_type !== "Driver");
          return (
            <div key={target} className="card overflow-x-auto">
              <h3 className="font-semibold text-paper-900 mb-2">{TARGET_LABEL[target] ?? target}</h3>
              <table className="w-full text-sm min-w-[40rem]">
                <thead className="text-paper-500">
                  <tr>
                    <th className="text-left font-normal py-1">Group</th>
                    <th className="text-right font-normal">laps</th>
                    {cols.map((c) => (
                      <th key={c} className="text-right font-normal">
                        {c.replace(/_/g, " ")}
                      </th>
                    ))}
                    <th className="text-left font-normal pl-3">note</th>
                  </tr>
                </thead>
                <tbody className="text-paper-700">
                  {rows.map((r) => (
                    <tr key={`${r.group_type}-${r.group}`} className="border-t border-track-300">
                      <td className="py-1">
                        {r.group_type === "overall" ? <strong>{r.group}</strong> : `${r.group_type}: ${r.group}`}
                      </td>
                      <td className="text-right tabular-nums">{r.n_laps}</td>
                      {cols.map((c) => (
                        <td key={c} className="text-right tabular-nums">
                          {typeof r[c] === "number" ? (Number.isInteger(r[c]) ? r[c] : r[c].toFixed(3)) : "—"}
                        </td>
                      ))}
                      <td
                        className={`pl-3 text-xs ${
                          r.sample_note === "OK" ? "text-paper-400" : "text-paper-500"
                        }`}
                      >
                        {r.sample_note}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="text-xs text-paper-400 mt-2">
                Per-driver rows are in <code>artifacts/xai/fairness_assessment.csv</code> and the figure below.
              </p>
              {s.figure && (
                <ArtifactImage
                  src={artifactUrl(s.figure)}
                  alt={`${target} stratification`}
                  className="w-full rounded mt-3"
                />
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
