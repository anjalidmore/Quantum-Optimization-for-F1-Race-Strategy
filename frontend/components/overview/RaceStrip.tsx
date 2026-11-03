import type { CSSProperties } from "react";
import { SearchSummary } from "@/lib/api";
import { TyreBadge } from "@/components/TyreBadge";
import { F1Car } from "@/components/overview/F1Art";

/**
 * The track across the foot of the Overview: the A* plan drawn to scale. Each
 * lap of the search problem is one equal step, so the pit box sits where the
 * plan actually stops and the band under the car changes compound there. The
 * road is a lap ruler, not a circuit: no track outline exists (DESIGN.md).
 * Without a plan the road stays empty rather than showing an invented stop.
 */
export function RaceStrip({ plan }: { plan: SearchSummary | null }) {
  const totalLaps = plan?.problem.total_laps ?? 0;
  const stop = plan?.pit_stops?.[0] ?? null;
  const at = (lap: number) => `${((lap - 1) / Math.max(totalLaps - 1, 1)) * 100}%`;

  const label = !plan
    ? "Race track. No strategy plan has been generated yet."
    : stop
      ? `Best plan over ${totalLaps} laps: start on ${plan.problem.start_compound}, pit on lap ${stop.lap} onto ${stop.compound}.`
      : `Best plan over ${totalLaps} laps: no stop, on ${plan.problem.start_compound} throughout.`;

  return (
    <section className="race-strip home-bleed" role="img" aria-label={label}>
      <div className="race-strip__kerb" />
      <div className="race-strip__road">
        <div className="mx-auto h-full max-w-[1320px] px-5 sm:px-8">
          <div className="relative h-full" style={{ "--pit": stop ? at(stop.lap) : "100%" } as CSSProperties}>
            {plan && (
              <>
                {stop && <div className="race-strip__pitlane" />}
                {stop && <div className="race-strip__box" />}
                <div className="race-strip__grid" />
                <div className="race-strip__finish" />

                <div
                  className="race-strip__stint"
                  style={{ left: 0, width: stop ? at(stop.lap) : "100%", backgroundColor: compoundVar(plan.problem.start_compound) }}
                />
                {stop && (
                  <div
                    className="race-strip__stint"
                    style={{ left: at(stop.lap), right: 0, backgroundColor: compoundVar(stop.compound) }}
                  />
                )}

                <span className="race-strip__mark" style={{ left: 0 }}>
                  <TyreBadge compound={plan.problem.start_compound} showLabel={false} /> L1
                </span>
                {stop && (
                  <span className="race-strip__mark race-strip__mark--pit" style={{ left: at(stop.lap) }}>
                    <TyreBadge compound={stop.compound} showLabel={false} /> L{stop.lap}
                  </span>
                )}
                <span className="race-strip__mark race-strip__mark--end" style={{ left: "100%" }}>
                  L{totalLaps}
                </span>
              </>
            )}
            <F1Car className={`race-strip__car ${stop ? "race-strip__car--pit" : plan ? "race-strip__car--end" : ""}`} />
          </div>
        </div>
      </div>
      <div className="race-strip__kerb" />
    </section>
  );
}

function compoundVar(compound: string) {
  return `var(--compound-${compound.trim().toLowerCase() === "intermediate" ? "inter" : compound.trim().toLowerCase()})`;
}
