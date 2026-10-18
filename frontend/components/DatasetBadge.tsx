import { DatasetSource } from "@/lib/api";

/**
 * Data provenance, as a statement of fact.
 *
 * It used to be amber, which reads as a warning — real telemetry is not a
 * problem — and it repeated the word "DATASET" that the tile label already
 * carried. It is now neutral, one line, and states the session.
 */
export function datasetLabel(source: DatasetSource): string {
  if (source.source === "real_fastf1") {
    return `Real · ${source.event} ${source.year} · ${source.session === "R" ? "Race" : source.session}`;
  }
  return "Synthetic demonstration data";
}

export function DatasetBadge({ source }: { source: DatasetSource }) {
  const real = source.source === "real_fastf1";
  return (
    <span className="badge badge-info">
      <span
        aria-hidden
        className={`block h-[5px] w-[5px] rounded-full ${real ? "bg-accent" : "border border-edge"}`}
      />
      {real ? (
        <>
          Real&nbsp;<b className="font-semibold text-paper-900">{source.event} {source.year}</b>
          <span className="text-paper-400">{source.session === "R" ? "Race" : source.session}</span>
          {"n_laps" in source && source.n_laps ? <span className="text-paper-400">{source.n_laps} laps</span> : null}
        </>
      ) : (
        "Synthetic demonstration data"
      )}
    </span>
  );
}
