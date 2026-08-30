import { DatasetSource } from "@/lib/api";

export function datasetLabel(source: DatasetSource): string {
  if (source.source === "real_fastf1") {
    return `DATASET: Real FastF1 — ${source.event} ${source.year} (${source.session})`;
  }
  return "DATASET: Synthetic Demonstration";
}

export function DatasetBadge({ source }: { source: DatasetSource }) {
  return <span className="badge badge-warning">{datasetLabel(source)}</span>;
}
