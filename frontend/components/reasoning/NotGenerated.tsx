/** Shown instead of a number when an artifact has not been generated yet. */
export function NotGenerated({ title, reason }: { title: string; reason: string }) {
  return (
    <section className="card">
      <h2 className="font-semibold text-white">{title}</h2>
      <p className="text-sm text-white/50 mt-2">Not generated yet.</p>
      <p className="text-xs text-white/40 mt-1">{reason}</p>
    </section>
  );
}
