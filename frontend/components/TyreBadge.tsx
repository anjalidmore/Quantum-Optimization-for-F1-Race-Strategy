/**
 * A tyre compound, shown as a colour chip plus its name.
 *
 * The letter inside the chip is deliberate: compound colour must never be the
 * only carrier of meaning (WCAG 1.4.1), and a red square alone is unreadable
 * to anyone who cannot separate it from the accent. Chip ink is --compound-ink,
 * which flips with the theme: dark ink on the bright dark-mode chips (4.8:1 at
 * worst), white ink on the darkened light-mode ones (5.0:1 at worst).
 *
 * These five colours appear nowhere else in the app.
 */
const COMPOUND: Record<string, { letter: string; color: string; label: string }> = {
  SOFT: { letter: "S", color: "var(--compound-soft)", label: "Soft" },
  MEDIUM: { letter: "M", color: "var(--compound-medium)", label: "Medium" },
  HARD: { letter: "H", color: "var(--compound-hard)", label: "Hard" },
  INTERMEDIATE: { letter: "I", color: "var(--compound-inter)", label: "Intermediate" },
  WET: { letter: "W", color: "var(--compound-wet)", label: "Wet" },
};

export function TyreBadge({
  compound,
  showLabel = true,
}: {
  compound: string | null | undefined;
  showLabel?: boolean;
}) {
  if (!compound) return <span className="text-paper-400">—</span>;
  const key = compound.trim().toUpperCase();
  const c = COMPOUND[key];

  // An unmapped compound is shown as its own text rather than guessed at.
  if (!c) return <span className="text-paper-700">{compound}</span>;

  return (
    <span className="inline-flex items-center gap-2 whitespace-nowrap">
      <span
        aria-hidden
        style={{ backgroundColor: c.color, color: "var(--compound-ink)" }}
        className="grid h-[15px] w-[15px] place-items-center rounded-sm font-display text-[10px] font-semibold leading-none"
      >
        {c.letter}
      </span>
      {showLabel ? <span className="text-paper-700">{c.label}</span> : <span className="sr-only">{c.label}</span>}
    </span>
  );
}
