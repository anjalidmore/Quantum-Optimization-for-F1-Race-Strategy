"use client";

/** Small building blocks shared by the three RAI sections. */

export function RaiHead({ id, kicker, title }: { id: string; kicker: string; title: string }) {
  return (
    <div>
      <p className="font-display text-[12px] font-bold uppercase tracking-[0.12em] text-[var(--papaya)]">{kicker}</p>
      <h2 id={id} className="mt-0.5 font-display text-[24px] font-bold leading-tight text-paper-900">{title}</h2>
    </div>
  );
}

/** A labelled segmented control. Options always come from the data. */
export function Seg<T extends string>({ label, value, options, onChange }: {
  label: string; value: T; options: { value: T; label: string }[]; onChange: (v: T) => void;
}) {
  return (
    <div>
      <p className="stat-label">{label}</p>
      <div role="radiogroup" aria-label={label} className="mt-1.5 inline-flex flex-wrap gap-1 rounded-md border border-track-300 bg-track-100 p-1">
        {options.map((o) => (
          <button key={o.value} type="button" role="radio" aria-checked={value === o.value}
                  onClick={() => onChange(o.value)}
                  className={`rounded-sm px-3 py-1.5 text-[12px] font-medium transition-colors duration-[120ms] ${
                    value === o.value ? "bg-track-300 text-paper-900" : "text-paper-500 hover:text-paper-900"}`}>
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

/** Expandable "what does this measure" note. */
export function Explainer({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <details className="group rounded-md border border-track-300 bg-track-100 px-4 py-3">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 text-[13px] font-semibold text-paper-900">
        {title}
        <span aria-hidden className="text-paper-500 transition-transform duration-[180ms] group-open:rotate-45">+</span>
      </summary>
      <p className="t-body mt-2 text-[13px]">{children}</p>
    </details>
  );
}

export function InsufficientData({ className = "", label = "Insufficient Data" }: { className?: string; label?: string }) {
  return (
    <span className={`inline-flex items-center gap-1.5 font-display text-[16px] font-bold uppercase tracking-[0.04em] text-paper-500 ${className}`}>
      <svg width="13" height="13" viewBox="0 0 13 13" aria-hidden><circle cx="6.5" cy="6.5" r="5.6" fill="none" stroke="currentColor" strokeWidth="1.3" /><path d="M3 10 10 3" stroke="currentColor" strokeWidth="1.3" /></svg>
      {label}
    </span>
  );
}
