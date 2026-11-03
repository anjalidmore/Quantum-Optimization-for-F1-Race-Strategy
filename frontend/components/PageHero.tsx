/** The header every page but Overview opens with: task, title, one short paragraph. */
export function PageHero({ eyebrow, title, blurb }: { eyebrow: string; title: string; blurb: string }) {
  return (
    <section className="border-b border-track-300 pb-6 pt-10">
      <p className="stat-label">{eyebrow}</p>
      <h1 className="t-page-title mt-1.5 !text-[clamp(28px,3.4vw,40px)]">{title}</h1>
      <p className="t-body mt-3 max-w-measure">{blurb}</p>
    </section>
  );
}
