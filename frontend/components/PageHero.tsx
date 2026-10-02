import Image from "next/image";

/**
 * The compact masthead used on every page but Overview and Strategy (which
 * have their own full treatments). Same photographic language — real image,
 * ember gradient, veil — at a third of the height, so a reader reaches the
 * dense tables in one scroll instead of a second hero.
 */
export function PageHero({
  photo,
  focus = "50% 50%",
  eyebrow,
  title,
  blurb,
}: {
  photo?: string;
  focus?: string;
  eyebrow: string;
  title: string;
  blurb: string;
}) {
  return (
    <section
      className="hero-stage mt-8 min-h-[220px]"
      style={{ ["--hero-focus" as string]: focus }}
    >
      {photo && <Image src={photo} alt="" fill sizes="100vw" className="hero-stage__photo" priority />}
      <div className="hero-stage__ember" aria-hidden />
      <div className="hero-stage__veil" aria-hidden />
      <div className="hero-stage__content p-7 md:p-10">
        <p className="stat-label">{eyebrow}</p>
        <h1 className="t-page-title mt-1.5 !text-[clamp(28px,3.4vw,40px)]">{title}</h1>
        <p className="t-body mt-3 max-w-measure text-[13px]">{blurb}</p>
      </div>
    </section>
  );
}
