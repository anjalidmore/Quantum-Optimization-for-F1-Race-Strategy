/** Blocks at the final height of the content they replace. No shimmer. */
export default function Loading() {
  return (
    <div className="pt-14" aria-busy="true" aria-live="polite">
      <span className="sr-only">Loading</span>
      <div className="skeleton h-[26px] w-[280px]" />
      <div className="mt-4 skeleton h-[76px] w-[380px]" />
      <div className="mt-6 skeleton h-[60px] w-full max-w-measure" />
      <div className="mt-10 grid gap-x-12 gap-y-8 lg:grid-cols-[1.55fr_1fr]">
        <div className="skeleton h-[420px]" />
        <div className="skeleton h-[420px]" />
      </div>
    </div>
  );
}
