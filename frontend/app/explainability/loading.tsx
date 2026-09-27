/** Blocks at the final height of the content they replace. No shimmer. */
export default function Loading() {
  return (
    <div className="pt-14" aria-busy="true" aria-live="polite">
      <span className="sr-only">Loading</span>
      <div className="skeleton h-[40px] w-[320px]" />
      <div className="mt-5 skeleton h-[44px] w-full max-w-measure" />
      <div className="mt-10 space-y-6">
        <div className="skeleton h-[220px]" />
        <div className="skeleton h-[320px]" />
      </div>
    </div>
  );
}
