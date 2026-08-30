"use client";

export function Modal({
  title,
  onClose,
  children,
}: {
  title: string;
  onClose: () => void;
  children: React.ReactNode;
}) {
  return (
    <div
      className="fixed inset-0 z-40 bg-black/60 flex items-start justify-center overflow-y-auto py-10 px-4"
      onClick={onClose}
    >
      <div
        className="bg-panel border border-white/10 rounded-xl max-w-2xl w-full p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between mb-4">
          <h2 className="text-lg font-semibold text-white">{title}</h2>
          <button onClick={onClose} className="text-white/50 hover:text-white text-xl leading-none">
            ×
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}
