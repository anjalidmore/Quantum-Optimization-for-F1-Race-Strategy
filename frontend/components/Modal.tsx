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
      className="fixed inset-0 z-40 bg-track-000/70 flex items-start justify-center overflow-y-auto py-10 px-4"
      onClick={onClose}
    >
      <div
        className="bg-panel border border-track-300 rounded-xl max-w-2xl w-full p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between mb-4">
          <h2 className="text-lg font-semibold text-paper-900">{title}</h2>
          <button onClick={onClose} className="text-paper-500 hover:text-paper-900 text-xl leading-none">
            ×
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}
