"use client";

import { useState } from "react";

export function Tooltip({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <span className="relative inline-block ml-1 align-middle">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        onBlur={() => setOpen(false)}
        aria-label="More information"
        className="w-4 h-4 rounded-full bg-white/15 hover:bg-white/25 text-white/70 text-[10px] leading-4 inline-flex items-center justify-center"
      >
        ?
      </button>
      {open && (
        <span className="absolute z-30 left-1/2 -translate-x-1/2 top-6 w-56 text-xs font-normal text-white/90 bg-panel2 border border-white/15 rounded-lg p-2.5 shadow-xl">
          {text}
        </span>
      )}
    </span>
  );
}
