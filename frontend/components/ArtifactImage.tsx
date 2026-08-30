"use client";

import { useState } from "react";

export function ArtifactImage({ src, alt, className }: { src: string; alt: string; className?: string }) {
  const [failed, setFailed] = useState(false);

  if (failed) {
    return (
      <div className={`flex items-center justify-center bg-panel2 border border-dashed border-white/15 rounded text-white/40 text-sm py-10 ${className ?? ""}`}>
        Artifact unavailable
      </div>
    );
  }

  // eslint-disable-next-line @next/next/no-img-element
  return <img src={src} alt={alt} className={className} onError={() => setFailed(true)} />;
}
