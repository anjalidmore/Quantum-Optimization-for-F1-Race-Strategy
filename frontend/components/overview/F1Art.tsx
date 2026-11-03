/**
 * The Overview page's drawings, by hand so they take the theme tokens and add
 * no icon dependency. The car and flag are illustration, not data: nothing here
 * is read from an artifact, and RaceStrip is what places the car on real laps.
 */

/** A car from above, nose to the right, the way it runs along the race strip. */
export function F1Car({ className }: { className?: string }) {
  const tyre = "#0b0b10";
  const carbon = "#1d1d26";
  return (
    <svg className={className} viewBox="0 0 240 100" aria-hidden focusable="false">
      {/* Suspension arms, under everything else. */}
      <path
        d="M50 28 66 42M62 28 76 40M50 72 66 58M62 72 76 60M166 29 150 44M178 29 166 45M166 71 150 56M178 71 166 55"
        stroke="#8e8e9c"
        strokeWidth="2.2"
        strokeLinecap="round"
        fill="none"
      />
      <g fill={tyre} stroke="#5f5f70" strokeWidth="1">
        <rect x="38" y="8" width="36" height="21" rx="4" />
        <rect x="38" y="71" width="36" height="21" rx="4" />
        <rect x="158" y="12" width="30" height="17" rx="4" />
        <rect x="158" y="71" width="30" height="17" rx="4" />
      </g>
      {/* Floor. */}
      <path d="M30 37 64 30 84 23 132 27 148 41V59L132 73 84 77 64 70 30 63Z" fill={carbon} />
      {/* Body: tapered tail, sidepods, then the long narrow nose. */}
      <path
        d="M26 44C44 44 56 40 72 31 82 26 100 25 112 29 124 33 130 38 142 41L214 46C217 47 217 53 214 54L142 59C130 62 124 67 112 71 100 75 82 74 72 69 56 60 44 56 26 56Z"
        fill="var(--ferrari)"
      />
      <path d="M30 48 104 45V55L30 52ZM146 46.5 214 48.5V51.5L146 53.5Z" fill="var(--papaya-fill)" />
      {/* Sidepod inlets. */}
      <path d="M106 30 116 33V38L106 35ZM106 70 116 67V62L106 65Z" fill={tyre} />
      {/* Rear wing. */}
      <rect x="8" y="16" width="20" height="68" rx="2" fill={carbon} />
      <rect x="12" y="19" width="9" height="62" fill="var(--ferrari)" />
      {/* Front wing, curved back to the endplates. */}
      <path d="M208 9H219C224 24 230 38 231 50 230 62 224 76 219 91H208C212 76 217 62 217 50 217 38 212 24 208 9Z" fill="var(--ferrari)" />
      <path d="M205 6H220V11H205ZM205 89H220V94H205Z" fill={carbon} />
      {/* Cockpit, helmet, halo. */}
      <ellipse cx="128" cy="50" rx="15" ry="8" fill={tyre} />
      <circle cx="124" cy="50" r="5" fill="#f7f4f1" />
      <path d="M146 50H137C129 41 117 41 111 44M137 50C129 59 117 59 111 56" stroke="#3a3a46" strokeWidth="2.6" strokeLinecap="round" fill="none" />
    </svg>
  );
}

/**
 * A chequered flag on a pole. The squares are computed along a sine so the
 * cloth reads as waving while every square stays a crisp quad, no gradient.
 */
export function ChequeredFlag({ className }: { className?: string }) {
  const cols = 13;
  const rows = 8;
  const size = 26;
  const wave = (col: number) => 12 * Math.sin(col * 0.62) + col * 1.1;
  const point = (col: number, row: number) => `${16 + col * size},${(row * size + 18 + wave(col)).toFixed(1)}`;

  const squares = [];
  for (let col = 0; col < cols; col++) {
    for (let row = 0; row < rows; row++) {
      squares.push(
        <polygon
          key={`${col}-${row}`}
          points={`${point(col, row)} ${point(col + 1, row)} ${point(col + 1, row + 1)} ${point(col, row + 1)}`}
          fill={(col + row) % 2 ? "#15151e" : "#f7f4f1"}
        />,
      );
    }
  }

  return (
    <svg className={className} viewBox="0 0 370 320" aria-hidden focusable="false">
      <rect x="9" y="4" width="7" height="312" rx="3" fill="var(--paper-500)" />
      {squares}
    </svg>
  );
}

const ICON = {
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.6,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
  focusable: false,
};

/** Line icons, one per page or figure. Size comes from the caller's class. */
export const Icons = {
  wheel: (c?: string) => (
    <svg {...ICON} className={c}>
      <circle cx="12" cy="12" r="9" />
      <circle cx="12" cy="12" r="2.5" />
      <path d="M3.2 10.5 9.6 11.3M20.8 10.5 14.4 11.3M12 14.5V21" />
    </svg>
  ),
  telemetry: (c?: string) => (
    <svg {...ICON} className={c}>
      <path d="M3 4v16h18" />
      <path d="M6.5 15.5 10 10l3.5 3.5L19 6.5" />
    </svg>
  ),
  tree: (c?: string) => (
    <svg {...ICON} className={c}>
      <circle cx="12" cy="4.5" r="2" />
      <circle cx="5" cy="19" r="2" />
      <circle cx="12" cy="19" r="2" />
      <circle cx="19" cy="19" r="2" />
      <path d="M12 6.5V17M12 11H6.5A1.5 1.5 0 0 0 5 12.5V17M12 11h5.5a1.5 1.5 0 0 1 1.5 1.5V17" />
    </svg>
  ),
  chip: (c?: string) => (
    <svg {...ICON} className={c}>
      <rect x="6" y="6" width="12" height="12" rx="1.5" />
      <rect x="10" y="10" width="4" height="4" />
      <path d="M9.5 2.5V6M14.5 2.5V6M9.5 18v3.5M14.5 18v3.5M2.5 9.5H6M2.5 14.5H6M18 9.5h3.5M18 14.5h3.5" />
    </svg>
  ),
  podium: (c?: string) => (
    <svg {...ICON} className={c}>
      <path d="M3 21v-7h6v7M9 21V9h6v12M15 21v-5h6v5M2 21h20" />
      <path d="m12 2.5 1 2 2.2.3-1.6 1.5.4 2.2-2-1-2 1 .4-2.2-1.6-1.5 2.2-.3Z" />
    </svg>
  ),
  lens: (c?: string) => (
    <svg {...ICON} className={c}>
      <circle cx="10.5" cy="10.5" r="7" />
      <path d="m20.5 20.5-5-5" />
      <path d="M7.5 12.5v-1.5M10.5 12.5v-5M13.5 12.5v-3" />
    </svg>
  ),
  shield: (c?: string) => (
    <svg {...ICON} className={c}>
      <path d="M12 2.5 4.5 5.5V11c0 5 3.2 8.7 7.5 10.5 4.3-1.8 7.5-5.5 7.5-10.5V5.5Z" />
      <path d="m8.5 12 2.5 2.5 4.5-5" />
    </svg>
  ),
  stopwatch: (c?: string) => (
    <svg {...ICON} className={c}>
      <circle cx="12" cy="14" r="7.5" />
      <path d="M12 14V9.5M9.5 2.5h5M12 2.5v4M18.5 6l1.5 1.5" />
    </svg>
  ),
  flag: (c?: string) => (
    <svg {...ICON} className={c}>
      <path d="M5 21.5V3" />
      <path d="M5 4h15v10H5" />
      <path d="M5 9h15M10 4v10M15 4v10" />
      <path d="M5 4h5v5H5ZM15 4h5v5h-5ZM10 9h5v5h-5Z" fill="currentColor" stroke="none" />
    </svg>
  ),
  rules: (c?: string) => (
    <svg {...ICON} className={c}>
      <rect x="4.5" y="3.5" width="15" height="18" rx="1.5" />
      <path d="m8 9 1.2 1.2L11 8M8 15l1.2 1.2L11 14M13.5 9.5H16.5M13.5 15.5H16.5" />
    </svg>
  ),
};
