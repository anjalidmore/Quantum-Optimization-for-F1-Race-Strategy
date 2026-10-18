"""Line icons (24×24, currentColor) shared by the study guide and the task summary sheet."""
from __future__ import annotations

# Line icons, 24×24, drawn with currentColor.
ICONS = {
    "network": '<circle cx="5" cy="6" r="2.2"/><circle cx="19" cy="5" r="2.2"/><circle cx="12" cy="12" r="2.4"/>'
               '<circle cx="5" cy="19" r="2.2"/><circle cx="19" cy="18" r="2.2"/><path d="M7 7l3 3.5M17 6.5l-3 4M7 17.5l3-3.5M17 16.5l-3-3"/>',
    "ifthen": '<rect x="3" y="3" width="13" height="18" rx="1.5"/><path d="M6 8h7M6 16h7"/><path d="M16 8h4v8h-4"/>'
              '<path d="M18 14l-2 2 2 2"/>',
    "tree": '<circle cx="12" cy="4.5" r="2"/><circle cx="6" cy="12" r="2"/><circle cx="18" cy="12" r="2"/>'
            '<circle cx="3.5" cy="20" r="1.8"/><circle cx="9" cy="20" r="1.8"/><circle cx="18" cy="20" r="1.8"/>'
            '<path d="M11 6.2L7 10.4M13 6.2l4 4.2M5.3 13.8l-1.2 4.4M6.8 13.8l1.6 4.4M18 14v4.2"/>'
            '<path d="M18 18V15M18 9.5V7l3 1-3 1"/>',
    "database": '<ellipse cx="11" cy="5" rx="7" ry="2.6"/><path d="M4 5v13c0 1.5 3.1 2.6 7 2.6M18 5v6"/>'
                '<path d="M4 11.5c0 1.5 3.1 2.6 7 2.6"/><circle cx="18" cy="17.5" r="2.6"/>'
                '<path d="M18 13.5v1.4M18 20.1v1.4M14 17.5h1.4M20.6 17.5H22"/>',
    "gears": '<circle cx="8" cy="9" r="3"/><path d="M8 3.5v2M8 12.5v2M2.5 9h2M11.5 9h2M4.1 5.1l1.4 1.4M10.5 11.5l1.4 1.4'
             'M4.1 12.9l1.4-1.4M10.5 6.5l1.4-1.4"/><rect x="13" y="13" width="9" height="8.5" rx="1"/>'
             '<path d="M15 16l1 1 2-2M15 19.5h5"/>',
    "brain": '<path d="M9.5 4.5a3 3 0 0 0-5 2.5 3 3 0 0 0-1.5 5 3 3 0 0 0 1.5 4.5 3 3 0 0 0 5 2.5V4.5z"/>'
             '<path d="M14.5 4.5a3 3 0 0 1 5 2.5 3 3 0 0 1 1.5 5 3 3 0 0 1-1.5 4.5 3 3 0 0 1-5 2.5V4.5z"/>'
             '<path d="M9.5 9H7M9.5 14H6.5M14.5 9H17M14.5 14h3"/>',
    "neural": '<circle cx="4" cy="6" r="1.8"/><circle cx="4" cy="12" r="1.8"/><circle cx="4" cy="18" r="1.8"/>'
              '<circle cx="12" cy="8" r="1.8"/><circle cx="12" cy="16" r="1.8"/><circle cx="20" cy="12" r="1.8"/>'
              '<path d="M5.7 6.6l4.6 1M5.7 11.4l4.6-2.8M5.7 12.6l4.6 2.8M5.7 17.4l4.6-1M5.7 7.2l4.6 7.6M13.7 8.7l4.6 2.6'
              'M13.7 15.3l4.6-2.6"/>',
    "xai": '<circle cx="10" cy="10" r="6.5"/><path d="M15 15l6 6"/><path d="M7 12.5v-2M10 12.5V7.5M13 12.5v-3.5"/>',
    "puzzle": '<path d="M4 4h6a2 2 0 1 1 4 0h6v6a2 2 0 1 0 0 4v6h-6a2 2 0 1 0-4 0H4v-6a2 2 0 1 1 0-4z"/>',
    "clipshield": '<rect x="3" y="4" width="12" height="17" rx="1.5"/><path d="M6.5 4V2.5h5V4M6 9h6M6 12.5h4"/>'
                  '<path d="M17.5 11l4 1.5V16c0 2.6-1.8 4.3-4 5-2.2-.7-4-2.4-4-5v-3.5z"/><path d="M15.7 16l1.3 1.3 2.4-2.6"/>',
    "doc": '<path d="M6 2.5h8l4 4V21.5H6z"/><path d="M14 2.5v4h4M9 11h6M9 14.5h6M9 18h4"/>',
    "folder": '<path d="M3 6.5V19h18V8.5H11L9 6.5z"/><path d="M3 10h18"/>',
    "table": '<rect x="3" y="4" width="18" height="16" rx="1"/><path d="M3 9h18M3 14h18M9 9v11M15 9v11"/>',
    "stopwatch": '<circle cx="12" cy="13.5" r="7.5"/><path d="M12 13.5V9.5M10 2.5h4M12 2.5v3.5M18 7l1.5-1.5"/>',
    "cubes": '<path d="M12 3l7 4v8l-7 4-7-4V7z"/><path d="M5 7l7 4 7-4M12 11v8"/>',
    "monitor": '<rect x="2.5" y="3.5" width="19" height="13" rx="1"/><path d="M8 21h8M12 16.5V21"/>'
               '<path d="M6 13l3-3 3 2 4-5 2 2"/>',
    "flagpath": '<circle cx="4" cy="18" r="2"/><circle cx="12" cy="13" r="2"/><circle cx="20" cy="17" r="2"/>'
                '<path d="M5.7 17l4.6-3M13.8 13.9l4.4 2.2M12 11V3l5 2-5 2"/>',
    "laptop": '<rect x="4" y="4" width="16" height="11" rx="1"/><path d="M2 19h20l-2-4H4z"/>'
              '<path d="M12 7.5v5M9.5 10h5"/>',
    "pie": '<path d="M11 3a9 9 0 1 0 9 9h-9z"/><path d="M14 2.5V10h7.5A7.5 7.5 0 0 0 14 2.5z"/>',
    "heatmap": '<rect x="3" y="3" width="18" height="18" rx="1"/><path d="M3 9h18M3 15h18M9 3v18M15 3v18"/>',
    "trophy": '<path d="M7 3h10v6a5 5 0 0 1-10 0z"/><path d="M7 5H3.5v1.5A3.5 3.5 0 0 0 7 10M17 5h3.5v1.5A3.5 3.5 0 0 1 17 10'
              'M12 14v4M8 21h8M9 18h6"/>',
    "shield": '<path d="M12 2.5l8 3v6c0 5-3.5 8.5-8 10-4.5-1.5-8-5-8-10v-6z"/><path d="M8.5 12l2.5 2.5 4.5-5"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5.5"/><circle cx="12" cy="12" r="2"/>'
              '<path d="M12 12l8-8M17 4h3v3"/>',
    "flag": '<path d="M5 21V3"/><path d="M5 4h14v9H5"/><path d="M5 4h3.5v3H5zM12 4h3.5v3H12zM8.5 7H12v3H8.5zM15.5 7H19v3h-3.5z'
            'M5 10h3.5v3H5zM12 10h3.5v3H12z" fill="currentColor" stroke="none"/>',
    "report": '<path d="M5 2.5h9l4 4v15H5z"/><path d="M14 2.5v4h4"/><path d="M8 18v-3M11 18v-5M14 18v-7"/>',
}

# UI and category icons used by the study guide.
ICONS.update({
    "back": '<path d="M19 12H5M11 18l-6-6 6-6"/>',
    "chevron": '<path d="M9 6l6 6-6 6"/>',
    "bolt": '<path d="M13 2L4 14h7l-1 8 9-12h-7z"/>',
    "layers": '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/>',
    "flow": '<rect x="3" y="3" width="7" height="5" rx="1"/><rect x="14" y="16" width="7" height="5" rx="1"/>'
            '<path d="M6.5 8v4.5h11V16"/>',
    "rocket": '<path d="M12 15l-3-3c1.5-4.5 4.5-8 10-9-1 5.5-4.5 8.5-9 10z"/><path d="M9 12l-4 1-2 3 4 1M12 15l-1 4 3 2 1-4"/>'
              '<circle cx="15" cy="8" r="1.4"/>',
    "box": '<rect x="3" y="7" width="18" height="13" rx="1.5"/><path d="M3 11h18M9 7V4h6v3"/>',
    "loop": '<path d="M4 12a8 8 0 0 1 14-5.3M20 12a8 8 0 0 1-14 5.3"/><path d="M18 3v4h-4M6 21v-4h4"/>',
    "decision": '<path d="M12 2.5l9.5 9.5-9.5 9.5L2.5 12z"/><path d="M10 9.5a2 2 0 1 1 2.6 1.9c-.4.2-.6.5-.6 1v.6M12 15.5v.2"/>',
    "calendar": '<rect x="3" y="5" width="18" height="16" rx="1.5"/><path d="M3 10h18M8 3v4M16 3v4"/>',
    "data": '<path d="M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3z"/><path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    "atom": '<circle cx="12" cy="12" r="1.6"/><ellipse cx="12" cy="12" rx="10" ry="4"/><ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(60 12 12)"/>'
            '<ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(120 12 12)"/>',
})


def svg(name: str, size: int = 20, stroke: float = 1.8, color: str = "currentColor", label: str | None = None) -> str:
    """One inline icon. Decorative (aria-hidden) unless ``label`` is given."""
    a11y = f'role="img" aria-label="{label}"' if label else 'aria-hidden="true" focusable="false"'
    return (f'<svg class="ic" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" {a11y}>{ICONS[name]}</svg>')
