---
name: F1 Race Strategy Intelligence
description: Race-engineering analytics for one Grand Prix — dense, near-black, one racing accent
colors:
  track-000: "#08090B"
  track-050: "#0E1013"
  track-100: "#15181C"
  track-200: "#1D2126"
  track-300: "#2A2F36"
  paper-900: "#F4F1EA"
  paper-700: "#D8D4CA"
  paper-500: "#A9A59B"
  paper-400: "#8B8780"
  accent: "#FF2D16"
  accent-ink: "#0A0B0D"
  edge: "#5E6773"
  focus: "#F4F1EA"
  compound-soft: "#E8382B"
  compound-medium: "#F2D13C"
  compound-hard: "#EDEAE3"
  compound-inter: "#3FB950"
  compound-wet: "#3B8EEA"
typography:
  display:
    fontFamily: "Saira Condensed, ui-sans-serif, sans-serif"
    fontSize: "76px"
    fontWeight: 700
    lineHeight: 0.92
    letterSpacing: "-0.02em"
  body:
    fontFamily: "IBM Plex Sans, ui-sans-serif, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.55
    letterSpacing: "normal"
  data:
    fontFamily: "IBM Plex Sans, ui-sans-serif, sans-serif"
    fontSize: "13px"
    fontWeight: 500
    lineHeight: 1.4
    fontVariantNumeric: "tabular-nums"
  label:
    fontFamily: "IBM Plex Sans, ui-sans-serif, sans-serif"
    fontSize: "12px"
    fontWeight: 500
    lineHeight: 1.2
    letterSpacing: "0.01em"
rounded:
  sm: "2px"
  md: "4px"
  lg: "6px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "16px"
  lg: "24px"
  xl: "32px"
  xxl: "48px"
  xxxl: "72px"
motion:
  fast: "120ms"
  base: "180ms"
  ease: "cubic-bezier(0.2, 0, 0.3, 1)"
---

# Design System: F1 Race Strategy Intelligence

## Overview

**Creative North Star: "The pit wall at 3am"**

A race engineer's screen during a long run: near-black so the room stays dark, numbers large enough
to read at a glance and aligned enough to compare down a column, and exactly one red — the colour
the sport already uses to mean *act now*. Nothing on screen is decoration. Every figure traces to a
generated artifact, and where a figure would be misleading the interface says so in words rather
than dressing it up.

Density is high and motion is low. This is an instrument you read, not a page you scroll through.
The information is genuinely dense — 17 drivers, 10 models, 32 rules, 360 trust rows — so the design
job is hierarchy and alignment, not filling space. Hierarchy comes from size and weight contrast: a
76px figure against a 13px table row, not a 12px label inside another bordered box. The current UI
gets this wrong with 59 equal cards and everything set between 11px and 14px.

Typography carries the personality. A squared condensed face (Saira Condensed) does the headlines
and the big figures, because narrow industrial type is the vernacular of timing screens and it lets
a large number sit in a narrow column. A technical humanist face (IBM Plex Sans) does everything
else, with tabular figures throughout.

Confirmed rejections: no glassmorphism beyond the one justified surface (the nav), no gradients anywhere, no
purple or blue as interface colour, no Inter/Roboto/Arial/Space Grotesk, no driver photos, team
logos or liveries, no live/streaming indicators, and no circuit map — the data for one does not
exist (see Constraints).

**Key characteristics:**
- Near-black in four steps; warm off-white text so the dark never reads blue-grey
- One accent, bright enough to pass AA as text, used once per view
- Tyre-compound colours reserved strictly for compound meaning, never chrome
- Tabular figures on every number; units dimmed and smaller than the value
- Sharp corners (2–6px) — instrumentation, not a SaaS card
- Elevation by surface step and hairline, never by shadow

## Colors

Four near-blacks, a four-step warm off-white, one racing red, and five compound colours that only
ever mean a compound. Ratios below are measured against the surface each token is used on.

### Primary

`--accent: #FF2D16` — **5.1:1 on `--track-050`**, so it is legible as text and not only as a fill.
The old `#E10600` failed at 3.86:1, which is why it could only ever be a button.

Used for: the primary action, the single figure a page exists to deliver, the marker on a
best/worst row, and the 3px rule in the wordmark. **One of those per view.** Label on an accent
fill is `--accent-ink` (5.3:1).

### Neutral

| Token | Value | Role | Contrast |
|---|---|---|---|
| `--track-000` | `#08090B` | page ground, chip ink | — |
| `--track-050` | `#0E1013` | primary surface | `paper-900` 16.9:1 |
| `--track-100` | `#15181C` | raised: tables, panels | `paper-900` 15.8:1 |
| `--track-200` | `#1D2126` | input wells, row hover | — |
| `--track-300` | `#2A2F36` | decorative hairline | non-text, exempt |
| `--edge` | `#5E6773` | control boundary | **3.3:1** — meets non-text AA |
| `--focus` | `#F4F1EA` | focus ring | 16.9:1 |
| `--paper-900` | `#F4F1EA` | headlines, primary figures | 15.8–17.7:1 |
| `--paper-700` | `#D8D4CA` | body | 12.0:1 |
| `--paper-500` | `#A9A59B` | labels, secondary | 7.2:1 |
| `--paper-400` | `#8B8780` | dimmed units, footnotes | 5.0:1 |

### Named Rules

- **`compound-*` is data, never decoration.** Soft `#E8382B`, Medium `#F2D13C`, Hard `#EDEAE3`,
  Intermediate `#3FB950`, Wet `#3B8EEA`. They appear only inside a compound chip, and the chip
  always carries the letter (S/M/H/I/W) as well as the colour — colour is never the only carrier of
  meaning. Chip ink is `--track-000`, 4.8:1 on the worst case (soft).
- **`--soft` and `--accent` are different reds on purpose.** Accent is interface; soft is data. They
  never sit adjacent at the same size.
- **No emerald.** The two existing "Completed" pills become `--paper-500` text with a filled square.
  The detector flags emerald (SD8) as the reflex accent once purple is ruled out, and here it
  carries no meaning that a filled square does not.
- **Amber is not a status colour.** The current dataset badge is amber, which reads as a warning
  when it is stating a fact. Provenance is a bordered pill in `--paper-500`.

## Typography

Two families, clearly distinct, loaded through `next/font/google` (self-hosted at build; no CDN
request at runtime).

| Role | Family | Weights |
|---|---|---|
| Display + figures | **Saira Condensed** | 500, 600, 700 |
| Text + tables | **IBM Plex Sans** | 400, 500, 600 |

### Hierarchy

| Token | Size / line | Family, weight | Use |
|---|---|---|---|
| `display-hero` | 76 / 0.92, −0.02em | Display 700 | The one figure or headline a page exists for |
| `display-lg` | 44 / 1.0 | Display 600 | Section figure |
| `display-md` | 28 / 1.1 | Display 600 | Panel figure, stat tile |
| `title` | 19 / 1.3 | Text 600 | Page title, panel heading |
| `body` | 14 / 1.55 | Text 400 | Prose |
| `data` | 13 / 1.4, tabular | Text 500 | Every table cell and figure |
| `label` | 12 / 1.2, 0.01em | Text 500, `paper-500` | Field labels, column heads — sentence case |
| `micro` | 11 / 1.3 | Text 400, `paper-400` | Units, footnotes, artifact provenance |

### Named Rules

- **Every number is `font-variant-numeric: tabular-nums`.** No exceptions. A figure that updates
  must not reflow its column.
- **Units are dimmed and smaller**: `0.565` in `paper-900` at `display-*`, `s` in `paper-400` at
  `micro`. The value is content; the unit is caption.
- **No tracked all-caps labels.** Sentence case in `paper-500`. The audit found 52; target zero.
- **One display element per page maximum.** If two things are 44px+, the page has no focus.
- **Driver codes use the display face** (600, +0.03em) — ALB, VER, NOR read as identifiers, not prose.

## Spacing & Layout

4px base, applied strictly: `sm` inside a table row, `md` inside a panel, `xl` between panels,
`xxl`/`xxxl` above a page's focal element. **Vertical rhythm is deliberately uneven** — the focal
element gets more air than anything else, which is how the eye finds it.

Radii: `sm 2px` for chips and inputs, `md 4px` for panels, `lg 6px` for the single feature panel per
page. Sharp by intent; the current `rounded-xl` on every card is what makes this read as a template.

Elevation is a surface step plus a hairline, never a shadow:

| Level | Treatment |
|---|---|
| 0 | `--track-050`, no border — the page |
| 1 | `--track-100` + `1px solid var(--track-300)` — panels, tables |
| 2 | `--track-100` + `1px solid var(--edge)` — the focal panel, a focused control |

Grid: a 1320px content measure, 32px gutters. The Overview and Strategy pages are two columns
(≈62/38 and ≈55/45); the table-first pages are single-column with full-width tables.

## Components

**Focus — every interactive element.** `:focus-visible { outline: 2px solid var(--focus);
outline-offset: 2px }`. The audit found zero focus styles; this is the largest accessibility fix.

**Data table.** `--track-100` surface. Column heads in `label` above a `--edge` hairline. Text
left-aligned, figures right-aligned and tabular. Rows separated by `1px var(--track-300)` — **no
zebra striping**. Hover lifts the row to `--track-200`. A best/worst row is marked by a 2px
`--accent` inset edge *and* a text marker, never colour alone.

**Tyre compound chip.** 14px square in the compound colour, letter in `--track-000`, name in `body`.

**Stat tile.** `label` above, figure in `display-md` with a dimmed unit. **No border, no card** —
tiles are separated by space and one hairline. 59 bordered boxes is the current mistake.

**Status pill.** Real state only. Permitted: dataset provenance (`Real · Bahrain 2023 · 995 laps`),
build state (`Not generated yet`), a model verdict (`Overfitting`). Banned: `LIVE`, viewer counts,
anything implying a running session.

**Empty state.** `Not generated yet` in `body` plus the exact command in `micro`. Already correct in
the reasoning cards; extend to every data component.

**Loading.** A `--track-100` block at the final height of the content it replaces. No shimmer. Add
`loading.tsx` per route — there are none today.

**Error.** Keep the current wording, which is honest and actionable; restyle to an `--accent`
hairline rather than an amber fill.

**Form (Strategy).** Grouped by what an engineer changes together — *Car*, *Tyre*, *Track*, *Race* —
with group headings in `label`. Inputs sized to their content: a 2-digit lap number does not need
380px. One "About these inputs" disclosure replaces eleven per-field question-mark buttons.

## Motion

Low, and always interruptible.

- Permitted: colour/border/opacity on hover and focus (`--t-fast`), disclosure height
  (`--t-base`), and a single 180ms fade for content that has just arrived.
- Banned: entrance animations, staggered reveals, **number count-ups** (they make real data look
  fake), parallax, anything driven by scroll.
- Every transition sits inside a `prefers-reduced-motion: reduce` guard that collapses duration to
  0.01ms. There are currently zero such guards against five `transition` utilities.

## Page Patterns

**Overview — dominant figure left, stacked detail right.** Two columns ≈62/38. Left: the single most
consequential figure with its provenance. Right: dense per-task rows, not ten cards. Below: the real
per-driver error table as a leaderboard. The workflow chip strip is removed — it carries no data.

**Strategy Simulator — grouped form left, result right.** The result column is the focal element and
sticks as the form beside it changes. It is a solid panel: nothing scrolls beneath it, so glass
there would be decoration.

**Models / Explainability / Reasoning / Data — table-first.** Title, one sentence, then tables.
Section headings in `title` weight. Charts stay as images from `artifacts/`, framed on `--track-100`
with a `micro` caption naming the source file.

**Every page:** one focal element, then density. Hierarchy by size and weight — never by adding
another box.

## Constraints

**No circuit or position map.** Checked: no X/Y or track-outline data exists anywhere in the
project. `data/raw/circuits.csv` holds six *synthetic* circuits with a single lat/lng point each, and
Bahrain — the session actually analysed — is not among them. One coordinate cannot draw a circuit,
and an invented outline would be decoration pretending to be data.

**No sparklines** on the Overview or Models pages: there is no per-lap series behind those figures.

**One glass surface, and it is the nav.** `--nav-surface` is the page ground at 78% with
`backdrop-filter: blur(10px) saturate(1.4)` and a `--edge` hairline beneath. It earns the treatment
because content genuinely passes under a sticky header, and the blur says so. Nothing else in the
app may use `backdrop-filter`; a panel that does not have content moving beneath it gets a solid
surface step instead.

The opacity is bounded from below by contrast, not by taste. Nav text must clear AA against the
lightest thing that can scroll behind it, which is `--paper-900` body text: at 78% that composites
to `#414242`, where `--paper-700` measures **6.8:1** and `--paper-500` only **4.1:1**. So the nav's
idle links are `--paper-700`. Dropping the surface to 70% would put even `--paper-700` at 5.6:1 and
the active/idle distinction under strain; raising it past 85% stops reading as glass at all.
`@supports` falls back to a solid `--track-050` where `backdrop-filter` is unavailable, because
unreadable text over moving content is worse than no effect.

Two things in the nav measure below 3.0 against a pure-white backdrop (a matplotlib PNG scrolling
under it, which composites the surface to `#434547`), and both are deliberate:

- The 3px accent bar beside the wordmark reaches 2.59:1. It is ornament, not a control or a
  graphical object needed to understand anything; the wordmark beside it measures 8.5:1 and carries
  the meaning on its own.
- The `--edge` hairline reaches 1.68:1 there. The nav's boundary is carried by whichever of the two
  has contrast to spare: over dark content the hairline does it at 3.3:1, and over light content the
  surface's own luminance step is unmissable. There is no backdrop where both are weak at once.

**Justified detector exception.** `avoid-ai-design` flags SD2 — *near-black ground plus one
vermilion accent* — as a second-order default. It is kept, with reason: red on black is the
subject's own livery language, the accent is used as data emphasis rather than as a decorative
escape from purple, and the catalogue's own guidance is to judge clusters rather than single hits
(SD1/SD4/SD5/SD6 are all absent). Recorded in code as `avoid-ai-design-ignore: SD2` with this
justification.

## Current State Before This System

Recorded so the change is measurable. Screenshots: `design-screenshots/before/` (1440px and 390px).

| Aspect | Before |
|---|---|
| Typeface | **none set** — browser default (Helvetica) |
| Accent | `#E10600`, 6 uses, 4 of them buttons; fails AA as text (3.86:1) |
| Type scale | `text-sm` ×109, `text-xs` ×64, `text-2xl` ×7; every `<h1>` identical |
| Layout | 37 `grid-cols-*`, 59 bare `.card` |
| Labels | 52 tracked all-caps eyebrows |
| Tabular figures | 18 uses; `.stat-value` had none |
| Focus states | 0 |
| Reduced-motion guards | 0 (against 5 transitions) |
| Mobile nav | none (`hidden lg:flex`, no alternative) |
| Compound colour | unused |
| Slop scan | **12 findings · P0 0 · P1 7 · P2 5** |

## Tool Reconciliation

| Input | Taken | Rejected, and why |
|---|---|---|
| `ui-ux-pro-max` | Density 9/10, motion 3/10, its AA requirements (4.5:1 text, keyboard, visible focus, reduced motion), "Data-Dense Dashboard" style keywords | Its palette — `#1E40AF` blue on `#F8FAFC` light — and its "Enterprise Gateway" pattern with *Contact Sales* CTAs: this is not a marketing site, and blue is the tell the brief rules out. Its Fira Code/Fira Sans pairing: no display voice, and a code face for headlines is the mono-chrome tell |
| `avoid-ai-design` | All 12 findings; the second-order (SD) cluster warning | Nothing. SD2 kept with the justification above |
| `frontend-design` | Ground the design in the subject; type as an active element; avoid the default big-number-plus-gradient hero | Nothing |
| `impeccable` | This file's structure (official DESIGN.md spec, token frontmatter, canonical headings); the craft floor's bans | Its binary was not run — it downloads from the network, so its playbooks were followed by hand |
| Inspiration brief | Editorial hero with a small real status pill; dominant-left / stacked-right; dense tabular leaderboard; compact tiles with dimmed units; one red on near-black | The circuit map (no data — see Constraints) |
