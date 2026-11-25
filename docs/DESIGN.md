# VMI Analysis workbench design

Act as a product designer and design engineer for a **scientific instrument
workbench**. The VMI Analysis GUI is a desktop-class web application
(NiceGUI + Quasar, served locally) that physicists use to configure, run,
and inspect velocity-map-imaging analysis pipelines: centroiding, pBasex
inversion, circularization, calibration, binning, PES, and batch pipelines.
Every design decision must serve someone who is comparing detector images,
reading fit quality, and babysitting long-running jobs — often for hours.

Make the interface precise, calm, dense, and legible. Build confidence
through clear state, honest feedback, and command of the data. Never
manufacture confidence through decoration, novelty, or motion.

This document is prescriptive: follow it when adding panels, restyling
components, or extending the theme. It states principles and the shared
vocabulary; it does not re-document the implementation. The authoritative
sources are:

- [Agent-first design](agent-first-design.md) — principles for the embedded
  Agent, shared panel operations, and reusable analyses.

- `src/vmi_analysis/gui/assets/theme.css` — the token scales. **Tokens are
  the design system.** If a value is not a token, it does not belong in
  component CSS.
- `src/vmi_analysis/gui/theme.py` — Quasar palette glue and the component
  helper vocabulary (`panel_header`, `form_section`, `run_button`,
  `status_badge`, `set_badge`, `log_console`).
- `src/vmi_analysis/gui/panels/common.py` — panel scaffolding (`JobView`,
  `field_grid`, `check_grid`, `action_bar`, `path_input`, zoom overlay).
- [architecture.md](architecture.md) — implemented panels and previews.
- [AGENTS.md](../AGENTS.md) — analysis and GUI invariants that the UI must
  never violate.

## Product context

- **Audience:** expert users (researchers, students in the lab), not
  consumers. They tolerate density; they do not tolerate ambiguity about
  what a job is doing or where its output went.
- **Session shape:** configure → run → watch log → inspect results → adjust
  → rerun. A session can outlive the browser tab; jobs belong to the server.
- **Data first:** detector images, fit curves, and radial spectra are the
  payload. Chrome exists to frame them and disappears under them.
- **One window, one workspace:** this is a workbench, not a site. There is
  no marketing surface, no hero, no onboarding carousel.

## Priority order

When requirements compete, protect them in this order:

1. **Analysis integrity.** Never let styling or convenience change what a
   job computes, writes, or overwrites. Honor every invariant in AGENTS.md
   (subprocess isolation, resource reservations, one run button per job,
   preview-only image manipulation, preserved user data).
2. **State honesty.** The user must always be able to answer: is something
   running, what stage, can I safely click this, where did the output go?
3. **Legibility of data.** Plots, images, logs, and paths stay readable at
   working distance; nothing decorative competes with them.
4. **Token and helper consistency.** Use the shared tokens and helpers;
   a one-off style is a defect even when it looks better in isolation.
5. **Flourish.** Motion, whitespace generosity, and polish — valuable, but
   never purchased with any of the above.

## The visual system

The theme is a **light Analytics Dashboard**: cool slate canvas
(`#f8fafc`), white card surfaces, deep blue ink (`#1e3a8a`), primary action
blue (`#1e40af`), amber reserved for warnings and fit overlays. Fira Code
is the display face (titles, status values, console, data identifiers);
Fira Sans is body and control copy. It reads as a laboratory instrument
panel, not a SaaS landing page.

### Token discipline

- Reference `--vmi-*` tokens everywhere: color, spacing
  (`--vmi-space-1…8`, 4px base), radii (`--vmi-radius-sm…lg`), shadows,
  shell metrics, text sizes, motion (`--vmi-speed`, `--vmi-ease`).
- Never inline a hex value in component CSS or inline styles. If the right
  value does not exist, add a token to `theme.css` and use it. One-off
  values fork the system silently.
- Keep Quasar's palette in sync through `theme.apply_theme()`'s
  `ui.colors(...)`; that is the single bridge between Quasar props
  (`color="primary"`) and the token palette.
- Respect the `VMI_GUI_OFFLINE=1` path: fonts must degrade to the
  monospace/sans fallback stacks without breaking layout.

### Surfaces, borders, elevation

- The canvas is flat. Cards (`vmi-panel`) carry a 1px hairline and the
  faintest shadow (`--vmi-shadow-1`). Depth comes from surface contrast,
  not from stacked shadows.
- Earn every boundary. Prefer spacing and alignment before a border; a
  border before a shadow. Never nest bordered cards inside bordered cards
  to repair weak grouping — regroup the content instead.
- Semantic color is scarce and always paired with a non-color cue: status
  badges pair tint + label text; notifications pair a tinted surface + a
  3px semantic left border + message text; the status dot pairs color + a
  text label. Color alone never carries meaning.

### Typography

- Hierarchy is set by role classes (`vmi-title-*`, `vmi-subtitle`,
  `vmi-caption`, `vmi-overline`), never by ad-hoc font sizes or weights.
- Fira Code marks *data and system voice*: panel titles, the sidebar word,
  status values, console output, paths and identifiers. Fira Sans is
  *human voice*: descriptions, labels, buttons. Do not mix the voices —
  a paragraph in mono or a button label in the display face is off-system.
- Numbers that change while watched (counts, progress, queue depth) use
  tabular figures (`vmi-numeric` / `font-variant-numeric: tabular-nums`)
  so they do not jitter.
- Buttons and tabs are sentence case, never all-caps (`text-transform:
  none` is part of the theme; do not reintroduce Quasar's uppercase
  default).

## Shell composition

The [Agent-native design guide](agent-first-design.md) defines which operations
the Agent can prepare or inspect. Use the same workbench controls for manual
and Agent-approved work.

The shell is a persistent workbench with four regions, and only four:

- **Header** — application name and shared-account identity when applicable.
  Navigation and Settings live in the rail.
- **Left rail** — the tool index, with independently collapsible *Electron*
  and *Ion* groups. *Acquire* and *Analyze* are section captions
  inside *Electron*. Each tool has an icon and a short name. The active tool
  has a colored label and a 3px indicator at the leading edge.
  The rail can collapse to icons on desktop, with tooltips and accessible
  names for every tool. Its bottom action is Settings, with a stronger blue
  treatment than the tool entries. Keep Settings available on narrow screens.
  The Ion group includes a nested Fallback disclosure. It starts closed and
  contains the three manual comparison tools. The Agent tab appears above
  Settings in local and sandboxed shared mode.
- **Workspace** — exactly one active panel. Panels switch; they do not
  stack, scroll into each other, or open side by side.
- **Status footer** — one compact line of global state: active jobs, queued
  jobs, job limit, server mode (with the keep/auto-stop dot). It never
  scrolls away and never grows to two lines.

The author's signature lives under *Settings → About*, never in the chrome.
Persistent branding, greeting text, and decorative headers are rejected:
the shell frames work, it does not introduce itself.

## Panel anatomy

Every job panel follows **one anatomy**, assembled from the shared helpers.
A panel that invents its own layout vocabulary is a defect.

1. **Header** — `panel_header(title, subtitle, view)`: the task name in
   `vmi-title-sm` with the job status cluster (badge, spinner, progress,
   wait reason) inline in the same row, and one short sentence saying what
   the stage does — without naming the Python module.
2. **Sections** — `form_section(caption)`: a small-caps overline with a
   hairline rule, grouping related fields. Sections order as the user
   thinks: inputs → parameters → output.
3. **Fields** — `field_grid(columns)` for even input columns, `check_grid()`
   for option rows, `path_input()` for every filesystem path (full-width,
   with a keyboard-operable Browse button in the append slot).
4. **Action row** — `action_bar(view, label)`: **one** primary Run button
   that becomes Stop/Cancel while work is active (`JobView.run_button`).
   Secondary and destructive actions sit beside it; destructive actions use
   the flat `vmi-action-danger` treatment and always confirm first.
5. **Console + results** — `place_console()` at the panel bottom: a
   collapsible, keyboard-operable console dock, collapsed at rest,
   auto-opening once per job, with a monospaced log and a results area
   that shows an explicit empty state ("No results yet — …") before the
   first result.


### State and feedback rules

- **One button owns the job.** A panel never has separate Run and Stop
  buttons; the same control toggles, its label follows the job
  (`Run` → `Stop` / `Cancel queued` → `Stopping…`), and a stale Stop can
  never launch a new job. The button's `job_name` matches the submitted
  job so a reopened page reconnects to server-owned work.
- **Status is always visible, never noisy.** The badge states
  (`idle/queued/running/done/failed/cancelled/skipped`) use the shared
  `vmi-badge-*` tints; spinner and progress bar appear only while running;
  a queued job explains itself with its wait reason.
- **The console answers "what is it doing"; results answer "what did I
  get."** Logs stream; results build once at terminal state. Deleting
  outputs restores the empty-state placeholder instead of leaving stale
  imagery.
- **Notifications are quiet surfaces.** Quasar paints notification roots
  with layered `!important` utilities that outrank author styles — so the
  theme styles the inner wrapper, never fights the root with more
  `!important`. When styling any Quasar component, check whether its own
  utilities are layered before writing an override.
- **Failures are actionable.** Error notifications say what failed and
  where to look; destructive confirmations list the exact files they will
  touch.

### Agent conversation and approval

- Use the workspace for the conversation and keep one page scroll surface.
  Keep the composer at the bottom of the viewport and **Connection** next
  to the Agent title. Connection contains diagnostics and model setup.
- Place ongoing activity in the conversation. Keep the composer controls
  for input, attachments, model selection, and Send or Stop.
- Show resolved inputs, parameters, and outputs beside **Approve and run**
  and **Decline**. For interactive review, return to the source panel and
  keep the original run and selection visible while the user adjusts it.
- Use the ordinary panel header, console, and results for approved jobs.
  Present a live receiver or Agent session as an interactive view with its
  own lifecycle.

## Data, imagery, and plots

- **Images are the payload.** Result images render with the shared
  `render_image` settings (viridis, log norm, nearest-neighbor, axes off)
  so every panel presents detector data identically.
- **Inspection is preview-only.** The shared `image_zoom.js` overlay
  (pan, wheel zoom, crop-to-region, reset) never mutates data or the saved
  settings — it only changes what the user sees. Reset always restores the
  original image. Any new image interaction follows the same rule.
- **Overlays are annotations, not chrome.** Polar guides and fit overlays
  draw in the image layer, use the warning amber only where they already
  do, and stay pointer-transparent except on their handles.
- **Plots earn their ink.** Axes, legends, and gridlines exist to be read;
  truncate nothing, label units, and keep plot chrome out of the panel's
  accent palette so data series own the color.
- **Empty states are explicit.** Before the first run, a dashed
  `vmi-results-empty` placeholder with one caption line — never a blank
  rectangle, never a skeleton that implies content is loading.

## Motion and density

- Density is high (dashboard class): 4px base unit, compact controls
  (`dense` inputs, 34–40px rows), tight gutters from the spacing scale.
  Whitespace separates groups; it does not pad for comfort.
- Motion is functional and short (`--vmi-speed` 160ms, `--vmi-ease`):
  the panel-enter fade, the console chevron rotation, hover color
  transitions. Nothing bounces, slides across the screen, pulses, or draws
  attention to itself.
- Honor `prefers-reduced-motion`: the theme collapses all animation and
  transition durations. Any new animation must respect the same media
  query.

## Accessibility

- Every interactive element is reachable and operable by keyboard. Use a
  named button for Browse. The custom console header has `role="button"`,
  `tabindex="0"`, and Enter/Space handlers.
- Focus is visible: the shared `:focus-visible` outline (2px primary, 2px
  offset) applies to every new control.
- State is announced, not implied: `aria-expanded` on the console header,
  `aria-pressed` on the overlay toolbar toggles, `aria-label` on icon-only
  controls.
- Contrast meets WCAG AA on every surface; muted text is for genuinely
  subordinate copy only. Verify any new tint pair, do not inherit
  assumptions.
- Text stays readable when zoomed or when fonts fall back (offline mode);
  layouts use `min-width: 0` and reflow instead of clipping.

## Responsive behavior

- Desktop-first (this is a lab workstation tool), then degrade gracefully:
  below 900px the rail becomes a horizontal scrollable strip (section
  captions and the rail footer hide), the workspace padding tightens, and
  footer notes disappear so the status bar stays one line.
- Below 600px the header subtitle hides and title sizes step down through
  the role classes — never by shrinking individual elements with one-off
  rules.
- Grids reduce columns rather than squeezing fields; logs and wide images
  scroll within their container rather than wrapping.

## Do / Don't

### Do

- Build every panel from `panel_header` / `form_section` / `field_grid` /
  `action_bar` / `place_console`, in that order.
- Add a `--vmi-*` token when a value is needed in more than one place —
  and when it is needed in one place.
- Keep summaries to one short sentence describing the task.
- Pair every color signal with text, an icon, a border, or a label.
- Run the browser regression checks (`tests/browser/`) after any change to
  shell, panel anatomy, console, or notification styling.

### Don't

- Don't inline hex values, pixel paddings, or font sizes in components.
- Don't create per-panel button styles, badge colors, or log treatments.
- Don't add shadows, gradients, glass effects, or rounded-pill decoration
  to raise visual interest — raise hierarchy instead.
- Don't let preview interactions (pan, zoom, crop) touch data or saved
  settings.
- Don't add a second Run/Stop control, a second status badge, or a second
  console to one job panel.
- Don't fight Quasar's layered utilities with author `!important`; move
  the styling to an element Quasar does not paint.
- Don't introduce uppercase button/tab labels, emojis as icons, or
  marketing-page patterns (heroes, card grids of features, CTAs).

## Extending the system

1. **One component at a time.** Identify the helper or token that owns the
   change; edit there, not at the call site. If no helper owns it, add one
   to `theme.py` or `common.py` following the existing naming
   (`vmi-<part>`, `vmi-<part>-<variant>`).
2. **States live beside their base.** Badge, notification, and button
   variants are sibling classes (`vmi-badge-running`, …) driven by the
   same token set — never a forked copy of the component.
3. **Keep the two sources of truth honest.** Token changes go in
   `theme.css`; Quasar palette changes go in `theme.apply_theme()`. If
   they disagree, the UI forks — change both in the same commit.
4. **Update the docs that describe what you changed:** this file for
   principles, `architecture.md` for implemented panels, `AGENTS.md` for
   invariants and workflows.
5. **Verify in the browser, both states.** Launch the GUI and check the
   panel at rest and while a job runs; run `tests/browser/`; check the
   narrow-width collapse and keyboard traversal before considering the
   change done.

## Pre-delivery checklist

- [ ] Panel uses the shared anatomy helpers; no local layout vocabulary.
- [ ] Every value resolves to a `--vmi-*` token or a helper class.
- [ ] Status, progress, and console behave at rest, running, queued, and
      terminal states — including reconnecting to a running job.
- [ ] Empty state present before first results; clear/delete restores it.
- [ ] Keyboard: all controls reachable, focus visible, custom targets have
      role/tabindex/key handlers and correct aria state.
- [ ] Color never the sole carrier of meaning; new tints meet AA contrast.
- [ ] Motion is token-timed, subtle, and collapses under reduced motion.
- [ ] Narrow viewports: rail strip scrolls, status bar stays one line,
      nothing clips.
- [ ] `uv run python -m unittest discover -s tests -v` and the relevant
      `tests/browser/` checks pass.
