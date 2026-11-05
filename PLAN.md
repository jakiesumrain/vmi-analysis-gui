# Plan: NiceGUI web interface for the VMI analysis scripts

## Goal

A minimal, comfortable NiceGUI web app (uv-managed) with one tab panel per script
plus a pipeline panel, themed by the validated `DESIGN.md` (warm cream canvas,
coral CTAs, dark log consoles). Scripts stay standalone-runnable; the GUI drives
them as subprocesses with live log streaming, progress, cancellation, and
image/video result display.

## Key facts established during exploration

- uv project root: `E:\work\projects\VMI-FEL` (`pyproject.toml`, Python 3.12.9,
  `.venv` present). `nicegui` not installed yet. Scripts already run in this venv.
- `circularization/circularization.py` already has a complete argparse CLI — no
  changes needed; its panel wraps it directly.
- `centroiding/vmi_histogram_gmd_bins.py` (added after the first plan draft)
  completes the pipeline:
  `build_gmd_map.py → process_andor_vmi_PICASSO_v5_h5_batch.py →
  vmi_histogram_gmd_bins.py → process_gmd_bins.py`
  (the latter invokes `circularization.py` internally per bin).
  `h5_play_frames_movie.py` and `h5_accumulate_fast_gmd.py` are QC/side tools.
- `process_gmd_bins.py` must run sequentially (shared MATLAB `examples/image.txt`),
  and reads constants from its module top (`WEIGHT_SUFFIX`, `INVERSION_BACKEND`, ...).
- `process_andor_vmi_PICASSO_v5_h5_batch.py` reads `config.toml` (cwd-relative)
  and ends with `plt.show()` — harmless under `MPLBACKEND=Agg`.
- `h5_play_frames_movie.py` plays a matplotlib `FuncAnimation`; GUI mode needs a
  headless render path (write mp4/gif, skip `plt.show()`).
- `DESIGN.md` lints clean (0 errors). Tokens: canvas `#faf9f5`, primary coral
  `#cc785c`, ink `#141413`, surface-card `#efe9de`, surface-dark `#181715`,
  success/warning/error, Cormorant Garamond (Copernicus substitute) + Inter +
  JetBrains Mono stacks.

## Layout (new files, all inside `data_20261104/`)

```
gui/
├── app.py            # entry: theme + tabs + ui.run()
├── theme.py          # parse DESIGN.md YAML -> ui.colors + CSS + component helpers
├── theme.css         # static CSS mapped from DESIGN.md tokens (hand-written, no node at runtime)
├── jobs.py           # JobManager: subprocess queue, log capture, progress parse, cancel
└── panels/
    ├── __init__.py
    ├── gmd_map.py      # build_gmd_map.py form
    ├── movie.py        # h5_play_frames_movie.py + video/frame-slider display
    ├── accumulate.py   # h5_accumulate_fast_gmd.py + live partial image
    ├── centroid.py     # config.toml form editor + Picasso run
    ├── binning.py      # vmi_histogram_gmd_bins.py form + bins PNG
    ├── circularize.py  # circularization.py CLI form
    ├── pes.py          # process_gmd_bins.py form + result PNG gallery
    └── pipeline.py     # ordered stage runner with per-stage status/resume
```

## Step 1 — Dependencies

At `E:\work\projects\VMI-FEL`: `uv add nicegui pyyaml`.
GUI and job subprocesses both use the venv interpreter (`sys.executable`) — no
`uv run` at runtime (avoids the interpreter-query issue documented in AGENTS.md).

## Step 2 — Minimal CLI refactors (standalone behavior unchanged)

All edits wrap the existing `--- config ---` blocks in argparse with **defaults
equal to current values**; running the scripts bare behaves exactly as today.

1. `build_gmd_map.py` — args: `--h5-dir --gmd-path --output-path --n-files`.
2. `h5_play_frames_movie.py` — args mirroring its config block plus
   `--headless` (skip playback, just write the movie) and `--dump-frames DIR`
   (write per-frame PNGs for the GUI scrub slider).
3. `h5_accumulate_fast_gmd.py` — args mirroring its config block plus
   `--no-show` and `--live-path PATH` (save the partial accumulated image every
   N completed files so the GUI can show it building up).
4. `centroiding/process_andor_vmi_PICASSO_v5_h5_batch.py` — accept
   `--config PATH` (default `config.toml`, cwd-relative as today). GUI edits a
   config file from the form and passes it; `MPLBACKEND=Agg` in the subprocess
   env neutralizes `plt.show()`.
5. `circularization/process_gmd_bins.py` — args: `--weight-suffix --bin-dir
   --output-dir --inversion-backend --pyabel-method --use-cache --no-show`.
   Replace the hardcoded `uv run circularization.py` subprocess with
   `sys.executable circularization.py` (same behavior, no uv dependency).
6. `circularization/circularization.py` — **no change**.
7. `centroiding/vmi_histogram_gmd_bins.py` — args: `--config --thresholds
   3.75e-8,5.5e-8,7.25e-8 --gmd-col --weight-mode --weight-power --no-show`.
   Also make the matplotlib backend respect `MPLBACKEND` (currently hardcoded
   `TkAgg`), falling back to TkAgg only when unset.

## Step 3 — `gui/theme.py` + `gui/theme.css`

- Parse `DESIGN.md` YAML front matter with PyYAML; resolve `{colors.x}` refs.
- `apply_theme()`: `ui.colors(primary=#cc785c, secondary=#6c6a64, accent=#e8a55a,
  dark=#181715, positive=#5db872, warning=#d4a017, negative=#c64545)` and inject
  `theme.css` via `ui.add_head_html()` + Google Fonts links (Cormorant Garamond,
  Inter, JetBrains Mono) with the token fallback stacks if offline.
- `theme.css`: cream canvas body, `feature-card`-style panels (12px radius,
  `#efe9de`), hairline borders, dark `code-window-card` log console, coral
  primary buttons, pill badges, category-tab styling for the tab bar.
- Helper components used by all panels: `panel_card()`, `run_button()`,
  `secondary_button()`, `status_badge(status)`, `log_console()` (dark card +
  `ui.log`), `section_title()` (serif, weight 400, negative tracking).

## Step 4 — `gui/jobs.py` (JobManager)

- One job at a time (MATLAB shared-state + network-drive bandwidth constraints);
  a queue with per-job: id, name, cmd, cwd, env, status
  (queued/running/done/failed/cancelled), started/finished timestamps, return code.
- stdout+stderr merged, read on a background thread into a ring buffer;
  tqdm `\r` lines parsed into a progress fraction (last `NN%` match).
- `cancel()` → `proc.terminate()` then `kill()` after timeout
  (`creationflags=CREATE_NEW_PROCESS_GROUP` on Windows).
- Panels poll job state with `ui.timer` (0.5–1 s) to refresh logs, progress bar,
  status badge, and live images (cache-busted reload).

## Step 5 — Panels (one function `build(panel)` each)

Common pattern per panel: parameter form (prefilled with each script's current
defaults) → Run/Cancel → status badge + progress → dark log console → results.
- **GMD map**: paths + `n_files`; shows summary stats parsed from output.
- **Movie**: paths/scale params; Run renders headless; result shown as
  `ui.video` (mp4/gif) plus a frame slider scrubbing `--dump-frames` PNGs.
- **Accumulate**: threshold/denoise/GMD/dark params; live `ui.image` reloading
  `--live-path` partial image during the run; final image after.
- **Centroiding**: form bound to all `[vmi_picasso]` config.toml keys
  (paths, box_size, min_net_gradient, camera params, processing, histogram,
  gmd); writes TOML, runs with `--config`; shows `01_tuning_check_picasso.png` /
  `04_final_summary.png` when produced.
- **Circularize**: direct form over its argparse (input/output/center/shape/
  extracted/reassign-file/interp-order/radial-model/max-trig-n/no-jacobian/
  plots); shows circularized image + trig-fit plots.
- **PES (process_gmd_bins)**: weight suffix, backend (matlab/pyabel), method,
  bin/output dirs, cache toggle; sequential note in the panel; shows the three
  result PNGs (`gmd_bins-energy-spectrum{,-normalized}.png`, `-betas.png`).
- **Binning (vmi_histogram_gmd_bins)**: thresholds, GMD column, weight mode/
  power; shows the `vmi_gmd_bins*.png` grid and per-bin event counts parsed
  from the log.
- **Pipeline**: stage cards for gmd_map → centroiding → binning → pes
  (circularization runs inside PES); per-stage status badge + log link; "skip
  stages whose outputs already exist" toggle; runs as one queued job chain,
  stopping on first failure.

## Step 6 — `gui/app.py`

`ui.tabs` + `ui.tab_panels` (9 tabs), header with serif title, footer job-queue
indicator. `ui.run(title='VMI Analysis', port=8080, reload=False)`.
Launch: `uv run python -m gui.app` from `vmi-analysis-gui/` (documented).

## Step 7 — Verification

1. `python -m compileall` on `gui/` and all five edited scripts; `--help`
   smoke test on each refactored script.
2. Headless movie render on a few frames of `F:/1103/1103_8_Aele` (if the
   drive is reachable) → confirms subprocess + video output path end-to-end.
3. Start the app, `curl http://localhost:8080` returns the NiceGUI page.
4. Jobs that need full data (centroiding, PES) are verified up to launch +
   log streaming only; their numerics are unchanged from current behavior.

## Step 8 — Rewrite `AGENTS.md`

Replace the imported R0090/R0102 documentation with a concise file for **this**
project: the six scripts + what each does, the pipeline order, the GUI (how to
launch, panel map), config entry points (config.toml + per-script CLI), and the
constraints that still apply (sequential MATLAB step, `config.toml` cwd
requirement). The old physics/calibration content is dropped as not applicable
to the new F:/1103 datasets.

## Explicitly out of scope

- No authentication/multi-user; local tool UI.
- No live histogram for the Picasso run in v1 (log + progress only; histogram
  is built once at the end from the checkpoint).
- No changes to numerical behavior of any script.
