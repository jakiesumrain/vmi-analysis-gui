# Design notes: NiceGUI web interface for the VMI analysis scripts

## Goal

A minimal, comfortable NiceGUI web app (uv-managed) with one tab panel per
script plus a pipeline panel, themed by the validated `docs/DESIGN.md` (warm cream
canvas, coral CTAs, dark log consoles). Scripts stay standalone-runnable; the
GUI drives them as subprocesses with live log streaming, progress,
cancellation, and image/video result display.

## Architecture decisions

- **NiceGUI** was chosen because the analysis scripts need live-updating
  plots (progressive accumulation, logs, progress bars) without writing any
  JavaScript; it serves a real browser UI from the same Python process.
- **Scripts run as subprocesses, never imports.** Isolation (a crashing job
  cannot take down the UI), clean cancellation (terminate the process
  group), and zero coupling between the web process and heavy numerical
  state. All job subprocesses use `sys.executable` — no `uv run` at
  runtime.
- **Configurable concurrent jobs.** `JobManager` admits jobs up to a saved
  positive-integer limit (default 2). Submitted command snapshots declare file
  reads/writes and exclusive resources in `gui/resources.py`. Shared readers
  can run together; overlapping writes and producer/reader conflicts wait.
  MATLAB CPBASEX uses an exclusive lock for its shared `examples/image.txt`.
  One active job per panel preserves its Run/Stop and reconnect controls.
  Pipeline stages remain ordered and reserve the union of their resources.
- **CLI refactors preserve standalone behavior.** Each script's former
  hardcoded config block became argparse arguments whose defaults match the
  old values (or, for data paths, are empty + validated as required — no
  machine-specific paths live in the repository). Running a script bare
  behaves as before once paths are supplied.
- **GUI fields are the source of job inputs.** All panels persist changes
  in NiceGUI general storage. Centroiding uses the shared defaults/CLI schema
  in `src/vmi_analysis/centroiding/options.py`; commands contain explicit argument snapshots.
  `src/vmi_analysis/gui/settings.py` migrates legacy TOML values once, read-only. Scripts accept
  optional explicit `--config` defaults for compatibility, with CLI overrides.
  Binning has explicit paths; Pipeline combines saved settings with its own
  dataset/GMD map. Result paths come from the submitted job, not live settings.
- **Theme = DESIGN.md → static CSS + `ui.colors`.** No node/build step at
  runtime: the design tokens are mapped by hand into `src/vmi_analysis/gui/theme.css`, with
  Google Fonts links and local fallback stacks if fonts are unreachable.
- **Server-side file picker.** The app and the data live on the same
  machine, so `src/vmi_analysis/gui/panels/picker.py` browses the server's filesystem
  directly (drive dropdown on Windows, dir/file/save modes, extension
  filters) instead of upload widgets.

## Layout

```
src/vmi_analysis/gui/
├── app.py            # entry: theme + tabs + ui.run()
├── theme.py          # DESIGN.md tokens -> ui.colors + component helpers
├── theme.css         # static CSS mapped from the tokens
├── jobs.py           # concurrent scheduler, log capture, progress, cancel
├── resources.py      # command file-access declarations and conflict checks
├── settings.py       # saved centroid values, one-time legacy migration, stage inputs
└── panels/
    ├── common.py       # form helpers + JobView (badge/progress/log/results)
    ├── picker.py       # LocalFilePicker dialog
    ├── gmd_map.py      # build_gmd_map.py form
    ├── movie.py        # h5_play_frames_movie.py + video/frame-slider display
    ├── accumulate.py   # h5_accumulate_fast_gmd.py + live partial image
    ├── centroid.py     # automatically saved fields + explicit Picasso CLI arguments
    ├── binning.py      # vmi_histogram_gmd_bins.py form + bins PNG
    ├── circularize.py  # circularization.py CLI form
    ├── pes.py          # process_gmd_bins.py form + result PNG gallery
    └── pipeline.py     # ordered stage runner with per-stage status/resume
```

Common pattern per panel: section title → status badge/progress →
parameter form and Run/Stop control (folder buttons open `LocalFilePicker`) →
dark log console at the bottom → results area.

The form's single Run button becomes Stop for its active job, Cancel queued
while waiting, and disabled Stopping during cleanup. `JobView` restores its
panel's active job after reload by job name. A separate cancel button is not
used. When a job ends, the same control returns to its original Run label.

## Job plumbing (`src/vmi_analysis/gui/jobs.py`)

- Queue with per-job: id, name, cmd, cwd, env, status, resource reservations
  (queued/running/done/failed/cancelled), timestamps, return code.
- stdout+stderr merged, read on a background thread into a ring buffer;
  tqdm `\r` lines parsed into a progress fraction (last `NN%` match).
- `cancel()` terminates the process tree (`taskkill /T /F` on Windows;
  a separate process group with SIGTERM/SIGKILL on POSIX).
- Scheduling can bypass a blocked job for unrelated work. Earlier waiting
  resource claims prevent later readers from starving a writer. Paths are
  resolved, case-normalized, and compared by ancestor/descendant overlap.
- Shutdown prevents queue advancement, cancels queued work, handles a
  cancellation during subprocess startup, and waits for all worker cleanup.
- GUI output deletion checks queued/running reservations under the scheduler
  lock. Movie frame dumps use unique per-run directories.
- The last socket disconnect schedules shutdown after a one-second grace
  period; reconnecting cancels that check. The app shutdown hook also
  cleans up jobs on terminal/server shutdown.
- Panels poll with `ui.timer` (~0.7 s) to refresh logs, progress, badges,
  and live images (cache-busted reload).
- Pipeline jobs are one job with ordered stages, per-stage output
  fingerprints for "skip existing", stop on first failure.

## Out of scope (deliberately)

- No authentication/multi-user; local tool UI.
- No live histogram for the Picasso run (log + progress only; the histogram
  is built once at the end from the checkpoint).
- No changes to the numerical behavior of any script.

## Source layout and working data

The project is an editable-installed `src/vmi_analysis` package. `uv sync`
installs it; `uv run vmi-analysis` starts the GUI. Panels launch analysis tools
as `[sys.executable, "-m", "vmi_analysis.<module>", ...]`, preserving explicit
CLI snapshots and the former working directories. Analysis tools run through
their package modules; old script paths and `python -m gui.app` were removed.

`vmi_analysis.paths` distinguishes the workspace from package source. A checkout
retains its original centroiding/circularization data roots; installed copies
use the launch directory unless `VMI_WORKSPACE_DIR` is set. GUI CSS/JavaScript
are package resources, previews use `.cache/vmi-analysis/`, and existing
`.nicegui` state and user data stay in place. `CPBASEX_DIR` overrides the MATLAB
checkout; otherwise an existing legacy checkout is preferred, then
`vendor/pbasex-matlab`. The MATLAB basis remains an external experiment input.

Tests are grouped under `tests/unit`, `tests/integration`, and `tests/browser`.
`python -m unittest discover -s tests -v` discovers the Python suite recursively.
