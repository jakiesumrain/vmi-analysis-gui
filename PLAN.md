# Design notes: NiceGUI web interface for the VMI analysis scripts

## Goal

A minimal, comfortable NiceGUI web app (uv-managed) with one tab panel per
script plus a pipeline panel, themed by the validated `DESIGN.md` (warm cream
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
- **One job at a time.** The MATLAB CPBASEX step shares global state
  (hardcoded `examples/image.txt`), and raw data typically sits on slow
  drives; a serial `JobManager` queue is the correct concurrency model.
- **CLI refactors preserve standalone behavior.** Each script's former
  hardcoded config block became argparse arguments whose defaults match the
  old values (or, for data paths, are empty + validated as required — no
  machine-specific paths live in the repository). Running a script bare
  behaves as before once paths are supplied.
- **`config.toml` is the centroiding source of truth.** The Centroiding
  panel edits it in place with tomlkit (comments preserved) and passes
  `--config`; the Binning and Pipeline panels read the same file.
- **Theme = DESIGN.md → static CSS + `ui.colors`.** No node/build step at
  runtime: the design tokens are mapped by hand into `gui/theme.css`, with
  Google Fonts links and local fallback stacks if fonts are unreachable.
- **Server-side file picker.** The app and the data live on the same
  machine, so `gui/panels/picker.py` browses the server's filesystem
  directly (drive dropdown on Windows, dir/file/save modes, extension
  filters) instead of upload widgets.

## Layout

```
gui/
├── app.py            # entry: theme + tabs + ui.run()
├── theme.py          # DESIGN.md tokens -> ui.colors + component helpers
├── theme.css         # static CSS mapped from the tokens
├── jobs.py           # JobManager: subprocess queue, log capture, progress, cancel
└── panels/
    ├── common.py       # form helpers + JobView (badge/progress/log/results)
    ├── picker.py       # LocalFilePicker dialog
    ├── gmd_map.py      # build_gmd_map.py form
    ├── movie.py        # h5_play_frames_movie.py + video/frame-slider display
    ├── accumulate.py   # h5_accumulate_fast_gmd.py + live partial image
    ├── centroid.py     # config.toml form editor + Picasso run
    ├── binning.py      # vmi_histogram_gmd_bins.py form + bins PNG
    ├── circularize.py  # circularization.py CLI form
    ├── pes.py          # process_gmd_bins.py form + result PNG gallery
    └── pipeline.py     # ordered stage runner with per-stage status/resume
```

Common pattern per panel: section title → status badge/progress/cancel →
parameter form (neutral defaults; folder buttons open `LocalFilePicker`) →
dark log console at the bottom → results area.

## Job plumbing (`gui/jobs.py`)

- Queue with per-job: id, name, cmd, cwd, env, status
  (queued/running/done/failed/cancelled), timestamps, return code.
- stdout+stderr merged, read on a background thread into a ring buffer;
  tqdm `\r` lines parsed into a progress fraction (last `NN%` match).
- `cancel()` terminates the process tree (`taskkill /T /F` on Windows;
  a separate process group with SIGTERM/SIGKILL on POSIX).
- Shutdown prevents queue advancement, cancels queued work, handles a
  cancellation during subprocess startup, and waits for worker cleanup.
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
