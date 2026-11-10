# AGENTS.md

Guidance for coding agents working in this repository: VMI (velocity-map
imaging) photoelectron analysis pipeline, plus a NiceGUI web interface that
drives every script and the full pipeline from one browser UI.

## Datasets

**No data is bundled with this repository.** Datasets are user-supplied
folders of `RAW-*.h5` files (selected in the GUI via the file pickers, or
passed via CLI). Each file holds ~126 frames of 1024×1024 uint16:

| dataset | shape | content |
|---|---|---|
| `raw/data` | (N, 1024, 1024) uint16 | image frames |
| `raw/meta` | (N, 5) uint64 | per-frame metadata |

`raw/meta` columns: 0 = frame counter, 1 = high-res hardware timestamp,
2 = status flag, 3 = Unix seconds, 4 = ns offset → `frame_time = col3 + col4/1e9`.
Camera and FEL both run at ~25 Hz. Some frames carry zeroed metadata
(`col3 = col4 = 0`); their timestamps are meaningless.

GMD binning additionally needs a GMD timestamp file
(`current-time-both.txt`: index, timestamp, value 1, value 2).

Per-dataset analysis outputs (checkpoint `.h5`, `vmi_2d_raw.npy`, PNGs) live
in the selected output directory (relative paths use the script cwd) and are **not**
committed to git.

## Scripts and pipeline

End-to-end order:

```
src/vmi_analysis/gmd/build_gmd_map.py       # 1. frame → GMD correspondence (timestamp match)
src/vmi_analysis/centroiding/process_andor_vmi_PICASSO_v5_h5_batch.py
                                          # 2. Picasso localization → checkpoint + 2D VMI histogram
src/vmi_analysis/centroiding/vmi_histogram_gmd_bins.py # 3. split events into GMD bins
src/vmi_analysis/circularization/process_gmd_bins.py  # 4. per bin: hole patch → rotate → center →
                                          #    circularization.py → apodize → Abel inversion → E, IE, betas
```

Side tools (QC / helpers, not pipeline stages):

- `src/vmi_analysis/imaging/h5_play_frames_movie.py` — play the first N frames as a movie; save mp4/gif.
- `src/vmi_analysis/imaging/h5_accumulate_fast_gmd.py` — fast pixel accumulation (threshold + morphology,
  optional GMD weighting); no sub-pixel localization. Superseded by the Picasso
  path for analysis, kept for quick looks.
- `src/vmi_analysis/centroiding/compute_dark.py` — precompute a mean/median dark frame
  (`dark_{method}.npy`); point the GUI's dark-file field or `--dark-file` at the result to
  skip per-run dark computation in the centroiding script.

| script | configuration |
|---|---|
| `build_gmd_map.py` | argparse CLI (`--h5-dir --gmd-path --output-path --n-files`; first two required) |
| `h5_play_frames_movie.py` | argparse CLI (`--h5-dir` required, `--headless`, `--dump-frames`, ...) |
| `h5_accumulate_fast_gmd.py` | argparse CLI (`--h5-dir` required, `--no-show`, `--live-path`, ...) |
| `process_andor_vmi_PICASSO_v5_h5_batch.py` | explicit CLI (`--h5-dir`, `--output-dir`, detection/processing/calibration flags); optional `--config PATH` for legacy defaults; `--spectrum-x-min/max` and `--spectrum-y-min/max` control summary axes (negative = auto) |
| `compute_dark.py` | argparse CLI (`--dark-dir`, `--method mean\|median`, `--frames`, `--output-dir`); optional explicit `--config PATH` defaults |
| `vmi_histogram_gmd_bins.py` | explicit `--checkpoint-path`, `--gmd-map-path`, `--output-dir`, resolution/filter/weighting flags; optional `--config PATH` defaults |
| `process_gmd_bins.py` | argparse CLI (`--weight-suffix`, `--inversion-backend`, `--pyabel-method`, `--use-cache`, `--no-show`) |
| `circularization/circularization.py` | full argparse CLI — no changes needed |

## Constraints and traps

- Centroiding accepts `--alpha`, `--fit-xe-4d`, `--fit-min-px2`, and
  `--fit-max-px2`. The legacy optional `[vmi_picasso.energy_calibration]` section accepts
  positive `alpha` (eV per histogram pixel²), `fit_xe_4d` (default false), and
  optional `fit_min_px2`/`fit_max_px2`. The Xe doublet is fitted in pixel²;
  its gas-phase IPs (4d3/2: 69.537 eV, 4d5/2: 67.548 eV) determine alpha and
  photon energy via `alpha*r² = hv - IP`. Fitted values are console advice;
  only the user-supplied alpha changes the plotted/CSV energy coordinates.
  Intensity is preserved when alpha changes (no division by alpha); label
  the y-axis "Intensity (arb. units)". The CSV's last column is `intensity`.
  Fit windows always use pixel²; plot limits use the displayed units.

- **`process_gmd_bins.py` must run sequentially.** The MATLAB CPBASEX step
  (`new_10.m`) hardcodes `examples/image.txt` as a shared input; bins are
  processed one at a time and outputs copied out immediately. The default
  PyAbel backend (`INVERSION_BACKEND = "pyabel"`) avoids MATLAB entirely.
- **Circularization calibration files are required** by `process_gmd_bins.py`:
  `extracted-FOC100eV-20260710.txt` and `reassignment-FOC100eV-20260710.txt`
  (trig-distortion fits + radial reassignment map) in `circularization/`.
  They are per-experiment inputs. The GUI's **Trig Calibration** tab builds
  the extracted trig-fit file from point-sampled polar intensities with
  optional inverse Abel per angle and filtering before a Gaussian ridge fit
  in the selected band. The
  reassignment file is supplied separately. The reassignment file for the
  reference dataset was the identity map — only the azimuthal (trig) terms
  act, so any radial nonlinearity propagates into the energy axis.
- **TOML is optional and never loaded implicitly by scripts.** Explicit CLI
  values override an explicitly supplied `--config PATH`. Relative analysis
  paths resolve from the subprocess cwd (GUI centroiding/binning jobs use
  `centroiding/`). The GUI migrates legacy values once, read-only, to
  `.nicegui/storage-general.json`; all subsequent field edits autosave there.
  Never rewrite legacy TOML from GUI events. Keep submitted result paths in
  job metadata rather than reading current form values when jobs finish.
- **`vmi_histogram_gmd_bins.py` prefers the TkAgg backend** for its live plot
  window; override via `MPLBACKEND` (e.g. Agg) when running headless.
- **GMD weighting is a per-frame scalar** — it corrects yield, never shape.
- **MATLAB + CPBASEX**: the CPBASEX code is bundled at `vendor/pbasex-matlab/`
  (MIT, Elio Gilbert Champenois). Only needed for the `matlab` inversion
  backend; `CPBASEX_DIR` overrides the bundled location. The backend
  requires the basis file `vendor/pbasex-matlab/examples/G_r512_k512_l10.h5`,
  generated once in MATLAB with `examples/save_gData.m` (time-consuming) —
  `run_matlab_cpbasex()` fails with an explicit error while it is missing.

## Environment

- uv-managed project (`pyproject.toml`, Python ≥ 3.12). Run scripts with
  `uv run python -m vmi_analysis.<module>` from the repo root, or use the venv interpreter
  in `.venv/` directly.
- `uv sync` after cloning installs everything, including the GUI.

## Git workflow

All changes land on `main` through pull requests, never direct pushes:

The user has authorized agents to handle this workflow end to end for requested
repository changes: branch, commit, push, open a PR, squash-merge, and update
local `main` after appropriate local checks. Do not add or wait for CI unless
the user requests it. Preserve unrelated local edits, including dataset settings.

1. Branch off `main` (`feature/<name>` or `fix/<name>`), commit there.
2. Push with `HTTPS_PROXY=http://127.0.0.1:7890` if direct GitHub access is
   blocked (repo-local `http.proxy` may already cover plain `git push`).
3. Open the PR with `gh pr create`, squash-merge with
   `gh pr merge --squash`; the head branch is deleted automatically.

Squash is the only enabled merge method; merge/rebase merges are disabled.
GitHub branch *protection* is unavailable (private repo, free plan) — the
no-direct-push rule is enforced by convention.

## GUI

`src/vmi_analysis/gui/` — NiceGUI app, one tab per script plus a pipeline tab, themed by
`docs/DESIGN.md`. See `docs/architecture.md` for the architecture. Launch:
`uv run vmi-analysis` → http://localhost:3157
(`VMI_GUI_PORT` overrides the port,
`VMI_GUI_NO_SHOW=1` suppresses the browser auto-open). The server shuts
itself down once the last browser client is gone (checked after NiceGUI's
one-second reconnect grace period; faster reloads survive). Shutdown stops
queue advancement, cancels queued jobs, and waits for the running process
tree to be terminated. Silent socket loss still needs heartbeat detection.

Rules for GUI-related changes:

- All `zoomable_image` outputs share `src/vmi_analysis/gui/assets/image_zoom.js`: Pan, wheel zoom,
  Crop area (drag a rectangle to create a cropped browser preview), Reset,
  and Escape handling. Exclude outside pixels even when zooming out; repeated
  crops operate on the current preview, and Reset restores the original image.
  Never modify the saved output image or analysis data.
  Browser regression checks use `tests/browser/zoom_fixture.py` and
  `tests/browser/zoom_overlay.browser.js`; generated artifacts stay in `.cache/vmi-analysis/`.

- Every panel uses `JobView.run_button`, a single Run → Stop control with
  Cancel queued / Stopping states. Set `job_name` to the exact submitted job
  name so an active job can be recovered after page reload. Do not add a
  separate panel Stop button or submit a second job from a stale Stop click.

- `src/vmi_analysis/centroiding/options.py` owns neutral defaults and CLI serialization;
  `src/vmi_analysis/gui/settings.py` owns persistence/migration and shared stage inputs. Pipeline
  uses its selected dataset/map and disables tuning-only in its job snapshot,
  without changing the saved Centroiding form. No GUI job passes `--config`.

- Scripts stay standalone-runnable; CLI args default to sane neutral values
  (no machine-specific paths — data locations always come from the user).
- The GUI runs scripts as subprocesses (isolation + cancellation), never
  imports them into the web process.
- Jobs can run concurrently up to the saved `jobs.max_concurrent` limit (default
  2, any positive integer). One active job per panel; each retains its Stop,
  progress, logs and reconnect state. Lowering the limit does not kill jobs.
- `gui/resources.py` declares file reads/writes from submitted module commands.
  Keep these declarations in sync with CLI interfaces. Block overlapping writes
  and producer/reader conflicts; shared reads are allowed. Pipeline reservations
  cover all stages for the whole run. MATLAB/CPBASEX stays mutually exclusive.
- Clear-output handlers must use `delete_outputs`, which checks reservations
  atomically. Shutdown must cancel and join every running job's process tree.

## Source layout

- All application implementations live under `src/vmi_analysis/`. Old command
  launchers have been removed; do not reintroduce root shims.
- Run `uv sync` before commands/tests; the package is installed editably.
  `uv run vmi-analysis` and `python -m vmi_analysis.gui.app` are the primary GUI
  entry points. Panels run analysis modules with `sys.executable -m`.
- Use `vmi_analysis.paths` for workspace locations; never infer writable paths
  from a source module's parent directory. Preserve legacy centroiding and
  circularization data roots, saved `.nicegui` keys, and CLI relative paths.
- CSS/JS are resources under `gui/assets` in the package. Disposable artifacts
  go in `.cache/vmi-analysis/`; user analysis outputs remain where selected.
- `centroiding/config.toml` is a local ignored legacy input; the neutral template
  is `examples/config.example.toml`. Do not overwrite existing local config.
- CPBASEX is vendored under `vendor/pbasex-matlab`; `CPBASEX_DIR` and an existing
  old `pbasex-matlab` checkout remain supported. Never move user basis files.
- Python tests: `uv run python -m unittest discover -s tests -v`. Test groups:
  `tests/unit/`, `tests/integration/`, and `tests/browser/`. No CI unless requested.
