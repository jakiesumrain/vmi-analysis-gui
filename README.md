# vmi-analysis-gui

VMI (velocity-map imaging) photoelectron analysis toolbox: raw camera HDF5
frames → event centroiding (Picasso) → GMD binning → circularization + Abel
inversion → photoelectron energy spectra and angular anisotropies.

Every step is a standalone Python CLI script, and the repository ships a
NiceGUI web interface with one panel per script plus a pipeline panel that
runs the whole chain in order.

## Requirements

- **Python ≥ 3.12** and [**uv**](https://docs.astral.sh/uv/) (the project
  manager used for everything below).
- Optional: **ffmpeg** on PATH — for MP4 output in the Movie tool
  (without it the tool writes a GIF instead).
- Optional: **MATLAB** — only for the MATLAB inversion backend in the PES
  stage. The default backend (PyAbel) is pure Python and needs nothing
  extra. The CPBASEX code is bundled at `vendor/pbasex-matlab/` (MIT,
  Elio Gilbert Champenois — see `vendor/pbasex-matlab/LICENSE.txt`), so no
  separate checkout is needed. **Before the first MATLAB run you must
  generate the basis data once**: in MATLAB, `cd` to
  `vendor/pbasex-matlab/examples` and run `save_gData.m`. ⚠ **This is
  time-consuming** (expect tens of minutes) and writes
  `G_r512_k512_l10.h5` next to the script. Set the `CPBASEX_DIR`
  environment variable only to use a different CPBASEX checkout.

## Quickstart (fresh clone)

```bash
git clone <this-repo-url>
cd vmi-analysis-gui
uv sync                          # creates .venv and installs everything
uv run vmi-analysis         # → http://localhost:3157 (opens a browser)
                                 # override port: VMI_GUI_PORT=3158 uv run vmi-analysis
                                 # don't open a browser: VMI_GUI_NO_SHOW=1 uv run vmi-analysis
```

If the requested port is unavailable, startup exits with a clear error and
prints PowerShell commands using a checked alternative port. The app does not
switch ports automatically; run the suggested commands to relaunch it.

The server stops automatically about 1–2 seconds after the last browser tab
showing the app disconnects. Reloads and brief disconnects have a one-second
reconnect grace period; a slower reload may require restarting the server.
Silent connection loss takes longer to detect because it relies on socket
heartbeats. `Ctrl+C` in the terminal works too. Shutdown cancels queued jobs,
terminates the running job's process tree, and waits for cleanup before exit.

That is the whole setup. The app starts with one tab per script:

| tab | wraps | purpose |
|---|---|---|
| GMD Map | `vmi_analysis.gmd.build_gmd_map` | match each camera frame to its GMD pulse-energy record by timestamp |
| Movie | `vmi_analysis.imaging.h5_play_frames_movie` | render the first N frames as a movie + frame scrubber (QC) |
| Dark Frame | `vmi_analysis.centroiding.compute_dark` | mean/median dark frame from dark HDF5 files → `dark_{method}.npy` |
| Accumulate | `vmi_analysis.imaging.h5_accumulate_fast_gmd` | fast threshold+denoise accumulation with live preview (QC) |
| Centroiding | `vmi_analysis.centroiding.process_andor_vmi_PICASSO_v5_h5_batch` | Picasso localization → checkpoint + 2D VMI histogram |
| Binning | `vmi_analysis.centroiding.vmi_histogram_gmd_bins` | split events into GMD bins, one 2D histogram per bin |
| Circularize | `vmi_analysis.circularization.circularization` | correct radial distortion of a single image |
| PES | `vmi_analysis.circularization.process_gmd_bins` | per-bin chain → Abel inversion → spectra + betas |
| Pipeline | all four pipeline stages | GMD map → centroiding → binning → PES, in one job |

Jobs run as independent subprocesses; logs stream live into the dark
console at the bottom of each panel, and results (images, movies, spectra)
render underneath it. Every path field has a folder button that browses
your local disks — no typing of paths needed.
The picker's **New folder** button creates a named folder in the directory
you are viewing and opens it, ready to select or save into. Existing files
and folders are never overwritten.

**Maximum concurrent jobs** in the header controls how many jobs may run at
once (default 2, dropdown choices 1–9, one active job per panel). Changes apply immediately and
survive restart. Increasing the limit opens more slots; lowering it lets
existing jobs finish and delays new starts. CPU worker counts inside individual
panels still apply, so choose a limit appropriate for your RAM and disk speed.

Read-only inputs can be shared. Jobs wait when another job writes an overlapping
file or directory, or reads something they need to overwrite. Waiting jobs show
the reason; unrelated jobs can use free slots. One job per panel runs at once.
Pipeline stages stay sequential and reserve their inputs/outputs for the whole
pipeline run. MATLAB/CPBASEX jobs also share an exclusive lock because their
temporary files are shared. Clear-output actions respect these reservations.
The protection covers jobs launched by this GUI, including its Pipeline panel.

Each panel uses a single action button: **Run** changes to **Stop** while
processing. Click Stop to terminate that job and its subprocess tree. A queued
job shows **Cancel queued**, and cancellation in progress shows disabled
**Stopping…**. The button returns to Run when the job finishes or stops.
Reloading the page restores controls for that panel's active job. Other queued
jobs start when their slots and resources are available; stopping one job leaves
other running panels alone. Closing the last browser tab stops all jobs.

Click an output image to open the inspection overlay. Choose **Crop area**,
drag a rectangle over the region you want to inspect, and release. The overlay
displays only that cropped region, enlarged to fit; surrounding pixels are
excluded even when you zoom out or pan. You can crop again within the preview.
The overlay returns to **Pan** afterward, and the mouse wheel or +/− zooms
the cropped preview. **reset** restores the original whole image. Escape
cancels an active selection; when not selecting, Escape closes the overlay.
Cropping is temporary in the browser and never changes the saved output file.

## What you need to bring

The repository contains **no data**. You need:

1. **A folder of raw HDF5 files** (`RAW-*.h5`), each holding
   - `raw/data` — `(N, 1024, 1024)` uint16 image frames
   - `raw/meta` — `(N, 5)` uint64 metadata (col 3 = Unix seconds,
     col 4 = ns offset → frame timestamp)
2. **A GMD timestamp file** (`current-time-both.txt`) if you want GMD
   weighting/binning — columns: index, timestamp, GMD value 1, GMD value 2.
3. **Circularization calibration files** for the PES stage, placed in
   `circularization/`: `extracted-FOC100eV-20260710.txt` (trig-fit
   coefficients) and `reassignment-FOC100eV-20260710.txt` (radial
   reassignment map; an identity map is valid). These are per-experiment
   inputs; the PES panel warns while they are missing.

## A first end-to-end run

1. **GMD Map tab** — pick your HDF5 folder and GMD timestamp file, choose
   an output map path, Run. Produces `gmd_map_all.txt`.
2. **Centroiding tab** — field edits save automatically and are passed as CLI inputs.
   Set the HDF5 directory and output directory (folder buttons), keep the
   defaults otherwise. Tick **Tuning mode only** first and Run to check spot
   detection on a single frame; then untick and run the full batch.
   Produces a checkpoint `.h5` and a 2D histogram in the output directory.
3. **Binning tab** — choose the checkpoint, GMD map, and output directory; Run produces one
   GMD-binned 2D histogram `.npy` per bin.
4. **PES tab** — drop the two calibration files into `circularization/`,
   keep the PyAbel backend, Run. Produces energy spectra, betas, and an
   `.npz` with everything under `circularization/gmd_bins_results_*/`.

Or press **Run pipeline** on the Pipeline tab after step 1 — it chains
stages 1–4 and skips stages whose outputs already exist.

All panel values are stored in `.nicegui/storage-general.json`, including
Centroiding edits made without clicking Run. This storage survives server
restarts and is shared by browsers connected to the same server. Keep `.nicegui/`
to retain your settings. Jobs capture their inputs at submission, so editing a
form afterward does not change a queued job or where its results are displayed.

`config.toml` is no longer required by the GUI or scripts. On first use after
upgrading, the GUI copies existing centroiding TOML values into saved settings
without changing the file; existing saved values take precedence. After that
one-time migration, removing or editing the TOML file does not affect the GUI.
With no file, neutral built-in defaults are used. Binning also migrates its
previously selected config path into explicit saved inputs once, if available.
Binning has explicit input
fields; PES initially suggests the saved Centroiding output directory. Pipeline
uses saved Centroiding parameters with its own dataset and generated GMD map,
forces a full batch run, and uses the Binning tab's saved thresholds/column/floor.

Centroiding's **Tuning check** and **Final summary** can be inspected as Plotly
figures: hover for numerical values, drag within a subplot to zoom, choose Pan
from the plot toolbar to move the view, and double-click or use Reset axes to
restore it. Image axes keep equal pixel scales. The log VMI color scale masks
nonpositive pixels and shows original counts on hover; the spectrum retains
its saved energy calibration and unchanged intensity.
Separate **Colormap** dropdowns control the tuning image, raw VMI, and Abel
image. Each remembers its selection and preserves the current zoom.

Runs save `01_tuning_check_picasso.npz` and `04_final_summary.npz` alongside the
usual PNG files. These contain numeric plotting data and JSON metadata, so
the GUI does not import or rerun analysis scripts to render a figure. Use
**View saved results** to reopen plots from the selected output directory.
Older runs without these sidecars still show their PNGs; rerun the relevant
stage once to produce interactive data. The **Saved PNG** expansion retains
the original downloadable/croppable figure, and standalone PNG export is unchanged.

Accumulate uses the same interactive controls for its live preview and final
image, with equal pixel scales and original counts on hover. The final view
reads the saved `.txt` array directly; **View saved result** opens existing
outputs without rerunning. Its Viridis log color range matches the standalone
script, and the original PNG remains available under **Saved PNG**.
The **Colormap** dropdown above the image changes colors immediately without
rerunning or resetting the zoom, and remembers the selection for future views.

## Command-line use

In the Centroiding tab, **Energy calibration / Xe 4d fit** provides an optional
**α** in eV/pixel². A positive value displays `E = α r²` in eV while preserving
the original intensity; leaving it empty keeps the squared-radius axis in pixel².
The y-axis is **Intensity (arb. units)** in both cases, not a density per eV.
The CSV columns are `r_px,r2_px2,energy,dN_per_dr,intensity`; the final column
was renamed from `dN_per_dE` to reflect its unchanged intensity values.
Only the `energy` column changes with alpha. Alpha refers to histogram
pixels, so it must be recalibrated if the histogram resolution factor changes.
Reset the spectrum x-axis limits to auto when switching units, or enter limits
in the new displayed units. Y-axis limits keep the same meaning.

Enable **Fit Xe 4d doublet** to fit the two most prominent peaks in pixel²
using two independent Gaussians plus a linear background. Optional fit min/max
limits always use pixel² and should isolate the doublet from other lines. The
console prints the fitted centers and derives alpha and photon energy from
`α q_3/2 = hν − 69.537 eV` and `α q_5/2 = hν − 67.548 eV`, using the gas-phase
Xe reference energies from [Elettra's PES reference table](https://www.elettra.eu/lightsources/elettra/elettra-beamlines/gas-phase/useful-reference-datafor-pes.html).
The higher-pixel² peak is assigned to 4d₅/₂. Fitting assumes these are the Xe
4d lines; unresolved or poor fits produce a console message without a calibration.
Reported uncertainties are formal fit errors and exclude systematic errors.
The fit does not change your alpha: copy the fitted value into the alpha field
and run again to apply it. Existing localization checkpoints are reused.

For standalone runs, use `--alpha`, `--fit-xe-4d`, `--fit-min-px2`, and
`--fit-max-px2`. For older workflows, passing `--config PATH` explicitly reads
optional TOML defaults; explicit CLI arguments override them, including false
boolean options such as `--no-abel` and `--no-gmd-weighting`. Files are never
loaded implicitly based on the working directory. The legacy equivalent is:

```toml
[vmi_picasso.energy_calibration]
# alpha = 0.0004          # omit for an uncalibrated pixel² axis
fit_xe_4d = true
fit_min_px2 = 25000.0     # optional: search window around the two Xe peaks
fit_max_px2 = 45000.0
```

Every script is standalone; `--help` lists all options. Examples:

```bash
uv run python -m vmi_analysis.gmd.build_gmd_map \
    --h5-dir /path/to/dataset --gmd-path /path/to/current-time-both.txt \
    --output-path gmd_map_all.txt

uv run python -m vmi_analysis.centroiding.process_andor_vmi_PICASSO_v5_h5_batch \
    --h5-dir /path/to/dataset --output-dir /path/to/results \
    --alpha 0.000731 --fit-xe-4d

uv run python -m vmi_analysis.centroiding.vmi_histogram_gmd_bins \
    --checkpoint-path /path/to/results/vmi_results_raw.h5 \
    --gmd-map-path /path/to/gmd_map_all.txt --output-dir /path/to/results \
    --weight-mode gmd_norm --no-show

uv run python -m vmi_analysis.circularization.process_gmd_bins \
    --inversion-backend pyabel --pyabel-method daun --no-show

uv run python -m vmi_analysis.imaging.h5_play_frames_movie \
    --h5-dir /path/to/dataset --headless --save-movie

uv run python -m vmi_analysis.imaging.h5_accumulate_fast_gmd \
    --h5-dir /path/to/dataset --output-path accumulated.txt --no-show
```

The two path-valued options marked "required" have no defaults — no dataset
is bundled, so point them at your own files.

## Repository layout

```text
src/vmi_analysis/       # installed Python package; all application implementations
  paths.py             # source-independent workspace and resource locations
  gmd/                 # frame/GMD mapping
  imaging/             # movie and fast accumulation tools
  centroiding/         # localization, calibration, binning, dark-frame tools
  circularization/     # distortion correction and PES processing
  gui/                 # NiceGUI panels, jobs, settings and plot builders
    assets/            # packaged CSS and image-overlay JavaScript
tests/
  unit/                # plotting and data checks
  integration/         # CLI, packaging, jobs, calibration and server lifecycle
  browser/             # Playwright checks and image-overlay fixture
docs/                  # architecture and design tokens
examples/              # neutral optional TOML template
vendor/pbasex-matlab/   # third-party MATLAB backend, with its own license
```

After `uv sync`, launch with `uv run vmi-analysis` or
`uv run python -m vmi_analysis.gui.app`. Analysis tools use
`uv run python -m vmi_analysis.<module>` as in the examples above; the former
root-level and working-directory launchers have been removed.
Run the installed command from another directory
using the environment's `vmi-analysis` executable.

Existing `centroiding/` and `circularization/` directories remain working-data
locations, including relative output defaults and experiment calibration files.
No datasets, movies, saved paths, or result folders are moved. The legacy local
`centroiding/config.toml` is ignored; `examples/config.example.toml` is the
versioned template. Existing local config files still support explicit
`--config` inputs and the GUI's one-time read-only migration.

`.nicegui/` retains saved settings. Disposable GUI previews now use the ignored
`.cache/vmi-analysis/`, outside application source. A checkout uses its repository
root as the workspace; an installed wheel uses the launch directory.
`VMI_WORKSPACE_DIR` explicitly selects a different workspace, and
`NICEGUI_STORAGE_PATH` can still override settings storage. For a wheel install,
set `CPBASEX_DIR` to a separate CPBASEX checkout if using MATLAB; the Python
wheel does not include third-party MATLAB code or experiment basis data.

## Further reading

Run the shutdown regression checks with `uv run python -m unittest discover -s tests -v`.
They use temporary local servers and dummy subprocesses; no dataset is required.

For browser checks of the shared image overlay, with Playwright CLI installed:

```bash
uv run python tests/browser/zoom_fixture.py
uv run python -m http.server 18765 --bind 127.0.0.1 --directory .cache/vmi-analysis
# In another terminal:
playwright-cli -s=vmi-area-zoom open http://127.0.0.1:18765/zoom_check.html --browser=chrome
playwright-cli -s=vmi-area-zoom run-code --filename=tests/browser/zoom_overlay.browser.js
playwright-cli -s=vmi-area-zoom close
```

The fixture and screenshots are generated under the ignored `.cache/vmi-analysis/`
directory. Stop the temporary HTTP server with Ctrl+C after checking.

- `docs/architecture.md` — architecture and design decisions (why NiceGUI, why
  subprocesses, theme mapping from `docs/DESIGN.md`).
- `docs/DESIGN.md` — the UI design tokens the theme is built from.
- `AGENTS.md` — contributor/agent guidance: pipeline details, constraints
  (sequential MATLAB step, legacy path compatibility), git workflow.

## License

MIT — see `LICENSE`. The bundled CPBASEX code under `vendor/pbasex-matlab/` is
third-party software, also MIT, copyright Elio Gilbert Champenois (2017);
see `vendor/pbasex-matlab/LICENSE.txt`.
