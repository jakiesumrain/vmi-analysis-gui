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
  extra. The CPBASEX code is bundled at `pbasex-matlab/` (MIT,
  Elio Gilbert Champenois — see `pbasex-matlab/LICENSE.txt`), so no
  separate checkout is needed. **Before the first MATLAB run you must
  generate the basis data once**: in MATLAB, `cd` to
  `pbasex-matlab/examples` and run `save_gData.m`. ⚠ **This is
  time-consuming** (expect tens of minutes) and writes
  `G_r512_k512_l10.h5` next to the script. Set the `CPBASEX_DIR`
  environment variable only to use a different CPBASEX checkout.

## Quickstart (fresh clone)

```bash
git clone <this-repo-url>
cd vmi-analysis-gui
uv sync                          # creates .venv and installs everything
uv run python -m gui.app         # → http://localhost:8090 (opens a browser)
                                 # override port: VMI_GUI_PORT=8080 uv run python -m gui.app
                                 # don't open a browser: VMI_GUI_NO_SHOW=1 uv run python -m gui.app
```

The server stops automatically about 1–2 seconds after the last browser tab
showing the app disconnects. Reloads and brief disconnects have a one-second
reconnect grace period; a slower reload may require restarting the server.
Silent connection loss takes longer to detect because it relies on socket
heartbeats. `Ctrl+C` in the terminal works too. Shutdown cancels queued jobs,
terminates the running job's process tree, and waits for cleanup before exit.

That is the whole setup. The app starts with one tab per script:

| tab | wraps | purpose |
|---|---|---|
| GMD Map | `build_gmd_map.py` | match each camera frame to its GMD pulse-energy record by timestamp |
| Movie | `h5_play_frames_movie.py` | render the first N frames as a movie + frame scrubber (QC) |
| Accumulate | `h5_accumulate_fast_gmd.py` | fast threshold+denoise accumulation with live preview (QC) |
| Dark Frame | `centroiding/compute_dark.py` | mean/median dark frame from dark HDF5 files → `dark_{method}.npy` |
| Centroiding | `centroiding/process_andor_vmi_PICASSO_v5_h5_batch.py` | Picasso localization → checkpoint + 2D VMI histogram |
| Binning | `centroiding/vmi_histogram_gmd_bins.py` | split events into GMD bins, one 2D histogram per bin |
| Circularize | `circularization/circularization.py` | correct radial distortion of a single image |
| PES | `circularization/process_gmd_bins.py` | per-bin chain → Abel inversion → spectra + betas |
| Pipeline | all four pipeline stages | GMD map → centroiding → binning → PES, in one job |

Jobs run one at a time as subprocesses; logs stream live into the dark
console at the bottom of each panel, and results (images, movies, spectra)
render underneath it. Every path field has a folder button that browses
your local disks — no typing of paths needed.

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
2. **Centroiding tab** — the form edits `centroiding/config.toml` directly.
   Set the HDF5 directory and output directory (folder buttons), keep the
   defaults otherwise. Tick **Tuning mode only** first and Run to check spot
   detection on a single frame; then untick and run the full batch.
   Produces a checkpoint `.h5` and a 2D histogram in the output directory.
3. **Binning tab** — defaults read the same `config.toml`; Run produces one
   GMD-binned 2D histogram `.npy` per bin.
4. **PES tab** — drop the two calibration files into `circularization/`,
   keep the PyAbel backend, Run. Produces energy spectra, betas, and an
   `.npz` with everything under `circularization/gmd_bins_results_*/`.

Or press **Run pipeline** on the Pipeline tab after step 1 — it chains
stages 1–4 and skips stages whose outputs already exist.

## Command-line use

Every script is standalone; `--help` lists all options. Examples:

```bash
uv run python build_gmd_map.py \
    --h5-dir /path/to/dataset --gmd-path /path/to/current-time-both.txt \
    --output-path gmd_map_all.txt

uv run python centroiding/process_andor_vmi_PICASSO_v5_h5_batch.py \
    --config centroiding/config.toml

uv run python centroiding/vmi_histogram_gmd_bins.py \
    --config centroiding/config.toml --weight-mode gmd_norm --no-show

uv run python circularization/process_gmd_bins.py \
    --inversion-backend pyabel --pyabel-method daun --no-show

uv run python h5_play_frames_movie.py \
    --h5-dir /path/to/dataset --headless --save-movie

uv run python h5_accumulate_fast_gmd.py \
    --h5-dir /path/to/dataset --output-path accumulated.txt --no-show
```

The two path-valued options marked "required" have no defaults — no dataset
is bundled, so point them at your own files.

## Repository layout

```
build_gmd_map.py                      # pipeline stage 1
h5_play_frames_movie.py               # QC tool
h5_accumulate_fast_gmd.py             # QC tool
centroiding/
├── config.toml                       # centroiding + binning configuration
├── process_andor_vmi_PICASSO_v5_h5_batch.py   # stage 2
└── vmi_histogram_gmd_bins.py         # stage 3
circularization/
├── circularization.py                # single-image distortion correction
└── process_gmd_bins.py               # stage 4 (calls circularization.py)
pbasex-matlab/                        # bundled CPBASEX (MATLAB backend, own license)
├── pbasex-MATLAB/                    # the inversion library
└── examples/                         # new_10.m (run by stage 4), save_gData.m
gui/                                  # NiceGUI app (see PLAN.md)
├── app.py  theme.py  theme.css  jobs.py
└── panels/                           # one module per tab + file picker
```

Per-dataset analysis outputs (checkpoints, histograms, result folders) are
gitignored.

## Further reading

Run the shutdown regression checks with `uv run python -m unittest discover -s tests -v`.
They use temporary local servers and dummy subprocesses; no dataset is required.

- `PLAN.md` — architecture and design decisions (why NiceGUI, why
  subprocesses, theme mapping from `DESIGN.md`).
- `DESIGN.md` — the UI design tokens the theme is built from.
- `AGENTS.md` — contributor/agent guidance: pipeline details, constraints
  (sequential MATLAB step, cwd-relative config), git workflow.

## License

MIT — see `LICENSE`. The bundled CPBASEX code under `pbasex-matlab/` is
third-party software, also MIT, copyright Elio Gilbert Champenois (2017);
see `pbasex-matlab/LICENSE.txt`.
