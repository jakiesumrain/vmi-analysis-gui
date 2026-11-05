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
in `centroiding/<output_dir>/` (from `config.toml`) and are **not**
committed to git.

## Scripts and pipeline

End-to-end order:

```
build_gmd_map.py                          # 1. frame → GMD correspondence (timestamp match)
centroiding/process_andor_vmi_PICASSO_v5_h5_batch.py
                                          # 2. Picasso localization → checkpoint + 2D VMI histogram
centroiding/vmi_histogram_gmd_bins.py     # 3. split events into GMD bins → one 2D histogram per bin
circularization/process_gmd_bins.py       # 4. per bin: hole patch → rotate → center →
                                          #    circularization.py → apodize → Abel inversion → E, IE, betas
```

Side tools (QC, not pipeline stages):

- `h5_play_frames_movie.py` — play the first N frames as a movie; save mp4/gif.
- `h5_accumulate_fast_gmd.py` — fast pixel accumulation (threshold + morphology,
  optional GMD weighting); no sub-pixel localization. Superseded by the Picasso
  path for analysis, kept for quick looks.

| script | configuration |
|---|---|
| `build_gmd_map.py` | argparse CLI (`--h5-dir --gmd-path --output-path --n-files`; first two required) |
| `h5_play_frames_movie.py` | argparse CLI (`--h5-dir` required, `--headless`, `--dump-frames`, ...) |
| `h5_accumulate_fast_gmd.py` | argparse CLI (`--h5-dir` required, `--no-show`, `--live-path`, ...) |
| `process_andor_vmi_PICASSO_v5_h5_batch.py` | `config.toml` `[vmi_picasso]` + `--config PATH` |
| `vmi_histogram_gmd_bins.py` | config.toml paths + argparse CLI (`--thresholds`, `--weight-mode`, `--weight-power`, `--no-show`) |
| `process_gmd_bins.py` | argparse CLI (`--weight-suffix`, `--inversion-backend`, `--pyabel-method`, `--use-cache`, `--no-show`) |
| `circularization/circularization.py` | full argparse CLI — no changes needed |

## Constraints and traps

- **`process_gmd_bins.py` must run sequentially.** The MATLAB CPBASEX step
  (`new_10.m`) hardcodes `examples/image.txt` as a shared input; bins are
  processed one at a time and outputs copied out immediately. The default
  PyAbel backend (`INVERSION_BACKEND = "pyabel"`) avoids MATLAB entirely.
- **Circularization calibration files are required** by `process_gmd_bins.py`:
  `extracted-FOC100eV-20260710.txt` and `reassignment-FOC100eV-20260710.txt`
  (trig-distortion fits + radial reassignment map) in `circularization/`.
  They are per-experiment inputs, not generated here; supply them before
  running stage 4. The reassignment file for the reference dataset was the
  identity map — only the azimuthal (trig) terms act, so any radial
  nonlinearity propagates into the energy axis.
- **`config.toml` is read cwd/script-relative.** Run the centroiding scripts
  from the `centroiding/` directory (or pass `--config` where supported).
- **`vmi_histogram_gmd_bins.py` prefers the TkAgg backend** for its live plot
  window; override via `MPLBACKEND` (e.g. Agg) when running headless.
- **GMD weighting is a per-frame scalar** — it corrects yield, never shape.
- **MATLAB + CPBASEX are external** (not bundled). Only needed for the
  `matlab` inversion backend; point the `CPBASEX_DIR` environment variable
  at the checkout.

## Environment

- uv-managed project (`pyproject.toml`, Python ≥ 3.12). Run scripts with
  `uv run python <script>` from the repo root, or use the venv interpreter
  in `.venv/` directly.
- `uv sync` after cloning installs everything, including the GUI.

## Git workflow

All changes land on `main` through pull requests, never direct pushes:

1. Branch off `main` (`feature/<name>` or `fix/<name>`), commit there.
2. Push with `HTTPS_PROXY=http://127.0.0.1:7890` if direct GitHub access is
   blocked (repo-local `http.proxy` may already cover plain `git push`).
3. Open the PR with `gh pr create`, squash-merge with
   `gh pr merge --squash`; the head branch is deleted automatically.

Squash is the only enabled merge method; merge/rebase merges are disabled.
GitHub branch *protection* is unavailable (private repo, free plan) — the
no-direct-push rule is enforced by convention.

## GUI

`gui/` — NiceGUI app, one tab per script plus a pipeline tab, themed by
`DESIGN.md`. See `PLAN.md` for the architecture. Launch:
`uv run python -m gui.app` → http://localhost:8080.

Rules for GUI-related changes:

- Scripts stay standalone-runnable; CLI args default to sane neutral values
  (no machine-specific paths — data locations always come from the user).
- The GUI runs scripts as subprocesses (isolation + cancellation), never
  imports them into the web process.
- One job at a time (the sequential MATLAB constraint and disk bandwidth).
