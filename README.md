# vmi-analysis-gui

VMI (velocity-map imaging) photoelectron analysis pipeline for the 2026-11
F:/1103 datasets, with a NiceGUI web interface that drives every script — and
the full pipeline — from one browser UI.

## Setup

```bash
uv sync
```

## Pipeline

```
build_gmd_map.py                                # 1. frame → GMD map
centroiding/process_andor_vmi_PICASSO_v5_h5_batch.py  # 2. Picasso → checkpoint
centroiding/vmi_histogram_gmd_bins.py           # 3. GMD-binned 2D histograms
circularization/process_gmd_bins.py             # 4. circularize + Abel → spectra
```

Side tools: `h5_play_frames_movie.py` (frame movie), `h5_accumulate_fast_gmd.py`
(fast accumulation).

## GUI

```bash
uv run python -m gui.app   # → http://localhost:8080
```

One tab per script plus a pipeline tab. See `PLAN.md` (build plan), `DESIGN.md`
(UI theme tokens), and `AGENTS.md` (repository guidance and constraints).
