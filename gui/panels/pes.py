"""Panel: process_gmd_bins.py — GMD bins → PES + betas."""

import sys
from pathlib import Path

from nicegui import ui

from ..jobs import Job, JobManager
from .. import theme
from ..settings import centroid_settings, output_dir as centroid_output_dir
from .common import (JobView, REPO_ROOT, file_url, open_folder_button,
                     path_input, persist, zoomable_image)

CIRC_DIR = REPO_ROOT / "circularization"
SCRIPT = CIRC_DIR / "process_gmd_bins.py"
CENTROID_DIR = REPO_ROOT / "centroiding"
PBASEX_GDATA = (REPO_ROOT / "pbasex-matlab" / "examples"
                / "G_r512_k512_l10.h5")
CALIBRATION_FILES = [
    "extracted-FOC100eV-20260710.txt",
    "reassignment-FOC100eV-20260710.txt",
]
PYABEL_METHODS = ["daun", "hansenlaw", "three_point", "two_point",
                  "basex", "rbasex", "linbasex", "onion_bordas", "direct"]


def default_output_dir(suffix: str, backend: str, method: str) -> Path:
    name = f"gmd_bins_results{suffix}" if suffix else "gmd_bins_results"
    if backend == "pyabel":
        name += f"_pyabel_{method}"
    return CIRC_DIR / name


def _default_bin_dir() -> str:
    """Use the saved centroiding output directory as the initial default."""
    return str(centroid_output_dir(centroid_settings()))


def build(jobs: JobManager) -> None:
    theme.section_title(
        "PES / Abel Inversion",
        "Hole patch → rotate → center → circularize → apodize → Abel "
        "inversion for each GMD bin (process_gmd_bins.py). "
        "Bins run sequentially — the MATLAB backend shares global state.")

    missing = [f for f in CALIBRATION_FILES if not (CIRC_DIR / f).exists()]
    if missing:
        ui.label("⚠ Calibration files missing in circularization/: "
                 + ", ".join(missing)
                 + " — supply them before running.").classes("text-negative")

    view = JobView(jobs, results_builder=_results)

    with theme.panel_card():
        weight_suffix = persist(ui.input(
            "Weight suffix", value="_w-gmd_norm",
            placeholder="'' = unweighted").classes("w-64")
            .props("outlined dense"), "pes.weight_suffix")
        bin_dir = persist(path_input(
            "Bin image directory", _default_bin_dir(), browse="dir"),
            "pes.bin_dir")
        output_dir = persist(path_input(
            "Output directory (empty = automatic from suffix/backend)", "",
            browse="dir"), "pes.output_dir")
        with ui.row().classes("gap-4 flex-wrap"):
            backend = persist(ui.select(
                ["pyabel", "matlab"], value="pyabel",
                label="Inversion backend").props("outlined dense"),
                "pes.backend")
            method = persist(ui.select(
                PYABEL_METHODS, value="daun",
                label="PyAbel method").props("outlined dense"), "pes.method")
            use_cache = persist(ui.checkbox("Use cached inversions",
                                            value=False), "pes.use_cache")
        gdata_warn = ui.label(
            "⚠ MATLAB basis data missing — generate "
            "pbasex-matlab/examples/G_r512_k512_l10.h5 first: in MATLAB, "
            "cd to pbasex-matlab/examples and run save_gData.m "
            "(time-consuming, tens of minutes).").classes("text-negative")

        def _sync_backend():
            gdata_warn.visible = (backend.value == "matlab"
                                  and not PBASEX_GDATA.exists())

        backend.on_value_change(lambda: _sync_backend())
        _sync_backend()

        def run():
            cmd = [sys.executable, str(SCRIPT),
                   "--weight-suffix", weight_suffix.value or "",
                   "--bin-dir", bin_dir.value,
                   "--inversion-backend", backend.value,
                   "--pyabel-method", method.value,
                   "--no-show"]
            if output_dir.value:
                cmd += ["--output-dir", output_dir.value]
            if use_cache.value:
                cmd.append("--use-cache")
            out = (Path(output_dir.value) if output_dir.value
                   else default_output_dir(weight_suffix.value or "",
                                           backend.value, method.value))
            job = jobs.submit(Job(name="PES", cmd=cmd, cwd=str(CIRC_DIR),
                                  extra={"output_dir": str(out)}))
            view.attach(job)

        theme.run_button(on_click=run)

    view.place_console()


def _results(job: Job):
    if job.status != "done":
        return
    out_dir = Path(job.extra["output_dir"])
    open_folder_button(out_dir, "open results folder")
    for name in ("gmd_bins-energy-spectrum.png",
                 "gmd_bins-energy-spectrum-normalized.png",
                 "gmd_bins-betas.png"):
        png = out_dir / name
        if png.exists():
            ui.label(name).classes("vmi-caption")
            zoomable_image(file_url(png)).classes("w-full max-w-4xl")
    npz = out_dir / "gmd_bins_results.npz"
    if npz.exists():
        ui.label(f"Data: {npz}").classes("vmi-subtitle")
