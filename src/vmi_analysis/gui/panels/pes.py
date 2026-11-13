"""Panel: process_gmd_bins.py — GMD bins → PES + betas."""

import re
import sys
from pathlib import Path

from nicegui import ui

from ..jobs import Job, JobManager
from .. import theme
from ..settings import centroid_settings, output_dir as centroid_output_dir
from ..resources import canonical_path
from .binning import BIN_LABEL_PATTERN
from .common import (JobView, REPO_ROOT, file_url, open_folder_button,
                     path_input, persist, zoomable_image, confirm_clear_outputs)

CIRC_DIR = REPO_ROOT / "circularization"
CENTROID_DIR = REPO_ROOT / "centroiding"
from vmi_analysis.paths import CPBASEX_DIR
PBASEX_GDATA = CPBASEX_DIR / "examples" / "G_r512_k512_l10.h5"
CALIBRATION_FILES = [
    "extracted-FOC100eV-20260710.txt",
    "reassignment-FOC100eV-20260710.txt",
]
PYABEL_METHODS = ["daun", "hansenlaw", "three_point", "two_point",
                  "basex", "rbasex", "linbasex", "onion_bordas", "direct"]
PES_PLOTS = ("gmd_bins-energy-spectrum.png", "gmd_bins-energy-spectrum-normalized.png", "gmd_bins-betas.png")


def _output_files(directory: Path, suffix: str, bin_directory: Path | None = None):
    stems = ("image", "image_circularized", "image_cleaned", "energy_l10", "energy_spec_l10", "betas10", "px-py_l10")
    pattern = re.compile(rf"(?:{'|'.join(stems)})_GMD_{BIN_LABEL_PATTERN}{re.escape(suffix)}\.txt")
    current = []
    if bin_directory is not None:
        for source in bin_directory.glob("vmi_2d_gmd_bin*.npy"):
            stem = source.stem
            if stem.endswith(suffix) and "_w-" not in stem.removesuffix(suffix):
                label = stem.removeprefix("vmi_2d_gmd_")
                current.extend(directory / f"{prefix}_GMD_{label}.txt" for prefix in stems)
    return [*(directory / name for name in (*PES_PLOTS, "gmd_bins_results.npz")),
            *current,
            *(path for path in sorted(directory.glob("*_GMD_bin*.txt")) if pattern.fullmatch(path.name))]


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
        "Apply Abel inversion to GMD-binned images to extract energy spectra and anisotropy parameters. "
        "This panel is immature and needs restructuring.")

    missing = [f for f in CALIBRATION_FILES if not (CIRC_DIR / f).exists()]
    if missing:
        ui.label("⚠ Calibration files missing in circularization/: "
                 + ", ".join(missing)
                 + " — supply them before running.").classes("text-negative")

    view = JobView(jobs, job_name='PES', results_builder=_results)

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
            cmd = [sys.executable, "-m", "vmi_analysis.circularization.process_gmd_bins",
                   "--weight-suffix", weight_suffix.value or "",
                   "--bin-dir", bin_dir.value,
                   "--inversion-backend", backend.value,
                   "--pyabel-method", method.value,
                   "--no-show"]
            if output_dir.value:
                cmd += ["--output-dir", output_dir.value]
            if use_cache.value:
                cmd.append("--use-cache")
            out = result_directory()
            job = jobs.submit(Job(name="PES", cmd=cmd, cwd=str(CIRC_DIR),
                                  extra={"output_dir": str(out)}))
            view.attach(job)

        def result_directory():
            return (canonical_path(output_dir.value, CIRC_DIR) if output_dir.value
                    else default_output_dir(weight_suffix.value or "", backend.value, method.value))

        with ui.row().classes("gap-2"):
            view.run_button(on_click=run)
            ui.button("Clear outputs", icon="delete", on_click=lambda: confirm_clear_outputs(
                _output_files(result_directory(), weight_suffix.value or "", canonical_path(bin_dir.value, CIRC_DIR)),
                "Delete PES outputs?",
                on_cleared=view.results.clear)).props("flat color=negative")

    view.place_console()


def _results(job: Job):
    if job.status != "done":
        return
    out_dir = Path(job.extra["output_dir"])
    open_folder_button(out_dir, "open results folder")
    for name in PES_PLOTS:
        png = out_dir / name
        if png.exists():
            ui.label(name).classes("vmi-caption")
            zoomable_image(file_url(png)).classes("w-full max-w-4xl")
    npz = out_dir / "gmd_bins_results.npz"
    if npz.exists():
        ui.label(f"Data: {npz}").classes("vmi-subtitle")
