"""Panel: circularization.py — single-image circularization."""

import sys
from pathlib import Path

from nicegui import ui

from ..jobs import Job, JobManager
from .. import theme
from .common import (GENERATED, JobView, REPO_ROOT, file_url, num_input,
                     path_input, render_txt_image, serve_dir, zoomable_image)

CIRC_DIR = REPO_ROOT / "circularization"
SCRIPT = CIRC_DIR / "circularization.py"


def build(jobs: JobManager) -> None:
    theme.section_title(
        "Circularize",
        "Correct radial distortion of one VMI image using the Gascooke "
        "trig-fit calibration (circularization.py).")

    view = JobView(jobs, results_builder=_results)

    with theme.panel_card():
        inp = path_input("Input image (.txt)",
                         str(CIRC_DIR / "image.txt"),
                         browse="file", filters=[".txt"])
        out = path_input("Output image (.txt)",
                         str(CIRC_DIR / "image_circularized.txt"),
                         browse="save", filters=[".txt"],
                         default_name="image_circularized.txt")
        with ui.row().classes("gap-4 flex-wrap"):
            shape_y = num_input("Shape NY", 1023, step=1)
            shape_x = num_input("Shape NX", 1023, step=1)
            center_x = num_input("Center X", 511.0, step=1)
            center_y = num_input("Center Y", 511.0, step=1)
        extracted = path_input(
            "Extracted trig-fit file",
            str(CIRC_DIR / "extracted-FOC100eV-20260710.txt"),
            browse="file", filters=[".txt"])
        reassign = path_input(
            "Reassignment file (optional)",
            str(CIRC_DIR / "reassignment-FOC100eV-20260710.txt"),
            browse="file", filters=[".txt"])
        with ui.row().classes("gap-4 flex-wrap"):
            radial_model = ui.select(["interp", "poly"], value="interp",
                                     label="Radial model").props("outlined dense")
            poly_order = num_input("Poly order", 2, step=1)
            interp_order = num_input("Interp order", 3, step=1)
            max_trig_n = num_input("Max trig n (0 = auto)", 0, step=1)
            polar_dr = num_input("Polar dr", 1.0, step=0.5)
        with ui.row().classes("gap-4 flex-wrap"):
            no_jacobian = ui.checkbox("No Jacobian", value=True)
            no_plots = ui.checkbox("No diagnostic plots", value=False)
            summary_only = ui.checkbox("Summary only", value=False)
        plot_dir = path_input("Plot output dir", str(CIRC_DIR / "trig_fit_plots"),
                              browse="dir")

        def run():
            cmd = [sys.executable, str(SCRIPT),
                   "--input", inp.value,
                   "--output", out.value,
                   "--shape", str(int(shape_y.value)), str(int(shape_x.value)),
                   "--center", str(center_x.value), str(center_y.value),
                   "--extracted", extracted.value,
                   "--radial-model", radial_model.value,
                   "--poly-order", str(int(poly_order.value)),
                   "--interp-order", str(int(interp_order.value)),
                   "--polar-dr", str(polar_dr.value),
                   "--plot-dir", plot_dir.value]
            if max_trig_n.value:
                cmd += ["--max-trig-n", str(int(max_trig_n.value))]
            if reassign.value:
                cmd += ["--reassign-file", reassign.value]
            if no_jacobian.value:
                cmd.append("--no-jacobian")
            if no_plots.value:
                cmd.append("--no-plots")
            if summary_only.value:
                cmd.append("--summary-only")
            job = jobs.submit(Job(name="Circularize", cmd=cmd,
                                  cwd=str(CIRC_DIR),
                                  extra={"output": out.value,
                                         "plot_dir": plot_dir.value}))
            view.attach(job)

        theme.run_button(on_click=run)

    view.place_console()


def _results(job: Job):
    if job.status != "done":
        return
    out = Path(job.extra["output"])
    if out.exists():
        png = GENERATED / f"circ_{job.id}.png"
        try:
            render_txt_image(out, png)
            ui.label("Circularized image:").classes("vmi-caption")
            zoomable_image(file_url(png)) \
                .classes("w-full max-w-2xl")
        except Exception as e:
            ui.label(f"Could not render {out}: {e}").classes("vmi-subtitle")
    plots = sorted(Path(job.extra["plot_dir"]).glob("*.png"))
    if plots:
        ui.label("Diagnostic plots:").classes("vmi-caption")
        with ui.row().classes("gap-4 flex-wrap"):
            for p in plots:
                zoomable_image(file_url(p)).classes("w-80")
