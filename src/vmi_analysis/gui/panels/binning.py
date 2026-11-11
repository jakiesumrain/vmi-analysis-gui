"""Panel: vmi_histogram_gmd_bins.py — GMD-binned 2D histograms."""

import sys
from pathlib import Path

from nicegui import ui

from ..jobs import Job, JobManager
from .. import theme
from ..settings import binning_inputs
from .common import (JobView, REPO_ROOT, file_url, num_input,
                     open_folder_button, path_input, persist, zoomable_image)

CENTROID_DIR = REPO_ROOT / "centroiding"


def build(jobs: JobManager) -> None:
    theme.section_title(
        "GMD Binning",
        "Group centroided events by GMD pulse energy and build one image per bin. "
        "This panel is immature and needs restructuring.")

    view = JobView(jobs, job_name='GMD binning', results_builder=_results)
    settings = binning_inputs()

    with theme.panel_card():
        checkpoint = persist(path_input("Localization checkpoint (.h5)",
                                        settings["checkpoint"],
                                        browse="file", filters=[".h5"]), "binning.checkpoint")
        output = persist(path_input("Output directory", settings["output_dir"], browse="dir"),
                         "binning.output_dir")
        gmd_map = persist(path_input("Frame-to-GMD map", settings["gmd_map"],
                                    browse="file", filters=[".txt"]), "binning.gmd_map")
        resolution = persist(num_input("Histogram resolution factor", settings["resolution_factor"], step=1),
                             "binning.resolution_factor")
        thresholds = persist(ui.input("GMD thresholds (comma-separated)",
                                      value="3.75e-8,5.5e-8,7.25e-8")
                             .classes("w-full").props("outlined dense"),
                             "binning.thresholds")
        with ui.row().classes("gap-4 flex-wrap"):
            gmd_col = persist(ui.select([4, 5], value=4,
                                        label="GMD column (4=m1, 5=m2)")
                              .props("outlined dense"), "binning.gmd_col")
            weight_mode = persist(ui.select(["none", "gmd", "gmd_norm"],
                                            value="gmd_norm",
                                            label="Weight mode")
                                  .props("outlined dense"),
                                  "binning.weight_mode")
            weight_power = persist(num_input("Weight power", 1.0, step=0.1),
                                   "binning.weight_power")
            gmd_floor = persist(num_input("GMD floor", 1e-12, format="%.1e"),
                                "binning.gmd_floor")
            weight_min = persist(num_input("GMD weight min", settings["gmd_weight_min"]),
                                 "binning.gmd_weight_min")
            weight_max = persist(num_input("GMD weight max", settings["gmd_weight_max"]),
                                 "binning.gmd_weight_max")

        def run():
            cmd = [sys.executable, "-m", "vmi_analysis.centroiding.vmi_histogram_gmd_bins",
                   "--checkpoint-path", checkpoint.value,
                   "--output-dir", output.value,
                   "--gmd-map-path", gmd_map.value,
                   "--resolution-factor", str(int(resolution.value or 1)),
                   "--gmd-weight-min", str(weight_min.value or 0),
                   "--gmd-weight-max", str(weight_max.value or 0),
                   "--thresholds", thresholds.value,
                   "--gmd-col", str(gmd_col.value),
                   "--weight-mode", weight_mode.value,
                   "--weight-power", str(weight_power.value),
                   "--gmd-floor", str(gmd_floor.value),
                   "--no-show"]
            job = jobs.submit(Job(name="GMD binning", cmd=cmd,
                                  cwd=str(CENTROID_DIR),
                                  extra={"weight_mode": weight_mode.value,
                                         "weight_power": weight_power.value,
                                         "output_dir": str(Path(output.value) if Path(output.value).is_absolute()
                                                           else CENTROID_DIR / output.value)}))
            view.attach(job)

        view.run_button(on_click=run)

    view.place_console()


def _suffix(mode: str, power: float) -> str:
    if mode == "none":
        return ""
    if mode == "gmd_norm" and power != 1.0:
        return f"_w-gmd_norm_p{power}"
    return f"_w-{mode}"


def _results(job: Job):
    if job.status != "done":
        return
    out_dir = Path(job.extra["output_dir"])
    suffix = _suffix(job.extra.get("weight_mode", "gmd_norm"),
                     job.extra.get("weight_power", 1.0))
    open_folder_button(out_dir, "open output folder")
    png = out_dir / f"vmi_gmd_bins{suffix}.png"
    if png.exists():
        ui.label(f"Bins overview ({png.name}):").classes("vmi-caption")
        zoomable_image(file_url(png)).classes("w-full max-w-5xl")
    else:
        ui.label(f"Expected plot not found: {png}").classes("vmi-subtitle")
