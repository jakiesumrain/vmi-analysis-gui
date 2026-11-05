"""Panel: vmi_histogram_gmd_bins.py — GMD-binned 2D histograms."""

import sys
import tomllib
from pathlib import Path

from nicegui import ui

from ..jobs import Job, JobManager
from .. import theme
from .common import (JobView, REPO_ROOT, file_url, num_input, path_input,
                     zoomable_image)

CENTROID_DIR = REPO_ROOT / "centroiding"
SCRIPT = CENTROID_DIR / "vmi_histogram_gmd_bins.py"
CONFIG_PATH = CENTROID_DIR / "config.toml"


def build(jobs: JobManager) -> None:
    theme.section_title(
        "GMD Binning",
        "Partition checkpoint events into GMD bins, one 2D histogram per bin "
        "(vmi_histogram_gmd_bins.py). Run after centroiding.")

    view = JobView(jobs, results_builder=_results)

    with theme.panel_card():
        config = path_input("Config file", str(CONFIG_PATH),
                            browse="file", filters=[".toml"])
        thresholds = ui.input("GMD thresholds (comma-separated)",
                              value="3.75e-8,5.5e-8,7.25e-8") \
            .classes("w-full").props("outlined dense")
        with ui.row().classes("gap-4 flex-wrap"):
            gmd_col = ui.select([4, 5], value=4,
                                label="GMD column (4=m1, 5=m2)") \
                .props("outlined dense")
            weight_mode = ui.select(["none", "gmd", "gmd_norm"],
                                    value="gmd_norm", label="Weight mode") \
                .props("outlined dense")
            weight_power = num_input("Weight power", 1.0, step=0.1)
            gmd_floor = num_input("GMD floor", 1e-12, format="%.1e")

        def run():
            cmd = [sys.executable, str(SCRIPT),
                   "--config", config.value,
                   "--thresholds", thresholds.value,
                   "--gmd-col", str(gmd_col.value),
                   "--weight-mode", weight_mode.value,
                   "--weight-power", str(weight_power.value),
                   "--gmd-floor", str(gmd_floor.value),
                   "--no-show"]
            job = jobs.submit(Job(name="GMD binning", cmd=cmd,
                                  cwd=str(CENTROID_DIR),
                                  extra={"weight_mode": weight_mode.value,
                                         "weight_power": weight_power.value}))
            view.attach(job)

        theme.run_button(on_click=run)

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
    try:
        cfg = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        out_dir = CENTROID_DIR / cfg["vmi_picasso"]["output_dir"]
    except Exception as e:
        ui.label(f"Could not locate output dir: {e}").classes("vmi-subtitle")
        return
    suffix = _suffix(job.extra.get("weight_mode", "gmd_norm"),
                     job.extra.get("weight_power", 1.0))
    png = out_dir / f"vmi_gmd_bins{suffix}.png"
    if png.exists():
        ui.label(f"Bins overview ({png.name}):").classes("vmi-caption")
        zoomable_image(file_url(png)).classes("w-full max-w-5xl")
    else:
        ui.label(f"Expected plot not found: {png}").classes("vmi-subtitle")
