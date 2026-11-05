"""Panel: pipeline — gmd_map → centroiding → binning → pes in one job."""

import sys
import tomllib
from pathlib import Path

from nicegui import ui

from ..jobs import Job, JobManager, Stage
from .. import theme
from .common import JobView, REPO_ROOT, num_input, path_input
from .pes import default_output_dir
from .binning import _suffix

CENTROID_DIR = REPO_ROOT / "centroiding"
CIRC_DIR = REPO_ROOT / "circularization"
CONFIG_PATH = CENTROID_DIR / "config.toml"


def build(jobs: JobManager) -> None:
    theme.section_title(
        "Pipeline",
        "Run the full chain as one sequential job: GMD map → Picasso "
        "centroiding → GMD binning → PES/Abel inversion. Configure the "
        "centroiding parameters on its own tab first — they are read from "
        "centroiding/config.toml.")

    view = JobView(jobs)

    stage_badges: list[ui.badge] = []
    with ui.row().classes("gap-2 items-center"):
        ui.label("Stages:").classes("vmi-caption")
        stage_names = ("GMD map", "Centroiding", "Binning", "PES")
        for i, name in enumerate(stage_names):
            b = theme.status_badge("idle")
            stage_badges.append(b)
            if i < len(stage_names) - 1:
                ui.icon("arrow_forward").classes("text-grey-5") \
                    .style("font-size: 14px")
    current_job: dict = {"job": None}

    def refresh_badges():
        job = current_job["job"]
        if job is None or job.stages is None:
            return
        for badge, stage in zip(stage_badges, job.stages):
            theme.set_badge(badge, stage.status)

    ui.timer(0.7, refresh_badges)

    with theme.panel_card():
        h5_dir = path_input("HDF5 directory (dataset)",
                            r"F:\1103\1103_9_Aele", browse="dir")
        gmd_path = path_input("GMD timestamp file",
                              r"F:\1103\current-time-both.txt",
                              browse="file", filters=[".txt"])
        map_out = path_input("GMD map output",
                             str(REPO_ROOT / "gmd_map_all.txt"),
                             browse="save", filters=[".txt"],
                             default_name="gmd_map_all.txt")
        with ui.row().classes("gap-4 flex-wrap"):
            weight_mode = ui.select(["none", "gmd", "gmd_norm"],
                                    value="gmd_norm", label="Weight mode") \
                .props("outlined dense")
            weight_power = num_input("Weight power", 1.0, step=0.1)
            skip_existing = ui.checkbox(
                "Skip stages whose outputs exist", value=True)

        def run():
            suffix = _suffix(weight_mode.value, weight_power.value)
            try:
                cfg = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                vmi = cfg["vmi_picasso"]
                out_dir = CENTROID_DIR / vmi["output_dir"]
                checkpoint = out_dir / vmi.get(
                    "checkpoint_file", "vmi_results_raw.h5")
            except Exception as e:
                ui.notify(f"Cannot read {CONFIG_PATH}: {e}", type="negative")
                return

            py = sys.executable
            stages = [
                Stage(
                    name="GMD map",
                    cmd=[py, str(REPO_ROOT / "build_gmd_map.py"),
                         "--h5-dir", h5_dir.value,
                         "--gmd-path", gmd_path.value,
                         "--output-path", map_out.value],
                    cwd=str(REPO_ROOT),
                    outputs=[map_out.value],
                ),
                Stage(
                    name="Centroiding",
                    cmd=[py, str(CENTROID_DIR /
                                 "process_andor_vmi_PICASSO_v5_h5_batch.py"),
                         "--config", str(CONFIG_PATH)],
                    cwd=str(CENTROID_DIR),
                    outputs=[str(checkpoint)],
                ),
                Stage(
                    name="Binning",
                    cmd=[py, str(CENTROID_DIR / "vmi_histogram_gmd_bins.py"),
                         "--config", str(CONFIG_PATH),
                         "--weight-mode", weight_mode.value,
                         "--weight-power", str(weight_power.value),
                         "--no-show"],
                    cwd=str(CENTROID_DIR),
                    outputs=[str(out_dir / f"vmi_2d_gmd_all{suffix}.npy")],
                ),
                Stage(
                    name="PES",
                    cmd=[py, str(CIRC_DIR / "process_gmd_bins.py"),
                         "--weight-suffix", suffix,
                         "--bin-dir", str(out_dir),
                         "--inversion-backend", "pyabel",
                         "--no-show"],
                    cwd=str(CIRC_DIR),
                    outputs=[str(default_output_dir(suffix, "pyabel", "daun")
                                 / "gmd_bins_results.npz")],
                ),
            ]
            job = jobs.submit(Job(name="Pipeline", stages=stages,
                                  skip_existing=bool(skip_existing.value)))
            current_job["job"] = job
            view.attach(job)

        theme.run_button("Run pipeline", on_click=run)
        ui.label("Watch each stage's detailed parameters on its own tab; "
                 "this panel uses the same defaults.").classes("vmi-subtitle")

    view.place_console()
