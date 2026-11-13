"""Panel: pipeline — gmd_map → centroiding → binning → pes in one job."""

import sys
from pathlib import Path

from nicegui import app, ui
from vmi_analysis.centroiding.options import centroid_cli_args

from ..jobs import Job, JobManager, Stage
from .. import theme
from ..settings import centroid_settings, output_dir as centroid_output_dir, binning_cli_args
from .common import JobView, REPO_ROOT, confirm_clear_outputs, num_input, path_input, persist
from .pes import default_output_dir, _output_files as pes_outputs
from .binning import _suffix, _output_files as binning_outputs
from .centroid import OUTPUT_NAMES as CENTROID_OUTPUTS

CENTROID_DIR = REPO_ROOT / "centroiding"
CIRC_DIR = REPO_ROOT / "circularization"


def build(jobs: JobManager) -> None:
    theme.section_title(
        "Pipeline",
        "Run GMD mapping, Centroiding, Binning, and PES in sequence. "
        "This panel is immature and needs restructuring.")

    view = JobView(jobs, job_name='Pipeline')

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
        h5_dir = persist(path_input(
            "HDF5 directory (dataset)", "", browse="dir",
            placeholder="folder containing RAW-*.h5 files"), "pipeline.h5_dir")
        gmd_path = persist(path_input(
            "GMD timestamp file", "", browse="file", filters=[".txt"],
            placeholder="current-time-both.txt"), "pipeline.gmd_path")
        map_out = persist(path_input(
            "GMD map output", str(REPO_ROOT / "gmd_map_all.txt"),
            browse="save", filters=[".txt"],
            default_name="gmd_map_all.txt"), "pipeline.map_out")
        with ui.row().classes("gap-4 flex-wrap"):
            weight_mode = persist(ui.select(
                ["none", "gmd", "gmd_norm"], value="gmd_norm",
                label="Weight mode").props("outlined dense"),
                "pipeline.weight_mode")
            weight_power = persist(num_input("Weight power", 1.0, step=0.1),
                                   "pipeline.weight_power")
            skip_existing = persist(ui.checkbox(
                "Skip stages whose outputs exist", value=True),
                "pipeline.skip_existing")

        def run():
            try:
                power = float(weight_power.value) if weight_power.value is not None else 1.0
                suffix = _suffix(weight_mode.value, power)
                cfg = centroid_settings()
                vmi = cfg["vmi_picasso"]
                # The pipeline's chosen inputs must reach all stages, even
                # when the Centroiding tab was last used for another dataset.
                if not h5_dir.value:
                    raise ValueError("Choose an HDF5 dataset directory")
                vmi["h5_dir"] = str(Path(h5_dir.value).resolve())
                vmi["gmd_map_path"] = str(Path(map_out.value).resolve())
                vmi["processing"]["tuning_only"] = False
                out_dir = centroid_output_dir(cfg)
                checkpoint = out_dir / vmi["checkpoint_file"]
                centroid_args = centroid_cli_args(cfg)
                if not gmd_path.value or not map_out.value:
                    raise ValueError("Choose a GMD timestamp file and map output path")
            except ValueError as e:
                ui.notify(str(e), type="negative")
                return

            py = sys.executable
            stages = [
                Stage(
                    name="GMD map",
                    cmd=[py, "-m", "vmi_analysis.gmd.build_gmd_map",
                         "--h5-dir", h5_dir.value,
                         "--gmd-path", gmd_path.value,
                         "--output-path", map_out.value],
                    cwd=str(REPO_ROOT),
                    outputs=[map_out.value],
                ),
                Stage(
                    name="Centroiding",
                    cmd=[py, "-m", "vmi_analysis.centroiding.process_andor_vmi_PICASSO_v5_h5_batch",
                         *centroid_args],
                    cwd=str(CENTROID_DIR),
                    outputs=[str(checkpoint)],
                ),
                Stage(
                    name="Binning",
                    cmd=[py, "-m", "vmi_analysis.centroiding.vmi_histogram_gmd_bins",
                         *binning_cli_args(cfg),
                         "--thresholds", str(app.storage.general.get("binning.thresholds", "3.75e-8,5.5e-8,7.25e-8")),
                         "--gmd-col", str(app.storage.general.get("binning.gmd_col", 4)),
                         "--gmd-floor", str(app.storage.general.get("binning.gmd_floor", 1e-12)),
                         "--weight-mode", weight_mode.value,
                         "--weight-power", str(power),
                         "--no-show"],
                    cwd=str(CENTROID_DIR),
                    outputs=[str(out_dir / f"vmi_2d_gmd_all{suffix}.npy")],
                ),
                Stage(
                    name="PES",
                    cmd=[py, "-m", "vmi_analysis.circularization.process_gmd_bins",
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
                                  skip_existing=bool(skip_existing.value),
                                  extra={"output_dir": str(out_dir), "suffix": suffix}))
            current_job["job"] = job
            view.attach(job)

        def clear():
            job = view.job or next((job for job in reversed(jobs.jobs()) if job.name == "Pipeline"), None)
            if job is None or job.finished is None:
                ui.notify("Clear outputs after a Pipeline run finishes.", type="info")
                return
            confirm_clear_outputs(_written_outputs(job), "Delete outputs written by this Pipeline run?",
                                  on_cleared=view.results.clear)

        with ui.row().classes("gap-2"):
            view.run_button("Run pipeline", on_click=run)
            ui.button("Clear outputs", icon="delete", on_click=clear).props("flat color=negative")
        ui.label("Watch each stage's detailed parameters on its own tab; "
                 "this panel uses the same defaults.").classes("vmi-subtitle")

    view.place_console()


def _written_outputs(job):
    if job.started is None or job.finished is None or not job.stages:
        return []
    directory = Path(job.extra["output_dir"])
    suffix = job.extra["suffix"]
    targets = []
    for stage in job.stages:
        if stage.started is None or stage.finished is None:
            continue
        candidates = [Path(stage.cwd or REPO_ROOT) / path for path in stage.outputs]
        if stage.name == "Centroiding":
            candidates.extend(directory / name for name in CENTROID_OUTPUTS)
        elif stage.name == "Binning":
            candidates.extend(binning_outputs(directory, suffix))
        elif stage.name == "PES":
            candidates.extend(pes_outputs(candidates[0].parent, suffix, directory))
        for path in candidates:
            try:
                if path.is_file() and stage.started <= path.stat().st_mtime <= stage.finished:
                    targets.append(path)
            except OSError:
                continue
    return targets
