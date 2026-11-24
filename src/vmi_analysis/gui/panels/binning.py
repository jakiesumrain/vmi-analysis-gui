"""Panel: vmi_histogram_gmd_bins.py — GMD-binned 2D histograms."""

import asyncio
import re
import sys
from functools import partial
from pathlib import Path

import numpy as np
from nicegui import ui

from vmi_analysis.centroiding.binning_options import (
    DEFAULT_GMD_COLUMN, DEFAULT_GMD_FLOOR, DEFAULT_THRESHOLDS, DEFAULT_WEIGHT_MODE,
    DEFAULT_WEIGHT_POWER, WEIGHT_MODES, histogram_filenames, parse_thresholds, weight_suffix as _suffix,
)
from vmi_analysis.paths import validate_distinct_output

from ..agent_panels import PanelOperation, number, require_completed_job, resolve_path
from ..shared import centroid_dir, checked_path
from ..accumulate_plot import FRAME_HISTOGRAM_HEIGHT, frame_histogram_figure
from ..centroid_plots import plot_options
from ..gmd_histograms import gmd_histogram_samples
from ..jobs import Job, JobManager, ResourceBusyError
from .. import theme
from ..settings import binning_inputs
from ..resources import canonical_path
from .common import (JobView, REPO_ROOT, action_bar, adopt_agent_form, field_grid, file_url,
                     num_input, open_folder_button, path_input, persist,
                     zoomable_image, confirm_clear_outputs)

CENTROID_DIR = REPO_ROOT / "centroiding"
JOB_NAME = "GMD binning"
PANEL_TITLE = "GMD Binning"
BIN_LABEL_PATTERN = r"bin\d+_(?:minusinf|[+-]?\d\.\d{2}e[+-]\d+|[+-]?inf|nan)_(?:plusinf|[+-]?\d\.\d{2}e[+-]\d+|[+-]?inf|nan)"


def _resolve(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else centroid_dir(CENTROID_DIR) / path


def binning_job(checkpoint, gmd_map, output_dir, *, thresholds=DEFAULT_THRESHOLDS,
                resolution_factor=1, gmd_col=DEFAULT_GMD_COLUMN, weight_mode=DEFAULT_WEIGHT_MODE,
                weight_power=DEFAULT_WEIGHT_POWER, gmd_floor=DEFAULT_GMD_FLOOR,
                gmd_weight_min=0.0, gmd_weight_max=4.0):
    cwd = centroid_dir(CENTROID_DIR)
    if any(not isinstance(path, (str, Path)) or not str(path).strip()
           for path in (checkpoint, gmd_map, output_dir)):
        raise ValueError("Choose a localization checkpoint, GMD map, and output directory.")
    thresholds = thresholds.strip() if isinstance(thresholds, str) else thresholds
    limits = parse_thresholds(thresholds)
    resolution = number(1 if resolution_factor is None else resolution_factor)
    if resolution < 1 or resolution != int(resolution):
        raise ValueError("Histogram resolution must be a positive whole number.")
    if type(gmd_col) is not int or gmd_col not in (4, 5):
        raise ValueError("Choose GMD map column 4 or 5.")
    if weight_mode not in WEIGHT_MODES:
        raise ValueError("Choose an available weight mode.")
    weight_power = float(number(weight_power))
    gmd_floor = number(gmd_floor)
    if gmd_floor <= 0:
        raise ValueError("GMD floor must be positive.")
    lower, upper = (number(0.0 if value is None else value) for value in (gmd_weight_min, gmd_weight_max))
    if weight_mode == "gmd_norm" and lower > 0 and upper > 0 and lower > upper:
        raise ValueError("Enter increasing GMD weight limits, or use zero for an open limit.")

    directory = checked_path(output_dir, write=True, cwd=cwd)
    if directory.exists() and not directory.is_dir():
        raise ValueError("Choose an output directory.")
    suffix = _suffix(weight_mode, weight_power)
    names = {"all_bins_image": f"vmi_2d_gmd_all{suffix}.npy", "overview": f"vmi_gmd_bins{suffix}.png",
             **{f"bin_{index}": filename for index, filename in enumerate(histogram_filenames(limits, suffix))}}
    inputs = (("checkpoint", checkpoint), ("GMD map", gmd_map))
    paths = {key: str(validate_distinct_output(checked_path(directory / filename, write=True),
                                               inputs, cwd=cwd, output_label="Binning output"))
             for key, filename in names.items()}
    if len(set(paths.values())) != len(paths):
        raise ValueError("Choose separate paths for the Binning outputs.")
    form = dict(checkpoint=str(checkpoint), gmd_map=str(gmd_map), output_dir=str(output_dir),
                thresholds=thresholds, resolution_factor=int(resolution), gmd_col=gmd_col,
                weight_mode=weight_mode, weight_power=weight_power, gmd_floor=gmd_floor,
                gmd_weight_min=lower, gmd_weight_max=upper)
    cmd = [sys.executable, "-m", "vmi_analysis.centroiding.vmi_histogram_gmd_bins",
           "--checkpoint-path", str(checkpoint), "--output-dir", str(output_dir), "--gmd-map-path", str(gmd_map),
           "--resolution-factor", str(int(resolution)), "--gmd-weight-min", str(lower), "--gmd-weight-max", str(upper),
           f"--thresholds={thresholds}", "--gmd-col", str(gmd_col), "--weight-mode", weight_mode,
           "--weight-power", str(weight_power), "--gmd-floor", str(gmd_floor), "--no-show"]
    return Job(JOB_NAME, cmd=cmd, cwd=str(cwd), extra={
        **paths, "output_dir": str(directory), "weight_mode": weight_mode, "weight_power": weight_power,
        "panel_form": form, "binning_artifacts": tuple(paths),
    })


def _prepare_agent(jobs, params):
    inputs = dict(params)
    for path_key, id_key, job_name, artifact in (
        ("checkpoint", "centroid_job_id", "Centroiding", "checkpoint"),
        ("gmd_map", "gmd_map_job_id", "GMD map", "output_path"),
    ):
        if id_key in inputs:
            if path_key in inputs:
                raise ValueError(f"Supply either {path_key} or {id_key}.")
            parent = require_completed_job(jobs, inputs.pop(id_key), name=job_name)
            if artifact not in parent.extra:
                raise ValueError(f"Choose a completed {job_name} job that produced {path_key}.")
            inputs[path_key] = parent.extra[artifact]
        inputs[path_key] = resolve_path(inputs.get(path_key))
        if not inputs[path_key].is_file():
            raise ValueError(f"Choose an existing {path_key.replace('_', ' ')} file.")
    inputs["output_dir"] = resolve_path(inputs.get("output_dir"), write=True)
    if isinstance(inputs.get("thresholds"), list):
        inputs["thresholds"] = ",".join(str(number(value)) for value in inputs["thresholds"])
    job = binning_job(**inputs)
    form = job.extra["panel_form"]
    summary = {"Localization checkpoint": form["checkpoint"], "Frame-to-GMD map": form["gmd_map"],
               "Output directory": job.extra["output_dir"], "GMD thresholds": form["thresholds"],
               "GMD column": form["gmd_col"], "Histogram resolution factor": form["resolution_factor"],
               "GMD floor": form["gmd_floor"], "Weight mode": form["weight_mode"]}
    if form["weight_mode"] == "gmd_norm":
        summary.update({"Weight power": form["weight_power"],
                        "Weight limits (zero = open)": [form["gmd_weight_min"], form["gmd_weight_max"]]})
    return job, summary


def agent_operations():
    description = {
        "panel": PANEL_TITLE,
        "summary": "Group centroided events by GMD pulse energy and save one VMI image per bin and their sum.",
        "manual": PANEL_TITLE,
        "outputs": ["all_bins_image", "overview", "bin_0, bin_1, …"],
        "parameters": {
            "checkpoint": {"type": "path", "description": "Localization checkpoint. Alternative to centroid_job_id."},
            "centroid_job_id": {"type": "job ID", "description": "Completed batch Centroiding job"},
            "gmd_map": {"type": "path", "description": "Frame-to-GMD map. Alternative to gmd_map_job_id."},
            "gmd_map_job_id": {"type": "job ID", "description": "Completed GMD Map job"},
            "output_dir": {"type": "path", "description": "Output directory in your workspace"},
            "thresholds": {"type": "list of numbers or comma-separated string", "default": DEFAULT_THRESHOLDS,
                           "description": "Strictly increasing GMD thresholds. N thresholds make N + 1 bins."},
            "resolution_factor": {"type": "integer", "default": 1},
            "gmd_col": {"type": "integer", "default": DEFAULT_GMD_COLUMN, "choices": [4, 5]},
            "weight_mode": {"type": "string", "default": DEFAULT_WEIGHT_MODE, "choices": list(WEIGHT_MODES)},
            "weight_power": {"type": "number", "default": DEFAULT_WEIGHT_POWER},
            "gmd_floor": {"type": "number", "default": DEFAULT_GMD_FLOOR},
            "gmd_weight_min": {"type": "number", "default": 0.0},
            "gmd_weight_max": {"type": "number", "default": 4.0},
        },
    }
    return (PanelOperation("binning", description, _prepare_agent, (JOB_NAME,),
                           artifacts=lambda job: job.extra.get("binning_artifacts", ())),)


def build(jobs: JobManager) -> None:
    def restore_form(job):
        for key, value in adopt_agent_form(job, "binning").items():
            if key in fields:
                fields[key].value = value

    view = JobView(jobs, job_name=JOB_NAME, results_builder=_results, attach_builder=restore_form)
    theme.panel_header(
        PANEL_TITLE,
        "Group centroided events by GMD pulse energy and build one image per bin. "
        "This panel is immature and needs restructuring.",
        view=view)
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
                                      value=DEFAULT_THRESHOLDS)
                             .classes("w-full").props("outlined dense"),
                             "binning.thresholds")
        with field_grid(3):
            gmd_col = persist(ui.select(
                {4: "GMD value 1 (method 1)", 5: "GMD value 2 (method 2)"},
                value=DEFAULT_GMD_COLUMN, label="GMD column",
            ).props("outlined dense"), "binning.gmd_col")
            gmd_col.tooltip("Which GMD value from the map drives binning, "
                            "weights and the histograms: 4 = gmd_val1, 5 = gmd_val2.")
            weight_mode = persist(ui.select(list(WEIGHT_MODES),
                                            value=DEFAULT_WEIGHT_MODE,
                                            label="Weight mode")
                                  .props("outlined dense"),
                                  "binning.weight_mode")
            weight_power = persist(num_input("Weight power", DEFAULT_WEIGHT_POWER, step=0.1),
                                   "binning.weight_power")
            gmd_floor = persist(num_input("GMD floor", DEFAULT_GMD_FLOOR, format="%.1e"),
                                "binning.gmd_floor")
            weight_min = persist(num_input("GMD weight min", settings["gmd_weight_min"]),
                                 "binning.gmd_weight_min")
            weight_max = persist(num_input("GMD weight max", settings["gmd_weight_max"]),
                                 "binning.gmd_weight_max")

        with ui.expansion("GMD histograms", icon="bar_chart", value=False).classes("w-full").props(
            'header-class="vmi-section-toggle text-primary" expand-icon-class="text-primary"'
        ):
            ui.label("Diagnostic distributions from the GMD map above. The GMD "
                     "floor and weight min/max are omitted. Leave bin settings blank for automatic values.") \
                .classes("vmi-subtitle")

            ui.label("GMD values histogram").classes("vmi-caption")
            with ui.row().classes("gap-3 items-end flex-wrap"):
                gmd_hist_min = persist(
                    num_input("GMD bin min", None, step="any").classes("w-36")
                    .props('placeholder="auto"'), "binning.gmd_hist_min")
                gmd_hist_max = persist(
                    num_input("GMD bin max", None, step="any").classes("w-36")
                    .props('placeholder="auto"'), "binning.gmd_hist_max")
                gmd_hist_width = persist(
                    num_input("GMD bin size", None, step="any").classes("w-36")
                    .props('placeholder="auto"'), "binning.gmd_hist_size")
                gmd_values_button = ui.button("Histogram GMD values", icon="bar_chart") \
                    .props("flat color=primary")
            gmd_area = ui.column().classes("w-full gap-1")

            ui.label("Weight histogram").classes("vmi-caption")
            with ui.row().classes("gap-3 items-end flex-wrap"):
                weight_hist_min = persist(
                    num_input("Weight bin min", None, step="any").classes("w-36")
                    .props('placeholder="auto"'), "binning.weight_hist_min")
                weight_hist_max = persist(
                    num_input("Weight bin max", None, step="any").classes("w-36")
                    .props('placeholder="auto"'), "binning.weight_hist_max")
                weight_hist_width = persist(
                    num_input("Weight bin size", None, step="any").classes("w-36")
                    .props('placeholder="auto"'), "binning.weight_hist_size")
                weights_button = ui.button("Histogram weights", icon="bar_chart") \
                    .props("flat color=primary")
            weights_area = ui.column().classes("w-full gap-1")
            gmd_area.visible = weights_area.visible = False

        def hide_histogram(area):
            area.clear()
            area.visible = False

        def histogram_limits(low, high, width):
            """Blank fields use automatic settings."""
            return dict(
                x_min=None if low.value is None else float(low.value),
                x_max=None if high.value is None else float(high.value),
                bin_width=None if width.value is None else float(width.value),
            )

        async def show_gmd_histogram(kind):
            sample = "GMD values" if kind == "gmd" else "weights"
            path_text = (gmd_map.value or "").strip()
            if not path_text:
                message = "Set the frame-to-GMD map first."
                view.log.push(f"[GMD histogram] {message}")
                ui.notify(message, type="warning")
                return
            map_path = _resolve(path_text)
            area, button = ((gmd_area, gmd_values_button) if kind == "gmd"
                            else (weights_area, weights_button))
            button.disable()
            button.props("loading")
            try:
                limits = (histogram_limits(gmd_hist_min, gmd_hist_max,
                                           gmd_hist_width) if kind == "gmd"
                          else histogram_limits(weight_hist_min,
                                                weight_hist_max, weight_hist_width))
                column = int(gmd_col.value or 4)
                mode = weight_mode.value or "gmd_norm"
                power = 1.0 if weight_power.value is None else float(weight_power.value)
                view.log.push(f"[GMD histogram] Reading {map_path}…")
                with jobs.preview_read_access([map_path]):
                    gmd_values, weights, mean_gmd = await asyncio.to_thread(
                        gmd_histogram_samples, map_path, column, mode, power)
                if kind == "gmd":
                    caption = (f"{len(gmd_values):,} mapped frames · mean GMD "
                               f"{mean_gmd:.3e} · map column {column}")
                    values = gmd_values
                    title = f"GMD value distribution · column {column}"
                    x_label = f"GMD value (map column {column})"
                else:
                    finite = int(np.isfinite(weights).sum())
                    caption = (f"{len(weights):,} mapped frames · mode {mode} · "
                               f"{len(weights) - finite:,} non-finite values omitted")
                    values = weights
                    title = f"GMD weight distribution · {mode}"
                    x_label = f"GMD weight ({mode})"
                figure = await asyncio.to_thread(
                    frame_histogram_figure, values, title=title, x_label=x_label,
                    y_label="Frames", count_label="frames", auto_bins=True, log_y=False,
                    x_tickformat=".1e" if kind == "gmd" else None, **limits)
            except (OSError, ValueError, KeyError, ResourceBusyError) as exc:
                view.log.push(f"[GMD histogram] Error: {exc}")
                ui.notify(str(exc), type="negative")
                return
            finally:
                button.props(remove="loading")
                button.enable()

            area.clear()
            with area:
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label(caption).classes("vmi-caption")
                    ui.button("Hide", icon="visibility_off",
                              on_click=partial(hide_histogram, area)) \
                        .props("flat dense color=secondary")
                ui.plotly(plot_options(figure)).classes("w-full") \
                    .style(f"height: {FRAME_HISTOGRAM_HEIGHT}px")
            area.visible = True
            view.log.push(f"[GMD histogram] Plotted {sample} · {caption}")

        gmd_values_button.on_click(partial(show_gmd_histogram, "gmd"))
        weights_button.on_click(partial(show_gmd_histogram, "weights"))

        fields = dict(checkpoint=checkpoint, output_dir=output, gmd_map=gmd_map,
                      resolution_factor=resolution, thresholds=thresholds, gmd_col=gmd_col,
                      weight_mode=weight_mode, weight_power=weight_power, gmd_floor=gmd_floor,
                      gmd_weight_min=weight_min, gmd_weight_max=weight_max)

        def run():
            try:
                job = binning_job(**{key: widget.value for key, widget in fields.items()})
                view.attach(jobs.submit(job))
            except (OSError, ValueError, OverflowError) as exc:
                ui.notify(str(exc), type="warning")

        def clear():
            if not output.value:
                ui.notify("Set an output directory first.", type="warning")
                return
            directory = canonical_path(output.value, centroid_dir(CENTROID_DIR))
            if weight_power.value is None:
                ui.notify("Enter a weight power first.", type="warning")
                return
            suffix = _suffix(weight_mode.value, float(weight_power.value))
            confirm_clear_outputs(_output_files(directory, suffix), "Delete Binning outputs?",
                                  on_cleared=view.clear_results)

        with action_bar(view, on_click=run):
            ui.button("Clear outputs", icon="delete", on_click=clear) \
                .props("flat color=negative").classes("vmi-action-danger")

    view.place_console()


def _output_files(directory: Path, suffix: str):
    pattern = re.compile(rf"vmi_2d_gmd_{BIN_LABEL_PATTERN}{re.escape(suffix)}\.npy")
    return [directory / f"vmi_2d_gmd_all{suffix}.npy", directory / f"vmi_gmd_bins{suffix}.png",
            *(path for path in sorted(directory.glob("vmi_2d_gmd_bin*.npy")) if pattern.fullmatch(path.name))]


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
