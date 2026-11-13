"""Panel: vmi_histogram_gmd_bins.py — GMD-binned 2D histograms."""

import asyncio
import re
import sys
from functools import partial
from pathlib import Path

import numpy as np
from nicegui import ui

from ..accumulate_plot import FRAME_HISTOGRAM_HEIGHT, frame_histogram_figure
from ..centroid_plots import plot_options
from ..gmd_histograms import gmd_histogram_samples
from ..jobs import Job, JobManager, ResourceBusyError
from .. import theme
from ..settings import binning_inputs
from ..resources import canonical_path
from .common import (JobView, REPO_ROOT, file_url, num_input,
                     open_folder_button, path_input, persist, zoomable_image, confirm_clear_outputs)

CENTROID_DIR = REPO_ROOT / "centroiding"
BIN_LABEL_PATTERN = r"bin\d+_(?:minusinf|[+-]?\d\.\d{2}e[+-]\d+|[+-]?inf|nan)_(?:plusinf|[+-]?\d\.\d{2}e[+-]\d+|[+-]?inf|nan)"


def _resolve(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else CENTROID_DIR / path


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
            gmd_col = persist(ui.select(
                {4: "GMD value 1 (method 1)", 5: "GMD value 2 (method 2)"},
                value=4, label="GMD column",
            ).props("outlined dense").classes("w-64"), "binning.gmd_col")
            gmd_col.tooltip("Which GMD value from the map drives binning, "
                            "weights and the histograms: 4 = gmd_val1, 5 = gmd_val2.")
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

        with ui.expansion("GMD histograms", icon="bar_chart", value=False).classes("w-full").props(
            'header-class="vmi-section-toggle text-primary" expand-icon-class="text-primary"'
        ):
            ui.label("Diagnostic distributions from the GMD map above. The GMD "
                     "floor and weight min/max are not applied here; set the "
                     "min, max and bin size for each histogram.") \
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
            """Blank fields are passed as None so NumPy chooses them."""
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
                power = float(weight_power.value or 1.0)
                view.log.push(f"[GMD histogram] Reading {map_path}…")
                with jobs.preview_read_access([map_path]) as stopping:
                    gmd_values, weights, mean_gmd = await asyncio.to_thread(
                        gmd_histogram_samples, map_path, column, mode, power)
            except (OSError, ValueError, KeyError, ResourceBusyError) as exc:
                view.log.push(f"[GMD histogram] Error: {exc}")
                ui.notify(str(exc), type="negative")
                return
            finally:
                button.props(remove="loading")
                button.enable()

            area.clear()
            with area:
                if kind == "gmd":
                    caption = (f"{len(gmd_values):,} mapped frames · mean GMD "
                               f"{mean_gmd:.3e} · map column {column}")
                    figure = frame_histogram_figure(
                        gmd_values, title=f"GMD value distribution · column {column}",
                        x_label=f"GMD value (map column {column})",
                        y_label="Frames", count_label="frames",
                        auto_bins=True, **limits)
                else:
                    finite = int(np.isfinite(weights).sum())
                    caption = (f"{len(weights):,} mapped frames · mode {mode} · "
                               f"{len(weights) - finite:,} non-finite values omitted")
                    figure = frame_histogram_figure(
                        weights, title=f"GMD weight distribution · {mode}",
                        x_label=f"GMD weight ({mode})",
                        y_label="Frames", count_label="frames",
                        auto_bins=True, **limits)
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

        def clear():
            if not output.value:
                ui.notify("Set an output directory first.", type="warning")
                return
            directory = canonical_path(output.value, CENTROID_DIR)
            if weight_power.value is None:
                ui.notify("Enter a weight power first.", type="warning")
                return
            suffix = _suffix(weight_mode.value, float(weight_power.value))
            confirm_clear_outputs(_output_files(directory, suffix), "Delete Binning outputs?",
                                  on_cleared=view.results.clear)

        with ui.row().classes("gap-2"):
            view.run_button(on_click=run)
            ui.button("Clear outputs", icon="delete", on_click=clear).props("flat color=negative")

    view.place_console()


def _suffix(mode: str, power: float) -> str:
    if mode == "none":
        return ""
    if mode == "gmd_norm" and power != 1.0:
        return f"_w-gmd_norm_p{power}"
    return f"_w-{mode}"


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
