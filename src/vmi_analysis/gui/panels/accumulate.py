"""Panel: h5_accumulate_fast_gmd.py — fast accumulation with live preview."""

import asyncio
import os
import sys
import time
from functools import partial
from operator import not_
from pathlib import Path

import numpy as np

from vmi_analysis.paths import validate_distinct_output
from nicegui import ui
from plotly.colors import get_colorscale

from ..agent_panels import PanelOperation, boolean, number, resolve_path
from .. import shared
from ..shared import WorkspaceDenied, checked_path, generated_dir, workspace_root
from ..accumulate_plot import (ACCUMULATE_HEIGHT, COLORMAPS,
                               FRAME_HISTOGRAM_HEIGHT, accumulation_figure,
                               frame_histogram_figure)
from ..centroid_plots import RESULT_COLORMAP_DEFAULTS, plot_options
from ..frame_preview import (average_raw_frames, list_raw_frames, load_raw_frame,
                             pick_frame, subtract_dark)
from ..jobs import Job, JobManager, ResourceBusyError
from .. import theme
from .common import (GENERATED, JobView, REPO_ROOT, action_bar, adopt_agent_form, check_grid,
                     field_grid, file_url, num_input, open_folder_button,
                     path_input, persist, zoomable_image,
                     confirm_clear_outputs)


JOB_NAME = "Accumulate"
PANEL_TITLE = "Fast Accumulate"
DEFAULTS = {
    "im_threshold": 5, "do_denoise": True, "simple_sum": False, "max_workers": 6, "frame_limit": 0,
    "enable_gmd": False, "gmd_w_min": 0.0, "gmd_w_max": 4.0,
}


def _whole_number(value, minimum, label):
    if ((type(value) is not int and (type(value) is not float or not value.is_integer()))
            or value < minimum):
        raise ValueError(f"{label} must be a whole number of {minimum} or more.")
    return int(value)


def accumulate_job(h5_dir, output_path, *, dark_path=None, gmd_map_path=None, reuse_existing=True, **settings):
    cwd = workspace_root(REPO_ROOT)
    options = DEFAULTS | settings
    for key in ("simple_sum", "do_denoise", "enable_gmd"):
        options[key] = boolean(options[key])
    frame_limit = _whole_number(0 if options["frame_limit"] is None else options["frame_limit"], 0, "Frame limit")
    workers = _whole_number(1 if options["max_workers"] is None else options["max_workers"], 1, "Workers")
    if not h5_dir or not str(h5_dir).strip():
        raise ValueError("Choose an HDF5 directory.")
    if not output_path or not str(output_path).strip():
        raise ValueError("Choose an output image file.")
    source = checked_path(h5_dir, cwd=cwd)
    output = checked_path(output_path, write=True, cwd=cwd)
    png = output.with_suffix(".png")
    if output == png:
        raise ValueError("Choose an output path for the image text file, such as accumulated_result.txt.")
    checked_path(png, write=True, cwd=cwd)
    if not reuse_existing and (os.path.lexists(output) or os.path.lexists(png)):
        raise ValueError("The output image or PNG already exists. Choose a new output path or request reuse.")
    if not output.is_file() and not source.is_dir():
        raise ValueError("Choose an existing HDF5 directory.")
    dark = checked_path(dark_path, cwd=cwd) if dark_path else None
    requested_map = gmd_map_path if gmd_map_path is not None else cwd / "gmd_map_all.txt"
    gmd = checked_path(requested_map, cwd=cwd) if requested_map and options["enable_gmd"] else None
    if not output.is_file():
        if dark is not None and not dark.is_file():
            raise ValueError("Choose an existing dark frame .npy file or leave the field blank.")
        if options["enable_gmd"] and (gmd is None or not gmd.is_file()):
            raise ValueError("Choose an existing GMD map for GMD weighting.")
    inputs = (("HDF5 directory", source), ("dark frame", dark), ("GMD map", gmd))
    validate_distinct_output(output, (*inputs, ("PNG output", png)), output_label="accumulated image output")
    validate_distinct_output(png, inputs, output_label="accumulation PNG output")
    if options["enable_gmd"]:
        low, high = map(number, (options["gmd_w_min"], options["gmd_w_max"]))
        if min(low, high) < 0 or (high and low > high):
            raise ValueError("Enter nonnegative GMD weight limits in increasing order. Zero disables a limit.")
    if not options["simple_sum"]:
        number(options["im_threshold"])

    job = Job(JOB_NAME, cwd=str(cwd))
    live = generated_dir(GENERATED) / f"accum_live_{job.id}.npy"
    job.cmd = [sys.executable, "-m", "vmi_analysis.imaging.h5_accumulate_fast_gmd",
               "--h5-dir", str(source), "--output-path", str(output), "--max-workers", str(workers),
               "--frame-limit", str(frame_limit), "--no-show", "--live-path", str(live)]
    if not reuse_existing:
        job.cmd.append("--require-new-output")
    if dark is not None:
        job.cmd.extend(["--dark-path", str(dark)])
    if options["simple_sum"]:
        job.cmd.append("--simple-sum")
    else:
        job.cmd.extend(["--im-threshold", str(options["im_threshold"]),
                        "--do-denoise" if options["do_denoise"] else "--no-do-denoise"])
    if options["enable_gmd"]:
        job.cmd.extend(["--enable-gmd", "--gmd-map-path", str(gmd) if gmd is not None else "",
                        "--gmd-w-min", str(low), "--gmd-w-max", str(high)])
    job.extra.update(output_path=str(output), plot_png=str(png), live_path=str(live),
                     panel_form=dict(options, h5_dir=str(source), output_path=str(output),
                                     frame_limit=frame_limit, max_workers=workers,
                                     dark_path=str(dark) if dark is not None else "",
                                     gmd_map_path=str(gmd or requested_map)))
    return job


def _prepare_agent(_jobs, params):
    inputs = dict(params)
    reuse = boolean(inputs.pop("reuse_existing", False))
    inputs["h5_dir"] = resolve_path(inputs.get("h5_dir"))
    inputs["output_path"] = resolve_path(inputs.get("output_path"), write=True)
    for key in ("dark_path", "gmd_map_path"):
        if inputs.get(key):
            inputs[key] = resolve_path(inputs[key])
        elif key in inputs and inputs[key] not in ("", None):
            raise ValueError("Enter a path or leave the optional input blank.")
    job = accumulate_job(**inputs, reuse_existing=reuse)
    form = job.extra["panel_form"]
    summary = {
        "HDF5 directory": form["h5_dir"],
        "Existing image policy": "Reuse the saved matrix if present. The calculation settings apply to new images."
                                 if reuse else "Calculate a new image.",
        "Frame limit": form["frame_limit"] or "All frames",
        "Workers": form["max_workers"],
        "Mode": "Simple sum" if form["simple_sum"] else "Thresholded accumulation",
        "Threshold": None if form["simple_sum"] else form["im_threshold"],
        "Denoise": None if form["simple_sum"] else form["do_denoise"],
        "Dark frame": form["dark_path"] or "Off",
        "GMD weighting": "On" if form["enable_gmd"] else "Off",
        "GMD map": form["gmd_map_path"] if form["enable_gmd"] else None,
        "GMD weight limits (zero = unbounded)": [form["gmd_w_min"], form["gmd_w_max"]] if form["enable_gmd"] else None,
    }
    return job, summary


def agent_operations():
    parameters = {
        "h5_dir": {"type": "path", "description": "Directory containing camera HDF5 files"},
        "output_path": {"type": "path", "description": "Text file for the accumulated matrix"},
        "simple_sum": {"type": "boolean", "description": "Skip thresholding and denoising"},
        "im_threshold": {"type": "number", "unit": "counts"},
        "do_denoise": {"type": "boolean", "description": "Apply a 2 × 2 morphological opening"},
        "max_workers": {"type": "integer", "description": "Maximum number of subprocess workers"},
        "frame_limit": {"type": "integer", "description": "Zero uses all frames. Positive limits use file order."},
        "dark_path": {"type": "path", "description": "Optional precomputed dark frame (.npy)"},
        "enable_gmd": {"type": "boolean", "description": "Weight frames by mean GMD divided by frame GMD"},
        "gmd_map_path": {"type": "path", "description": "When weighting is enabled, defaults to gmd_map_all.txt in the workspace"},
        "gmd_w_min": {"type": "number", "description": "Minimum accepted weight. Zero disables this limit."},
        "gmd_w_max": {"type": "number", "description": "Maximum accepted weight. Zero disables this limit."},
        "reuse_existing": {"type": "boolean", "default": False,
                           "description": "Allow replotting a saved matrix with the same frame limit"},
    }
    for key, value in DEFAULTS.items():
        parameters[key]["default"] = value
    description = {
        "panel": PANEL_TITLE,
        "summary": "Sum camera frames into one image with optional thresholding, denoising, dark subtraction, "
                   "and GMD weighting.",
        "parameters": parameters,
        "manual": PANEL_TITLE,
    }
    return (PanelOperation("accumulate", description, _prepare_agent, (JOB_NAME,),
                           outputs=("output_path", "plot_png")),)


def build(jobs: JobManager) -> None:
    with ui.column().classes("w-full") as live_area:
        live_img, live_colors = _interactive_plot()
    live_area.visible = False
    live_state = {"stamp": 0.0}

    def _live(job: Job):
        npy = job.extra.get("live_path")
        if not npy or not Path(npy).exists():
            return
        stamp = os.path.getmtime(npy)
        if stamp <= live_state["stamp"]:
            return
        try:
            figure = accumulation_figure(np.load(npy, allow_pickle=False), title="Accumulation in progress",
                                         revision=job.id, colormap=live_colors.value)
            live_img.update_figure(plot_options(figure))
            live_area.visible = True
            live_state["stamp"] = stamp
        except (OSError, ValueError):
            pass  # file may be mid-write; next refresh picks it up

    def finished(job):
        live_area.visible = False
        live_path = job.extra.get("live_path")
        if live_path:
            try:
                Path(live_path).unlink(missing_ok=True)
            except OSError as exc:
                view.log.push(f"[gui] Could not remove live preview: {exc}")
        _results(job)

    def restore_form(job):
        live_state["stamp"] = 0.0
        live_area.visible = False
        for key, value in adopt_agent_form(job, "accumulate").items():
            if key in fields:
                fields[key].value = value

    view = JobView(jobs, job_name=JOB_NAME, results_builder=finished, live_builder=_live,
                   attach_builder=restore_form)
    theme.panel_header(
        PANEL_TITLE,
        "Sum camera frames into one image with optional thresholding, denoising, and GMD weighting.",
        view=view)

    with theme.panel_card():
        h5_dir = persist(path_input(
            "HDF5 directory", "", browse="dir",
            placeholder="folder containing RAW-*.h5 files"),
            "accumulate.h5_dir")
        output_path = persist(path_input(
            "Output image (.txt)", "", browse="save", filters=[".txt"],
            default_name="accumulated_result.txt",
            placeholder="accumulated_result.txt"), "accumulate.output_path")
        with field_grid(3):
            im_threshold = persist(num_input("Threshold", DEFAULTS["im_threshold"], step=1),
                                   "accumulate.im_threshold")
            max_workers = persist(num_input("Workers", DEFAULTS["max_workers"], step=1),
                                  "accumulate.max_workers")
            frame_limit = persist(num_input("Frame limit (0 = all)", DEFAULTS["frame_limit"], min=0, step=1),
                                  "accumulate.frame_limit")
        with check_grid():
            do_denoise = persist(ui.checkbox("Denoise (2×2 opening)",
                                             value=DEFAULTS["do_denoise"]),
                                 "accumulate.do_denoise")
            simple_sum = persist(ui.checkbox("Simple sum (no threshold)",
                                             value=DEFAULTS["simple_sum"]),
                                 "accumulate.simple_sum")
            im_threshold.bind_enabled_from(simple_sum, "value", not_)
            do_denoise.bind_enabled_from(simple_sum, "value", not_)
        dark_path = persist(path_input("Dark frame .npy (optional)", "",
                                       browse="file", filters=[".npy"]),
                            "accumulate.dark_path")
        with ui.expansion("Histogram previews", icon="bar_chart", value=False).classes("w-full").props(
            'header-class="vmi-section-toggle text-primary" expand-icon-class="text-primary"'
        ):
            histogram_subtract_dark = persist(ui.checkbox("Subtract dark frame in previews", value=True),
                                              "accumulate.histogram_subtract_dark")
            with ui.row().classes("gap-3 items-end flex-wrap"):
                histogram_x_min = persist(
                    num_input("Histogram X min", None, step="any").classes("w-36")
                    .props('placeholder="Auto"'), "accumulate.histogram_x_min")
                histogram_x_max = persist(
                    num_input("Histogram X max", None, step="any").classes("w-36")
                    .props('placeholder="Auto"'), "accumulate.histogram_x_max")
                histogram_bin_width = persist(
                    num_input("Bin width (counts)", None, step="any").classes("w-36")
                    .props('placeholder="Auto"'), "accumulate.histogram_bin_width")
            with ui.row().classes("gap-3 items-end flex-wrap"):
                frame_index = persist(num_input("Frame index (0-based)", None).classes("w-56")
                                      .props('placeholder="blank = random"'), "accumulate.frame_index")
                frame_button = ui.button("Histogram frame", icon="bar_chart").props("flat color=primary")
                file_count = persist(num_input("Files to average", 1, step=1).classes("w-36"),
                                     "accumulate.histogram_file_count")
                average_button = ui.button("Histogram average", icon="bar_chart").props("flat color=primary")
            frame_area = ui.column().classes("w-full gap-1")
            average_area = ui.column().classes("w-full gap-1")
            frame_area.visible = average_area.visible = False

        def hide_histogram(area):
            area.clear()
            area.visible = False

        def histogram_data(folder, dark, selection, average, limits):
            paths = [folder, dark] if dark is not None else [folder]
            with jobs.preview_read_access(paths) as stopping:
                if average:
                    values, count = average_raw_frames(folder, selection, stopping=stopping)
                    caption = f"{count:,} frames from {selection:,} distinct HDF5 files"
                    title = "Average frame histogram"
                    x_label = "Mean pixel value per frame (counts)"
                else:
                    refs = list_raw_frames(folder)
                    ref = pick_frame(refs, selection)
                    values = load_raw_frame(ref)
                    caption = (f"Frame {ref.global_index + 1} of {len(refs)} · "
                               f"{ref.file_name} frame {ref.frame_index} "
                               f"(0-based index {ref.global_index})")
                    kind = "Raw frame" if dark is None else "Frame"
                    title = f"{kind} histogram · {ref.file_name} frame {ref.frame_index}"
                    x_label = "Raw pixel value (counts)"
                if dark is not None:
                    values = subtract_dark(values, dark)
                    caption += f" · dark-subtracted ({dark.name})"
                    x_label = ("Dark-subtracted mean pixel value per frame (counts)" if average
                               else "Dark-subtracted pixel value (counts)")
            return frame_histogram_figure(values, title=title, x_label=x_label, **limits), caption

        async def show_histogram(average):
            preview = "Histogram average" if average else "Histogram frame"
            folder = (h5_dir.value or "").strip()
            if not folder:
                message = "Set the HDF5 directory first."
                view.log.push(f"[{preview}] {message}")
                ui.notify(message, type="warning")
                return
            area, button = (average_area, average_button) if average else (frame_area, frame_button)
            button.disable()
            button.props("loading")
            started = time.monotonic()
            try:
                selection = file_count.value if average else frame_index.value
                if average and (selection is None or selection < 1 or not float(selection).is_integer()):
                    raise ValueError("Files to average must be a positive whole number.")
                if not average and selection is not None and not float(selection).is_integer():
                    raise ValueError("Frame index must be a whole number.")
                selection = None if selection is None else int(selection)
                if average:
                    source = f"{selection:,} distinct HDF5 files"
                else:
                    source = "a random frame" if selection is None else f"frame index {selection:,}"
                view.log.push(f"[{preview}] Reading {source} from {folder}…")
                dark = (dark_path.value or "").strip() if histogram_subtract_dark.value else ""
                limits = dict(x_min=histogram_x_min.value, x_max=histogram_x_max.value,
                              bin_width=histogram_bin_width.value)
                figure, caption = await asyncio.to_thread(
                    histogram_data, workspace_root(REPO_ROOT) / Path(folder).expanduser(),
                    workspace_root(REPO_ROOT) / Path(dark).expanduser() if dark else None,
                    selection, average, limits)
            except (OSError, ValueError, KeyError, IndexError, ResourceBusyError) as exc:
                view.log.push(f"[{preview}] Error: {exc}")
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
            view.log.push(f"[{preview}] Plotted in {time.monotonic() - started:.1f}s · {caption}")

        frame_button.on_click(partial(show_histogram, False))
        average_button.on_click(partial(show_histogram, True))
        with check_grid():
            enable_gmd = persist(ui.checkbox("GMD weighting", value=DEFAULTS["enable_gmd"]),
                                 "accumulate.enable_gmd")
        gmd_map_path = persist(path_input(
            "GMD map file", str(workspace_root(REPO_ROOT) / "gmd_map_all.txt"),
            browse="file", filters=[".txt"]), "accumulate.gmd_map_path")
        with field_grid():
            gmd_w_min = persist(num_input("Weight min", DEFAULTS["gmd_w_min"], step=0.5),
                                "accumulate.gmd_w_min")
            gmd_w_max = persist(num_input("Weight max", DEFAULTS["gmd_w_max"], step=0.5),
                                "accumulate.gmd_w_max")

        fields = dict(h5_dir=h5_dir, output_path=output_path, im_threshold=im_threshold,
                      max_workers=max_workers, frame_limit=frame_limit, do_denoise=do_denoise,
                      simple_sum=simple_sum, dark_path=dark_path, enable_gmd=enable_gmd,
                      gmd_map_path=gmd_map_path, gmd_w_min=gmd_w_min, gmd_w_max=gmd_w_max)

        def run():
            try:
                job = accumulate_job(**{key: widget.value for key, widget in fields.items()})
                view.attach(jobs.submit(job))
            except (OSError, ValueError, OverflowError) as exc:
                ui.notify(str(exc), type="warning")

        def show_saved():
            if view.job and view.job.status in ("running", "queued"):
                ui.notify("Wait for the running job to finish.", type="warning")
                return
            view.results.clear()
            with view.results:
                _result_file(str(workspace_root(REPO_ROOT) / output_path.value) if output_path.value else "")
            view.open_console()
        with action_bar(view, on_click=run):
            ui.button("View saved result", icon="analytics",
                      on_click=show_saved).props("flat color=secondary")
            ui.button("Clear output", icon="delete",
                      on_click=lambda: _clear_output(output_path, view, (dark_path.value, gmd_map_path.value))) \
                .props("flat color=negative").classes("vmi-action-danger")

    view.place_console()
    live_area.move()


def _clear_output(output_path, view, inputs) -> None:
    """Delete the accumulated .txt (and its .png) so the next run recomputes.

    The script skips accumulation whenever the output file exists
    ("[Resume] ... plotting directly"), so a stale result must be removed
    explicitly.
    """
    txt = (output_path.value or "").strip()
    if not txt:
        ui.notify("Set an output path first.", type="warning")
        return
    target = workspace_root(REPO_ROOT) / txt
    targets = [target, target.with_suffix(".png")]
    try:
        for path in targets:
            validate_distinct_output(path, zip(("dark frame", "GMD map"), inputs), cwd=workspace_root(REPO_ROOT))
    except (OSError, ValueError) as exc:
        ui.notify(str(exc), type="warning")
        return
    for job in view.jobs.jobs():
        if (job.name == JOB_NAME and job.extra.get("live_path")
                and (Path(job.cwd or workspace_root(REPO_ROOT)) / job.extra["output_path"]).resolve() == target.resolve()):
            targets.append(Path(job.extra["live_path"]))
    confirm_clear_outputs(targets, "Delete accumulated outputs?", on_cleared=view.clear_results)


def _results(job: Job):
    if job.status == "done":
        output = job.extra.get("output_path")
        if output:
            _result_file(str(Path(job.cwd or workspace_root(REPO_ROOT)) / output))


def _interactive_plot(figure=None):
    key = "accumulate.colormap"
    colors = persist(ui.select(list(COLORMAPS), value=RESULT_COLORMAP_DEFAULTS[key], label="Colormap")
                     .classes("w-48").props("outlined dense"), key)
    if figure is not None:
        figure.update_traces(colorscale=colors.value)
    plot = ui.plotly(plot_options(figure) if figure is not None else {"data": [], "layout": {}}) \
        .classes("w-full").style(f"height: {ACCUMULATE_HEIGHT}px")

    def recolor(event):
        if plot.figure["data"]:
            scale = get_colorscale(event.value)
            plot.figure["data"][0]["colorscale"] = scale
            # Only send the color change; keep the current zoom and full image data.
            # Python knows more named scales than plotly.js. Pass explicit stops,
            # wrapped once because restyle distributes array values over traces.
            plot.run_plot_method("restyle", {"colorscale": [scale]}, [0])

    colors.on_value_change(recolor)
    return plot, colors


def _result_file(out: str):
    if not out:
        ui.label("Set an output path to view a saved result.").classes("vmi-subtitle")
        return
    try:
        png = checked_path(os.path.splitext(out)[0] + ".png")
        out = checked_path(out)
    except WorkspaceDenied as exc:
        ui.label(str(exc)).classes("text-negative")
        return
    if Path(out).is_file():
        ui.label(f"Result: {out}").classes("vmi-subtitle")
        with ui.row().classes("items-center gap-2"):
            matrix_url = file_url(out)
            ui.button("Download image matrix", icon="download",
                      on_click=lambda: ui.download.from_url(matrix_url, filename=out.name)).props("flat")
            if not shared.SERVER:
                open_folder_button(out)
        try:
            figure = accumulation_figure(np.loadtxt(out))
            ui.label("Drag to zoom; wheel to zoom; use Pan to move; double-click to reset. Hover for original counts.") \
                .classes("vmi-subtitle")
            _interactive_plot(figure)
        except (OSError, ValueError) as exc:
            ui.label(f"Interactive data could not be loaded: {exc}").classes("text-negative")
    elif not Path(png).is_file():
        ui.label("No saved Accumulate result found at this path.").classes("vmi-subtitle")
    if Path(png).is_file():
        with ui.expansion("Saved PNG", icon="image").classes("w-full"):
            url = file_url(png)
            ui.button("Download PNG", icon="download",
                      on_click=lambda: ui.download.from_url(url, filename=png.name)).props("flat")
            zoomable_image(url).classes("w-full max-w-4xl")
