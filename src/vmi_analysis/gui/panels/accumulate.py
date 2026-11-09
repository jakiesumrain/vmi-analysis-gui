"""Panel: h5_accumulate_fast_gmd.py — fast accumulation with live preview."""

import asyncio
import os
import sys
import time
from pathlib import Path

import numpy as np
from nicegui import ui
from plotly.colors import get_colorscale

from ..accumulate_plot import (ACCUMULATE_HEIGHT, COLORMAPS,
                               FRAME_HISTOGRAM_HEIGHT, accumulation_figure,
                               frame_histogram_figure)
from ..centroid_plots import plot_options
from ..frame_preview import list_raw_frames, load_raw_frame, pick_frame
from ..jobs import Job, JobManager
from .. import theme
from .common import (GENERATED, JobView, REPO_ROOT, file_url, num_input,
                     open_folder_button, path_input, persist,
                     zoomable_image, delete_outputs)


def build(jobs: JobManager) -> None:
    theme.section_title(
        "Fast Accumulate",
        "Threshold + morphological denoise summed into one image, with "
        "optional GMD weighting (h5_accumulate_fast_gmd.py). "
        "Quick-look tool — the Picasso path is the precise one.")

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
        _results(job)

    view = JobView(jobs, job_name='Accumulate', results_builder=finished, live_builder=_live)

    with theme.panel_card():
        h5_dir = persist(path_input(
            "HDF5 directory", "", browse="dir",
            placeholder="folder containing RAW-*.h5 files"),
            "accumulate.h5_dir")
        output_path = persist(path_input(
            "Output image (.txt)", "", browse="save", filters=[".txt"],
            default_name="accumulated_result.txt",
            placeholder="accumulated_result.txt"), "accumulate.output_path")
        with ui.row().classes("gap-4 flex-wrap"):
            im_threshold = persist(num_input("Threshold", 5, step=1),
                                   "accumulate.im_threshold")
            max_workers = persist(num_input("Workers", 6, step=1),
                                  "accumulate.max_workers")
            do_denoise = persist(ui.checkbox("Denoise (2×2 opening)",
                                             value=True),
                                 "accumulate.do_denoise")
            simple_sum = persist(ui.checkbox("Simple sum (no threshold)",
                                             value=False),
                                 "accumulate.simple_sum")
        with ui.row().classes("gap-3 items-end flex-wrap"):
            frame_index = num_input("Frame index (0-based)",
                                    None).classes("w-56") \
                .props('placeholder="blank = random"')
            histogram_button = ui.button("Histogram frame", icon="bar_chart") \
                .props("flat color=primary")
        histogram_area = ui.column().classes("w-full gap-1")
        histogram_area.visible = False

        def hide_histogram():
            histogram_area.clear()
            histogram_area.visible = False

        async def show_histogram():
            folder = (h5_dir.value or "").strip()
            if not folder:
                ui.notify("Set the HDF5 directory first.", type="warning")
                return
            histogram_button.disable()
            histogram_button.props("loading")
            try:
                raw_index = frame_index.value
                index = None if raw_index is None else int(raw_index)
                refs = await asyncio.to_thread(list_raw_frames, folder)
                ref = pick_frame(refs, index)
                values = await asyncio.to_thread(load_raw_frame, ref)
                figure = await asyncio.to_thread(
                    frame_histogram_figure, values,
                    title=f"Raw frame histogram · {ref.file_name} frame {ref.frame_index}")
            except (OSError, ValueError, KeyError, IndexError) as exc:
                ui.notify(str(exc), type="negative")
                return
            finally:
                histogram_button.props(remove="loading")
                histogram_button.enable()

            caption = (f"Frame {ref.global_index + 1} of {len(refs)} · "
                       f"{ref.file_name} frame {ref.frame_index} "
                       f"(0-based index {ref.global_index})")
            histogram_area.clear()
            with histogram_area:
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label(caption).classes("vmi-caption")
                    ui.button("Hide", icon="visibility_off",
                              on_click=hide_histogram) \
                        .props("flat dense color=secondary")
                ui.plotly(plot_options(figure)).classes("w-full") \
                    .style(f"height: {FRAME_HISTOGRAM_HEIGHT}px")
            histogram_area.visible = True

        histogram_button.on_click(show_histogram)
        with ui.row().classes("gap-4 flex-wrap"):
            enable_gmd = persist(ui.checkbox("GMD weighting", value=False),
                                 "accumulate.enable_gmd")
            gmd_map_path = persist(path_input(
                "GMD map file", str(REPO_ROOT / "gmd_map_all.txt"),
                browse="file", filters=[".txt"]), "accumulate.gmd_map_path")
            gmd_w_min = persist(num_input("Weight min", 0.0, step=0.5),
                                "accumulate.gmd_w_min")
            gmd_w_max = persist(num_input("Weight max", 4.0, step=0.5),
                                "accumulate.gmd_w_max")
        dark_path = persist(path_input("Dark frame .npy (optional)", "",
                                       browse="file", filters=[".npy"]),
                            "accumulate.dark_path")

        def run():
            job_id_hint = str(time.time_ns())[-8:]
            live_path = GENERATED / f"accum_live_{job_id_hint}.npy"
            if live_path.exists():
                live_path.unlink()
            live_state["stamp"] = 0.0
            live_area.visible = False
            cmd = [sys.executable, "-m", "vmi_analysis.imaging.h5_accumulate_fast_gmd",
                   "--h5-dir", h5_dir.value,
                   "--output-path", output_path.value,
                   "--im-threshold", str(im_threshold.value),
                   "--max-workers", str(int(max_workers.value or 1)),
                   "--gmd-map-path", gmd_map_path.value,
                   "--gmd-w-min", str(gmd_w_min.value),
                   "--gmd-w-max", str(gmd_w_max.value),
                   "--dark-path", dark_path.value,
                   "--no-show",
                   "--live-path", str(live_path)]
            if do_denoise.value:
                cmd.append("--do-denoise")
            else:
                cmd.append("--no-do-denoise")
            if simple_sum.value:
                cmd.append("--simple-sum")
            if enable_gmd.value:
                cmd.append("--enable-gmd")
            job = jobs.submit(Job(name="Accumulate", cmd=cmd, cwd=str(REPO_ROOT),
                                  extra={"live_path": str(live_path),
                                         "output_path": output_path.value}))
            view.attach(job)

        with ui.row().classes("gap-2 items-center"):
            view.run_button(on_click=run)
            def show_saved():
                if view.job and view.job.status in ("running", "queued"):
                    ui.notify("Wait for the running job to finish.", type="warning")
                    return
                view.results.clear()
                with view.results:
                    _result_file((output_path.value or "").strip())
            ui.button("View saved result", icon="analytics", on_click=show_saved).props("flat")
            ui.button("Clear output", icon="delete",
                      on_click=lambda: _clear_output(output_path)) \
                .props("flat color=negative")

    view.place_console()
    live_area.move()


def _clear_output(output_path) -> None:
    """Delete the accumulated .txt (and its .png) so the next run recomputes.

    The script skips accumulation whenever the output file exists
    ("[Resume] ... plotting directly"), so a stale result must be removed
    explicitly.
    """
    txt = (output_path.value or "").strip()
    if not txt:
        ui.notify("Set an output path first.", type="warning")
        return
    targets = [Path(txt), Path(os.path.splitext(txt)[0] + ".png")]
    existing = [p for p in targets if p.exists()]
    if not existing:
        ui.notify(f"Nothing to delete at {txt}", type="info")
        return

    def do_delete():
        failed = delete_outputs(existing)
        if failed is None:
            return
        dlg.close()
        if failed:
            ui.notify("Could not delete: " + "; ".join(failed),
                      type="negative")
        else:
            ui.notify(f"Deleted {len(existing)} file(s) — "
                      "next run recomputes from scratch.", type="positive")

    with ui.dialog() as dlg, ui.card().classes("gap-2"):
        ui.label("Delete the existing accumulated output?") \
            .classes("vmi-caption")
        for p in existing:
            ui.label(str(p)).classes("text-sm") \
                .style("word-break: break-all")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=dlg.close).props("flat")
            ui.button("Delete", on_click=do_delete) \
                .props("unelevated color=negative")
    dlg.open()


def _results(job: Job):
    if job.status == "done":
        _result_file(job.extra.get("output_path", ""))


def _interactive_plot(figure=None):
    colors = persist(ui.select(list(COLORMAPS), value="Viridis", label="Colormap")
                     .classes("w-48").props("outlined dense"), "accumulate.colormap")
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
    png = os.path.splitext(out)[0] + ".png"
    if Path(out).is_file():
        ui.label(f"Result: {out}").classes("vmi-subtitle")
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
            zoomable_image(file_url(png)).classes("w-full max-w-4xl")
