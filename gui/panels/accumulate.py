"""Panel: h5_accumulate_fast_gmd.py — fast accumulation with live preview."""

import os
import sys
import time
from pathlib import Path

from nicegui import ui

from ..jobs import Job, JobManager
from .. import theme
from .common import (GENERATED, JobView, REPO_ROOT, file_url, num_input,
                     open_folder_button, path_input, render_npy,
                     zoomable_image)


def build(jobs: JobManager) -> None:
    theme.section_title(
        "Fast Accumulate",
        "Threshold + morphological denoise summed into one image, with "
        "optional GMD weighting (h5_accumulate_fast_gmd.py). "
        "Quick-look tool — the Picasso path is the precise one.")

    live_img = zoomable_image().classes("w-full max-w-2xl")
    live_img.visible = False
    live_state = {"stamp": 0.0}

    def _live(job: Job):
        npy = job.extra.get("live_path")
        if not npy or not Path(npy).exists():
            return
        stamp = os.path.getmtime(npy)
        if stamp <= live_state["stamp"]:
            return
        live_state["stamp"] = stamp
        png = GENERATED / f"accum_live_{job.id}.png"
        try:
            render_npy(npy, png)
            live_img.set_source(file_url(png))
            live_img.visible = True
        except Exception:
            pass  # file may be mid-write; next refresh picks it up

    view = JobView(jobs, results_builder=_results, live_builder=_live)

    with theme.panel_card():
        h5_dir = path_input("HDF5 directory", "", browse="dir",
                            placeholder="folder containing RAW-*.h5 files")
        output_path = path_input("Output image (.txt)", "",
                                 browse="save", filters=[".txt"],
                                 default_name="accumulated_result.txt",
                                 placeholder="accumulated_result.txt")
        with ui.row().classes("gap-4 flex-wrap"):
            im_threshold = num_input("Threshold", 5, step=1)
            max_workers = num_input("Workers", 6, step=1)
            do_denoise = ui.checkbox("Denoise (2×2 opening)", value=True)
            simple_sum = ui.checkbox("Simple sum (no threshold)", value=False)
        with ui.row().classes("gap-4 flex-wrap"):
            enable_gmd = ui.checkbox("GMD weighting", value=False)
            gmd_map_path = path_input("GMD map file",
                                      str(REPO_ROOT / "gmd_map_all.txt"),
                                      browse="file", filters=[".txt"])
            gmd_w_min = num_input("Weight min", 0.0, step=0.5)
            gmd_w_max = num_input("Weight max", 4.0, step=0.5)
        dark_path = path_input("Dark frame .npy (optional)", "",
                               browse="file", filters=[".npy"])

        def run():
            job_id_hint = str(time.time_ns())[-8:]
            live_path = GENERATED / f"accum_live_{job_id_hint}.npy"
            if live_path.exists():
                live_path.unlink()
            live_state["stamp"] = 0.0
            cmd = [sys.executable, str(REPO_ROOT / "h5_accumulate_fast_gmd.py"),
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
            theme.run_button(on_click=run)
            ui.button("Clear output", icon="delete",
                      on_click=lambda: _clear_output(output_path)) \
                .props("flat color=negative")

    view.place_console()


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
        failed = []
        for p in existing:
            try:
                p.unlink()
            except OSError as e:
                failed.append(f"{p}: {e}")
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
    out = job.extra.get("output_path", "")
    png = os.path.splitext(out)[0] + ".png"
    if job.status == "done" and Path(png).exists():
        ui.label(f"Result: {out}").classes("vmi-subtitle")
        open_folder_button(out)
        zoomable_image(file_url(png)).classes("w-full max-w-2xl")
