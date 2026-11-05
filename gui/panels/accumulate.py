"""Panel: h5_accumulate_fast_gmd.py — fast accumulation with live preview."""

import os
import sys
import time
from pathlib import Path

from nicegui import ui

from ..jobs import Job, JobManager
from .. import theme
from .common import (GENERATED, JobView, REPO_ROOT, file_url, num_input,
                     path_input, render_npy)


def build(jobs: JobManager) -> None:
    theme.section_title(
        "Fast Accumulate",
        "Threshold + morphological denoise summed into one image, with "
        "optional GMD weighting (h5_accumulate_fast_gmd.py). "
        "Quick-look tool — the Picasso path is the precise one.")

    live_img = ui.image().classes("w-full max-w-2xl")
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
            live_img.set_source(f"{file_url(png)}?t={time.time()}")
            live_img.visible = True
        except Exception:
            pass  # file may be mid-write; next refresh picks it up

    view = JobView(jobs, results_builder=_results, live_builder=_live)

    with theme.panel_card():
        h5_dir = path_input("HDF5 directory", r"F:\1103\1103_12_ele",
                            browse="dir")
        output_path = path_input("Output image (.txt)",
                                 r"F:\1103\1103_12_ele\accumulated_result.txt",
                                 browse="save", filters=[".txt"],
                                 default_name="accumulated_result.txt")
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

        theme.run_button(on_click=run)

    view.place_console()


def _results(job: Job):
    out = job.extra.get("output_path", "")
    png = os.path.splitext(out)[0] + ".png"
    if job.status == "done" and Path(png).exists():
        ui.label(f"Result: {out}").classes("vmi-subtitle")
        ui.image(file_url(png)).classes("w-full max-w-2xl")
