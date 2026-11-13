"""Panel: h5_play_frames_movie.py — headless movie render + frame scrubber."""

import json
import os
import sys
import uuid
from pathlib import Path

from fastapi.responses import JSONResponse, Response
from nicegui import app, ui

from ..jobs import Job, JobManager, MANAGER, TERMINAL
from .. import theme
from .common import (GENERATED, JobView, REPO_ROOT, file_url, num_input,
                     open_folder_button, path_input, persist,
                     zoomable_image, delete_outputs)

# Fetch the newest frame without storing image payloads in the UI event history.
_PLAYER_JS = """
<script>
window.vmiMovie = null;

function vmiMovieStop() {
  window.vmiMovie = null;
}
function vmiMovieStart(url) {
  const p = {url, index: 0, paused: false};
  window.vmiMovie = p;
  const img = document.getElementById('vmi-live-img');
  if (img) img.removeAttribute('src');
  const ctr = document.getElementById('vmi-live-counter');
  if (ctr) ctr.textContent = '';
  vmiMovieStep(p);
}
async function vmiMovieStep(p) {
  if (p !== window.vmiMovie) return;
  try {
    if (!p.paused) {
      const response = await fetch(p.url + '?after=' + p.index, {cache: 'no-store'});
      if (p !== window.vmiMovie || p.paused) return;
      if (response.status === 410) vmiMovieStop();
      if (response.status === 200) {
        const frame = await response.json();
        if (p !== window.vmiMovie || p.paused) return;
        const img = document.getElementById('vmi-live-img');
        if (img) img.src = 'data:image/png;base64,' + frame.png;
        const ctr = document.getElementById('vmi-live-counter');
        if (ctr) ctr.textContent = 'frame ' + frame.index + ' / ' + frame.total;
        p.index = frame.index;
        if (p.index === frame.total) vmiMovieStop();
      }
    }
  } catch {
    // The next poll can recover from a transient network interruption.
  } finally {
    if (p === window.vmiMovie) setTimeout(() => vmiMovieStep(p), 50);
  }
}
</script>
"""


@app.get("/movie-preview/{job_id}")
async def _preview_frame(job_id: str, after: int = 0):
    headers = {"Cache-Control": "no-store"}
    job = MANAGER.get(job_id)
    if job is None:
        return Response(status_code=410, headers=headers)
    status = job.status
    if job.cancel_requested or status in {"failed", "cancelled"}:
        return Response(status_code=410, headers=headers)
    frame = job.extra.get("preview_frame")
    if frame is None or frame["index"] <= after:
        return Response(status_code=410 if status == "done" else 204,
                        headers=headers)
    return JSONResponse(frame, headers=headers)


def _consume_preview_frame(job: Job, line: str) -> bool:
    prefix = "[preview-png] "
    if not line.startswith(prefix):
        return False
    index, total, png = line[len(prefix):].split(" ", 2)
    job.extra["preview_frame"] = {"index": int(index), "total": int(total), "png": png}
    return True


def build(jobs: JobManager) -> None:
    theme.section_title(
        "Frame Movie",
        "Preview camera frames in order or save them as a movie.")

    view = JobView(jobs, job_name='Frame movie', results_builder=_results)
    ui.add_body_html(_PLAYER_JS, shared=True)

    with theme.panel_card():
        h5_dir = persist(path_input(
            "HDF5 directory", "", browse="dir",
            placeholder="folder containing RAW-*.h5 files"), "movie.h5_dir")
        h5_dataset = persist(path_input("Frame dataset", "raw/data"),
                             "movie.h5_dataset")
        dark_path = persist(path_input("Dark frame .npy (optional)", "",
                                       browse="file", filters=[".npy"]),
                            "movie.dark_path")
        with ui.row().classes("gap-4 flex-wrap"):
            n_frames = persist(num_input("Frames", 20, step=1),
                               "movie.n_frames")
            interval_ms = persist(num_input("Interval (ms)", 500, step=50),
                                  "movie.interval_ms")
            scale_frames = persist(num_input("Scale frames", 10, step=1),
                                   "movie.scale_frames")
        with ui.row().classes("gap-4 flex-wrap"):
            cmap = persist(ui.input("Colormap", value="gray")
                           .props("outlined dense"), "movie.cmap")
            log_scale = persist(ui.checkbox("Log scale", value=False),
                                "movie.log_scale")
            vmin_frac = persist(num_input("vmin fraction", 0.5, step=0.1),
                                "movie.vmin_frac")
            vmax_pct = persist(num_input("vmax percentile", 99.9, step=0.1),
                               "movie.vmax_pct")
        with ui.row().classes("gap-4 flex-wrap items-center"):
            save_movie = persist(ui.checkbox("Save movie file", value=True),
                                 "movie.save_movie")
            movie_dir = persist(path_input("Movie output folder",
                                           str(REPO_ROOT / "movies"),
                                           browse="dir"), "movie.movie_dir")
            movie_ext = persist(ui.select([".mp4", ".gif"], value=".mp4",
                                          label="Format")
                                .props("outlined dense"), "movie.movie_ext")
            movie_fps = persist(num_input("Saved fps (0 = playback)", 0,
                                          step=1), "movie.movie_fps")
        for w in (movie_dir, movie_ext, movie_fps):
            w.bind_enabled_from(save_movie, "value")

        def run():
            saving = bool(save_movie.value)
            live_label.visible = False
            live_row.visible = False
            live_img.visible = False
            cmd = [sys.executable, "-m", "vmi_analysis.imaging.h5_play_frames_movie",
                   "--h5-dir", h5_dir.value,
                   "--h5-dataset", h5_dataset.value,
                   "--n-frames", str(int(n_frames.value or 0)),
                   "--interval-ms", str(int(interval_ms.value or 500)),
                   "--cmap", cmap.value,
                   "--vmin-frac", str(vmin_frac.value),
                   "--vmax-pct", str(vmax_pct.value),
                   "--scale-frames", str(int(scale_frames.value or 10)),
                   "--dark-path", dark_path.value,
                   "--movie-dir", movie_dir.value,
                   "--movie-ext", movie_ext.value,
                   "--movie-fps", str(movie_fps.value or 0),
                   "--headless",
                   "--save-movie" if saving else "--no-save-movie"]
            if log_scale.value:
                cmd.append("--log-scale")
            for previous in jobs.jobs():
                if previous.name == "Frame movie" and previous.status in TERMINAL:
                    previous.extra.pop("preview_frame", None)
            job = Job(name="Frame movie", cmd=cmd, cwd=str(REPO_ROOT))
            if saving:
                frames_dir = GENERATED / "movie_frames" / uuid.uuid4().hex
                cmd.extend(["--dump-frames", str(frames_dir)])
                job.extra["frames_dir"] = str(frames_dir)
            else:
                cmd.append("--preview-stream")
                job.consume_output = _consume_preview_frame
            jobs.submit(job)
            live_state["job"] = None if saving else job
            if saving:
                ui.run_javascript("vmiMovieStop()")
            else:
                live_label.visible = True
                live_row.visible = True
                live_img.visible = True
                pause_btn.text = "Pause"
                pause_btn.enable()
                ui.run_javascript(
                    f"vmiMovieStart({json.dumps(f'movie-preview/{job.id}')})")
            view.attach(job)

        with ui.row().classes("gap-2"):
            view.run_button(on_click=run)
            ui.button("Clear outputs", icon="delete",
                      on_click=lambda: _clear_outputs(
                          movie_dir, h5_dir)) \
                .props("flat color=negative")
        ui.label("Turn off Save movie file to preview frames without saving files.").classes("vmi-subtitle")

        live_label = ui.label("Live playback").classes("vmi-caption")
        live_label.visible = False
        with ui.row().classes("items-center gap-4") as live_row:
            live_row.visible = False
            pause_btn = ui.button("Pause", on_click=lambda: _toggle_pause()) \
                .props("flat dense color=primary")
            ui.label("").props("id=vmi-live-counter").classes("vmi-subtitle")
        # a plain <img>, not ui.image: the JS player drives src directly,
        # and Quasar's QImg wrapper div would swallow the updates
        live_img = ui.element("img").props("id=vmi-live-img") \
            .classes("w-full max-w-2xl")
        live_img.visible = False
        live_state = {"job": None}

        def _toggle_pause():
            if live_state["job"] is None:
                return
            ui.run_javascript(
                "if (window.vmiMovie) window.vmiMovie.paused = !window.vmiMovie.paused")
            pause_btn.text = ("Resume" if pause_btn.text == "Pause"
                              else "Pause")

        def _live_tick():
            job = live_state["job"]
            if job is None:
                return
            if job.cancel_requested or job.status in TERMINAL:
                live_state["job"] = None
                was_paused = pause_btn.text == "Resume"
                pause_btn.disable()
                if job.cancel_requested or job.status != "done" or was_paused:
                    ui.run_javascript("vmiMovieStop()")
                return
        ui.timer(0.1, _live_tick)

    view.place_console()


def _clear_outputs(movie_dir, h5_dir) -> None:
    """Delete the saved movie file(s) and the dumped scrubber frames."""
    name = (os.path.basename(os.path.normpath(h5_dir.value))
            if h5_dir.value else "")
    targets = []
    if name and movie_dir.value:
        for ext in (".mp4", ".gif"):
            p = Path(movie_dir.value) / (name + ext)
            if p.exists():
                targets.append(p)
    frames_dir = GENERATED / "movie_frames"
    frame_pngs = (sorted(frames_dir.rglob("frame_*.png"))
                  if frames_dir.exists() else [])
    if not targets and not frame_pngs:
        ui.notify("Nothing to delete.", type="info")
        return

    with ui.dialog() as dlg, ui.card():
        ui.label("Delete movie outputs?").classes("vmi-caption")
        for p in targets:
            ui.label(str(p)).style("font-family: monospace")
        if frame_pngs:
            ui.label(f"{len(frame_pngs)} dumped frame PNGs in {frames_dir}") \
                .style("font-family: monospace")
        with ui.row().classes("gap-2"):
            ui.button("Delete", color="negative", on_click=lambda: do_delete())
            ui.button("Cancel", on_click=dlg.close).props("flat")
    dlg.open()

    def do_delete():
        failed = delete_outputs(targets + frame_pngs)
        if failed is None:
            return
        if frame_pngs:
            try:
                frames_dir.rmdir()
            except OSError:
                pass
        dlg.close()
        if failed:
            ui.notify("Could not delete: " + "; ".join(failed),
                      type="negative")
        else:
            ui.notify(f"Deleted {len(targets) + len(frame_pngs)} file(s)",
                      type="positive")


def _results(job: Job):
    if job.status != "done" or "frames_dir" not in job.extra:
        return
    frames_dir = Path(job.extra["frames_dir"])
    result_path = frames_dir / "movie-path.txt"
    movie = result_path.read_text(encoding="utf-8") if result_path.is_file() else ""
    if movie and Path(movie).is_file():
        ui.label(f"Movie: {movie}").classes("vmi-subtitle")
        open_folder_button(movie)
        if movie.lower().endswith(".mp4"):
            ui.video(file_url(movie)).classes("w-full max-w-2xl")
        else:
            zoomable_image(file_url(movie)).classes("w-full max-w-2xl")
    frames = sorted(frames_dir.glob("frame_*.png"))
    if frames:
        ui.label(f"Scrub frames ({len(frames)}):").classes("vmi-caption")
        img = zoomable_image(file_url(frames[0])).classes("w-full max-w-2xl")

        def show(e):
            img.set_source(file_url(frames[int(e.value)]))

        ui.slider(min=0, max=len(frames) - 1, value=0, on_change=show) \
            .classes("w-full max-w-2xl")
