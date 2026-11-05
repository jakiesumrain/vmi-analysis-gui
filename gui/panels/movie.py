"""Panel: h5_play_frames_movie.py — headless movie render + frame scrubber."""

import os
import sys
from pathlib import Path

from nicegui import ui

from ..jobs import Job, JobManager, TERMINAL
from .. import theme
from .common import (GENERATED, JobView, REPO_ROOT, file_url, num_input,
                     open_folder_button, path_input, persist, serve_dir,
                     zoomable_image)


def build(jobs: JobManager) -> None:
    theme.section_title(
        "Frame Movie",
        "Render the first N frames as a movie and scrub them frame by frame "
        "(h5_play_frames_movie.py). Uncheck 'Save movie file' to watch the "
        "movie play live in this page instead of writing a file.")

    view = JobView(jobs, results_builder=_results)

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
            frames_dir = GENERATED / "movie_frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            for old in frames_dir.glob("frame_*.png"):
                old.unlink()  # stale frames would confuse the live view
            live_state["src"] = None
            live_label.visible = False
            live_img.visible = False
            cmd = [sys.executable, str(REPO_ROOT / "h5_play_frames_movie.py"),
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
                   "--save-movie" if save_movie.value else "--no-save-movie",
                   "--dump-frames", str(frames_dir)]
            if log_scale.value:
                cmd.append("--log-scale")
            if not save_movie.value:
                # embedded live playback: frames stream in at playback pace
                cmd.append("--stream-pace")
            name = os.path.basename(os.path.normpath(h5_dir.value))
            job = jobs.submit(Job(
                name="Frame movie", cmd=cmd, cwd=str(REPO_ROOT),
                extra={"frames_dir": str(frames_dir),
                       "movie_path": str(Path(movie_dir.value) / (name + movie_ext.value)),
                       "movie_path_gif": str(Path(movie_dir.value) / (name + ".gif"))}))
            live_state["job"] = None if save_movie.value else job
            view.attach(job)

        with ui.row().classes("gap-2"):
            theme.run_button(on_click=run)
            ui.button("Clear outputs", icon="delete",
                      on_click=lambda: _clear_outputs(
                          movie_dir, h5_dir)) \
                .props("flat color=negative")
        ui.label("Tip: with 'Save movie file' unchecked the movie plays "
                 "live below as the frames stream in (at the Interval "
                 "pace).").classes("vmi-subtitle")

        live_label = ui.label("").classes("vmi-caption")
        live_label.visible = False
        live_img = ui.image("").classes("w-full max-w-2xl")
        live_img.visible = False
        live_state = {"job": None, "src": None}

        def _live_tick():
            job = live_state["job"]
            if job is None:
                return
            frames_dir = Path(job.extra["frames_dir"])
            pngs = sorted(frames_dir.glob("frame_*.png"))
            if job.status in TERMINAL:
                live_state["job"] = None
            elif len(pngs) > 1:
                # one behind the newest: that file may still be mid-write
                pngs = pngs[:-1]
            if pngs:
                latest = pngs[-1]
                if str(latest) != live_state["src"]:
                    live_state["src"] = str(latest)
                    live_img.set_source(file_url(latest))
                    live_label.visible = True
                    live_img.visible = True
                live_label.text = (
                    f"Live playback: frame {int(latest.stem.split('_')[1]) + 1}")

        ui.timer(0.3, _live_tick)

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
    frame_pngs = (sorted(frames_dir.glob("frame_*.png"))
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
        failed = []
        for p in targets + frame_pngs:
            try:
                p.unlink()
            except OSError as e:
                failed.append(f"{p.name}: {e}")
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
    if job.status != "done":
        return
    movie = job.extra["movie_path"]
    if not Path(movie).exists() and Path(job.extra["movie_path_gif"]).exists():
        movie = job.extra["movie_path_gif"]  # ffmpeg fallback
    if Path(movie).exists():
        ui.label(f"Movie: {movie}").classes("vmi-subtitle")
        open_folder_button(movie)
        if movie.lower().endswith(".mp4"):
            ui.video(file_url(movie)).classes("w-full max-w-2xl")
        else:
            zoomable_image(file_url(movie)).classes("w-full max-w-2xl")
    frames = sorted(Path(job.extra["frames_dir"]).glob("frame_*.png"))
    if frames:
        ui.label(f"Scrub frames ({len(frames)}):").classes("vmi-caption")
        img = zoomable_image(file_url(frames[0])).classes("w-full max-w-2xl")

        def show(e):
            img.set_source(file_url(frames[int(e.value)]))

        ui.slider(min=0, max=len(frames) - 1, value=0, on_change=show) \
            .classes("w-full max-w-2xl")
