"""Panel: h5_play_frames_movie.py — headless movie render + frame scrubber."""

import os
import sys
from pathlib import Path

from nicegui import ui

from ..jobs import Job, JobManager
from .. import theme
from .common import (GENERATED, JobView, REPO_ROOT, file_url, num_input,
                     path_input, serve_dir)


def build(jobs: JobManager) -> None:
    theme.section_title(
        "Frame Movie",
        "Render the first N frames as a movie and scrub them frame by frame "
        "(h5_play_frames_movie.py, headless mode).")

    view = JobView(jobs, results_builder=_results)

    with theme.panel_card():
        h5_dir = path_input("HDF5 directory", r"F:\1103\1103_8_Aele")
        h5_dataset = path_input("Frame dataset", "raw/data")
        dark_path = path_input("Dark frame .npy (optional)", "")
        with ui.row().classes("gap-4 flex-wrap"):
            n_frames = num_input("Frames", 20, step=1)
            interval_ms = num_input("Interval (ms)", 500, step=50)
            scale_frames = num_input("Scale frames", 10, step=1)
        with ui.row().classes("gap-4 flex-wrap"):
            cmap = ui.input("Colormap", value="gray").props("outlined dense")
            log_scale = ui.checkbox("Log scale", value=False)
            vmin_frac = num_input("vmin fraction", 0.5, step=0.1)
            vmax_pct = num_input("vmax percentile", 99.9, step=0.1)
        with ui.row().classes("gap-4 flex-wrap"):
            movie_dir = path_input("Movie output folder",
                                   str(REPO_ROOT / "movies"))
            movie_ext = ui.select([".mp4", ".gif"], value=".mp4",
                                  label="Format").props("outlined dense")
            movie_fps = num_input("Saved fps (0 = playback)", 0, step=1)

        def run():
            frames_dir = GENERATED / "movie_frames"
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
                   "--headless", "--save-movie",
                   "--dump-frames", str(frames_dir)]
            if log_scale.value:
                cmd.append("--log-scale")
            name = os.path.basename(os.path.normpath(h5_dir.value))
            job = jobs.submit(Job(
                name="Frame movie", cmd=cmd, cwd=str(REPO_ROOT),
                extra={"frames_dir": str(frames_dir),
                       "movie_path": str(Path(movie_dir.value) / (name + movie_ext.value)),
                       "movie_path_gif": str(Path(movie_dir.value) / (name + ".gif"))}))
            view.attach(job)

        theme.run_button(on_click=run)


def _results(job: Job):
    if job.status != "done":
        return
    movie = job.extra["movie_path"]
    if not Path(movie).exists() and Path(job.extra["movie_path_gif"]).exists():
        movie = job.extra["movie_path_gif"]  # ffmpeg fallback
    if Path(movie).exists():
        ui.label(f"Movie: {movie}").classes("vmi-subtitle")
        if movie.lower().endswith(".mp4"):
            ui.video(file_url(movie)).classes("w-full max-w-2xl")
        else:
            ui.image(file_url(movie)).classes("w-full max-w-2xl")
    frames = sorted(Path(job.extra["frames_dir"]).glob("frame_*.png"))
    if frames:
        ui.label(f"Scrub frames ({len(frames)}):").classes("vmi-caption")
        img = ui.image(file_url(frames[0])).classes("w-full max-w-2xl")

        def show(e):
            img.set_source(f"{file_url(frames[int(e.value)])}?t={id(e.value)}")

        ui.slider(min=0, max=len(frames) - 1, value=0, on_change=show) \
            .classes("w-full max-w-2xl")
