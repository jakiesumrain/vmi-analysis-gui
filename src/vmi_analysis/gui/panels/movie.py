"""Panel: h5_play_frames_movie.py — headless movie render + frame scrubber."""

import json
import os
import sys
import re
from pathlib import Path

from fastapi.responses import JSONResponse, Response
from nicegui import app, ui

from ..agent_panels import PanelOperation, boolean, number, resolve_path
from ..shared import checked_path, generated_dir, workspace_root, workspace_timer
from ..jobs import Job, JobManager, MANAGER, TERMINAL
from ..resources import canonical_path
from .. import shared, theme
from .common import (GENERATED, JobView, REPO_ROOT, action_bar, adopt_agent_form, check_grid,
                     field_grid, file_url, num_input, open_folder_button,
                     path_input, persist, zoomable_image, delete_outputs)

JOB_NAME = "Frame movie"
PANEL_TITLE = "Frame Movie"
DEFAULTS = {"h5_dataset": "raw/data", "n_frames": 20, "interval_ms": 500, "scale_frames": 10,
            "cmap": "gray", "vmin_frac": 0.5, "vmax_pct": 99.9, "log_scale": False,
            "save_movie": True, "movie_ext": ".mp4", "movie_fps": 0}


# Fetch the newest frame without storing image payloads in the UI event history.
_PLAYER_JS = """
<script>
window.vmiMovie = null;

function vmiMovieStop() {
  window.vmiMovie?.observer?.disconnect();
  window.vmiMovie = null;
}
function vmiMovieStart(url) {
  vmiMovieStop();
  const p = {url, index: 0, paused: false};
  window.vmiMovie = p;
  const img = document.getElementById('vmi-live-img');
  if (img) img.replaceChildren();
  const ctr = document.getElementById('vmi-live-counter');
  if (ctr) ctr.textContent = '';
  vmiMovieStep(p);
}
function vmiMovieDraw(svg, frame) {
  const {width, height} = frame;
  if (svg.dataset.shape !== width + 'x' + height || !svg.firstChild) {
    svg.dataset.shape = width + 'x' + height;
    svg.replaceChildren();
    const scale = 570 / Math.max(width, height);
    const w = width * scale, h = height * scale;
    const left = 65, top = 14;
    svg.setAttribute('viewBox', `0 0 ${left + w + 20} ${top + h + 54}`);
    const add = (tag, attrs, label) => {
      const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
      for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
      if (label !== undefined) el.textContent = label;
      svg.append(el);
    };
    const line = (x1, y1, x2, y2) =>
      add('line', {x1, y1, x2, y2, stroke: 'currentColor'});
    const text = (x, y, label, anchor = 'middle', transform = '') =>
      add('text', {x, y, 'text-anchor': anchor, transform, fill: 'currentColor', 'font-size': 13}, label);
    add('image', {x: left, y: top, width: w, height: h});
    line(left, top, left, top + h);
    line(left, top + h, left + w, top + h);
    const ticks = size => [...new Set(Array.from({length: 5},
      (_, i) => Math.round(i * (size - 1) / 4)))];
    for (const tick of ticks(width)) {
      const x = left + (tick + .5) * scale;
      line(x, top + h, x, top + h + 5);
      text(x, top + h + 20, tick);
    }
    for (const tick of ticks(height)) {
      const y = top + (tick + .5) * scale;
      line(left - 5, y, left, y);
      text(left - 9, y + 4, tick, 'end');
    }
    text(left + w / 2, top + h + 41, 'X (pixels)');
    text(17, top + h / 2, 'Y (pixels)', 'middle', `rotate(-90 17 ${top + h / 2})`);
  }
  svg.firstChild.setAttribute('href', 'data:image/png;base64,' + frame.png);
}
async function vmiMovieStep(p) {
  if (p !== window.vmiMovie) return;
  try {
    const img = document.getElementById('vmi-live-img');
    const ctr = document.getElementById('vmi-live-counter');
    if (!p.paused && img?.getClientRects().length && ctr) {
      const response = await fetch(p.url + '?after=' + p.index, {cache: 'no-store'});
      if (p !== window.vmiMovie || p.paused) return;
      if (response.status === 410) vmiMovieStop();
      if (response.status === 200) {
        const frame = await response.json();
        if (p !== window.vmiMovie || p.paused || !img.getClientRects().length) return;
        const label = 'frame ' + frame.index + ' / ' + frame.total;
        vmiMovieDraw(img, frame);
        ctr.textContent = label;
        p.index = frame.index;
        if (p.index === frame.total) {
          p.complete = true;
          // Quasar remounts the plot when Movie is selected again.
          p.observer = new MutationObserver(() => {
            const target = document.getElementById('vmi-live-img');
            const counter = document.getElementById('vmi-live-counter');
            if (target && !target.firstChild) vmiMovieDraw(target, frame);
            if (counter && counter.textContent !== label) counter.textContent = label;
          });
          p.observer.observe(document.body, {childList: true, subtree: true});
        }
      }
    }
  } catch {
    // The next poll can recover from a transient network interruption.
  } finally {
    if (p === window.vmiMovie && !p.complete) setTimeout(() => vmiMovieStep(p), 50);
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
    if line.startswith("Movie written: "):
        path = canonical_path(line.removeprefix("Movie written: "), job.cwd)
        if path in job.extra.get("movie_candidates", ()):
            job.extra["movie_file"] = str(path)
    elif match := re.match(r"  writing movie: frame (\d+)/(\d+) ", line):
        frame, total = map(int, match.groups())
        job.progress = frame / total
    if not line.startswith(prefix):
        return False
    index, total, width, height, png = line[len(prefix):].split(" ", 4)
    job.progress = int(index) / int(total)
    job.extra["preview_frame"] = {"index": int(index), "total": int(total),
                                  "width": int(width), "height": int(height), "png": png}
    return True


def movie_job(h5_dir, *, dark_path="", movie_dir=None, **settings):
    cwd = workspace_root(REPO_ROOT)
    values = DEFAULTS | settings
    for key in ("log_scale", "save_movie"):
        boolean(values[key])
    source = checked_path(h5_dir, cwd=cwd) if h5_dir and str(h5_dir).strip() else None
    if source is None or not source.is_dir():
        raise ValueError("Choose an existing HDF5 directory.")
    dark = checked_path(dark_path, cwd=cwd) if dark_path else None
    if dark is not None and (not dark.is_file() or dark.suffix.lower() != ".npy"):
        raise ValueError("Choose an existing dark frame .npy file or leave it blank.")
    for key in ("h5_dataset", "cmap"):
        value = values[key]
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValueError(f"Enter a valid {key.replace('_', ' ')}.")
    for key in ("n_frames", "interval_ms", "scale_frames"):
        value = values[key]
        if ((type(value) is not int and (type(value) is not float or not value.is_integer())) or value < 1):
            raise ValueError(f"{key.replace('_', ' ').capitalize()} must be a positive whole number.")
        values[key] = int(value)
    floor, percentile = map(number, (values["vmin_frac"], values["vmax_pct"]))
    if not 0 <= floor < 1 or (values["log_scale"] and floor == 0) or not 0 < percentile <= 100:
        raise ValueError("Use a vmin fraction below one and a vmax percentile above zero up to 100. "
                         "Log scale requires a positive vmin fraction.")

    default_dir = cwd / "movies"
    folder = movie_dir if movie_dir is not None else default_dir
    job = Job(JOB_NAME, cwd=str(cwd), consume_output=_consume_preview_frame)
    cmd = [sys.executable, "-m", "vmi_analysis.imaging.h5_play_frames_movie", "--h5-dir", str(source),
           "--headless", "--save-movie" if values["save_movie"] else "--no-save-movie"]
    for key in ("h5_dataset", "n_frames", "interval_ms", "scale_frames", "cmap", "vmin_frac", "vmax_pct"):
        cmd.append(f"--{key.replace('_', '-')}={values[key]}")
    if dark is not None:
        cmd += ["--dark-path", str(dark)]
    if values["log_scale"]:
        cmd.append("--log-scale")
    if values["save_movie"]:
        fps = number(0 if values["movie_fps"] is None else values["movie_fps"])
        if fps < 0 or values["movie_ext"] not in (".mp4", ".gif"):
            raise ValueError("Choose a supported movie format and a nonnegative frame rate.")
        folder = checked_path(folder or cwd, write=True, cwd=cwd)
        if folder.exists() and not folder.is_dir():
            raise ValueError("Choose a movie output folder.")
        extensions = (".mp4", ".gif") if values["movie_ext"] == ".mp4" else (".gif",)
        targets = [folder / (source.name + ext) for ext in extensions]
        for path in targets:
            checked_path(path, write=True, cwd=cwd)
            if path.is_dir():
                raise ValueError("Choose a movie output path that is a file.")
        frames = generated_dir(GENERATED) / "movie_frames" / job.id
        cmd += ["--movie-dir", str(folder), "--movie-ext", values["movie_ext"],
                "--movie-fps", str(fps), "--dump-frames", str(frames)]
        values["movie_fps"] = fps
        job.extra.update(frames_dir=str(frames), movie_candidates=tuple(map(canonical_path, targets)))
    else:
        cmd.append("--preview-stream")
    job.cmd = cmd
    job.extra["panel_form"] = dict(values, h5_dir=str(source), dark_path=str(dark) if dark else "",
                                   movie_dir=str(folder))
    return job


def _prepare_agent(_jobs, params):
    options = dict(params)
    options["h5_dir"] = resolve_path(options.get("h5_dir"))
    if options.get("dark_path"):
        options["dark_path"] = resolve_path(options["dark_path"])
    if options.get("movie_dir"):
        options["movie_dir"] = resolve_path(options["movie_dir"], write=True)
    try:
        job = movie_job(**options)
    except OverflowError as exc:
        raise ValueError("Enter finite Movie settings.") from exc
    values = job.extra["panel_form"]
    summary = {
        "Mode": "Export movie and frames" if values["save_movie"] else "Preview without saving files",
        "HDF5 directory": values["h5_dir"], "Frame dataset": values["h5_dataset"],
        "Frames": f"First {values['n_frames']} in file order",
        "Playback interval (ms)": values["interval_ms"], "Scale frames": values["scale_frames"],
        "Colormap": values["cmap"], "Log scale": values["log_scale"],
        "vmin fraction": values["vmin_frac"], "vmax percentile": values["vmax_pct"],
        "Dark frame": values["dark_path"] or "Off",
    }
    if values["save_movie"]:
        summary.update({"Output directory": values["movie_dir"], "Format": values["movie_ext"],
                        "Saved fps": values["movie_fps"] or "Match playback",
                        "Output paths (existing files may be replaced)": [str(p) for p in job.extra["movie_candidates"]]})
    return job, summary


def agent_operations():
    parameters = {
        "h5_dir": {"type": "path", "description": "Directory of camera HDF5 files"},
        "dark_path": {"type": "path", "description": "Optional precomputed dark frame"},
        "movie_dir": {"type": "path", "description": "Movie output folder. Defaults to movies in the workspace."},
    }
    for key, value in DEFAULTS.items():
        types = {bool: "boolean", int: "integer", float: "number", str: "string"}
        parameters[key] = {"type": types[type(value)], "default": value}
    parameters["movie_ext"]["choices"] = [".mp4", ".gif"]
    parameters["n_frames"]["description"] = "First N frames in file order"
    parameters["save_movie"]["description"] = "False streams a preview to the Movie panel without writing files"
    parameters["movie_fps"].update(type="number", description="Zero matches playback speed")
    return (PanelOperation("movie", {
        "panel": PANEL_TITLE, "manual": "Movie", "parameters": parameters,
        "summary": "Preview camera frames or export a movie. MP4 falls back to GIF when ffmpeg is unavailable.",
    }, _prepare_agent, (JOB_NAME,), outputs=("movie_file",)),)


def build(jobs: JobManager) -> None:
    def attach_movie(job):
        for key, value in adopt_agent_form(job, "movie").items():
            if key in fields:
                fields[key].value = value
        preview = "--preview-stream" in (job.cmd or ()) and (
            job.status not in TERMINAL or job.status == "done" and "preview_frame" in job.extra)
        live_state["job"] = job if preview else None
        live_label.visible = live_row.visible = live_img.visible = preview
        pause_btn.text = "Pause"
        pause_btn.set_enabled(preview and job.status != "done")
        if preview:
            ui.run_javascript(f"vmiMovieStart({json.dumps(f'movie-preview/{job.id}')})")
        else:
            ui.run_javascript("vmiMovieStop()")
        for previous in jobs.jobs():
            if previous.created < job.created and previous.name == JOB_NAME and previous.status in TERMINAL:
                previous.extra.pop("preview_frame", None)

    view = JobView(jobs, job_name=JOB_NAME, results_builder=_results, attach_builder=attach_movie)
    theme.panel_header(
        PANEL_TITLE,
        "Preview camera frames in order or save them as a movie.",
        view=view)
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
        theme.form_section("Playback")
        with field_grid(3):
            n_frames = persist(num_input("Frames", 20, step=1),
                               "movie.n_frames")
            interval_ms = persist(num_input("Interval (ms)", 500, step=50),
                                  "movie.interval_ms")
            scale_frames = persist(num_input("Scale frames", 10, step=1),
                                   "movie.scale_frames")
        with field_grid(3):
            cmap = persist(ui.input("Colormap", value="gray")
                           .props("outlined dense"), "movie.cmap")
            vmin_frac = persist(num_input("vmin fraction", 0.5, step=0.1),
                                "movie.vmin_frac")
            vmax_pct = persist(num_input("vmax percentile", 99.9, step=0.1),
                               "movie.vmax_pct")
        with check_grid():
            log_scale = persist(ui.checkbox("Log scale", value=False),
                                "movie.log_scale")
        theme.form_section("Save movie")
        with check_grid():
            save_movie = persist(ui.checkbox("Save movie file", value=True),
                                 "movie.save_movie")
        movie_dir = persist(path_input("Movie output folder",
                                       str(workspace_root(REPO_ROOT) / "movies"),
                                       browse="dir"), "movie.movie_dir")
        with field_grid():
            movie_ext = persist(ui.select([".mp4", ".gif"], value=".mp4",
                                          label="Format")
                                .props("outlined dense"), "movie.movie_ext")
            movie_fps = persist(num_input("Saved fps (0 = playback)", 0,
                                          step=1), "movie.movie_fps")
        for w in (movie_dir, movie_ext, movie_fps):
            w.bind_enabled_from(save_movie, "value")

        fields = dict(h5_dir=h5_dir, h5_dataset=h5_dataset, dark_path=dark_path, n_frames=n_frames,
                      interval_ms=interval_ms, scale_frames=scale_frames, cmap=cmap, vmin_frac=vmin_frac,
                      vmax_pct=vmax_pct, log_scale=log_scale, save_movie=save_movie, movie_dir=movie_dir,
                      movie_ext=movie_ext, movie_fps=movie_fps)

        def run():
            try:
                job = movie_job(**{key: widget.value for key, widget in fields.items()})
                view.attach(jobs.submit(job))
            except (OSError, ValueError, OverflowError) as exc:
                ui.notify(str(exc), type="warning")

        with action_bar(view, on_click=run):
            ui.button("Clear outputs", icon="delete",
                      on_click=lambda: _clear_outputs(movie_dir, h5_dir, view, clear_preview)) \
                .props("flat color=negative").classes("vmi-action-danger")
        ui.label("Turn off Save movie file to preview frames without saving files.").classes("vmi-subtitle")

        live_label = ui.label("Live playback").classes("vmi-caption")
        live_label.visible = False
        with ui.row().classes("items-center gap-4") as live_row:
            live_row.visible = False
            pause_btn = ui.button("Pause", on_click=lambda: _toggle_pause()) \
                .props("flat dense color=primary")
            ui.label("").props("id=vmi-live-counter").classes("vmi-subtitle")
        live_img = ui.element("svg").props('id=vmi-live-img role=img aria-label="Movie preview with pixel axes"') \
            .classes("w-full max-w-2xl")
        live_img.visible = False
        live_state = {"job": None}

        def clear_preview():
            if view.job is None or view.job.status not in TERMINAL:
                return False
            had_preview = view.job.extra.pop("preview_frame", None) is not None or live_img.visible
            if had_preview:
                ui.run_javascript("vmiMovieStop()")
                live_state["job"] = None
                live_label.visible = live_row.visible = live_img.visible = False
            return had_preview

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
        workspace_timer(0.1, _live_tick)

    view.place_console()


def _saved_frame_dirs(dataset: Path, output_dir: Path):
    for source in sorted((generated_dir(GENERATED) / "movie_frames").glob("*/movie-source.json")):
        if source.parent.is_symlink() or source.parent.is_junction() or source.is_symlink():
            continue
        try:
            metadata = json.loads(source.read_text(encoding="utf-8"))
            if (canonical_path(metadata["dataset"]) == dataset
                    and canonical_path(metadata["output_dir"]) == output_dir):
                yield source.parent
        except (OSError, ValueError, KeyError, TypeError):
            continue


def _clear_outputs(movie_dir, h5_dir, view: JobView, clear_preview) -> None:
    """Clear the selected movie and its identified saved-frame dumps."""
    if not h5_dir.value:
        if clear_preview():
            ui.notify("Movie preview cleared.", type="positive")
            return
        ui.notify("Choose the dataset whose saved outputs you want to clear.", type="info")
        return
    dataset = canonical_path(h5_dir.value, workspace_root(REPO_ROOT))
    output_dir = canonical_path(movie_dir.value or workspace_root(REPO_ROOT), workspace_root(REPO_ROOT))
    name = os.path.basename(os.path.normpath(h5_dir.value))
    movies = [path for ext in (".mp4", ".gif")
              if (path := output_dir / (name + ext)).is_file()]
    frame_dirs = list(_saved_frame_dirs(dataset, output_dir))
    frame_pngs = [path for folder in frame_dirs for path in sorted(folder.glob("frame_*.png"))]
    sidecars = [path for folder in frame_dirs for filename in ("movie-source.json", "movie-path.txt")
                if (path := folder / filename).is_file()]
    targets = {path: path.lstat() for path in movies + frame_pngs + sidecars}
    if not targets:
        if clear_preview():
            ui.notify("Movie preview cleared.", type="positive")
        else:
            ui.notify("No saved movie outputs for this dataset. Preview runs save no files.", type="info")
        return

    with ui.dialog() as dlg, ui.card():
        ui.label("Delete movie outputs?").classes("vmi-caption")
        for p in movies:
            ui.label(str(p)).style("font-family: monospace")
        if frame_pngs:
            ui.label(f"{len(frame_pngs)} saved frame images for this dataset") \
                .style("font-family: monospace")
        with ui.row().classes("gap-2"):
            ui.button("Delete", color="negative", on_click=lambda: do_delete())
            ui.button("Cancel", on_click=dlg.close).props("flat")
    dlg.open()

    def do_delete():
        failed = delete_outputs([*movies, *frame_pngs], expected=targets)
        if failed is None:
            return
        if not failed:
            failed = delete_outputs(sidecars, expected=targets)
            if failed is None:
                return
        for folder in frame_dirs:
            try:
                folder.rmdir()
            except OSError:
                pass
        dlg.close()
        if failed:
            ui.notify("Could not delete: " + "; ".join(failed),
                      type="negative")
        else:
            view.clear_results()
            clear_preview()
            ui.notify("Saved movie outputs cleared.", type="positive")


def _results(job: Job):
    if job.status != "done" or "frames_dir" not in job.extra:
        return
    frames_dir = Path(job.extra["frames_dir"])
    result_path = frames_dir / "movie-path.txt"
    movie = result_path.read_text(encoding="utf-8") if result_path.is_file() else ""
    if movie and Path(movie).is_file():
        ui.label(f"Movie: {movie}").classes("vmi-subtitle")
        with ui.row().classes("items-center gap-2"):
            url = file_url(movie)
            ui.button("Download movie", icon="download",
                      on_click=lambda: ui.download.from_url(url, filename=Path(movie).name)).props("flat")
            if not shared.SERVER:
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
