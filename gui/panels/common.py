"""Shared scaffolding for script panels: form helpers, JobView, file serving."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from nicegui import app, ui

from ..jobs import Job, JobManager
from .. import theme

GUI_DIR = Path(__file__).parent.parent
REPO_ROOT = GUI_DIR.parent
GENERATED = GUI_DIR / "generated"
GENERATED.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# Static file serving (job outputs: PNGs, movies, frames)
# --------------------------------------------------------------------------

_SERVED: dict[str, str] = {}


def serve_dir(path) -> str:
    """Serve a directory over HTTP once; return its URL prefix."""
    key = str(Path(path).resolve())
    if key not in _SERVED:
        url = f"/files/{len(_SERVED)}"
        app.add_static_files(url, key)
        _SERVED[key] = url
    return _SERVED[key]


def file_url(path) -> str:
    """URL for a single file inside a served directory.

    Appends the file's mtime as a cache-buster so the browser refetches
    when a job rewrites the same filename (e.g. tuning check, result PNGs).
    """
    p = Path(path)
    try:
        stamp = p.stat().st_mtime_ns
    except OSError:
        stamp = 0
    return f"{serve_dir(p.parent)}/{p.name}?t={stamp}"


# --------------------------------------------------------------------------
# Open-in-file-manager
# --------------------------------------------------------------------------

def open_folder(path) -> None:
    """Open a file's containing folder (or the folder itself) in the OS
    file manager on the machine running the server."""
    p = Path(path)
    folder = p if p.is_dir() else p.parent
    try:
        os.startfile(str(folder))  # Windows
    except (OSError, AttributeError) as e:
        ui.notify(f"Could not open {folder}: {e}", type="negative")
    else:
        ui.notify(f"Opened {folder}")


def open_folder_button(path, label: str = "open containing folder"
                       ) -> ui.button:
    """Flat caption-style button that opens the folder in the file manager."""
    return ui.button(label, icon="folder_open",
                     on_click=lambda: open_folder(path)) \
        .props("flat dense").classes("vmi-caption")


# --------------------------------------------------------------------------
# Zoomable image lightbox
# --------------------------------------------------------------------------

_ZOOM_READY = False

_OVERLAY_HTML = """
<div id="vmi-zoom-overlay" style="display:none; position:fixed; inset:0;
     z-index:9999; background:rgba(20,19,18,0.93); cursor:grab;
     overflow:hidden; touch-action:none;">
  <img id="vmi-zoom-img" src="" alt="" style="position:absolute; left:50%;
       top:50%; max-width:none; user-select:none; -webkit-user-drag:none;"/>
  <div style="position:absolute; top:14px; right:16px; display:flex; gap:8px;
       font-family:monospace;">
    <button onclick="vmiZoomAt(event,1.25)" style="background:#cc785c;
      color:#faf9f5; border:none; border-radius:6px; width:34px; height:30px;
      font-size:18px; cursor:pointer;">+</button>
    <button onclick="vmiZoomAt(event,0.8)" style="background:#cc785c;
      color:#faf9f5; border:none; border-radius:6px; width:34px; height:30px;
      font-size:18px; cursor:pointer;">&minus;</button>
    <button onclick="vmiZoomReset()" style="background:#efe9de; color:#141413;
      border:none; border-radius:6px; padding:0 10px; height:30px;
      font-size:12px; cursor:pointer;">reset</button>
    <button onclick="vmiZoomClose()" style="background:#efe9de; color:#141413;
      border:none; border-radius:6px; width:34px; height:30px;
      font-size:14px; cursor:pointer;">&#10005;</button>
  </div>
</div>
<script>
(function () {
  let scale = 1, tx = 0, ty = 0, dragging = false, sx = 0, sy = 0;
  const overlay = document.getElementById('vmi-zoom-overlay');
  const img = document.getElementById('vmi-zoom-img');
  function apply() {
    img.style.transform = 'translate(calc(-50% + ' + tx + 'px), calc(-50% + '
      + ty + 'px)) scale(' + scale + ')';
  }
  img.addEventListener('load', function () {
    const fw = overlay.clientWidth * 0.92 / img.naturalWidth;
    const fh = overlay.clientHeight * 0.92 / img.naturalHeight;
    scale = Math.min(1, fw, fh); tx = 0; ty = 0; apply();
  });
  window.vmiZoomOpen = function (src) {
    overlay.style.display = 'block';
    img.src = src;   // onload fits to viewport
  };
  window.vmiZoomClose = function () {
    overlay.style.display = 'none'; img.src = '';
  };
  window.vmiZoomReset = function () {
    img.dispatchEvent(new Event('load'));
  };
  window.vmiZoomAt = function (e, f) {
    const cx = overlay.clientWidth / 2, cy = overlay.clientHeight / 2;
    const px = e.clientX - cx - tx, py = e.clientY - cy - ty;
    const ns = Math.min(30, Math.max(0.05, scale * f));
    tx -= px * (ns / scale - 1);
    ty -= py * (ns / scale - 1);
    scale = ns; apply();
  };
  overlay.addEventListener('wheel', function (e) {
    e.preventDefault();
    vmiZoomAt(e, e.deltaY < 0 ? 1.2 : 1 / 1.2);
  }, { passive: false });
  overlay.addEventListener('mousedown', function (e) {
    if (e.target.tagName === 'BUTTON') return;
    dragging = true; sx = e.clientX - tx; sy = e.clientY - ty;
    overlay.style.cursor = 'grabbing';
  });
  window.addEventListener('mousemove', function (e) {
    if (!dragging) return;
    tx = e.clientX - sx; ty = e.clientY - sy; apply();
  });
  window.addEventListener('mouseup', function () {
    dragging = false; overlay.style.cursor = 'grab';
  });
  overlay.addEventListener('click', function (e) {
    if (e.target === overlay) vmiZoomClose();
  });
  overlay.addEventListener('dblclick', function (e) {
    if (e.target.tagName !== 'BUTTON') vmiZoomReset();
  });
  window.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') vmiZoomClose();
  });
})();
</script>
"""


def init_zoom_overlay() -> None:
    global _ZOOM_READY
    if not _ZOOM_READY:
        _ZOOM_READY = True
        # shared=True: appended to every page; a per-client add would be
        # consumed by the auto-opened browser (ui.run(show=True)) and never
        # reach later clients.
        ui.add_body_html(_OVERLAY_HTML, shared=True)


def zoomable_image(src: str = "", **kwargs) -> ui.image:
    """ui.image that opens a fullscreen zoom/pan overlay when clicked."""
    init_zoom_overlay()
    img = ui.image(src, **kwargs)
    img.style("cursor: zoom-in")
    img.on("click", lambda: ui.run_javascript(
        f"vmiZoomOpen({json.dumps(img.source)})"))
    return img


# --------------------------------------------------------------------------
# Image rendering (npy / text images -> PNG for the browser)
# --------------------------------------------------------------------------

def render_image(data: np.ndarray, out_path: Path, cmap: str = "viridis",
                 log: bool = True, vmax_pct: float = 99.999,
                 vmin_pct: float | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    data = np.asarray(data, dtype=float)
    vmax = np.percentile(data[data > 0], vmax_pct) if (data > 0).any() else 1.0
    if log:
        norm = LogNorm(vmin=max(vmax * 1e-4, 1e-12), vmax=vmax)
    else:
        vmin = np.percentile(data, vmin_pct) if vmin_pct is not None else 0
        norm = Normalize(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(data, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
    ax.axis("off")
    fig.savefig(out_path, dpi=110, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out_path


def render_npy(npy_path, out_path: Path, **kw) -> Path:
    return render_image(np.load(npy_path), out_path, **kw)


def render_txt_image(txt_path, out_path: Path, **kw) -> Path:
    return render_image(np.loadtxt(txt_path), out_path, **kw)


# --------------------------------------------------------------------------
# Form helpers
# --------------------------------------------------------------------------

def path_input(label: str, value: str = "", browse: str | None = None,
               filters: list[str] | None = None, default_name: str = "",
               **kw) -> ui.input:
    """Path text field; browse adds a folder button opening LocalFilePicker.

    browse: None | "dir" | "file" | "save".
    """
    with ui.row().classes("w-full items-center no-wrap gap-1"):
        inp = ui.input(label, value=value, **kw).classes("grow") \
            .props("outlined dense")
        if browse:
            async def pick():
                from .picker import LocalFilePicker
                result = await LocalFilePicker(start=inp.value, mode=browse,
                                               filters=filters,
                                               default_name=default_name)
                if result:
                    inp.value = result
            ui.button(icon="folder_open", on_click=pick) \
                .props("flat dense round color=primary")
    return inp


def num_input(label: str, value, fmt: str | None = None, **kw) -> ui.number:
    kwargs = {"label": label, "value": value}
    if fmt:
        kwargs["format"] = fmt
    kwargs.update(kw)
    return ui.number(**kwargs).props("outlined dense")


def text_area(label: str, value: str = "") -> ui.input:
    return ui.input(label, value=value).classes("w-full").props("outlined dense")


# --------------------------------------------------------------------------
# JobView: status badge + progress + cancel + log console + results
# --------------------------------------------------------------------------

class JobView:
    """Tracks one job; refresh via a NiceGUI timer.

    results_builder(job) is called once when the job reaches a terminal state.
    live_builder(job) is called on every refresh while the job runs (e.g. to
    update a live partial image).
    """

    def __init__(self, jobs: JobManager, results_builder=None, live_builder=None,
                 interval: float = 0.7):
        self.jobs = jobs
        self.job: Job | None = None
        self._log_cursor = 0
        self._results_shown = False
        self._results_builder = results_builder
        self._live_builder = live_builder

        with ui.row().classes("items-center gap-3 w-full"):
            self.badge = theme.status_badge("idle")
            self.spinner = ui.spinner("dots", size="md", color="primary")
            self.spinner.visible = False
            self.progress = ui.linear_progress(value=0, show_value=False) \
                .classes("grow").props("color=primary track-color=grey-4")
            self.progress.visible = False
            self.cancel_btn = theme.cancel_button(on_click=self.cancel)
            self.cancel_btn.visible = False
        self.log = None
        self.results = None
        self._interval = interval

    def place_console(self) -> None:
        """Create the log console and results area; call at panel bottom."""
        self.log = theme.log_console()
        self.results = ui.column().classes("w-full gap-4")
        ui.timer(self._interval, self.refresh)

    def attach(self, job: Job) -> None:
        self.job = job
        self._log_cursor = 0
        self._results_shown = False
        self.log.clear()
        self.results.clear()
        self.refresh()

    def cancel(self) -> None:
        if self.job is not None:
            self.jobs.cancel(self.job.id)

    def refresh(self) -> None:
        j = self.job
        if j is None:
            theme.set_badge(self.badge, "idle")
            return
        theme.set_badge(self.badge, j.status)
        running = j.status == "running"
        self.spinner.visible = running
        self.cancel_btn.visible = running or j.status == "queued"
        self.progress.visible = running
        if running and j.progress is not None:
            self.progress.value = j.progress
        # stream new log lines
        n = len(j.log)
        if self._log_cursor > n:  # ring buffer wrapped; rebuild
            self.log.clear()
            self._log_cursor = 0
        while self._log_cursor < n:
            self.log.push(j.log[self._log_cursor])
            self._log_cursor += 1
        if running and self._live_builder is not None:
            try:
                self._live_builder(j)
            except Exception:
                pass
        if j.status in ("done", "failed", "cancelled") and not self._results_shown:
            self._results_shown = True
            if self._results_builder is not None:
                try:
                    with self.results:
                        self._results_builder(j)
                except Exception as e:
                    with self.results:
                        ui.label(f"Could not build results view: {e}") \
                            .classes("vmi-subtitle")
