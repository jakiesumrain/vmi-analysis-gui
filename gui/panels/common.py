"""Shared scaffolding for script panels: form helpers, JobView, file serving."""

from __future__ import annotations

import time
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
    """URL for a single file inside a served directory."""
    p = Path(path)
    return f"{serve_dir(p.parent)}/{p.name}"


# --------------------------------------------------------------------------
# Image rendering (npy / text images -> PNG for the browser)
# --------------------------------------------------------------------------

def render_image(data: np.ndarray, out_path: Path, cmap: str = "viridis",
                 log: bool = True, vmax_pct: float = 99.999) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    data = np.asarray(data, dtype=float)
    vmax = np.percentile(data[data > 0], vmax_pct) if (data > 0).any() else 1.0
    if log:
        norm = LogNorm(vmin=max(vmax * 1e-4, 1e-12), vmax=vmax)
    else:
        norm = Normalize(vmin=0, vmax=vmax)
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

def path_input(label: str, value: str = "", **kw) -> ui.input:
    return ui.input(label, value=value, **kw).classes("w-full").props("outlined dense")


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
