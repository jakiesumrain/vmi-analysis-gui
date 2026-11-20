"""Shared scaffolding for script panels: form helpers, JobView, file serving."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import contextmanager
from pathlib import Path
from time import monotonic, sleep

import numpy as np
from nicegui import app, ui

from ..jobs import Job, JobManager, MANAGER, ResourceBusyError
from .. import shared, theme
from ..workflow import register_field

from importlib.resources import files
from vmi_analysis.paths import WORKSPACE_ROOT as REPO_ROOT, GENERATED_DIR as GENERATED

GENERATED.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Static file serving (job outputs: PNGs, movies, frames)
# --------------------------------------------------------------------------

_SERVED: dict[str, str] = {}


def serve_dir(path) -> str:
    """Serve a directory over HTTP once; return its URL prefix."""
    if shared.SERVER:
        raise RuntimeError("Use an authorized file URL in shared mode.")
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
    p = shared.checked_path(path) if shared.SERVER else Path(path)
    try:
        stamp = p.stat().st_mtime_ns
    except OSError:
        stamp = 0
    if shared.SERVER:
        from urllib.parse import quote

        root = shared.workspace_root()
        area = "private" if p.is_relative_to(root) else "datasets"
        relative = p.relative_to(root if area == "private" else shared.SERVER.datasets)
        return f"/workspace-file/{area}/{quote(relative.as_posix())}?t={stamp}"
    return f"{serve_dir(p.parent)}/{p.name}?t={stamp}"


# --------------------------------------------------------------------------
# Open-in-file-manager
# --------------------------------------------------------------------------

def _open_windows_folder(folder: Path) -> None:
    os.startfile(str(folder))
    try:
        import pythoncom
        import pywintypes
        import win32con
        import win32gui
        from win32com.client import Dispatch
    except ImportError:
        return

    # Locate the actual folder window after Explorer handles the launch.
    # Its process ID may be shared by windows showing unrelated folders.
    expected = os.path.normcase(str(folder))
    pythoncom.CoInitialize()
    try:
        shell = Dispatch("Shell.Application")
        deadline = monotonic() + 1.5
        while monotonic() < deadline:
            for window in shell.Windows():
                try:
                    location = os.path.normcase(os.path.normpath(window.Document.Folder.Self.Path))
                    if location != expected:
                        continue
                    hwnd = window.HWND
                    if win32gui.IsIconic(hwnd):
                        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                    try:
                        win32gui.SetForegroundWindow(hwnd)
                    except pywintypes.error:
                        pass
                    if win32gui.GetForegroundWindow() != hwnd:
                        # Raise the window without taking keyboard focus when Windows blocks activation.
                        flags = win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE
                        win32gui.SetWindowPos(hwnd, win32con.HWND_TOP, 0, 0, 0, 0, flags)
                    return
                except (AttributeError, TypeError, pywintypes.com_error, pywintypes.error):
                    continue  # A Shell window may be closing or showing a virtual location.
            sleep(0.05)
    except pywintypes.com_error:
        pass  # Explorer already received the open request.
    finally:
        pythoncom.CoUninitialize()


async def open_folder(path) -> None:
    """Open a file's containing folder (or the folder itself) in the OS
    file manager on the machine running the server."""
    if shared.SERVER:
        ui.notify("Use the file picker to browse folders on the server.")
        return
    p = Path(path)
    folder = (p if p.is_dir() else p.parent).resolve()
    try:
        await asyncio.to_thread(_open_windows_folder, folder)
    except (OSError, AttributeError) as e:
        ui.notify(f"Could not open {folder}: {e}", type="negative")
    else:
        ui.notify(f"Opened {folder}")


def open_folder_button(path, label: str = "open containing folder"
                       ) -> ui.button:
    """Flat caption-style button that opens the folder in the file manager."""
    async def on_click():
        await open_folder(path)

    return ui.button(label, icon="folder_open",
                     on_click=on_click) \
        .props("flat dense").classes("vmi-caption")


# --------------------------------------------------------------------------
# Zoomable image lightbox
# --------------------------------------------------------------------------

_ZOOM_READY = False

_OVERLAY_HTML = """
<div id="vmi-zoom-overlay" style="display:none; position:fixed; inset:0;
     z-index:9999; background:rgba(248,250,252,0.97); cursor:grab;
     overflow:hidden; touch-action:none;">
  <img id="vmi-zoom-img" src="" alt="" style="position:absolute; left:50%;
       top:50%; max-width:none; user-select:none; -webkit-user-drag:none;"/>
  <div id="vmi-zoom-selection" style="display:none; position:absolute; pointer-events:none;
       border:2px solid #1e40af; background:rgba(30,64,175,0.10); box-sizing:border-box;"></div>
  <div id="vmi-zoom-toolbar" role="toolbar" aria-label="Image inspection"
       style="position:absolute; top:14px; right:16px; display:flex; gap:8px; flex-wrap:wrap; max-width:95%;
       font-family:'Fira Code', ui-monospace, monospace;">
    <button id="vmi-zoom-pan" onclick="vmiZoomPan()" aria-pressed="true"
      style="background:#1e40af; color:#ffffff; border:1px solid #1e40af; border-radius:6px;
      padding:0 10px; height:30px; cursor:pointer; transition:background-color .16s ease, color .16s ease;">Pan</button>
    <button id="vmi-zoom-select" onclick="vmiZoomSelect()" aria-pressed="false"
      style="background:#ffffff; color:#1e40af; border:1px solid #bfdbfe; border-radius:6px;
      padding:0 10px; height:30px; cursor:pointer; transition:background-color .16s ease, color .16s ease;">Crop area</button>
    <button onclick="vmiZoomAt(null,1.25)" style="background:#ffffff;
      color:#1e40af; border:1px solid #bfdbfe; border-radius:6px; width:34px; height:30px;
      font-size:18px; cursor:pointer;">+</button>
    <button onclick="vmiZoomAt(null,0.8)" style="background:#ffffff;
      color:#1e40af; border:1px solid #bfdbfe; border-radius:6px; width:34px; height:30px;
      font-size:18px; cursor:pointer;">&minus;</button>
    <button onclick="vmiZoomReset()" style="background:#ffffff; color:#1e40af;
      border:1px solid #bfdbfe; border-radius:6px; padding:0 10px; height:30px;
      font-size:12px; cursor:pointer;">reset</button>
    <button onclick="vmiZoomClose()" style="background:#ffffff; color:#1e40af;
      border:1px solid #bfdbfe; border-radius:6px; width:34px; height:30px;
      font-size:14px; cursor:pointer;">&#10005;</button>
  </div>
  <div id="vmi-zoom-hint" style="position:absolute; bottom:14px; left:16px;
       color:#475569; font:12px 'Fira Sans', sans-serif; pointer-events:none;">
    Drag to pan · Mouse wheel to zoom · Crop area to inspect only a region
  </div>
</div>
""" + "<script>" + (files("vmi_analysis.gui.assets") / "image_zoom.js").read_text(encoding="utf-8") + "</script>"

def init_zoom_overlay() -> None:
    global _ZOOM_READY
    if not _ZOOM_READY:
        _ZOOM_READY = True
        # shared=True: appended to every page; a per-client add would be
        # consumed by the auto-opened browser (ui.run(show=True)) and never
        # reach later clients.
        ui.add_body_html(_OVERLAY_HTML, shared=True)


def zoomable_image(src: str = "", **kwargs) -> ui.image:
    """Image opening a shared overlay with pan, wheel zoom, and view cropping."""
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


# --------------------------------------------------------------------------
# Form helpers
# --------------------------------------------------------------------------

def path_input(label: str, value: str = "", browse: str | None = None,
               filters: list[str] | None = None, default_name: str = "",
               **kw) -> ui.input:
    """Path text field; browse adds an in-field icon opening LocalFilePicker.

    browse: None | "dir" | "file" | "save".
    """
    inp = ui.input(label, value=value, **kw).classes("w-full") \
        .props("outlined dense")
    if browse:
        async def pick():
            from .picker import LocalFilePicker
            result = await LocalFilePicker(start=inp.value, mode=browse,
                                           filters=filters,
                                           default_name=default_name)
            if result:
                inp.value = result
        with inp.add_slot("append"):
            ui.button(icon="folder_open", on_click=pick).classes("vmi-browse-icon") \
                .props('flat dense round aria-label="Browse"')
    return inp


def num_input(label: str, value, fmt: str | None = None, **kw) -> ui.number:
    kwargs = {"label": label, "value": value}
    if fmt:
        kwargs["format"] = fmt
    kwargs.update(kw)
    return ui.number(**kwargs).props("outlined dense")


def persist(widget, key: str):
    """Restore and save settings in the active workspace."""
    store = shared.settings_store()
    register_field(key, widget)
    if key in store:
        widget.value = store[key]
    widget.on_value_change(lambda e: store.__setitem__(key, e.value))
    return widget


def delete_outputs(paths, *, expected=None):
    """Delete only when no running/queued job reads or writes these files."""
    try:
        with MANAGER.write_access(paths):
            failed = []
            for path in paths:
                try:
                    if expected is not None:
                        current, prior = path.lstat(), expected[path]
                        if (current.st_ino, current.st_size, current.st_mtime_ns) != (
                                prior.st_ino, prior.st_size, prior.st_mtime_ns):
                            failed.append(f"{path}: changed since confirmation opened")
                            continue
                    Path(path).unlink(missing_ok=True)
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    failed.append(f"{path}: {exc}")
            return failed
    except (ResourceBusyError, shared.WorkspaceDenied) as exc:
        ui.notify(str(exc), type="warning")
        return None


def confirm_clear_outputs(paths, title, *, on_cleared=None):
    """Confirm the exact files to remove, leaving folders and other files alone."""
    targets = {}
    for path in dict.fromkeys(map(Path, paths)):
        if shared.SERVER:
            try:
                path = shared.checked_path(path, write=True)
            except shared.WorkspaceDenied as exc:
                ui.notify(str(exc), type="warning")
                return
        if path.is_file():
            try:
                targets[path] = path.lstat()
            except OSError as exc:
                ui.notify(f"Could not inspect {path}: {exc}", type="negative")
                return
    if not targets:
        ui.notify("No saved outputs found for this selection.", type="info")
        return

    def clear():
        failed = delete_outputs(targets, expected=targets)
        if failed is None:
            return
        dialog.close()
        if on_cleared is not None:
            on_cleared()
        if failed:
            ui.notify("Could not delete: " + "; ".join(failed), type="negative")
        else:
            ui.notify(f"Cleared {len(targets)} output file(s).", type="positive")

    with ui.dialog() as dialog, ui.card().classes("w-full max-w-2xl"):
        ui.label(title).classes("vmi-caption")
        with ui.column().classes("w-full max-h-64 overflow-auto gap-1"):
            for target in targets:
                ui.label(str(target)).classes("break-all")
        with ui.row().classes("gap-2"):
            ui.button("Delete", color="negative", on_click=clear)
            ui.button("Cancel", on_click=dialog.close).props("flat")
    dialog.open()


def text_area(label: str, value: str = "") -> ui.input:
    return ui.input(label, value=value).classes("w-full").props("outlined dense")


def field_grid(columns: int = 2) -> ui.grid:
    """Form grid giving related inputs even columns and gutters."""
    return ui.grid(columns=columns).classes("w-full gap-x-4 vmi-field-grid")


def check_grid() -> ui.row:
    """Wrapping row for checkbox/toggle groups with even column gaps."""
    return ui.row().classes("vmi-check-grid w-full")


@contextmanager
def action_bar(view: "JobView", label: str = "Run", *, on_click):
    """Run/Stop button row; add secondary buttons inside the with block."""
    with ui.row().classes("vmi-action-bar w-full items-center gap-2"):
        yield view.run_button(label, on_click=on_click)


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
                 interval: float = 0.7, job_name: str | None = None, command_key: str | None = None):
        self.jobs = jobs
        self.job_name = job_name
        self._command_key = command_key
        self.command_label = None
        self.job: Job | None = None
        self.action_button = None
        self._run_label = "Run"
        self._log_cursor = 0
        self._results_shown = False
        self._results_builder = results_builder
        self._live_builder = live_builder
        # Status controls are created inline in the panel header by
        # place_status() (theme.panel_header passes the view there).
        self.badge = None
        self.spinner = None
        self.progress = None
        self.wait_label = None
        self.log = None
        self.console_hint = None
        self.results = None
        self._chevron = None
        self._console_head = None
        self._console_open = False
        self._console_opened_for_job = False
        self._results_placeholder = None
        self._interval = interval

    def place_status(self) -> None:
        """Inline status cluster for the panel header (theme.panel_header)."""
        self.badge = theme.status_badge("idle")
        self.spinner = ui.spinner("dots", size="sm", color="primary")
        self.spinner.visible = False
        self.progress = ui.linear_progress(value=0, show_value=False) \
            .classes("grow").props("color=primary track-color=grey-9")
        self.progress.visible = False
        self.wait_label = ui.label("").classes("vmi-caption")
        self.wait_label.visible = False

    def run_button(self, label="Run", *, on_click):
        """One button: Run when idle, Stop/Cancel queued while work is active."""
        self._run_label = label
        def toggle():
            # Honor the displayed action: a click on a stale Stop button must
            # never launch a new job if the old one just finished.
            if self.action_button.text != self._run_label:
                self.cancel()
            else:
                self.refresh()
                if self.job is None or self.job.status not in ("running", "queued"):
                    try:
                        on_click()
                    except shared.WorkspaceDenied as exc:
                        ui.notify(str(exc), type="warning")
        self.action_button = theme.run_button(label, on_click=toggle)
        self._sync_button()
        return self.action_button

    def _sync_button(self):
        if self.action_button is None:
            return
        j = self.job
        active = j is not None and j.status in ("running", "queued")
        stopping = active and j.cancel_requested
        self.action_button.text = ("Stopping…" if stopping else
                                   "Cancel queued" if active and j.status == "queued" else
                                   "Stop" if active else self._run_label)
        self.action_button.props("color=negative" if active else "color=primary")
        self.action_button.set_enabled(not stopping)

    def place_console(self, title: str = "Console") -> None:
        """Create the collapsible log console and results area; call at panel bottom."""
        with ui.column().classes("vmi-console w-full gap-2"):
            head = ui.row().classes(
                "vmi-console-head w-full items-center gap-2 no-wrap")
            head.props('role="button" tabindex="0" aria-expanded="false"')
            self._console_head = head
            with head:
                self._chevron = ui.icon("chevron_right") \
                    .classes("vmi-console-chevron")
                ui.label(title).classes("vmi-console-title")
                self.console_hint = ui.label("Output appears here after a run.") \
                    .classes("vmi-caption vmi-console-hint")
            head.on("click", self.toggle_console)
            head.on("keydown", self._on_console_key)
            if self._command_key is not None:
                self.command_label = ui.label("").classes("w-full font-mono text-xs break-all select-text")
            self.log = theme.log_console()
            self.results = ui.column().classes("w-full gap-4 vmi-results")
        self._reset_results()
        self._sync_console()
        shared.workspace_timer(self._interval, self.refresh)
        self.refresh()

    def toggle_console(self) -> None:
        self._console_open = not self._console_open
        self._sync_console()

    def open_console(self) -> None:
        if not self._console_open:
            self._console_open = True
            self._sync_console()

    def _on_console_key(self, e) -> None:
        key = e.args.get("key") if isinstance(e.args, dict) else None
        if key in ("Enter", " ", "Spacebar"):
            self.toggle_console()

    def _sync_console(self) -> None:
        if self.log is not None:
            self.log.visible = self._console_open
        if self.results is not None:
            self.results.visible = self._console_open
        self._sync_command()
        if self._chevron is not None:
            if self._console_open:
                self._chevron.classes(add="vmi-console-open")
            else:
                self._chevron.classes(remove="vmi-console-open")
        if self._console_head is not None:
            expanded = "true" if self._console_open else "false"
            self._console_head.props(f'aria-expanded="{expanded}"')

    def _reset_results(self) -> None:
        """Clear result content and show the empty-state placeholder."""
        if self.results is None:
            return
        self.results.clear()
        with self.results:
            self._results_placeholder = ui.row().classes(
                "vmi-results-empty w-full items-center justify-center gap-2")
            with self._results_placeholder:
                ui.icon("insights")
                ui.label("No results yet — run this stage or open saved results.") \
                    .classes("vmi-caption")

    def _sync_command(self) -> None:
        if self.command_label is not None:
            command = self.job.extra.get(self._command_key, "") if self.job else ""
            self.command_label.text = command
            self.command_label.visible = self._console_open and bool(command)

    def clear_results(self) -> None:
        """Public clear (e.g. after deleting outputs); restores the placeholder.

        Leaves _results_shown untouched so the next refresh does not rebuild
        results for a finished job whose outputs were just deleted.
        """
        self._reset_results()

    def attach(self, job: Job) -> None:
        self.job = job
        self._log_cursor = 0
        self._results_shown = False
        self._console_opened_for_job = False
        if self.log is not None:
            self.log.clear()
        self._reset_results()
        self.refresh()

    def cancel(self) -> None:
        if self.job is not None and not self.job.cancel_requested:
            self.jobs.cancel(self.job.id)
            self.refresh()

    def refresh(self) -> None:
        # Reconnect the new page's controls to a job still owned by the server.
        if self.job_name and (self.job is None or self.job.status in ("done", "failed", "cancelled")):
            owned = [j for j in self.jobs.jobs() if j.name == self.job_name]
            active = [j for j in owned if j.status in ("running", "queued")]
            if active:
                self.attach(next((j for j in active if j.status == "running"), active[-1]))
                return
            if owned and owned[-1].extra.get("agent_panel") and self.job is not owned[-1]:
                self.attach(owned[-1])
                return
            if shared.SERVER and self.job is None and owned:
                self.attach(owned[-1])
                return
        j = self.job
        self._sync_button()
        self._sync_command()
        if self.wait_label is not None:
            self.wait_label.visible = j is not None and j.status == "queued"
            self.wait_label.text = j.wait_reason if j is not None else ""
        if j is None:
            if self.badge is not None:
                theme.set_badge(self.badge, "idle")
            if self.console_hint is not None:
                self.console_hint.text = "Output appears here after a run."
            return
        if self.badge is not None:
            theme.set_badge(self.badge, j.status)
        if self.console_hint is not None:
            self.console_hint.text = {
                "running": "Streaming output…",
                "queued": "Queued — waiting for a slot",
                "done": "Finished — results below",
                "failed": "Failed — see log for details",
                "cancelled": "Cancelled",
            }.get(j.status, "Output appears here after a run.")
        running = j.status == "running"
        if not self._console_opened_for_job and (running or j.status in ("done", "failed", "cancelled")):
            self._console_opened_for_job = True
            self.open_console()
        if self.spinner is not None:
            self.spinner.visible = running
        if self.progress is not None:
            self.progress.visible = running
            if running and j.progress is not None:
                self.progress.value = j.progress
        # stream new log lines
        if self.log is not None:
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
            if self._results_placeholder is not None:
                self._results_placeholder.visible = False
            if self._results_builder is not None:
                try:
                    with self.results:
                        self._results_builder(j)
                except Exception as e:
                    with self.results:
                        ui.label(f"Could not build results view: {e}") \
                            .classes("vmi-subtitle")
