"""VMI Analysis GUI — one tab per script plus the pipeline.

Launch from the repository root:  uv run vmi-analysis
"""

import asyncio
import os

from nicegui import app, background_tasks, ui

from . import theme
from .jobs import MANAGER
from .startup import startup_port
from .panels import (accumulate, binning, centroid, circularize, dark,
                     gmd_map, movie, pes, pipeline)
from .panels.common import init_zoom_overlay

KEEP_ALIVE_KEY = "server.keep_alive_while_jobs"
DEFAULT_RECONNECT_TIMEOUT = 30.0
DEFAULT_SHUTDOWN_GRACE = 2.0

_shutdown_task: asyncio.Task | None = None


def _env_float(name: str, default: float, minimum: float = 0.1) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(value, minimum)


def _keep_alive_enabled() -> bool:
    return bool(app.storage.general.get(KEEP_ALIVE_KEY, True))


def _has_connected_client() -> bool:
    return any(client.has_socket_connection for client in app.clients())


def _cancel_shutdown_check() -> None:
    global _shutdown_task
    if _shutdown_task is not None:
        _shutdown_task.cancel()
        _shutdown_task = None


app.on_connect(_cancel_shutdown_check)


@app.on_shutdown
async def _cleanup_jobs() -> None:
    await asyncio.to_thread(MANAGER.shutdown)


def _shutdown_now() -> None:
    MANAGER.begin_shutdown()
    print("[gui] last client disconnected — shutting down", flush=True)
    app.shutdown()


async def _shutdown_after(delay: float) -> None:
    await asyncio.sleep(max(delay, 0.0))
    if app.is_stopping or _has_connected_client():
        return
    _shutdown_now()


async def _keep_alive_until_idle() -> None:
    """Keep the server and its jobs alive while no page is connected."""
    while not app.is_stopping:
        if _has_connected_client():
            return
        if not MANAGER.has_active():
            break
        await asyncio.sleep(1.0)
    await asyncio.sleep(_env_float("VMI_GUI_SHUTDOWN_GRACE", DEFAULT_SHUTDOWN_GRACE))
    if app.is_stopping or _has_connected_client():
        return
    _shutdown_now()


@app.on_disconnect
def _schedule_shutdown_after_last_client(client) -> None:
    """Stop after the last page is gone, unless jobs should survive a detach.

    With the header keep-alive option on, a disconnect while jobs are running
    leaves the server and the jobs alive so Edge sleeping tabs, locks or network
    drops do not kill a long job. The server exits after the queue becomes idle
    and no page reconnects. When the option is off, the old behaviour applies:
    wait a short grace period, then stop and cancel the jobs.
    """
    global _shutdown_task
    _cancel_shutdown_check()
    if app.is_stopping or _has_connected_client():
        return
    if _keep_alive_enabled() and MANAGER.has_active():
        _shutdown_task = background_tasks.create(
            _keep_alive_until_idle(), name="keep-alive-while-jobs-run")
        return
    _shutdown_task = background_tasks.create(
        _shutdown_after(_env_float("VMI_GUI_SHUTDOWN_GRACE", DEFAULT_SHUTDOWN_GRACE)),
        name="auto-shutdown-check")

TABS = [
    ("GMD Map", gmd_map),
    ("Movie", movie),
    ("Dark Frame", dark),
    ("Accumulate", accumulate),
    ("Centroiding", centroid),
    ("Binning", binning),
    ("Circularize", circularize),
    ("PES", pes),
    ("Pipeline", pipeline),
]


@ui.page("/")
def index() -> None:
    theme.apply_theme()

    def change_limit(event):
        if event.value is None:
            return
        try:
            MANAGER.set_max_concurrent(event.value)
        except ValueError as exc:
            ui.notify(str(exc), type="warning")
            concurrency.value = MANAGER.max_concurrent
            return
        app.storage.general["jobs.max_concurrent"] = MANAGER.max_concurrent

    def change_keep_alive(event):
        app.storage.general[KEEP_ALIVE_KEY] = bool(event.value)

    with ui.dialog() as settings_dialog, ui.card().classes("w-96 gap-3"):
        ui.label("Settings").classes("text-lg font-semibold")
        ui.separator()
        keep_alive = ui.checkbox(
            "Keep running on disconnect",
            value=_keep_alive_enabled(),
        ).props("dense").classes("vmi-caption")
        keep_alive.tooltip(
            "Applies to every panel. While a job is running, keep this server "
            "and the job alive when the page disconnects (closed tab, Edge "
            "sleeping tab, locked screen, network loss). Turn off to stop the "
            "server and cancel jobs when the last page closes.")
        keep_alive.on_value_change(change_keep_alive)
        limits = list(range(1, len(TABS) + 1))
        if MANAGER.max_concurrent not in limits:
            limits.append(MANAGER.max_concurrent)
        concurrency = ui.select(
            limits, label="Maximum concurrent jobs",
            value=MANAGER.max_concurrent,
        ).classes("w-full").props("outlined dense")
        concurrency.on_value_change(change_limit)
        concurrency.tooltip("Lowering the limit lets running jobs finish; "
                            "conflicts remain queued.")
        with ui.row().classes("w-full justify-end"):
            ui.button("Close", on_click=lambda: settings_dialog.close()).props("flat")

    with ui.header().classes("items-center justify-between px-6 py-3") \
            .style("background: #faf9f5; border-bottom: 1px solid #e6dfd8"):
        with ui.row().classes("items-baseline gap-3"):
            ui.label("VMI Analysis").classes("vmi-title vmi-title-md vmi-title-bold")
            ui.label("photoelectron imaging toolbox").classes("vmi-subtitle")
        with ui.row().classes("items-center gap-4"):
            queue_label = ui.label("").classes("vmi-caption")
            ui.button("Settings", icon="settings",
                      on_click=lambda: settings_dialog.open()).props("outline dense")

    def refresh_queue():
        running = MANAGER.running()
        n_queued = sum(1 for j in MANAGER.jobs() if j.status == "queued")
        queue_label.text = ("running: " + ", ".join(job.name for job in running)) if running else "idle"
        if n_queued:
            queue_label.text += f"  ·  {n_queued} queued"

    ui.timer(1.0, refresh_queue)

    with ui.column().classes("w-full max-w-6xl mx-auto gap-4 p-6"):
        with ui.tabs().classes("w-full") as tabs:
            tab_objs = [ui.tab(name) for name, _ in TABS]
        with ui.tab_panels(tabs, value=tab_objs[0]).classes("w-full bg-transparent"):
            for tab, (_, module) in zip(tab_objs, TABS):
                with ui.tab_panel(tab):
                    module.build(MANAGER)

    with ui.footer().classes("px-6 py-2 items-center justify-between") \
            .style("background: #181715"):
        ui.label("vmi-analysis-gui — independent jobs run concurrently; closing "
                 "the last tab stops the server unless 'Keep running on "
                 "disconnect' is on and a job is still running.") \
            .classes("vmi-subtitle").style("color: #a09d96")
        ui.link("Tao (jakiesumrain@163.com)", "mailto:jakiesumrain@163.com") \
            .classes("vmi-subtitle").style("color: #cc785c")


def main() -> None:
    port = startup_port()
    from vmi_analysis.paths import prepare_workspace
    prepare_workspace()
    try:
        MANAGER.set_max_concurrent(app.storage.general.get("jobs.max_concurrent", 2))
    except ValueError:
        MANAGER.set_max_concurrent(2)
    init_zoom_overlay()
    ui.run(title="VMI Analysis",
           port=port,
           reconnect_timeout=_env_float("VMI_GUI_RECONNECT_TIMEOUT",
                                        DEFAULT_RECONNECT_TIMEOUT, minimum=4.0),
           reload=False,
           show=os.environ.get("VMI_GUI_NO_SHOW", "") != "1")


if __name__ in {"__main__", "__mp_main__"}:
    main()
