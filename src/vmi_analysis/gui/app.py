"""VMI Analysis GUI — one tab per script plus the pipeline.

Launch from the repository root:  uv run vmi-analysis
"""

import asyncio

from nicegui import app, background_tasks, ui

from . import theme
from .jobs import MANAGER
from .panels import (accumulate, binning, centroid, circularize, dark,
                     gmd_map, movie, pes, pipeline)
from .panels.common import init_zoom_overlay

_shutdown_task: asyncio.Task | None = None


def _cancel_shutdown_check() -> None:
    global _shutdown_task
    if _shutdown_task is not None:
        _shutdown_task.cancel()
        _shutdown_task = None


app.on_connect(_cancel_shutdown_check)


@app.on_shutdown
async def _cleanup_jobs() -> None:
    await asyncio.to_thread(MANAGER.shutdown)


@app.on_disconnect
def _shutdown_after_last_client(client) -> None:
    """Stop the server once the last browser client is gone for good.

    on_disconnect fires on the socket drop, before NiceGUI's reconnect grace
    period ends — so re-check after that timeout: a page reload reconnects
    within it and survives. Each new connection cancels the pending check;
    each last-client disconnect starts a fresh grace period.
    """
    global _shutdown_task
    _cancel_shutdown_check()
    if app.is_stopping or any(c.has_socket_connection for c in app.clients()):
        return

    async def check() -> None:
        await asyncio.sleep(client.page.resolve_reconnect_timeout() + 0.1)
        if app.is_stopping or any(c.has_socket_connection for c in app.clients()):
            return
        MANAGER.begin_shutdown()
        print("[gui] last client disconnected — shutting down", flush=True)
        app.shutdown()

    _shutdown_task = background_tasks.create(check(), name="auto-shutdown-check")

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

    with ui.header().classes("items-center justify-between px-6 py-3") \
            .style("background: #faf9f5; border-bottom: 1px solid #e6dfd8"):
        with ui.row().classes("items-baseline gap-3"):
            ui.label("VMI Analysis").classes("vmi-title vmi-title-md")
            ui.label("photoelectron imaging toolbox").classes("vmi-subtitle")
        with ui.row().classes("items-center gap-4"):
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

            limits = list(range(1, len(TABS) + 1))
            if MANAGER.max_concurrent not in limits:
                limits.append(MANAGER.max_concurrent)
            concurrency = ui.select(limits, label="Maximum concurrent jobs", value=MANAGER.max_concurrent) \
                .classes("w-52").props("outlined dense")
            concurrency.on_value_change(change_limit)
            concurrency.tooltip("Lowering the limit lets running jobs finish; conflicts remain queued.")
            queue_label = ui.label("").classes("vmi-caption")

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
        ui.label("vmi-analysis-gui — independent jobs can run concurrently; "
                 "closing the browser tab stops the server.") \
            .classes("vmi-subtitle").style("color: #a09d96")
        ui.link("Tao (jakiesumrain@163.com)", "mailto:jakiesumrain@163.com") \
            .classes("vmi-subtitle").style("color: #cc785c")


def main() -> None:
    import os
    from vmi_analysis.paths import prepare_workspace
    prepare_workspace()
    try:
        MANAGER.set_max_concurrent(app.storage.general.get("jobs.max_concurrent", 2))
    except ValueError:
        MANAGER.set_max_concurrent(2)
    init_zoom_overlay()
    ui.run(title="VMI Analysis",
           port=int(os.environ.get("VMI_GUI_PORT", 8090)),
           reconnect_timeout=1.0,
           reload=False,
           show=os.environ.get("VMI_GUI_NO_SHOW", "") != "1")


if __name__ in {"__main__", "__mp_main__"}:
    main()
