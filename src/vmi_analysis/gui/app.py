"""VMI Analysis GUI.

Launch from the repository root:  uv run vmi-analysis
"""

import asyncio
from importlib.resources import files
from math import isfinite
import os

from nicegui import app, background_tasks, ui

from vmi_analysis.paths import WORKSPACE_ROOT

from . import theme, workflow
from .jobs import MANAGER, ResourceBusyError
from .startup import startup_port
from .panels import (accumulate, binning, calibration, centroid, circularize,
                     dark, gmd_map, movie, pes, pipeline)
from .panels.common import init_zoom_overlay
from .panels.picker import LocalFilePicker

# Keep the saved key so existing on/off preferences carry over.
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
    return max(value, minimum) if isfinite(value) else default


def _keep_alive_enabled() -> bool:
    return bool(app.storage.general.get(KEEP_ALIVE_KEY, False))


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


def _shutdown_now(reason: str) -> None:
    if app.is_stopping:
        return
    MANAGER.begin_shutdown()
    print(f"[gui] {reason} — shutting down", flush=True)
    app.shutdown()


def _stop_server() -> None:
    _shutdown_now("stop requested in Settings")


async def _shutdown_after(delay: float) -> None:
    await asyncio.sleep(max(delay, 0.0))
    if not _has_connected_client() and not _keep_alive_enabled():
        _shutdown_now("last client disconnected")


@app.on_disconnect
def _schedule_shutdown_after_last_client(client) -> None:
    """Schedule automatic shutdown only when the last client leaves and keep-alive is off."""
    global _shutdown_task
    _cancel_shutdown_check()
    if app.is_stopping or _has_connected_client() or _keep_alive_enabled():
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
    ("Trig Calibration", calibration),
    ("Circularize", circularize),
    ("PES", pes),
    ("Pipeline", pipeline),
]
SIDEBAR_GROUPS = (
    ("Acquire", ("GMD Map", "Movie", "Dark Frame", "Accumulate")),
    ("Analyze", ("Centroiding", "Binning", "Trig Calibration", "Circularize", "PES")),
    ("Automate", ("Pipeline",)),
)


@ui.page("/")
def index() -> None:
    theme.apply_theme()
    store = app.storage.general
    workflow_fields = workflow.result_fields(store)

    async def save_workflow():
        path = await LocalFilePicker(start=WORKSPACE_ROOT, mode="save", filters=[".json"],
                                     default_name="vmi-workflow.json")
        if not path:
            return
        try:
            with MANAGER.write_access([path]):
                workflow.save_workflow(path, workflow_fields)
        except (OSError, ValueError, ResourceBusyError) as exc:
            ui.notify(f"Could not save workflow. {exc}", type="negative")
            return
        ui.notify(f"Workflow saved to {path}", type="positive")

    async def load_workflow():
        path = await LocalFilePicker(start=WORKSPACE_ROOT, filters=[".json"])
        if not path:
            return
        try:
            settings = workflow.load_workflow(path, workflow_fields)
        except (OSError, ValueError) as exc:
            ui.notify(f"Could not load workflow. {exc}", type="negative")
            return
        store.update(settings)
        ui.navigate.reload()

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

    with ui.dialog() as stop_dialog, ui.card().classes("w-96 gap-3"):
        ui.label("Stop VMI Analysis?").classes("text-lg font-semibold")
        ui.label("This disconnects every browser and cancels running and queued jobs.")
        with ui.row().classes("w-full justify-end"):
            ui.button("Cancel", on_click=stop_dialog.close).props("flat")
            ui.button("Stop server", on_click=_stop_server).props("color=negative unelevated")

    with ui.dialog() as about_dialog, ui.card().classes("w-96 gap-3"):
        ui.label("About VMI Analysis").classes("text-lg font-semibold")
        ui.label("Photoelectron imaging analysis toolbox.").classes("vmi-subtitle")
        ui.separator()
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Author").classes("vmi-caption")
            ui.link("Tao · jakiesumrain@163.com", "mailto:jakiesumrain@163.com") \
                .classes("vmi-subtitle")
        with ui.row().classes("w-full justify-end"):
            ui.button("Close", on_click=lambda: about_dialog.close()).props("flat")

    def open_about():
        settings_dialog.close()
        about_dialog.open()

    with ui.dialog() as settings_dialog, ui.card().classes("w-96 gap-3"):
        ui.label("Settings").classes("text-lg font-semibold")
        ui.separator()
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Keep running on disconnect").classes("vmi-subtitle")
            keep_alive = ui.checkbox(
                value=_keep_alive_enabled(),
            ).props("dense").classes("vmi-setting-keep-alive")
        keep_alive.tooltip(
            "Keep the server available after the last browser disconnects, even "
            "when no jobs are running. Turn off to shut down and cancel jobs "
            "after the last browser disconnects.")
        keep_alive.on_value_change(change_keep_alive)
        limits = list(range(1, len(TABS) + 1))
        if MANAGER.max_concurrent not in limits:
            limits.append(MANAGER.max_concurrent)
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Maximum concurrent jobs").classes("vmi-subtitle")
            concurrency = ui.select(
                limits, value=MANAGER.max_concurrent,
            ).props("outlined dense aria-label=Maximum concurrent jobs") \
                .classes("w-24 vmi-setting-concurrency")
        concurrency.on_value_change(change_limit)
        concurrency.tooltip("Lowering the limit lets running jobs finish; "
                            "conflicts remain queued.")
        ui.separator()
        ui.label("Workflow files restore panel inputs and parameters.").classes("vmi-subtitle")
        with ui.row().classes("w-full gap-2"):
            ui.button("Save workflow", icon="save", on_click=save_workflow).props("flat")
            ui.button("Load workflow", icon="folder_open", on_click=load_workflow).props("flat")
        with ui.row().classes("w-full justify-between items-center"):
            with ui.row().classes("gap-1"):
                ui.button("About", icon="info",
                          on_click=open_about).props("flat")
                ui.button("Stop server", icon="power_settings_new",
                          on_click=stop_dialog.open).props("flat color=negative")
            ui.button("Close", on_click=lambda: settings_dialog.close()).props("flat")

    with ui.header().classes("vmi-header items-center justify-between px-6 py-3"):
        with ui.row().classes("items-baseline gap-3"):
            ui.label("VMI Analysis").classes("vmi-title vmi-title-md vmi-title-bold")
            ui.label("photoelectron imaging toolbox").classes("vmi-subtitle")
        with ui.row().classes("items-center gap-3"):
            settings_button = ui.button(
                "Settings", icon="settings",
                on_click=lambda: settings_dialog.open()).props("outline dense")
            settings_button.tooltip(
                "Stop server, keep-alive, concurrency and workflow files")

    def status_item(label, value, note):
        with ui.row().classes("vmi-status-item items-baseline gap-1"):
            ui.label(label).classes("vmi-status-label")
            value_label = ui.label(value).classes("vmi-status-value")
            note_label = ui.label(note).classes("vmi-status-note")
        return value_label, note_label

    def refresh_queue():
        running = MANAGER.running()
        n_queued = sum(1 for job in MANAGER.jobs() if job.status == "queued")
        running_value.text = str(len(running))
        running_note.text = (" · ".join(job.name for job in running)
                             if running else "idle")
        queued_value.text = str(n_queued)
        queued_note.text = "waiting for a slot" if n_queued else "none waiting"
        limit_value.text = str(MANAGER.max_concurrent)
        keep_value.text = "Keep-alive" if _keep_alive_enabled() else "Auto-stop"
        keep_note.text = ("after disconnect" if _keep_alive_enabled()
                          else "after last client")

    with ui.row().classes("vmi-workbench w-full no-wrap"):
        with ui.column().classes("vmi-sidebar"):
            ui.label("Workspace").classes("vmi-sidebar-title")
            with ui.tabs().classes("vmi-sidebar-tabs") as tabs:
                for group, names in SIDEBAR_GROUPS:
                    ui.label(group).classes("vmi-sidebar-group")
                    for name in names:
                        ui.tab(name)
            with ui.column().classes("vmi-sidebar-foot"):
                ui.label("Local workspace").classes("vmi-sidebar-foot-label")
                ui.label(f"{len(TABS)} tools").classes("vmi-sidebar-foot-note")

        with ui.column().classes("vmi-workspace"):
            with ui.tab_panels(tabs, value=TABS[0][0]).classes(
                    "vmi-tab-panels w-full bg-transparent"):
                def export_calibration(extracted_path, image_path, manual_center, **settings):
                    return use_calibration(extracted_path, image_path, manual_center, **settings)

                with workflow.collect_fields(workflow_fields):
                    for name, module in TABS:
                        with ui.tab_panel(name):
                            if module is circularize:
                                use_calibration = module.build(MANAGER)
                            elif module is calibration:
                                module.build(MANAGER, use_calibration=export_calibration)
                            else:
                                module.build(MANAGER)

    with ui.footer().classes(
            "vmi-footer vmi-status-footer px-6 py-2 items-center justify-between"):
        with ui.row().classes("vmi-statusbar items-center no-wrap"):
            running_value, running_note = status_item("Active jobs", "0", "idle")
            queued_value, queued_note = status_item("Queued", "0", "none waiting")
            limit_value, _ = status_item("Job limit", str(MANAGER.max_concurrent),
                                         "concurrent")
            keep_value, keep_note = status_item(
                "Server",
                "Keep-alive" if _keep_alive_enabled() else "Auto-stop",
                "after disconnect" if _keep_alive_enabled() else "after last client")
    refresh_queue()
    ui.timer(1.0, refresh_queue)


def main(*, host: str = "0.0.0.0", launch_command: str = "uv run vmi-analysis") -> None:
    port = startup_port(host=host, launch_command=launch_command)
    from vmi_analysis.paths import prepare_workspace
    prepare_workspace()
    try:
        MANAGER.set_max_concurrent(app.storage.general.get("jobs.max_concurrent", 2))
    except ValueError:
        MANAGER.set_max_concurrent(2)
    init_zoom_overlay()
    ui.run(title="VMI Analysis",
           favicon=str(files("vmi_analysis.gui.assets") / "electron-spectrum-measured.svg"),
           host=host,
           port=port,
           dark=False,
           reconnect_timeout=_env_float("VMI_GUI_RECONNECT_TIMEOUT",
                                        DEFAULT_RECONNECT_TIMEOUT, minimum=4.0),
           reload=False,
           show=os.environ.get("VMI_GUI_NO_SHOW", "") != "1")


if __name__ in {"__main__", "__mp_main__"}:
    main()
