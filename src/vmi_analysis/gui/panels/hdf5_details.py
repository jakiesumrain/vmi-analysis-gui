"""File details disclosure for HDF5 inputs and results."""

import asyncio
from pathlib import Path

from nicegui import ui

from vmi_analysis.hdf5_details import inspect_hdf5

from ..jobs import ResourceBusyError
from .common import REPO_ROOT, path_input


def file_details(jobs):
    def read(path):
        with jobs.preview_read_access([path]) as stopping:
            return inspect_hdf5(path, stopping)

    async def inspect_selected():
        selected = file_input.value.strip()
        if not selected:
            status.text = "Choose an HDF5 file."
            return
        output.clear()
        inspect_button.disable()
        status.text = "Inspecting HDF5 file…"
        try:
            path = (REPO_ROOT / Path(selected).expanduser()).resolve()
            rows, structure = await asyncio.to_thread(read, path)
            status.text = str(path)
            with output:
                if rows:
                    columns = [dict(name=key, field=key, label=label, align="left")
                               for key, label in (("dataset", "Dataset"), ("metric", "Metric"), ("value", "Value"))]
                    ui.table(columns=columns, rows=[dict(row, id=index) for index, row in enumerate(rows)],
                             pagination=10).classes("w-full").props("flat dense")
                else:
                    ui.label("No recognized LUNA datasets. Open HDF5 structure to inspect this file.").classes("vmi-subtitle")
                with ui.expansion("HDF5 structure", icon="account_tree").classes("w-full"):
                    ui.label(structure).classes("block w-full font-mono text-xs overflow-auto") \
                        .style("white-space: pre; max-height: 28rem")
        except (OSError, ValueError, ResourceBusyError) as exc:
            status.text = f"Could not inspect the file. {exc}"
        finally:
            inspect_button.enable()

    with ui.expansion("Inspect HDF5 files", icon="description").classes("w-full") \
            .style("scroll-margin-top: var(--vmi-header-h)") as details:
        with ui.row().classes("w-full items-center"):
            file_input = path_input("HDF5 file", browse="file", filters=[".h5", ".hdf5"]).classes("flex-1")
            inspect_button = ui.button("Inspect file", on_click=inspect_selected).props("outline")
        status = ui.label("").classes("vmi-subtitle break-all").props('role="status"')
        output = ui.column().classes("w-full")

    async def inspect_output(path):
        if not inspect_button.enabled:
            return
        details.open()
        file_input.value = str(path)
        ui.run_javascript(f'getHtmlElement({details.id}).scrollIntoView({{block: "start"}})')
        await inspect_selected()

    return inspect_output
