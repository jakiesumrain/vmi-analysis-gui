"""Timepix3 dump controls for standalone-script verification."""

from pathlib import Path
import sys

from nicegui import ui

from vmi_analysis.tpx3_analysis import UNWRAP_MODE

from .. import shared, theme
from ..jobs import Job
from .common import JobView, action_bar, path_input, persist


JOB_NAME = "TPX3 Dump"


def build(jobs):
    def show_result(job):
        if job.status == "done":
            ui.label(job.extra["output"]).classes("vmi-caption break-all")
            ui.label("Open this HDF5 file in Ion TOF, or choose Use latest dump there.").classes("vmi-subtitle")

    def run():
        try:
            field = directory if mode.value == "Directory" else single_file
            if not field.value.strip() or not output.value.strip():
                raise ValueError("Choose a TPX3 input and an output HDF5 file.")
            source = shared.checked_path(field.value.strip())
            if (mode.value == "Directory" and not source.is_dir()
                    or mode.value == "Single file" and (not source.is_file() or source.suffix.lower() != ".tpx3")):
                raise ValueError("Choose an existing TPX3 file or directory.")
            destination = shared.checked_path(output.value.strip(), write=True)
            if destination.suffix.lower() not in (".h5", ".hdf5"):
                raise ValueError("Use .h5 or .hdf5 for the verification dump.")
            job = Job(JOB_NAME, cwd=str(shared.workspace_root()))
            job.cmd = [sys.executable, "-m", "vmi_analysis.ion_verification", "dump",
                       "--input", str(source), "--output", str(destination), "--unwrap", unwrap.value]
            job.extra["output"] = str(destination)
            view.attach(jobs.submit(job))
        except (OSError, ValueError) as exc:
            ui.notify(str(exc), type="warning")

    view = JobView(jobs, job_name=JOB_NAME, results_builder=show_result)
    theme.panel_header(JOB_NAME, "Decode TPX3 files into per-frame TOFs and GMD readings for verification.", view=view)
    with theme.panel_card():
        mode = persist(ui.toggle(["Directory", "Single file"], value="Directory"), "ion_unpack.mode")
        directory = persist(path_input("TPX3 directory (one delay position)", browse="dir"), "ion_unpack.directory")
        single_file = persist(path_input("TPX3 file", browse="file", filters=[".tpx3"]), "ion_unpack.file")
        directory.bind_visibility_from(mode, "value", lambda value: value == "Directory")
        single_file.bind_visibility_from(mode, "value", lambda value: value == "Single file")
        output = persist(path_input("Output HDF5", str(Path(shared.workspace_root()) / "tpx3_verification.h5"),
                                    browse="save", filters=[".h5", ".hdf5"]), "ion_unpack.output")
        unwrap = persist(ui.select({"signed": "Signed (default)", "go4": "Go4"}, value=UNWRAP_MODE,
                                   label="Timestamp unwrapping").props("outlined dense"), "ion_unpack.unwrap")
        with action_bar(view, "Dump TPX3", on_click=run):
            pass
    view.place_console()
