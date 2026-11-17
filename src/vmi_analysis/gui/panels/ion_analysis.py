"""Connect the temporary TPX3 analysis commands to the Ion workbench."""

from fractions import Fraction
from math import isfinite
from pathlib import Path
import sys

from nicegui import ui

from vmi_analysis.tpx3_analysis import FRAME_SHIFT, GMD2_DEFAULT, SKIP_DIRS, parse_delay_from_name

from .. import shared, theme
from ..jobs import Job
from .common import (JobView, action_bar, field_grid, file_url, num_input,
                     open_folder_button, path_input, persist, zoomable_image)


JOB_NAME = "TPX3 Analysis"
TASKS = {
    "verify": "Verify reader",
    "all": "Run all analyses",
    "gmd": "GMD diagnostics",
    "tof": "Time-of-flight spectra",
    "yield": "Ion yield",
    "erf": "Fit rising edge",
}


def _dataset_paths(root):
    """Reserve the position folders whose TPX3 files the script will scan."""
    yield root
    if shared.SERVER:
        for position in root.iterdir():
            if position.name not in SKIP_DIRS and parse_delay_from_name(position.name) is not None:
                if position.is_dir():
                    yield shared.checked_path(position)


def _results(job):
    if job.status != "done":
        return
    if not job.extra["output_dirs"]:
        ui.label("See the console for the reader verification result.")
        return
    for directory in map(Path, job.extra["output_dirs"]):
        ui.label(str(directory)).classes("vmi-caption break-all")
        if not shared.SERVER:
            open_folder_button(directory, "Open output folder")
        outputs = sorted(path for path in directory.glob("*")
                         if path.suffix.lower() in {".png", ".csv"} and path.is_file()
                         and job.started <= path.stat().st_mtime <= job.finished)
        if not outputs:
            ui.label("No new plots or CSV files. See the console for details.")
        for path in outputs:
            url = file_url(path)
            if path.suffix.lower() == ".csv":
                ui.button(f"Download {path.name}", icon="download",
                          on_click=lambda url=url, name=path.name: ui.download.from_url(url, filename=name)).props("flat")
            else:
                with ui.expansion(path.name, icon="show_chart").classes("w-full"):
                    zoomable_image(url).classes("w-full max-w-4xl")


def build(jobs):
    view = JobView(jobs, job_name=JOB_NAME, results_builder=_results)
    theme.panel_header(JOB_NAME, "Temporary panel for Timepix3 verification, diagnostics, spectra, yield and fits.", view=view)

    def save(widget, key):
        return persist(widget, f"ion_analysis.{key}")

    def path(label, key, mode="dir", filters=None, default=""):
        return save(path_input(label, default, browse=mode, filters=filters), key)

    def visible(element, *tasks):
        return element.bind_visibility_from(task, "value", lambda value: value in tasks)

    def run():
        selected = task.value
        cwd = shared.workspace_root()
        args = [selected]
        reads, writes, output_dirs = [], [], []

        def input_path(widget, label, *, directory=False, optional=False):
            value = widget.value.strip()
            if not value and optional:
                return None
            if not value:
                raise ValueError(f"Choose {label}.")
            candidate = shared.checked_path(value, cwd=cwd)
            if not (candidate.is_dir() if directory else candidate.is_file()):
                raise ValueError(f"Could not find {label}: {candidate}")
            return candidate

        try:
            if selected == "verify":
                source = input_path(verify_file, "a reference TPX3 file", optional=not shared.SERVER)
                if source:
                    args += ["--sample", str(source)]
                    reads.append(source)
            elif selected == "all":
                for flag, widget in (("odl1", odl1), ("odl2", odl2)):
                    source = input_path(widget, f"the {flag.upper()} folder", directory=True, optional=True)
                    if source:
                        args += ["--" + flag, str(source)]
                        reads.extend(_dataset_paths(source))
                        output_dirs.append(flag.upper())
                if not output_dirs:
                    raise ValueError("Choose at least one ODL dataset.")
            elif selected == "erf":
                source = input_path(csv_file, "a yield CSV file")
                args += ["--csv", str(source)]
                reads.append(source)
            else:
                root = input_path(dataset, "the dataset folder", directory=True)
                args += ["--root", str(root)]
                reads.extend(_dataset_paths(root))

            if selected != "verify":
                if not output.value.strip():
                    raise ValueError("Choose an output folder.")
                target = shared.checked_path(output.value.strip(), write=True, cwd=cwd)
                if target.exists() and not target.is_dir():
                    raise ValueError("Choose a folder for the output.")
                args += ["--out", str(target)]
                output_dirs = [target / name for name in output_dirs] if selected == "all" else [target]
                writes.extend(output_dirs)
            if selected in {"gmd", "tof", "all"} and sample_folder.value.strip():
                args += ["--sample", sample_folder.value.strip()]
            if selected in {"tof", "yield", "erf"}:
                name = label.value.strip()
                if name:
                    if name in {".", ".."} or any(char in '/\\:*?"<>|' or ord(char) < 32 for char in name):
                        raise ValueError("Enter a label without path separators or special characters.")
                    args += ["--label", name]
            if selected in {"tof", "yield"}:
                if no_gmd.value:
                    args.append("--no-gmd")
                else:
                    bounds = gmd_min.value, gmd_max.value
                    if any(value is None or not isfinite(value) for value in bounds) or bounds[0] >= bounds[1]:
                        raise ValueError("Enter a GMD minimum smaller than the maximum.")
                    args += ["--gmd-min", str(bounds[0]), "--gmd-max", str(bounds[1])]
                if shift.value is None or not float(shift.value).is_integer():
                    raise ValueError("Enter a whole number for the GMD frame shift.")
                args += ["--shift", str(int(shift.value))]
            if selected == "yield" and fixed_window.value:
                args.append("--fixed-window")
            if selected in {"erf", "all"}:
                portion = fraction.value.strip()
                if "/" not in portion or not 0 < Fraction(portion) <= 1:
                    raise ValueError("Enter a fit fraction such as 1/3 or 1/1.")
                if nboot.value is None or not float(nboot.value).is_integer() or nboot.value < 1:
                    raise ValueError("Enter a positive whole number of bootstrap samples.")
                args += ["--fraction", portion, "--nboot", str(int(nboot.value))]

            job = jobs.submit(Job(JOB_NAME, cmd=[sys.executable, "-m", "vmi_analysis.tpx3_analysis", *args],
                                  cwd=str(cwd), reads=reads, writes=writes,
                                  extra={"output_dirs": [str(directory) for directory in output_dirs]}))
        except (OSError, ValueError, ZeroDivisionError) as exc:
            ui.notify(str(exc), type="warning")
            return
        view.attach(job)

    with theme.panel_card():
        task = save(ui.select(TASKS, value="verify", label="Task").props("outlined dense").classes("w-full"), "task")
        verify_label = "Reference TPX3 file" if shared.SERVER else "Reference TPX3 file (optional)"
        verify_file = visible(path(verify_label, "verify_file", "file", [".tpx3"]), "verify")
        odl1 = visible(path("ODL1 dataset folder", "odl1"), "all")
        odl2 = visible(path("ODL2 dataset folder", "odl2"), "all")
        dataset = visible(path("Dataset folder containing delay positions", "dataset"), "gmd", "tof", "yield")
        csv_file = visible(path("Yield CSV file", "csv", "file", [".csv"]), "erf")
        output = visible(path("Output folder", "output", default=str(shared.workspace_root() / "tpx3_analysis_out")),
                         "all", "gmd", "tof", "yield", "erf")
        visible(ui.label("Choose ODL1, ODL2, or both. A full run can take an hour on large datasets.")
                .classes("vmi-subtitle"), "all")
        with visible(ui.column().classes("w-full gap-3"), "tof", "yield"):
            theme.form_section("GMD filter")
            no_gmd = save(ui.checkbox("Use all frames"), "no_gmd")
            with field_grid():
                gmd_min = save(num_input("GMD minimum (nA)", GMD2_DEFAULT[0], step=0.1), "gmd_min")
                gmd_max = save(num_input("GMD maximum (nA)", GMD2_DEFAULT[1], step=0.1), "gmd_max")
            for field in (gmd_min, gmd_max):
                field.bind_enabled_from(no_gmd, "value", lambda value: not value)
        with visible(ui.column().classes("w-full gap-3"), "erf", "all"):
            theme.form_section("Rising-edge fit")
            with field_grid():
                fraction = save(path_input("Fit fraction", "1/3"), "fraction")
                nboot = save(num_input("Bootstrap samples", 4000, min=1, step=1), "nboot")
        with visible(ui.expansion("Analysis options", icon="tune").classes("w-full"), "all", "gmd", "tof", "yield", "erf"):
            sample_folder = visible(save(path_input("Delay-position folder name (optional)", ""), "sample_folder"),
                                    "all", "gmd", "tof")
            label = visible(save(path_input("Output label (optional)", ""), "label"), "tof", "yield", "erf")
            shift = visible(save(num_input("GMD frame shift", FRAME_SHIFT, step=1), "shift"), "tof", "yield")
            fixed_window = visible(save(ui.checkbox("Use fixed TOF windows"), "fixed_window"), "yield")
        with action_bar(view, on_click=run):
            pass
    view.place_console()
