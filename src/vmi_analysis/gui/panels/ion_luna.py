"""Workbench panel for Timepix3 conversion with ASI LUNA."""

import json
from pathlib import Path
import sys

from nicegui import ui

from vmi_analysis.ion_luna import LUNA_COMMAND_PREFIX, PROCESS_OPTIONS, process_arguments
from vmi_analysis.paths import validate_distinct_output
from vmi_analysis.ion_acquisition import manifest_path, segment_files

from ..shared import checked_path, workspace_root
from .. import shared, theme
from ..jobs import Job, JobManager
from .common import (REPO_ROOT, JobView, action_bar, check_grid, field_grid, num_input,
                     open_folder_button, path_input, persist)
from .hdf5_details import file_details

JOB_NAME = "TPX3 Dump"
DUMP = "Dump TPX3"
CORRECT = "Apply timewalk correction"
HDF5_TYPES = [".h5", ".hdf5"]
OPTIONS = {option.flag: option for option in PROCESS_OPTIONS}
PANEL_DEFAULTS = {"calculate-timewalk-statistics": True, "tof-tdc-reference": "TDC1Rising", "ctot-cut": 500}
COMMAND_KEY = "luna_command"
WSL_FIELDS = {"distribution": "WSL distribution", "executable": "Linux LUNA executable"}


def wsl_settings():
    if sys.platform != "win32" or shared.SERVER:
        return {}
    store = shared.settings_store()
    return {key: store.get(f"ion_luna.{key}", "") for key in WSL_FIELDS}


def wsl_arguments(settings):
    return [f"--wsl-{key}={value.strip()}" for key, value in settings.items()
            if isinstance(value, str) and value.strip()]


def capture_command(job, line):
    prefix = "Acquisition collection ready: "
    if line.startswith(prefix):
        job.extra["collection"] = line[len(prefix):].strip()
        return False
    if line.startswith(LUNA_COMMAND_PREFIX):
        job.extra[COMMAND_KEY] = line
        return True
    return False


def dump_job(source, output, *, prefixed=True, options=None, timewalk=None, wsl=()):
    cwd = workspace_root(REPO_ROOT)
    selected = checked_path(source, cwd=cwd)
    matrix = checked_path(timewalk, cwd=cwd) if timewalk else None
    target = validate_distinct_output(output, [("TPX3 input", selected), ("timewalk file", matrix)],
                                      cwd=cwd, output_label="HDF5 output")
    target = checked_path(target, write=True, cwd=cwd)
    options = options or {}
    cmd = [sys.executable, "-m", "vmi_analysis.ion_luna", "--input", str(selected),
           "--output", str(target), *process_arguments(options)]
    extra = {"output": str(target)}
    if prefixed:
        if options.get("layout", "single") != "single":
            raise ValueError("The 32-byte acquisition format requires a single-chip layout.")
        metadata = target.with_name(f"{target.stem}_metadata.h5")
        cmd += ["--metadata-output", str(metadata), "--split-acquisitions"]
        extra["metadata"] = str(metadata)
    if matrix:
        cmd += ["--timewalk-file", str(matrix)]
    return Job(JOB_NAME, cmd=[*cmd, *wsl], cwd=str(cwd), extra=extra, consume_output=capture_command)


def build(jobs: JobManager):
    def results(job):
        if job.status == "done":
            collection = job.extra.get("collection")
            path = Path(collection or job.extra["output"])
            if collection:
                ui.label("Each acquisition is an independent LUNA stream at the same experimental condition."
                         ).classes("vmi-subtitle")
                for index, (data, meta) in enumerate(segment_files(path), 1):
                    ui.label(f"Segment {index} · {data}").classes("vmi-caption break-all")
                    ui.button(f"Inspect segment {index}", on_click=lambda data=data: inspect_output(data)).props("flat")
                document = json.loads(path.read_text(encoding="utf-8"))
                for index, segment in enumerate(document["segments"], 1):
                    ui.label(f"Segment {index} · {segment['reason']}").classes("vmi-subtitle")
            label = "Corrected HDF5" if job.extra.get("corrected") else "LUNA output"
            ui.label(f"{label}: {path}").classes("break-all")
            metadata = None if collection else job.extra.get("metadata")
            if metadata:
                ui.label(f"Acquisition metadata: {metadata}").classes("break-all")
            with ui.row().classes("w-full items-center"):
                async def inspect_data():
                    await inspect_output(path)

                async def inspect_metadata():
                    await inspect_output(metadata)

                if not collection:
                    ui.button("Inspect output", on_click=inspect_data).props("flat")
                if metadata:
                    ui.button("Inspect metadata", on_click=inspect_metadata).props("flat")
                open_folder_button(path.parent, "Open output folder")

    def restore_form(job):
        """Show an Agent run's resolved inputs, as if the user had entered them."""
        form = job.extra.get("panel_form") if job.extra.get("agent_panel") else None
        if not isinstance(form, dict) or not form:
            return
        if not job.extra.get("corrected"):
            stage.value = DUMP
        for key, widget in (("input_mode", file_mode), ("input", source), ("output", output),
                            ("acquisition_prefix", prefixed), ("timewalk_file", timewalk),
                            ("correction_input", correction_input),
                            ("corrected_output", corrected_output)):
            if key in form:
                widget.value = form[key]
        for flag, value in form.items():
            if flag in fields:
                fields[flag].value = value

    view = JobView(jobs, job_name=JOB_NAME, results_builder=results, command_key=COMMAND_KEY,
                   attach_builder=restore_form)
    theme.panel_header(JOB_NAME, "Unpack and cluster Timepix3 data or apply a timewalk correction.", view=view)
    fields = {}

    def save(widget, key):
        return persist(widget, f"ion_luna.{key}")

    def option_field(flag):
        option = OPTIONS[flag]
        default = PANEL_DEFAULTS.get(flag, option.default)
        if option.choices:
            choices = list(option.choices) if option.default is not None else {"": "None", **{
                choice: choice for choice in option.choices
            }}
            widget = ui.select(choices, value=default, label=option.label).props("outlined dense")
        elif isinstance(default, bool):
            widget = ui.checkbox(option.label, value=default)
        elif type(default) is int:
            widget = num_input(option.label, default, step=1, min=1 if flag == "kuttruff-buffer-size" else 0)
        else:
            widget = ui.input(option.label, value=default).props("outlined dense")
        fields[flag] = save(widget, flag)
        return widget

    def cluster_enabled():
        return not (fields["raw-only"].value or fields["disable-clustering"].value)

    def calculate_stats():
        return cluster_enabled() and fields["calculate-timewalk-statistics"].value

    def option_value(flag):
        value = fields[flag].value
        if type(OPTIONS[flag].default) is int:
            if value is None or int(value) != value:
                raise ValueError(f"Enter a whole number for {OPTIONS[flag].label}.")
            return int(value)
        return value

    def wsl_options():
        if sys.platform == "win32" and not shared.SERVER:
            return wsl_arguments({key: field.value for key, field in wsl_fields.items()})
        return []

    def run():
        command = [sys.executable, "-m", "vmi_analysis.ion_luna"]
        try:
            if stage.value == CORRECT:
                input_value = correction_input.value.strip() or output.value.strip()
                if not input_value:
                    raise ValueError("Choose the HDF5 file with timewalk statistics.")
                selected = (workspace_root(REPO_ROOT) / Path(input_value).expanduser()).resolve()
                if not correction_input.value.strip() and manifest_path(selected).is_file():
                    selected = manifest_path(selected)
                target = corrected_output.value.strip() or selected.with_name(f"{selected.stem}_corrected.h5")
                target = validate_distinct_output(target, [("timewalk input", selected)],
                                                  cwd=workspace_root(REPO_ROOT), output_label="corrected HDF5 output")
                cmd = [*command, "--apply-timewalk-correction", "--input", str(selected), "--output", str(target),
                       "--ctot-cut", str(option_value("ctot-cut"))]
                view.attach(jobs.submit(Job(JOB_NAME, cmd=[*cmd, *wsl_options()],
                                            cwd=str(workspace_root(REPO_ROOT)),
                                            extra={"output": str(target), "corrected": True},
                                            consume_output=capture_command)))
                return
            if not source.value.strip() or not output.value.strip():
                raise ValueError("Choose a TPX3 input and an HDF5 output.")
            stats = calculate_stats()
            arguments = {flag: option_value(flag) for flag in ("layout", "raw-only", "disable-clustering")}
            if not arguments["raw-only"]:
                arguments["tof-tdc-reference"] = option_value("tof-tdc-reference")
            if cluster_enabled():
                arguments.update({flag: option_value(flag) for flag in ("clustering-algorithm", "eps-t", "eps-s")})
                if arguments["clustering-algorithm"] == "Kuttruff2024":
                    arguments["kuttruff-buffer-size"] = option_value("kuttruff-buffer-size")
            if stats:
                arguments.update({"calculate-timewalk-statistics": True, "ctot-cut": option_value("ctot-cut")})
            job = dump_job(source.value.strip(), output.value.strip(), prefixed=prefixed.value,
                           options=arguments, timewalk=timewalk.value.strip() if stats else None,
                           wsl=wsl_options())
            view.attach(jobs.submit(job))
        except (OSError, TypeError, ValueError) as exc:
            ui.notify(str(exc), type="warning")
            return

    async def browse_input():
        from .picker import LocalFilePicker

        folder = file_mode.value == "Folder"
        chosen = await LocalFilePicker(start=source.value, mode="dir" if folder else "file",
                                       filters=None if folder else [".tpx3"])
        if chosen:
            source.value = chosen

    with theme.panel_card():
        stage = ui.toggle([DUMP, CORRECT], value=DUMP)
        with ui.column().classes("w-full").bind_visibility_from(stage, "value", lambda value: value == DUMP):
            theme.form_section("Input and output")
            file_mode = save(ui.toggle(["File", "Folder"], value="File"), "input_mode")
            source = save(path_input("TPX3 input"), "input")
            with source.add_slot("append"):
                ui.button(icon="folder_open", on_click=browse_input).classes("vmi-browse-icon") \
                    .props('flat dense round aria-label="Browse TPX3 input"')
            output = save(path_input("HDF5 output", str(workspace_root(REPO_ROOT) / "ions.h5"), browse="save",
                                     filters=HDF5_TYPES, default_name="ions.h5"), "output")
            ui.label("For split acquisitions, this name sets the collection name and segment folder. "
                     "For example, ions.h5 creates ions.segments.json and separate HDF5 files in ions_segments."
                     ).classes("vmi-subtitle")
            prefixed = save(ui.checkbox("Strip 32-byte acquisition prefix", value=True), "acquisition_prefix")
            prefixed.tooltip("For single-chip recordings with timestamp and GMD metadata. Clear for standard ASI Serval files.")
            ui.label("Prefixed data are checked for acquisition restarts. A backward TDC clock jump starts "
                     "a new LUNA stream when the rollover interval disagrees with wall time by more than 1 s. "
                     "Each segment is saved as a separate HDF5 with matching metadata. Open the saved "
                     ".segments.json collection in Ion TOF, or use its latest-dump button."
                     ).classes("vmi-subtitle").bind_visibility_from(prefixed, "value")
            theme.form_section("Detector and timing")
            with field_grid():
                option_field("layout")
                reference = option_field("tof-tdc-reference")
            theme.form_section("Clustering")
            with check_grid():
                option_field("disable-clustering")
                option_field("raw-only")
            with ui.column().classes("w-full") as clustering_options:
                with field_grid():
                    algorithm = option_field("clustering-algorithm")
                    buffer_size = option_field("kuttruff-buffer-size")
                    buffer_size.bind_visibility_from(algorithm, "value", lambda value: value == "Kuttruff2024")
                    buffer_size.tooltip("Number of recent clusters Kuttruff compares with each incoming pixel hit.")
                with field_grid():
                    option_field("eps-t")
                    option_field("eps-s")
            theme.form_section("Timewalk statistics")
            with ui.column().classes("w-full") as stats_options:
                stats_checkbox = option_field("calculate-timewalk-statistics")
                timewalk = save(path_input("Seed with existing statistics HDF5 (optional)", browse="file",
                                           filters=HDF5_TYPES), "timewalk_file")
                timewalk.bind_visibility_from(stats_checkbox, "value")
            stats_unavailable = ui.label("Timewalk statistics require clustering.").classes("vmi-subtitle")
        with ui.column().classes("w-full").bind_visibility_from(stage, "value", lambda value: value == CORRECT):
            theme.form_section("Timewalk correction input and output")
            correction_input = save(path_input("Statistics HDF5 (blank = dump output)", browse="file",
                                               filters=[*HDF5_TYPES, ".json"]), "correction_input")
            corrected_output = save(path_input("Corrected HDF5 (blank = append _corrected)", browse="save",
                                               filters=HDF5_TYPES, default_name="ions_corrected.h5"), "corrected_output")
        with field_grid() as cut_options:
            option_field("ctot-cut")

        def update_options():
            clustering = cluster_enabled()
            clustering_options.set_visibility(clustering)
            reference.set_enabled(not fields["raw-only"].value)
            stats_options.set_visibility(clustering)
            stats_unavailable.set_visibility(not clustering)
            cut_options.set_visibility(stage.value == CORRECT or calculate_stats())

        for widget in (stage, fields["raw-only"], fields["disable-clustering"], stats_checkbox):
            widget.on_value_change(update_options)
        update_options()
        if sys.platform == "win32" and not shared.SERVER:
            with ui.expansion("WSL2 settings", icon="settings").classes("w-full"):
                with field_grid():
                    wsl_fields = {
                        key: save(ui.input(f"{label} (blank = {fallback})").props("outlined dense"), key)
                        for (key, label), fallback in zip(WSL_FIELDS.items(), ("default", "PATH"), strict=True)
                    }
        with action_bar(view, on_click=run):
            pass
    view.place_console()
    inspect_output = file_details(jobs)
