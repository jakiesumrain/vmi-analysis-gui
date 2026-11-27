"""Workbench panel for Timepix3 conversion with ASI LUNA."""

import json
from functools import partial
from pathlib import Path
import sys

from nicegui import ui

from vmi_analysis.ion_luna import LUNA_COMMAND_PREFIX, PROCESS_OPTIONS, input_files, process_arguments
from vmi_analysis.paths import validate_distinct_output
from vmi_analysis.ion_acquisition import manifest_path, segment_files

from ..shared import checked_path, workspace_root
from .. import shared, theme
from ..agent_panels import PanelOperation, boolean, resolve_path
from ..jobs import Job, JobManager, Stage
from ..resources import command_access, overlaps
from .common import (REPO_ROOT, JobView, action_bar, adopt_agent_form, check_grid, field_grid, num_input,
                     open_folder_button, path_input, persist)
from .hdf5_details import file_details

JOB_NAME = "TPX3 Dump"
TIMEWALK_WORKFLOW = "TPX3 Timewalk Calibration"
DUMP = "Dump TPX3"
CORRECT = "Apply timewalk correction"
HDF5_TYPES = [".h5", ".hdf5"]
OPTIONS = {option.flag: option for option in PROCESS_OPTIONS}
PANEL_DEFAULTS = {"calculate-timewalk-statistics": True, "tof-tdc-reference": "TDC1Rising", "ctot-cut": 500}
COMMAND_KEY = "luna_command"
WSL_FIELDS = {"distribution": "WSL distribution", "executable": "Linux LUNA executable"}


DUMP_OPTIONS = (
    "layout", "raw-only", "disable-clustering", "tof-tdc-reference",
    "clustering-algorithm", "eps-t", "eps-s", "kuttruff-buffer-size",
    "calculate-timewalk-statistics", "ctot-cut",
)


def _dump_artifacts(job):
    return ("collection",) if job.extra.get("collection") else ("output", "metadata")


def planned_dump_input(job):
    output = Path(job.extra["output"])
    return manifest_path(output) if "--split-acquisitions" in job.cmd else output


def completed_dump_input(job):
    extra = job.extra
    return extra.get("collection", extra["output"]), extra.get("collection", extra.get("metadata"))


def correction_job(source, output=None, *, ctot_cut=500, wsl=(), pending_input=False):
    cwd = workspace_root(REPO_ROOT)
    selected = checked_path(source, cwd=cwd)
    if not pending_input and not selected.is_file():
        raise ValueError("Choose an existing HDF5 file or acquisition collection with timewalk statistics.")
    if type(ctot_cut) is not int or ctot_cut < 0:
        raise ValueError("CTOT cut must be a nonnegative whole number.")
    target = output or selected.with_name(f"{selected.stem}_corrected.h5")
    target = validate_distinct_output(target, [("timewalk input", selected)], cwd=cwd,
                                      output_label="corrected HDF5 output")
    target = checked_path(target, write=True, cwd=cwd)
    if target.suffix.lower() not in HDF5_TYPES:
        raise ValueError("Choose an .h5 or .hdf5 corrected output path.")
    extra = {"output": str(target), "corrected": True,
             "panel_form": {"correction_input": str(selected), "corrected_output": str(target),
                            "ctot-cut": ctot_cut}}
    if selected.suffix.lower() == ".json":
        inputs = [("input collection", selected)]
        if not pending_input:
            inputs.extend((str(path), path) for pair in segment_files(selected) for path in pair)
        collection = validate_distinct_output(manifest_path(target), inputs, cwd=cwd,
                                              output_label="corrected collection")
        extra["collection"] = str(collection)
    cmd = [sys.executable, "-m", "vmi_analysis.ion_luna", "--apply-timewalk-correction",
           "--input", str(selected), "--output", str(target), "--ctot-cut", str(ctot_cut), *wsl]
    return Job(JOB_NAME, cmd=cmd, cwd=str(cwd), extra=extra, consume_output=capture_command)


def _prepare_correction(_jobs, params):
    source = resolve_path(params["input"])
    output = resolve_path(params["output"], write=True) if params.get("output") else None
    ctot_cut = params.get("ctot_cut", PANEL_DEFAULTS["ctot-cut"])
    if type(ctot_cut) is float and ctot_cut.is_integer():
        ctot_cut = int(ctot_cut)
    wsl = wsl_settings()
    job = correction_job(source, output, ctot_cut=ctot_cut, wsl=wsl_arguments(wsl))
    summary = {"Timewalk input": str(source), "Corrected HDF5 output": job.extra["output"],
               "CTOT cut": ctot_cut}
    if "collection" in job.extra:
        summary["Corrected collection"] = job.extra["collection"]
    summary.update({WSL_FIELDS[key]: value.strip() for key, value in wsl.items()
                    if isinstance(value, str) and value.strip()})
    return job, summary


def agent_operations():
    description = {
        "panel": JOB_NAME,
        "summary": "Decode TPX3 files with LUNA. Acquisition prefixes supply timestamps and GMD readings. "
                   "Detected clock resets split independent streams. Output is an HDF5 file or a collection.",
        "parameters": {
            "input": {"type": "path", "description": "TPX3 file or directory"},
            "output": {"type": "path", "description": "Workspace HDF5 file"},
            "prefixed": {"type": "boolean", "default": True},
            "timewalk": {"type": "path", "description": "Optional existing statistics HDF5 seed"},
            "options": {"type": "object", "default": {
                key: PANEL_DEFAULTS.get(key, OPTIONS[key].default) for key in DUMP_OPTIONS
            }},
        },
        "option_choices": {key: list(OPTIONS[key].choices) for key in DUMP_OPTIONS if OPTIONS[key].choices},
        "outputs": ["collection", "output", "metadata"],
        "manual": JOB_NAME,
    }
    correction = {
        "panel": JOB_NAME,
        "summary": "Apply the panel's LUNA timewalk correction to an existing statistics HDF5 file or acquisition collection.",
        "parameters": {
            "input": {"type": "path", "description": "Existing HDF5 file or acquisition collection with timewalk statistics"},
            "output": {"type": "path", "description": "Optional corrected workspace HDF5 file; defaults to _corrected.h5"},
            "ctot_cut": {"type": "integer", "default": PANEL_DEFAULTS["ctot-cut"]},
        },
        "outputs": ["output"],
        "manual": JOB_NAME,
    }
    return (PanelOperation("tpx3_dump", description, _prepare_dump, (JOB_NAME,), artifacts=_dump_artifacts),
            PanelOperation("tpx3_timewalk", correction, _prepare_correction, (),
                           proposal_job_names=(JOB_NAME,)))


def _active_dump_options(value):
    options = {flag: value(flag) for flag in ("layout", "raw-only", "disable-clustering")}
    if not options["raw-only"]:
        options["tof-tdc-reference"] = value("tof-tdc-reference")
    if not (options["raw-only"] or options["disable-clustering"]):
        options.update({flag: value(flag) for flag in ("clustering-algorithm", "eps-t", "eps-s")})
        if options["clustering-algorithm"] == "Kuttruff2024":
            options["kuttruff-buffer-size"] = value("kuttruff-buffer-size")
        options["calculate-timewalk-statistics"] = value("calculate-timewalk-statistics")
        if options["calculate-timewalk-statistics"]:
            options["ctot-cut"] = value("ctot-cut")
    return options


def _prepare_dump(_jobs, params, *, calibration=False, pending_seed=False):
    source = resolve_path(params["input"])
    if not source.is_dir() and not (source.is_file() and source.suffix.lower() == ".tpx3"):
        raise ValueError("Choose an existing TPX3 file or directory.")
    target = resolve_path(params["output"], write=True)
    if target.suffix.lower() not in HDF5_TYPES:
        raise ValueError("Choose an .h5 or .hdf5 output path.")
    prefixed = boolean(params.get("prefixed", True))
    requested = params.get("options", {})
    if not isinstance(requested, dict) or requested.keys() - set(DUMP_OPTIONS):
        raise ValueError("Use the TPX3 Dump options listed by describe.")
    values = {}
    for key in DUMP_OPTIONS:
        option = OPTIONS[key]
        default = PANEL_DEFAULTS.get(key, option.default)
        value = requested.get(key, default)
        if type(value) is not type(default) or (option.choices and value not in option.choices):
            raise ValueError(f"Choose a valid value for {option.label}.")
        if type(value) is int and value < (1 if key == "kuttruff-buffer-size" else 0):
            raise ValueError(f"{option.label} is below the permitted minimum.")
        if isinstance(value, str) and (not value.strip() or value.startswith("-") or "\x00" in value):
            raise ValueError(f"Choose a valid value for {option.label}.")
        values[key] = value
    values = _active_dump_options(values.__getitem__)
    matrix = resolve_path(params["timewalk"]) if params.get("timewalk") else None
    if calibration and not values.get("calculate-timewalk-statistics"):
        raise ValueError("Timewalk calibration requires clustering and timewalk statistics.")
    if matrix and (not values.get("calculate-timewalk-statistics") or (not pending_seed and not matrix.is_file())):
        raise ValueError("A timewalk seed requires statistics calculation and an existing HDF5 file.")
    wsl = wsl_settings()
    job = dump_job(source, target, prefixed=prefixed, options=values, timewalk=matrix,
                   wsl=wsl_arguments(wsl), calibration=calibration)
    summary = {"TPX3 input": str(source), "Strip acquisition prefix": prefixed,
               "Split detected acquisitions": prefixed and not calibration, "Timewalk seed": str(matrix) if matrix else None}
    if prefixed and not calibration:
        summary["Collection output"] = str(manifest_path(target))
        summary["Segment folder"] = str(target.with_name(f"{target.stem}_segments"))
    else:
        summary["HDF5 output"] = str(target)
        if prefixed:
            summary["Acquisition metadata"] = job.extra["metadata"]
    summary.update({OPTIONS[key].label: value for key, value in values.items()})
    summary.update({WSL_FIELDS[key]: value.strip() for key, value in wsl.items()
                    if isinstance(value, str) and value.strip()})
    job.extra["panel_form"] = {
        "input_mode": "Folder" if source.is_dir() else "File",
        "input": str(source), "output": str(target), "acquisition_prefix": prefixed,
        **values, "timewalk_file": str(matrix) if matrix else "",
    }
    return job, summary


def timewalk_workflow_operation(registry):
    description = {
        "panel": TIMEWALK_WORKFLOW,
        "summary": "Build a master LUNA timewalk LUT from ordered uninterrupted TPX3 recordings, "
                   "using each accumulated HDF5 file as the next seed. Optionally dump a separate target "
                   "with the master seed and apply its updated LUT.",
        "parameters": {
            "recordings": {"type": "array", "description": "Ordered list of distinct TPX3 files or directories."},
            "output": {"type": "path", "description": "Workspace path for the final master HDF5."},
            "prefixed": {"type": "boolean", "default": True, "description": "Whether inputs have the 32-byte acquisition prefix."},
            "options": {"type": "object", "description": "Common tpx3_dump options for all dumps. Keep clustering "
                                                         "and timewalk statistics enabled."},
            "target": {"type": "object", "description": "Optional separate recording. Provide input, dump_output "
                                                        "(uncorrected .h5 path), and optionally output "
                                                        "(corrected .h5 path, defaults to an _corrected name)."},
        },
    }
    return PanelOperation("tpx3_timewalk_calibration", description, partial(_prepare_timewalk_workflow, registry),
                          (TIMEWALK_WORKFLOW,), outputs=("master_calibration",), workflow_panels=(JOB_NAME,))


def _prepare_timewalk_workflow(registry, jobs, params):
    recordings = params.get("recordings")
    if not isinstance(recordings, list) or not recordings:
        raise ValueError("Provide an ordered list of one or more calibration recordings.")
    target = params.get("target")
    if "target" in params and (not isinstance(target, dict) or not {"input", "dump_output"} <= target.keys()
                               or target.keys() - {"input", "dump_output", "output"}):
        raise ValueError("Provide target input and dump_output, with an optional corrected output.")

    # Reject the same raw frames even if a file is also specified via its parent directory.
    selected_files = set()
    for entry in [*recordings, *([target["input"]] if target else [])]:
        source = resolve_path(entry)
        files = {checked_path(path) for path in input_files(source)}
        if files & selected_files:
            raise ValueError("A TPX3 file is selected more than once. Use disjoint calibration and target inputs.")
        selected_files.update(files)

    master = resolve_path(params.get("output"), write=True)
    common = {"prefixed": params.get("prefixed", True), "options": params.get("options", {})}
    stages = []
    reserved_outputs = set()
    summary = {}

    def add_stage(label, step):
        access = command_access(step.cmd, step.cwd, prior_outputs=reserved_outputs)
        if any(overlaps(path, earlier) for path in access.writes for earlier in reserved_outputs):
            raise ValueError("Choose distinct output paths for all workflow stages, including generated sidecars.")
        reserved_outputs.update(access.writes)
        stages.append(Stage(label, step.cmd, step.cwd, result=step))

    seed = None
    for index, entry in enumerate(recordings, 1):
        output = master if index == len(recordings) else master.with_name(f"{master.stem}_step_{index:03d}{master.suffix}")
        step, settings = registry.prepare_stage(
            jobs, "tpx3_dump", {**common, "input": entry, "output": str(output), "timewalk": seed},
            calibration=True, pending_seed=seed is not None,
        )
        label = f"Calibration {index}"
        add_stage(label, step)
        seed = step.extra["output"]
        summary[label] = {"input": settings["TPX3 input"], "seed": settings["Timewalk seed"],
                          "output": seed}
        if "metadata" in step.extra:
            summary[label]["metadata"] = step.extra["metadata"]

    summary["Common dump options"] = {label: settings[label] for label in settings
                                      if label not in {"TPX3 input", "Timewalk seed", "HDF5 output",
                                                       "Acquisition metadata", "Split detected acquisitions"}}
    if target:
        dump, dump_summary = registry.prepare_stage(
            jobs, "tpx3_dump", {**common, "input": target["input"], "output": target["dump_output"],
                              "timewalk": str(master)}, pending_seed=True,
        )
        add_stage("Target dump", dump)
        target_input = planned_dump_input(dump)
        dump_output = Path(dump.extra["output"])
        corrected_output = (resolve_path(target["output"], write=True) if "output" in target
                            else dump_output.with_name(f"{dump_output.stem}_corrected.h5"))
        corrected = correction_job(target_input, corrected_output, ctot_cut=dump.extra["panel_form"]["ctot-cut"],
                                   wsl=wsl_arguments(wsl_settings()), pending_input=True)
        corrected.extra["agent_panel"] = True
        add_stage("Target correction", corrected)
        summary["Target dump"] = {"input": dump_summary["TPX3 input"], "seed": str(master),
                                  "output": str(target_input), "split acquisitions": common["prefixed"]}
        summary["Target correction"] = {"input": str(target_input),
                                        "output": corrected.extra.get("collection", corrected.extra["output"]),
                                        "LUT": "Includes the target recording's statistics."}

    return Job(TIMEWALK_WORKFLOW, cwd=stages[0].cwd, stages=stages, extra={"master_calibration": str(master)}), summary


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


def dump_job(source, output, *, prefixed=True, options=None, timewalk=None, wsl=(), calibration=False):
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
        cmd += ["--metadata-output", str(metadata)]
        if not calibration:
            cmd.append("--split-acquisitions")
        extra["metadata"] = str(metadata)
    if calibration:
        cmd.append("--timewalk-calibration")
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
        form = adopt_agent_form(job, "ion_luna")
        if not form:
            return
        stage.value = CORRECT if job.extra.get("corrected") else DUMP
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
        try:
            if stage.value == CORRECT:
                input_value = correction_input.value.strip() or output.value.strip()
                if not input_value:
                    raise ValueError("Choose the HDF5 file with timewalk statistics.")
                selected = (workspace_root(REPO_ROOT) / Path(input_value).expanduser()).resolve()
                if not correction_input.value.strip() and manifest_path(selected).is_file():
                    selected = manifest_path(selected)
                job = correction_job(selected, corrected_output.value.strip() or None,
                                     ctot_cut=option_value("ctot-cut"), wsl=wsl_options())
                view.attach(jobs.submit(job))
                return
            if not source.value.strip() or not output.value.strip():
                raise ValueError("Choose a TPX3 input and an HDF5 output.")
            arguments = _active_dump_options(option_value)
            job = dump_job(source.value.strip(), output.value.strip(), prefixed=prefixed.value,
                           options=arguments, wsl=wsl_options(),
                           timewalk=timewalk.value.strip() if arguments.get("calculate-timewalk-statistics") else None)
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
