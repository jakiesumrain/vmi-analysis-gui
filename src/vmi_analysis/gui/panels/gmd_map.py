"""Panel: build_gmd_map.py — frame → GMD correspondence map."""

import re
import sys

from nicegui import ui
from vmi_analysis.gmd.build_gmd_map import (
    DEFAULT_H5_PATTERN, DEFAULT_N_FILES, DEFAULT_OUTPUT_PATH, validate_h5_pattern,
)
from vmi_analysis.paths import validate_distinct_output

from .. import shared, theme
from ..agent_panels import PanelOperation, resolve_path
from ..jobs import Job, JobManager
from .common import (JobView, REPO_ROOT, action_bar, adopt_agent_form, confirm_clear_outputs,
                     field_grid, file_url, num_input, open_folder_button, path_input, persist)


JOB_NAME = "GMD map"
PANEL_TITLE = "GMD Map"


def _capture_progress(job, line):
    if match := re.fullmatch(r"Scanned HDF5 files: (\d+)/(\d+)", line):
        scanned, total = map(int, match.groups())
        job.progress = scanned / total
    elif match := re.fullmatch(r"Frames: (\d+)", line):
        job.extra["frame_count"] = int(match[1])
    return False


def gmd_map_job(h5_dir, gmd_path, *, output_path=None, h5_pattern=DEFAULT_H5_PATTERN, n_files=DEFAULT_N_FILES):
    cwd = shared.workspace_root(REPO_ROOT)
    if not h5_dir or not str(h5_dir).strip():
        raise ValueError("Choose an HDF5 directory.")
    if not gmd_path or not str(gmd_path).strip():
        raise ValueError("Choose a GMD timestamp file.")
    directory = shared.checked_path(h5_dir, cwd=cwd)
    gmd = shared.checked_path(gmd_path, cwd=cwd)
    if not directory.is_dir():
        raise ValueError("Choose an existing HDF5 directory.")
    if not gmd.is_file():
        raise ValueError("Choose an existing GMD timestamp file.")
    pattern = h5_pattern.strip() if isinstance(h5_pattern, str) else h5_pattern
    if pattern == "" or pattern is None:
        pattern = DEFAULT_H5_PATTERN
    validate_h5_pattern(pattern)
    count = DEFAULT_N_FILES if n_files is None else n_files
    if (type(count) is not int and (type(count) is not float or not count.is_integer())) or count < 0:
        raise ValueError("Enter a whole file count of zero or more.")
    count = int(count)
    if output_path is not None and not str(output_path).strip():
        raise ValueError("Choose an output map file.")
    target = shared.checked_path(DEFAULT_OUTPUT_PATH if output_path is None else output_path, write=True, cwd=cwd)
    target = validate_distinct_output(target, (("GMD timestamp file", gmd), ("HDF5 directory", directory)),
                                      output_label="GMD map output")
    pattern_args = [f"--h5-pattern={pattern}"] if pattern.startswith("-") else ["--h5-pattern", pattern]
    cmd = [sys.executable, "-m", "vmi_analysis.gmd.build_gmd_map",
           "--h5-dir", str(directory), *pattern_args, "--gmd-path", str(gmd),
           "--output-path", str(target), "--n-files", str(count)]
    form = dict(h5_dir=str(directory), gmd_path=str(gmd), output_path=str(target),
                h5_pattern=pattern, n_files=count)
    return Job(JOB_NAME, cmd=cmd, cwd=str(cwd), consume_output=_capture_progress,
               extra={"output_path": str(target), "panel_form": form})


def _prepare_agent(_jobs, params):
    inputs = dict(params)
    for key in ("h5_dir", "gmd_path", "output_path"):
        if key in inputs or key != "output_path":
            inputs[key] = resolve_path(inputs.get(key), write=key == "output_path")
    job = gmd_map_job(**inputs)
    form = job.extra["panel_form"]
    return job, {"HDF5 directory": form["h5_dir"], "GMD timestamp file": form["gmd_path"],
                 "HDF5 pattern": form["h5_pattern"], "First N files": form["n_files"] or "All files"}


def agent_operations():
    description = {
        "panel": PANEL_TITLE,
        "summary": "Match camera frames to GMD pulse-energy records by timestamp.",
        "parameters": {
            "h5_dir": {"type": "path", "description": "Directory containing camera HDF5 files"},
            "gmd_path": {"type": "path", "description": "GMD timestamp file, usually current-time-both.txt"},
            "output_path": {"type": "path", "description": "Map file. Defaults to gmd_map_all.txt in the workspace."},
            "h5_pattern": {"type": "string", "default": DEFAULT_H5_PATTERN, "description": "Filename-only glob"},
            "n_files": {"type": "integer", "default": DEFAULT_N_FILES, "description": "Zero uses all matching files"},
        },
        "manual": PANEL_TITLE,
    }
    return (PanelOperation("gmd_map", description, _prepare_agent, (JOB_NAME,), outputs=("output_path",)),)


def build(jobs: JobManager) -> None:
    def restore_form(job):
        for key, value in adopt_agent_form(job, "gmd_map").items():
            if key in fields:
                fields[key].value = value

    def run():
        try:
            job = gmd_map_job(**{key: widget.value for key, widget in fields.items()})
            view.attach(jobs.submit(job))
        except (OSError, ValueError) as exc:
            ui.notify(str(exc), type="warning")

    view = JobView(jobs, job_name=JOB_NAME, results_builder=_results, attach_builder=restore_form)
    theme.panel_header(
        PANEL_TITLE,
        "Match camera frames to GMD pulse-energy records by timestamp.",
        view=view)

    with theme.panel_card():
        h5_dir = persist(path_input(
            "HDF5 directory", "", browse="dir",
            placeholder="folder containing RAW-*.h5 files"), "gmd_map.h5_dir")
        gmd_path = persist(path_input(
            "GMD timestamp file (current-time-both.txt)",
            "", browse="file", filters=[".txt"],
            placeholder="current-time-both.txt"), "gmd_map.gmd_path")
        output_path = persist(path_input(
            "Output map file", str(shared.workspace_root(REPO_ROOT) / DEFAULT_OUTPUT_PATH),
            browse="save", filters=[".txt"],
            default_name=DEFAULT_OUTPUT_PATH), "gmd_map.output_path")
        with field_grid(2):
            pattern = persist(ui.input("HDF5 pattern", value=DEFAULT_H5_PATTERN)
                              .props("outlined dense"), "gmd_map.h5_pattern")
            pattern.tooltip("Glob inside the HDF5 directory. Use *RAW-*.h5 "
                            "when the filenames have a prefix before RAW-.")
            n_files = persist(num_input("First N files (0 = all)", DEFAULT_N_FILES, step=1),
                              "gmd_map.n_files")
        fields = dict(h5_dir=h5_dir, gmd_path=gmd_path, output_path=output_path,
                      h5_pattern=pattern, n_files=n_files)
        with action_bar(view, on_click=run):
            ui.button("Clear output", icon="delete",
                      on_click=lambda: _clear_output(output_path, h5_dir, gmd_path, view)) \
                .props("flat color=negative").classes("vmi-action-danger")

    view.place_console()


def _clear_output(output_path, h5_dir, gmd_path, view) -> None:
    """Delete the output map file so the next run rewrites it from scratch."""
    txt = (output_path.value or "").strip()
    if not txt:
        ui.notify("Set an output path first.", type="warning")
        return
    try:
        target = validate_distinct_output(txt, (("timestamp file", gmd_path.value),
                                               ("HDF5 directory", h5_dir.value)),
                                          cwd=shared.workspace_root(REPO_ROOT), output_label="GMD map output")
    except (OSError, ValueError) as exc:
        ui.notify(str(exc), type="warning")
        return
    confirm_clear_outputs([target], "Delete GMD map file?", on_cleared=view.clear_results)


def _results(job: Job):
    out = job.extra.get("output_path", "")
    if job.status == "done" and out:
        out = shared.checked_path(out, cwd=job.cwd or shared.workspace_root(REPO_ROOT))
        if not out.is_file():
            return
        size = out.stat().st_size / 1e6
        frames = job.extra.get("frame_count")
        details = f"{frames:,} frames, {size:.1f} MB" if frames is not None else f"{size:.1f} MB"
        ui.label(f"Map written: {out}  ({details})").classes("vmi-subtitle")
        url = file_url(out)
        with ui.row().classes("items-center gap-2"):
            ui.button("Download GMD map", icon="download",
                      on_click=lambda: ui.download.from_url(url, filename=out.name)).props("flat")
            if not shared.SERVER:
                open_folder_button(out)
