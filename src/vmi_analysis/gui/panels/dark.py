"""Panel: compute_dark.py — mean/median dark frame from dark HDF5 files."""

from glob import iglob
import sys
from pathlib import Path

import numpy as np
from nicegui import ui

from vmi_analysis.centroiding.compute_dark import DEFAULT_DATASET, DEFAULT_FRAMES, DEFAULT_METHOD, DEFAULT_PATTERN

from .. import shared, theme
from ..agent_panels import PanelOperation, resolve_path
from ..jobs import Job, JobManager
from ..shared import generated_dir, workspace_root
from .common import (GENERATED, JobView, REPO_ROOT, action_bar, adopt_agent_form, confirm_clear_outputs,
                     field_grid, file_url, num_input, open_folder_button, path_input,
                     persist, render_image, zoomable_image)


JOB_NAME = "Dark frame"
PANEL_TITLE = "Dark Frame"
METHODS = ("median", "mean")


def _output_directory():
    return workspace_root(REPO_ROOT) / "centroiding"


def dark_frame_job(dark_dir, *, pattern=DEFAULT_PATTERN, dataset=DEFAULT_DATASET,
                   method=DEFAULT_METHOD, frames=DEFAULT_FRAMES, out_dir=None):
    cwd = workspace_root(REPO_ROOT)
    if not isinstance(dark_dir, (str, Path)) or not str(dark_dir).strip():
        raise ValueError("Choose a dark HDF5 directory.")
    source = shared.checked_path(dark_dir, cwd=cwd)
    if not source.is_dir():
        raise ValueError("Choose an existing dark HDF5 directory.")
    if method not in METHODS:
        raise ValueError("Choose median or mean for the dark frame.")
    if not isinstance(pattern, str) or not pattern.strip() or "\x00" in pattern:
        raise ValueError("Enter an HDF5 filename pattern.")
    if not isinstance(dataset, str) or not dataset.strip() or "\x00" in dataset:
        raise ValueError("Enter the HDF5 frame dataset.")
    if frames is None:
        frames = DEFAULT_FRAMES
    if (type(frames) is not int and (type(frames) is not float or not frames.is_integer())) or frames < 0:
        raise ValueError("Enter a whole dark frame count of zero or more.")
    frames = int(frames)
    if isinstance(out_dir, str) and not out_dir.strip():
        raise ValueError("Choose an output directory.")
    target = shared.checked_path(out_dir if out_dir is not None else _output_directory(), write=True, cwd=cwd)
    if target.exists() and not target.is_dir():
        raise ValueError("Choose an output directory.")
    npy = target / f"dark_{method}.npy"
    info = target / f"dark_{method}_info.txt"
    for output in (npy, info):
        shared.checked_path(output, write=True, cwd=cwd)
    cmd = [sys.executable, "-m", "vmi_analysis.centroiding.compute_dark",
           "--dark-dir", str(source), "--pattern", pattern, "--dataset", dataset,
           "--method", method, "--frames", str(frames), "--output-dir", str(target)]
    if shared.SERVER:
        shared.check_command(cmd)
    if not any(Path(path).is_file() for path in iglob(str(source / pattern))):
        raise ValueError("No HDF5 files match the pattern in the selected directory.")
    form = {"dark_dir": str(source), "pattern": pattern, "dataset": dataset, "method": method,
            "frames": frames, "out_dir": str(target)}
    return Job(JOB_NAME, cmd=cmd, cwd=str(cwd),
               extra={"dark_npy": str(npy), "info": str(info), "panel_form": form})


def _prepare_agent(_jobs, params):
    inputs = dict(params)
    inputs["dark_dir"] = resolve_path(inputs["dark_dir"])
    if "out_dir" in inputs:
        inputs["out_dir"] = resolve_path(inputs["out_dir"], write=True)
    job = dark_frame_job(**inputs)
    form = job.extra["panel_form"]
    return job, {
        "Dark HDF5 directory": form["dark_dir"],
        "HDF5 pattern": form["pattern"],
        "Frame dataset": form["dataset"],
        "Method": form["method"],
        "Dark frame count": form["frames"] or "All frames",
    }


def agent_operations():
    description = {
        "panel": PANEL_TITLE,
        "summary": "Build a mean or median dark frame from dark HDF5 images for background subtraction.",
        "parameters": {
            "dark_dir": {"type": "path", "description": "Directory containing dark HDF5 files"},
            "pattern": {"type": "string", "default": DEFAULT_PATTERN},
            "dataset": {"type": "string", "default": DEFAULT_DATASET},
            "method": {"type": "string", "default": DEFAULT_METHOD, "choices": list(METHODS)},
            "frames": {"type": "integer", "default": DEFAULT_FRAMES, "description": "Zero uses all frames"},
            "out_dir": {"type": "path", "description": "Output directory. Defaults to centroiding in the workspace."},
        },
        "manual": PANEL_TITLE,
    }
    return (PanelOperation("dark_frame", description, _prepare_agent, (JOB_NAME,), outputs=("dark_npy", "info")),)


def build(jobs: JobManager) -> None:
    def restore_form(job):
        for key, value in adopt_agent_form(job, "dark").items():
            if key in fields:
                fields[key].value = value

    view = JobView(jobs, job_name=JOB_NAME, results_builder=_results, attach_builder=restore_form)
    theme.panel_header(
        PANEL_TITLE,
        "Build a mean or median dark frame for background subtraction.",
        view=view)

    with theme.panel_card():
        dark_dir = persist(path_input(
            "Dark HDF5 directory", "", browse="dir",
            placeholder="folder containing dark RAW-*.h5 files"),
            "dark.dark_dir")
        with field_grid():
            pattern = persist(ui.input("HDF5 pattern", value=DEFAULT_PATTERN)
                              .props("outlined dense"), "dark.pattern")
            dataset = persist(ui.input("Frame dataset", value=DEFAULT_DATASET)
                              .props("outlined dense"), "dark.dataset")
            method = persist(ui.select(list(METHODS), value=DEFAULT_METHOD,
                                       label="Method")
                             .props("outlined dense"), "dark.method")
            frames = persist(num_input("Dark frame count (0 = all)", DEFAULT_FRAMES,
                                       step=100), "dark.frames")
        out_dir = persist(path_input("Output directory",
                                     str(_output_directory()),
                                     browse="dir"), "dark.out_dir")

        fields = dict(dark_dir=dark_dir, pattern=pattern, dataset=dataset, method=method,
                      frames=frames, out_dir=out_dir)

        def run():
            try:
                settings = {key: widget.value for key, widget in fields.items()}
                view.attach(jobs.submit(dark_frame_job(**settings)))
            except (OSError, ValueError) as exc:
                ui.notify(str(exc), type="warning")

        with action_bar(view, on_click=run):
            ui.button("Clear output", icon="delete",
                      on_click=lambda: _clear_output(out_dir, method, view)) \
                .props("flat color=negative").classes("vmi-action-danger")

    view.place_console()


def _clear_output(out_dir, method, view) -> None:
    """Delete dark_{method}.npy (+ info) so the next run recomputes."""
    folder = workspace_root(REPO_ROOT) / (out_dir.value or "").strip()
    if not out_dir.value or not folder.is_dir():
        ui.notify("Set an output directory first.", type="warning")
        return
    targets = [folder / f"dark_{method.value}.npy",
               folder / f"dark_{method.value}_info.txt"]
    for job in view.jobs.jobs():
        if (job.name == JOB_NAME and job.extra.get("dark_npy")
                and (Path(job.cwd or workspace_root(REPO_ROOT)) / job.extra["dark_npy"]).resolve() == targets[0].resolve()):
            targets.append(generated_dir(GENERATED) / f"dark_frame_{job.id}.png")
    confirm_clear_outputs(targets, "Delete dark frame outputs?", on_cleared=view.clear_results)


def _results(job: Job):
    if job.status != "done":
        return
    npy = Path(job.cwd or workspace_root(REPO_ROOT)) / job.extra["dark_npy"]
    if not npy.exists():
        return
    data = np.load(npy)
    ui.label(f"Dark frame: {npy}  —  mean={data.mean():.1f}, "
             f"std={data.std():.1f}, min={data.min():.0f}, "
             f"max={data.max():.0f}").classes("vmi-subtitle")
    open_folder_button(npy)
    png = render_image(data, generated_dir(GENERATED) / f"dark_frame_{job.id}.png",
                       cmap="gray", log=False, vmin_pct=0, vmax_pct=99.9)
    zoomable_image(file_url(png)).classes("w-full max-w-2xl")
    info = Path(job.cwd or workspace_root(REPO_ROOT)) / job.extra["info"]
    if info.exists():
        ui.label(info.read_text(encoding="utf-8").strip()) \
            .classes("vmi-caption")
