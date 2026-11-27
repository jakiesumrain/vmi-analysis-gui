"""Correct VMI images and electron localization checkpoints."""

import re
import sys
from functools import partial
from pathlib import Path

import numpy as np
from nicegui import ui
from plotly.subplots import make_subplots
from vmi_analysis.circularization.checkpoint import checkpoint_targets
from vmi_analysis.circularization.circularization import COORDINATE_CORRECTIONS
from vmi_analysis.paths import validate_distinct_output

from ..agent_panels import PanelOperation, boolean, number, require_completed_job, resolve_path
from ..shared import checked_path, circularization_dir, generated_dir
from ..centroid_plots import _heatmap, _image_axes, _style, plot_options
from ..jobs import Job, JobManager
from .. import shared, theme
from ..settings import centroid_settings, output_dir
from ..resources import canonical_path
from .common import (GENERATED, JobView, REPO_ROOT, action_bar, adopt_agent_form, check_grid,
                     field_grid, file_url, num_input, open_folder_button,
                     path_input, persist, zoomable_image,
                     confirm_clear_outputs)
from .polar_preview import (bind_percentiles, polar_controls, polar_preview,
                            set_polar_source, set_polar_view)

CIRC_DIR = REPO_ROOT / "circularization"


JOB_NAME = "Circularize"
CHECKPOINT_OPERATION = "circularize_checkpoint"
CHECKPOINT_OUTPUTS = ("corrected_checkpoint", "counts_output")
IMAGE_OUTPUTS = ("output", "polar_preview")
CHECKPOINT_GEOMETRY = ("shape_y", "shape_x", "center_x", "center_y")
DEFAULTS = {
    "shape_y": 1023, "shape_x": 1023, "center_x": 511.0, "center_y": 511.0,
    "brightest_center": True, "radial_model": "interp", "poly_order": 2,
    "interp_order": 3, "max_trig_n": 0, "polar_dr": 1.0,
    "no_jacobian": True, "no_plots": False, "summary_only": False, "resolution_factor": 1,
}
RADIAL_MODELS = ("interp", "poly")
CHECKPOINT_SUFFIXES = (".h5", ".hdf5")
BOOLEAN_FIELDS = ("brightest_center", "no_jacobian", "no_plots", "summary_only")


def _whole(value, minimum, label, maximum=None):
    if ((type(value) is not int and (type(value) is not float or not value.is_integer()))
            or value < minimum or (maximum is not None and value > maximum)):
        limit = f"between {minimum} and {maximum}" if maximum is not None else f"of {minimum} or more"
        raise ValueError(f"{label} must be a whole number {limit}.")
    return int(value)


def circularize_job(extracted="", *, inp="", out="", reassign="", plot_dir=None,
                    corrected_checkpoint="", counts_output="", coordinate_correction="circularize", **settings):
    cwd = circularization_dir(CIRC_DIR)
    options = DEFAULTS | settings
    for key in BOOLEAN_FIELDS:
        boolean(options[key])
    events = Path(inp or "").suffix.lower() in CHECKPOINT_SUFFIXES and not options["summary_only"]
    if coordinate_correction not in COORDINATE_CORRECTIONS or (coordinate_correction == "none" and not events):
        raise ValueError("Choose a valid coordinate correction mode for a localization checkpoint.")
    apply_correction = coordinate_correction == "circularize"
    if apply_correction and (not extracted or not str(extracted).strip()):
        raise ValueError("Choose an extracted trig-fit file.")
    plot_dir = cwd / "trig_fit_plots" if plot_dir is None else plot_dir
    cmd = [sys.executable, "-m", "vmi_analysis.circularization.circularization"]
    if apply_correction:
        cmd += ["--extracted", str(extracted)]
    order = _whole(options["max_trig_n"] or 0, 0, "Maximum trig order") if apply_correction else 0
    if order:
        cmd += ["--max-trig-n", str(order)]

    job = Job(JOB_NAME, cmd=cmd, cwd=str(cwd))
    job.extra["summary_only"] = options["summary_only"]
    if options["summary_only"]:
        cmd.append("--summary-only")
        job.extra["panel_form"] = {"extracted": str(extracted), "summary_only": True, "max_trig_n": order}
        return job

    if not inp or not str(inp).strip():
        raise ValueError("Choose a VMI image or localization checkpoint.")
    raw = Path(inp).suffix.lower() == ".npy"
    model = options["radial_model"]
    if apply_correction:
        if model not in RADIAL_MODELS:
            raise ValueError("Choose interpolation or polynomial radial fits.")
        cmd += ["--radial-model", model]
        if model == "poly":
            cmd += ["--poly-order", str(_whole(options["poly_order"], 1, "Polynomial order"))]
    preview = generated_dir(GENERATED) / f"circ_{job.id}_polar.npz"
    checked_path(preview, write=True)
    if events:
        source = checked_path(inp, cwd=cwd)
        if not source.is_file():
            raise ValueError("Choose an existing localization checkpoint.")
        if not corrected_checkpoint or not counts_output:
            raise ValueError("Choose a corrected checkpoint and an integer histogram output.")
        corrected, counts = checkpoint_targets(
            source, extracted, reassign, checked_path(corrected_checkpoint, write=True, cwd=cwd),
            checked_path(counts_output, write=True, cwd=cwd), cwd=cwd)
        ny, nx = [_whole(options[key], 1, f"Shape {key[-1].upper()}") for key in ("shape_y", "shape_x")]
        xc, yc = [number(options[key]) for key in ("center_x", "center_y")]
        if not (0 <= xc <= nx - 0.5 and 0 <= yc <= ny - 0.5):
            raise ValueError("Center must lie inside the calibration grid.")
        factor = _whole(options["resolution_factor"], 1, "Histogram resolution factor")
        cmd += ["--checkpoint", str(source), "--corrected-checkpoint", str(corrected),
                "--counts-output", str(counts), "--shape", str(ny), str(nx),
                "--center", str(xc), str(yc), "--resolution-factor", str(factor),
                "--coordinate-correction", coordinate_correction]
        if apply_correction:
            dr = number(options["polar_dr"])
            if dr <= 0:
                raise ValueError("Polar radial step must be positive.")
            cmd += ["--polar-dr", str(dr)]
            if reassign:
                cmd += ["--reassign-file", str(reassign)]
        cmd += ["--polar-preview", str(preview)]
        job.extra.update(corrected_checkpoint=str(corrected), counts_output=str(counts), polar_preview=str(preview),
                         panel_form=dict(options, inp=str(source), extracted=str(extracted),
                                         reassign=str(reassign), corrected_checkpoint=str(corrected),
                                         counts_output=str(counts), max_trig_n=order,
                                         coordinate_correction=coordinate_correction))
        if not apply_correction:
            for key in ("extracted", "reassign", "radial_model", "poly_order", "max_trig_n", "polar_dr"):
                job.extra["panel_form"].pop(key)
        return job

    inputs = (("input image", inp), ("extracted calibration file", extracted), ("radial reassignment file", reassign))
    if not out or not str(out).strip():
        raise ValueError("Choose an output image file.")
    target = validate_distinct_output(checked_path(out, write=True, cwd=cwd), inputs, cwd=cwd)
    cmd += ["--input", str(inp), "--output", str(target),
            "--interp-order", str(_whole(options["interp_order"], 0, "Interpolation order", 5)),
            "--polar-preview", str(preview)]
    if not raw:
        shape = [_whole(options[key], 1, f"Shape {key[-1].upper()}") for key in ("shape_y", "shape_x")]
        dr = number(options["polar_dr"])
        if dr <= 0:
            raise ValueError("Polar radial step must be positive.")
        cmd += ["--shape", *map(str, shape), "--polar-dr", str(dr)]
    if not raw or not options["brightest_center"]:
        if options["center_x"] is None or options["center_y"] is None:
            raise ValueError("Enter both Center X and Center Y.")
        cmd += ["--center", *(str(number(options[key])) for key in ("center_x", "center_y"))]
    if reassign:
        cmd += ["--reassign-file", str(reassign)]
    for key, flag in (("no_jacobian", "--no-jacobian"), ("no_plots", "--no-plots")):
        if options[key]:
            cmd.append(flag)
    if not options["no_plots"] or reassign:
        if not str(plot_dir).strip():
            raise ValueError("Choose an output directory for diagnostic plots.")
        directory = checked_path(plot_dir, write=True, cwd=cwd)
        if directory.exists() and not directory.is_dir():
            raise ValueError("Choose an output directory for diagnostic plots.")
        diagnostics = _diagnostic_files(directory) if not options["no_plots"] else [directory / "reassignment_map.png"]
        for path in diagnostics:
            validate_distinct_output(path, (*inputs, ("corrected image", target)), cwd=cwd)
        plot_dir = directory
        cmd += ["--plot-dir", str(directory)]
    job.extra.update(output=str(target), polar_preview=str(preview), plot_dir=str(plot_dir), reassignment=bool(reassign),
                     panel_form=dict(options, inp=str(inp), out=str(target), extracted=str(extracted),
                                     reassign=str(reassign), plot_dir=str(plot_dir), max_trig_n=order))
    return job


def _prepare_agent(jobs, params, *, checkpoint_only=False):
    inputs = dict(params)
    if checkpoint_only:
        missing = [key for key in (*CHECKPOINT_GEOMETRY, *CHECKPOINT_OUTPUTS) if inputs.get(key) is None]
        if missing:
            raise ValueError(f"Supply the checkpoint geometry and both output paths. Missing {', '.join(missing)}.")
    summary_only = boolean(inputs.get("summary_only", False))
    source_id = inputs.pop("source_job_id", None)
    bin_index = inputs.pop("bin_index", None)
    source = inputs.pop("input", "")
    event_outputs = checkpoint_only or any(inputs.get(key) for key in CHECKPOINT_OUTPUTS)
    if source_id is not None:
        if source:
            raise ValueError("Supply either input or source_job_id.")
        parent = jobs.get(source_id) if isinstance(source_id, str) else None
        if parent is None or parent.name not in ("Centroiding", "GMD binning"):
            raise ValueError("Choose a Centroiding or Binning job from your workspace.")
        parent = require_completed_job(jobs, source_id, name=parent.name)
        if bin_index is not None and parent.name != "GMD binning":
            raise ValueError("A bin index requires a completed Binning job.")
        if parent.name == "Centroiding":
            artifact = "checkpoint" if event_outputs else "raw_image"
        elif event_outputs:
            raise ValueError("A checkpoint input requires a completed Centroiding job.")
        else:
            artifact = "all_bins_image" if bin_index is None else f"bin_{_whole(bin_index, 0, 'Bin index')}"
        source = parent.extra.get(artifact)
        if not source:
            raise ValueError("The completed job does not have the requested image or checkpoint.")
    elif bin_index is not None:
        raise ValueError("A bin index requires a completed Binning job.")
    if not summary_only or source:
        source = resolve_path(source)
        if not source.is_file() or source.suffix.lower() not in (".npy", ".txt", *CHECKPOINT_SUFFIXES):
            raise ValueError("Choose an existing VMI image (.npy or .txt) or localization checkpoint (.h5 or .hdf5).")
    events = bool(source) and source.suffix.lower() in CHECKPOINT_SUFFIXES
    if event_outputs and not events:
        raise ValueError("Corrected checkpoint and count histogram outputs require a localization checkpoint.")
    apply_correction = inputs.get("coordinate_correction", "circularize") != "none"
    calibration = inputs.get("extracted")
    if apply_correction:
        if not calibration and source:
            calibration = source.with_name("extracted.txt")
        calibration = resolve_path(str(calibration) if isinstance(calibration, Path) else calibration)
        if not calibration.is_file():
            raise ValueError("Choose an existing extracted trig-fit file.")
        inputs["extracted"] = calibration
    else:
        inputs.pop("extracted", None)
    inputs["inp"] = source
    inputs["out"] = inputs.pop("output", "")
    inputs["reassign"] = inputs.pop("reassign_file", "") or ""
    if not apply_correction:
        inputs["reassign"] = ""
    for key in ("out", "plot_dir", "reassign", "corrected_checkpoint", "counts_output"):
        if inputs.get(key) or (key == "out" and not summary_only and not events):
            inputs[key] = resolve_path(inputs.get(key), write=key != "reassign")
    if inputs["reassign"] and not inputs["reassign"].is_file():
        raise ValueError("Choose an existing radial reassignment file.")

    try:
        job = circularize_job(**inputs)
    except OverflowError as exc:
        raise ValueError("Enter finite Circularize settings.") from exc
    form = job.extra["panel_form"]
    summary = {"Mode": "Calibration summary" if summary_only else "Correct radial distortion",
               "Extracted trig-fit file": str(calibration) if apply_correction else None,
               "Maximum trig order": (form["max_trig_n"] or "Automatic") if apply_correction else None}
    if not summary_only and events:
        summary.update({
            "Mode": COORDINATE_CORRECTIONS[form["coordinate_correction"]],
            "Source checkpoint": str(source),
            "Image shape (NY, NX)": [form["shape_y"], form["shape_x"]],
            "Center (X, Y)": [form["center_x"], form["center_y"]],
            "Histogram bins per detector pixel": form["resolution_factor"],
            "Polar radial step (pixels)": form["polar_dr"] if apply_correction else None,
            "Radial model": form["radial_model"] if apply_correction else None,
            "Polynomial order": form["poly_order"] if apply_correction and form["radial_model"] == "poly" else None,
            "Radial reassignment": (form["reassign"] or "Off") if apply_correction else None,
            "Corrected checkpoint": form["corrected_checkpoint"],
            "Integer histogram": form["counts_output"],
        })
    elif not summary_only:
        raw = source.suffix.lower() == ".npy"
        summary.update({
            "VMI image": str(source),
            "Image shape (NY, NX)": "Native NumPy shape" if raw else [form["shape_y"], form["shape_x"]],
            "Center": "Brightest pixel" if raw and form["brightest_center"]
                      else [form["center_x"], form["center_y"]],
            "Polar radial step (pixels)": 1.0 if raw else form["polar_dr"],
            "Radial model": form["radial_model"],
            "Polynomial order": form["poly_order"] if form["radial_model"] == "poly" else None,
            "Interpolation order": form["interp_order"],
            "Jacobian intensity correction": not form["no_jacobian"],
            "Radial reassignment": form["reassign"] or "Off",
            "Trig-fit diagnostic plots": not form["no_plots"],
            "Diagnostic output directory": job.extra["plot_dir"] if not form["no_plots"] or form["reassign"] else None,
        })
    return job, summary


def agent_operations():
    parameters = {
        "input": {"type": "path", "description": "VMI image (.npy or .txt) or localization checkpoint (.h5 or .hdf5), alternative to source_job_id"},
        "source_job_id": {"type": "job ID", "description": "Completed batch Centroiding or Binning job"},
        "bin_index": {"type": "integer", "description": "Optional zero-based Binning bin. Defaults to the combined image."},
        "extracted": {"type": "path", "description": "Trig-fit file. Defaults to extracted.txt beside the input image."},
        "output": {"type": "path", "description": "Corrected image text file in your workspace"},
        "corrected_checkpoint": {"type": "path", "description": "Output HDF5 file for corrected centroids"},
        "counts_output": {"type": "path", "description": "Output NumPy integer histogram for corrected centroids"},
        "coordinate_correction": {"type": "string", "choices": list(COORDINATE_CORRECTIONS), "default": "circularize",
                                  "description": "For checkpoints, use 'none' to keep measured coordinates without calibration."},
        "plot_dir": {"type": "path", "description": "Diagnostic output directory. Defaults to circularization/trig_fit_plots."},
        "reassign_file": {"type": "path", "description": "Optional two-column radial reassignment file"},
    }
    help_text = {
        "shape_y": "Rows in the checkpoint calibration grid or legacy text image",
        "shape_x": "Columns in the checkpoint calibration grid or legacy text image",
        "center_x": "Zero-based center column in pixels", "center_y": "Zero-based center row in pixels",
        "brightest_center": "Use the brightest pixel as center for NumPy input",
        "radial_model": "Model for radial dependence of the trig coefficients",
        "poly_order": "Order used for the polynomial radial model",
        "interp_order": "Remapping interpolation order from 0 to 5",
        "max_trig_n": "Highest trig term to use. Zero infers it from the calibration.",
        "polar_dr": "Radial step in pixels for checkpoint or legacy text input",
        "resolution_factor": "Histogram bins per detector pixel in the checkpoint calibration image",
        "no_jacobian": "Disable Jacobian intensity correction",
        "no_plots": "Skip the trig-fit diagnostic plots",
        "summary_only": "Read and summarize the calibration. An output image is unnecessary.",
    }
    types = {bool: "boolean", int: "integer", float: "number", str: "string"}
    for key, value in DEFAULTS.items():
        parameters[key] = {"type": types[type(value)], "default": value, "description": help_text[key]}
    parameters["radial_model"]["choices"] = list(RADIAL_MODELS)
    description = {"panel": JOB_NAME, "manual": JOB_NAME, "parameters": parameters,
                   "summary": "Correct radial distortion in a VMI image. Use circularize_checkpoint for individual centroids."}
    checkpoint_parameters = {
        key: dict(parameters[key])
        for key in ("input", "source_job_id", "extracted", "reassign_file", "coordinate_correction", *CHECKPOINT_GEOMETRY,
                    "polar_dr", "resolution_factor", "max_trig_n", "radial_model", "poly_order",
                    *CHECKPOINT_OUTPUTS)
    }
    checkpoint_parameters["input"]["description"] = "Centroiding localization checkpoint (.h5 or .hdf5). Supply this or source_job_id."
    checkpoint_parameters["source_job_id"]["description"] = "Completed Centroiding job whose checkpoint to use."
    checkpoint_parameters["extracted"]["description"] = (
        "Trig-fit file for circularization. Defaults to extracted.txt beside the checkpoint. Omit in 'none' mode.")
    for key in (*CHECKPOINT_GEOMETRY, *CHECKPOINT_OUTPUTS):
        checkpoint_parameters[key].pop("default", None)
        checkpoint_parameters[key]["required"] = True
    checkpoint = {
        "panel": JOB_NAME, "manual": JOB_NAME, "parameters": checkpoint_parameters,
        "summary": "Export measured or circularized centroids with a complete shot roster and an unweighted integer histogram.",
    }
    return (
        PanelOperation("circularize", description, _prepare_agent, (JOB_NAME,),
                       artifacts=lambda job: CHECKPOINT_OUTPUTS if "counts_output" in job.extra else IMAGE_OUTPUTS,
                       outputs=(*IMAGE_OUTPUTS, *CHECKPOINT_OUTPUTS)),
        PanelOperation(CHECKPOINT_OPERATION, checkpoint, partial(_prepare_agent, checkpoint_only=True), (),
                       outputs=CHECKPOINT_OUTPUTS, proposal_job_names=(JOB_NAME,)),
    )


def build(jobs: JobManager):
    def results(job):
        _results(job)
        if job.status == "done" and job.extra.get("polar_preview"):
            show_polar(job.extra["polar_preview"])

    def restore_form(job):
        polar_data.clear()
        polar_card.set_visibility(False)
        for key, value in adopt_agent_form(job, "circularize").items():
            if key in fields:
                fields[key].value = value

    view = JobView(jobs, job_name=JOB_NAME, results_builder=results, attach_builder=restore_form)
    theme.panel_header(
        JOB_NAME,
        "Correct radial distortion in a VMI image or in individual electron centroids.",
        view=view)

    with theme.panel_card():
        inp = persist(path_input("VMI image or localization checkpoint",
                                 str(output_dir(centroid_settings()) / "vmi_2d_raw.npy"),
                                 browse="file", filters=[".npy", ".txt", *CHECKPOINT_SUFFIXES]),
                      "circularize.inp")
        out = persist(path_input("Output image (.txt)",
                                 str(circularization_dir(CIRC_DIR) / "image_circularized.txt"),
                                 browse="save", filters=[".txt"],
                                 default_name="image_circularized.txt"),
                      "circularize.out")
        with ui.column().classes("w-full gap-2") as event_paths:
            coordinate_correction = persist(
                ui.select(COORDINATE_CORRECTIONS, value="circularize", label="Coordinate correction")
                .props("outlined dense").classes("w-full"), "circularize.coordinate_correction")
            corrected_checkpoint = persist(path_input(
                "Corrected checkpoint (.h5)", str(circularization_dir(CIRC_DIR) / "localizations_circularized.h5"),
                browse="save", filters=list(CHECKPOINT_SUFFIXES), default_name="localizations_circularized.h5"),
                "circularize.corrected_checkpoint")
            counts_output = persist(path_input(
                "Integer histogram (.npy)", str(circularization_dir(CIRC_DIR) / "image_circularized_counts.npy"),
                browse="save", filters=[".npy"], default_name="image_circularized_counts.npy"),
                "circularize.counts_output")
            ui.label("Match the grid shape, binning and center used to fit the calibration.").classes("vmi-subtitle")
        with field_grid():
            shape_y = persist(num_input("Shape NY", 1023, step=1),
                              "circularize.shape_y")
            shape_x = persist(num_input("Shape NX", 1023, step=1),
                              "circularize.shape_x")
            center_x = persist(num_input("Center X", 511.0, step="any"),
                               "circularize.center_x")
            center_y = persist(num_input("Center Y", 511.0, step="any"),
                               "circularize.center_y")
            resolution_factor = persist(num_input("Histogram bins per detector pixel", 1, min=1, step=1),
                                        "circularize.resolution_factor")
        with check_grid():
            brightest_center = persist(ui.checkbox("Use brightest pixel as center", value=True),
                                       "circularize.brightest_center")
        extracted = persist(path_input(
            "Extracted trig-fit file",
            str(circularization_dir(CIRC_DIR) / "extracted-FOC100eV-20260710.txt"),
            browse="file", filters=[".txt"]), "circularize.extracted")
        reassign = persist(path_input(
            "Reassignment file (optional)",
            browse="file", filters=[".txt"]), "circularize.reassign")
        with field_grid(3):
            radial_model = persist(ui.select(
                ["interp", "poly"], value="interp",
                label="Radial model").props("outlined dense"),
                "circularize.radial_model")
            poly_order = persist(num_input("Poly order", 2, step=1),
                                 "circularize.poly_order")
            poly_order.set_enabled(radial_model.value == "poly")
            radial_model.on_value_change(lambda e: poly_order.set_enabled(e.value == "poly"))
            interp_order = persist(num_input("Interp order", 3, step=1),
                                   "circularize.interp_order")
            max_trig_n = persist(num_input("Max trig n (0 = auto)", 0, step=1),
                                 "circularize.max_trig_n")
            polar_dr = persist(num_input("Polar dr", 1.0, step=0.5),
                               "circularize.polar_dr")
        with check_grid():
            no_jacobian = persist(ui.checkbox("No Jacobian", value=True),
                                  "circularize.no_jacobian")
            no_plots = persist(ui.checkbox("No diagnostic plots", value=False),
                               "circularize.no_plots")
            summary_only = persist(ui.checkbox("Summary only", value=False),
                                   "circularize.summary_only")
        plot_dir = persist(path_input(
            "Plot output dir", str(circularization_dir(CIRC_DIR) / "trig_fit_plots"),
            browse="dir"), "circularize.plot_dir")

        image_hint = ui.label("NumPy input keeps its original grid. Use the same center as Trig Calibration.") \
            .classes("vmi-subtitle")

        def raw_input():
            return Path(inp.value or "").suffix.lower() == ".npy"

        def checkpoint_input():
            return Path(inp.value or "").suffix.lower() in CHECKPOINT_SUFFIXES

        def update_geometry():
            raw = raw_input()
            events = checkpoint_input()
            measured = events and coordinate_correction.value == "none"
            event_paths.set_visibility(events)
            resolution_factor.set_visibility(events)
            for field in (out, interp_order, no_jacobian, no_plots, plot_dir, summary_only, image_hint):
                field.set_visibility(not events)
            for field in (shape_y, shape_x, polar_dr):
                field.set_enabled(not raw)
            for field in (extracted, reassign, radial_model, poly_order, max_trig_n, polar_dr):
                field.set_visibility(not measured)
            for field in (center_x, center_y):
                field.set_enabled(not raw or not brightest_center.value)
            brightest_center.set_visibility(raw)

        inp.on_value_change(lambda _: update_geometry())
        coordinate_correction.on_value_change(lambda _: update_geometry())
        brightest_center.on_value_change(lambda _: update_geometry())
        update_geometry()

        fields = dict(inp=inp, out=out, extracted=extracted, reassign=reassign, plot_dir=plot_dir,
                      corrected_checkpoint=corrected_checkpoint, counts_output=counts_output,
                      resolution_factor=resolution_factor, coordinate_correction=coordinate_correction,
                      shape_y=shape_y, shape_x=shape_x, center_x=center_x, center_y=center_y,
                      brightest_center=brightest_center, radial_model=radial_model, poly_order=poly_order,
                      interp_order=interp_order, max_trig_n=max_trig_n, polar_dr=polar_dr,
                      no_jacobian=no_jacobian, no_plots=no_plots, summary_only=summary_only)

        def run():
            try:
                options = {key: widget.value for key, widget in fields.items()}
                if checkpoint_input():
                    options["summary_only"] = False
                else:
                    options.pop("coordinate_correction")
                job = circularize_job(**options)
                view.attach(jobs.submit(job))
            except (OSError, ValueError, OverflowError) as exc:
                ui.notify(str(exc), type="warning")

        def cleared():
            view.clear_results()
            polar_data.clear()
            polar_card.set_visibility(False)

        def clear():
            if checkpoint_input():
                try:
                    checkpoint, counts = checkpoint_targets(inp.value, extracted.value, reassign.value,
                                                            corrected_checkpoint.value, counts_output.value,
                                                            cwd=circularization_dir(CIRC_DIR))
                except (OSError, ValueError) as exc:
                    ui.notify(str(exc), type="warning")
                    return
                targets = [checkpoint, counts]
                targets.extend(job.extra["polar_preview"] for job in jobs.jobs()
                               if job.name == JOB_NAME and job.extra.get("polar_preview")
                               and any(job.extra.get(key) and canonical_path(job.extra[key], job.cwd) == target
                                       for key, target in zip(CHECKPOINT_OUTPUTS, (checkpoint, counts))))
                confirm_clear_outputs(targets, "Delete the corrected checkpoint and integer histogram?",
                                      on_cleared=cleared)
                return
            if not out.value:
                ui.notify("Set an output image path first.", type="warning")
                return
            inputs = (("input image", inp.value), ("extracted calibration file", extracted.value),
                      ("radial reassignment file", reassign.value))
            try:
                target = validate_distinct_output(out.value, inputs, cwd=circularization_dir(CIRC_DIR))
                directory = canonical_path(plot_dir.value, circularization_dir(CIRC_DIR))
                diagnostics = [path for path in _diagnostic_files(directory) if path.is_file()]
                for path in diagnostics:
                    validate_distinct_output(path, inputs, cwd=circularization_dir(CIRC_DIR))
            except (OSError, ValueError) as exc:
                ui.notify(str(exc), type="warning")
                return
            targets = [target, *diagnostics]
            for job in jobs.jobs():
                if (job.name == JOB_NAME and job.extra.get("polar_preview") and job.extra.get("output")
                        and canonical_path(job.extra["output"], job.cwd) == canonical_path(target)):
                    targets.append(job.extra["polar_preview"])

            confirm_clear_outputs(targets, "Delete Circularize outputs and diagnostic plots?", on_cleared=cleared)

        with action_bar(view, on_click=run):
            ui.button("Clear outputs", icon="delete", on_click=clear) \
                .props("flat color=negative").classes("vmi-action-danger")

        polar_data, polar_widgets, polar_sections = {}, {}, {}
        with ui.column().classes("w-full gap-2 border-t border-stone-200 pt-4 polar-guide-group") as polar_card:
            polar_card.set_visibility(False)
            scale, colormap, guides = polar_controls("circularize", radius_only=True)
            for key, label in (("raw", "Raw polar"), ("circularized", "Circularized polar")):
                with ui.column().classes("w-full gap-2") as section:
                    polar_sections[key] = section
                    polar_widgets[key] = polar_preview(
                        "circularize", key, label, guides, lambda: ui.interactive_image(""),
                        radius_only=True)

    def refresh_polar(key):
        if key in polar_data:
            image, limits, status = polar_widgets[key]
            warning = set_polar_source(image, polar_data[key], colormap, scale, limits)
            status.text = f"Showing automatic linear colors. {warning}" if warning else ""
            status.set_visibility(bool(warning))

    def refresh_polar_pair():
        for key in polar_widgets:
            refresh_polar(key)

    def show_polar(path):
        if not Path(path).is_file():
            return  # The job console reports preview generation errors.
        with np.load(path, allow_pickle=False) as data:
            polar_data.clear()
            polar_data.update({key: data[key] for key in polar_widgets if key in data})
            dr = float(data["dr"])
        for key, section in polar_sections.items():
            section.set_visibility(key in polar_data)
        n_theta, n_r = polar_data["raw"].shape
        set_polar_view(polar_card, (0, n_r - 1), (0, n_theta - 1), dr, 360 / n_theta)
        refresh_polar_pair()
        polar_card.set_visibility(True)

    for key, (_, limits, _) in polar_widgets.items():
        bind_percentiles(limits, lambda key=key: refresh_polar(key))
    for control in (scale, colormap):
        control.on_value_change(lambda _: refresh_polar_pair())
    view.place_console()

    def use_calibration(extracted_path, image_path=None, manual_center=None, *,
                        radial_model_value=None, poly_order_value=None):
        extracted.value = extracted_path
        if image_path is not None:
            inp.value = image_path
            brightest_center.value = manual_center is None
            if manual_center is not None:
                center_x.value, center_y.value = manual_center
        if radial_model_value is not None:
            radial_model.value = radial_model_value
        if poly_order_value is not None:
            poly_order.value = poly_order_value
        return int(max_trig_n.value or 0)

    return use_calibration


def _diagnostic_files(directory):
    return [directory / "reassignment_map.png",
            *(path for path in sorted(directory.glob("trig_order_*.png"))
              if re.fullmatch(r"trig_order_\d+\.png", path.name))]


def _results(job: Job):
    if job.status != "done" or job.extra.get("summary_only"):
        return
    cwd = Path(job.cwd or circularization_dir(CIRC_DIR))
    counts = job.extra.get("counts_output")
    if counts:
        checkpoint = cwd / job.extra["corrected_checkpoint"]
        if checkpoint.is_file():
            url = file_url(checkpoint)
            ui.button("Download corrected checkpoint", icon="download",
                      on_click=lambda: ui.download.from_url(url, filename=checkpoint.name)).props("flat")
    out = cwd / (counts or job.extra["output"])
    if out.is_file():
        with ui.row().classes("items-center gap-2"):
            url = file_url(out)
            ui.button("Download integer histogram" if counts else "Download corrected image", icon="download",
                      on_click=lambda: ui.download.from_url(url, filename=out.name)).props("flat")
            if not shared.SERVER:
                open_folder_button(out)
        try:
            image = np.load(out, allow_pickle=False) if counts else np.loadtxt(out, ndmin=2)
            figure = _corrected_figure(image, quantity="Counts" if counts else "Intensity")
            ui.plotly(plot_options(figure)).classes("w-full max-w-4xl")
        except (OSError, ValueError) as e:
            ui.label(f"Could not render {out}: {e}").classes("vmi-subtitle")
    if job.extra.get("reassignment") and job.started is not None:
        reassignment_plot = cwd / job.extra["plot_dir"] / "reassignment_map.png"
        if reassignment_plot.is_file() and reassignment_plot.stat().st_mtime >= job.started:
            ui.label("Radial reassignment map").classes("vmi-caption")
            zoomable_image(file_url(reassignment_plot)).classes("w-full max-w-2xl")


def _corrected_figure(image, quantity="Intensity"):
    trace = _heatmap(image, logarithmic=True, colorbar={"title": {"text": quantity}})
    positive = image[np.isfinite(image) & (image > 0)]
    if positive.size:
        high = np.log10(np.percentile(positive, 99.999))
        low = high - 4
        ticks = list(range(int(np.ceil(low)), int(np.floor(high)) + 1))
        trace.update(zmin=low, zmax=high, colorbar={
            "tickvals": ticks, "ticktext": [f"10<sup>{tick}</sup>" for tick in ticks]})
    trace.update(colorscale="Viridis",
                 hovertemplate="X: %{x:.0f} px<br>Y: %{y:.0f} px<br>" + quantity + ": %{customdata:.6g}<extra></extra>")
    figure = make_subplots(rows=1, cols=1)
    figure.add_trace(trace, row=1, col=1)
    _image_axes(figure, image, 1, 1, "x")
    _style(figure, "Circularized image · log scale")
    figure.update_layout(height=740)
    return figure
