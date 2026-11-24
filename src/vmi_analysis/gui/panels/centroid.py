"""Panel: Picasso centroiding with saved fields and explicit CLI inputs."""

from math import isfinite
import os
import sys
from operator import not_
from pathlib import Path

from nicegui import ui
from plotly.colors import get_colorscale
from vmi_analysis.paths import validate_distinct_output
from vmi_analysis.centroiding.options import (
    centroid_cli_args, load_settings, options, section as _section, validate_settings,
)
from vmi_analysis.gmd.build_gmd_map import validate_h5_pattern
from vmi_analysis.centroiding.plot_data import (ABEL_IMAGE, RAW_IMAGE, SPECTRUM_CSV,
                                                SUMMARY_DATA, SUMMARY_PNG, TUNING_DATA, TUNING_PNG)

from ..agent_panels import PanelOperation, boolean, resolve_path
from ..shared import centroid_dir, checked_path, settings_store
from ..jobs import Job, JobManager
from .. import theme
from ..centroid_plots import (COLORMAPS, PLOT_HEIGHT, RESULT_COLORMAP_DEFAULTS,
                              linear_axis_limits, tuning_figure, summary_figure, plot_options)
from ..settings import centroid_settings, storage_key, output_dir as _output_dir
from .common import (JobView, REPO_ROOT, action_bar, adopt_agent_form, field_grid, file_url,
                     num_input, open_folder_button, path_input, persist, zoomable_image,
                     delete_outputs)

CENTROID_DIR = REPO_ROOT / "centroiding"
JOB_NAME = "Centroiding"
TUNING_FILES = {"tuning_plot": TUNING_PNG, "tuning_data": TUNING_DATA}
BATCH_FILES = {"raw_image": RAW_IMAGE, "summary_plot": SUMMARY_PNG, "summary_data": SUMMARY_DATA}
ABEL_FILES = {"abel_image": ABEL_IMAGE, "spectrum_csv": SPECTRUM_CSV}
ABEL_LOGS = {"[Save] Abel inverted: ": "abel_image", "[Spectrum] Saved: ": "spectrum_csv"}
OUTPUT_NAMES = (TUNING_PNG, SUMMARY_PNG, TUNING_DATA, SUMMARY_DATA, RAW_IMAGE, ABEL_IMAGE, SPECTRUM_CSV)

# (section path, key, kind, label) — kind: str/int/float/bool/select:opts
FIELDS = [
    ("vmi_picasso", "h5_dir", "dir", "HDF5 directory"),
    ("vmi_picasso", "h5_pattern", "str", "HDF5 pattern"),
    ("vmi_picasso", "h5_dataset", "str", "Frame dataset"),
    ("vmi_picasso", "h5_meta", "str", "Metadata dataset"),
    ("vmi_picasso", "gmd_map_path", "file:.txt", "GMD map file (empty = raw/meta)"),
    ("vmi_picasso", "dark_file", "file:.npy", "Pre-computed dark .npy"),
    ("vmi_picasso", "output_dir", "dir", "Output directory"),
    ("vmi_picasso", "checkpoint_file", "str", "Checkpoint filename"),
    ("vmi_picasso", "box_size", "int", "Box size (odd)"),
    ("vmi_picasso", "min_net_gradient", "float", "Min net gradient"),
    ("vmi_picasso", "camera_baseline", "float", "Camera baseline"),
    ("vmi_picasso", "camera_sensitivity", "float", "Camera sensitivity"),
    ("vmi_picasso", "camera_gain", "float", "Camera gain"),
    ("vmi_picasso", "camera_qe", "float", "Camera QE"),
    ("vmi_picasso", "mle_convergence", "float", "MLE convergence"),
    ("vmi_picasso", "mle_max_iterations", "int", "MLE max iterations"),
    ("vmi_picasso", "mle_method", "select:sigma,sigmaxy", "MLE method"),
    ("vmi_picasso", "fitting_method", "select:mle,lq", "Fitting method"),
    ("vmi_picasso", "tuning_frame_index", "int", "Tuning frame index (blank = random)"),
    ("processing", "enable_parallel", "bool", "Parallel processing"),
    ("processing", "num_cores", "int", "CPU cores"),
    ("processing", "enable_dark_subtraction", "bool", "Dark subtraction"),
    ("processing", "enable_median_subtraction", "bool", "Per-frame median subtraction"),
    ("processing", "enable_flat_field", "bool", "Flat field"),
    ("processing", "flat_field_file", "file:.txt,.npy", "Flat field file"),
    ("processing", "enable_circular_mask", "bool", "Circular mask"),
    ("processing", "mask_center_x", "float", "Mask center X (empty = image center)"),
    ("processing", "mask_center_y", "float", "Mask center Y (empty = image center)"),
    ("processing", "mask_radius", "float", "Mask radius (empty = automatic)"),
    ("processing", "tuning_only", "bool", "Tuning mode only"),
    ("processing", "frame_limit", "int", "Frame limit (0 = all)"),
    ("histogram", "enable_abel", "bool", "Abel inversion"),
    ("histogram", "abel_rotation_deg", "float", "Image rotation before Abel inversion (°)"),
    ("histogram", "resolution_factor", "int", "Histogram resolution factor"),
    ("energy_calibration", "alpha", "float", "α (eV/pixel²; empty = uncalibrated)"),
    ("energy_calibration", "fit_xe_4d", "bool", "Fit Xe 4d doublet → hν and α in console"),
    ("energy_calibration", "fit_min_px2", "float", "Xe fit min (pixel²; empty = auto)"),
    ("energy_calibration", "fit_max_px2", "float", "Xe fit max (pixel²; empty = auto)"),
    ("gmd", "enable_gmd_weighting", "bool", "GMD weighting"),
    ("gmd", "gmd_weight_min", "float", "GMD weight min"),
    ("gmd", "gmd_weight_max", "float", "GMD weight max"),
]

ADVANCED_KEYS = {
    "camera_baseline", "camera_sensitivity", "camera_gain", "camera_qe",
    "mle_convergence", "mle_max_iterations", "mle_method", "fitting_method",
}


def _capture_outputs(job, line):
    for prefix, artifact in ABEL_LOGS.items():
        if line.startswith(prefix):
            job.extra["generated_abel"].add(artifact)
    return False


def _artifacts(job):
    if job.extra.get("tuning_only"):
        return tuple(TUNING_FILES)
    return ("checkpoint", *BATCH_FILES, *(key for key in ABEL_FILES if key in job.extra.get("generated_abel", ())))


def centroid_job(snapshot, *, reuse_checkpoint=True):
    cwd = centroid_dir(CENTROID_DIR)
    vmi = snapshot["vmi_picasso"]
    processing = vmi["processing"]
    args = centroid_cli_args(snapshot)
    destination = checked_path(_output_dir(snapshot), write=True)
    if destination.exists() and not destination.is_dir():
        raise ValueError("Choose an output directory.")
    inputs = [("HDF5 directory", vmi["h5_dir"]),
              ("GMD map", vmi["gmd_map_path"] if vmi["gmd"]["enable_gmd_weighting"] else ""),
              ("dark frame", vmi["dark_file"] if processing["enable_dark_subtraction"] else ""),
              ("flat field", processing["flat_field_file"] if processing["enable_flat_field"] else "")]
    tuning = processing["tuning_only"]
    paths = TUNING_FILES if tuning else BATCH_FILES | ABEL_FILES
    extra = {key: str(validate_distinct_output(
        checked_path(destination / filename, write=True), inputs, cwd=cwd))
        for key, filename in paths.items()}
    if not tuning:
        checkpoint = checked_path(destination / vmi["checkpoint_file"], write=True)
        checkpoint = validate_distinct_output(
            checkpoint, [*inputs, *((key.replace("_", " "), path) for key, path in extra.items())],
            cwd=cwd, output_label="checkpoint",
        )
        if not reuse_checkpoint:
            if os.path.lexists(checkpoint):
                raise ValueError("The checkpoint already exists. Choose a new filename or explicitly request reuse.")
            args.append("--require-new-checkpoint")
        extra["checkpoint"] = str(checkpoint)
    form = {f"{group}.{key}": _section(snapshot, group)[key] for group, key, _ in options()}
    extra.update(output_dir=str(destination), tuning_only=tuning, panel_form=form, generated_abel=set())
    return Job(JOB_NAME, cwd=str(cwd), extra=extra, consume_output=_capture_outputs,
               cmd=[sys.executable, "-m",
                    "vmi_analysis.centroiding.process_andor_vmi_PICASSO_v5_h5_batch", *args])


def agent_operations():
    fields = {key: (kind, label) for _, key, kind, label in FIELDS}
    parameters = {}
    for _, key, default in options():
        if key == "meta_column_gmd":
            continue  # Deprecated by the CLI.
        kind, label = fields.get(key, ("", key.replace("_", " ").capitalize()))
        choices = kind.removeprefix("select:").split(",") if kind.startswith("select:") else None
        if kind == "dir" or kind.startswith("file:"):
            value_type = "path"
        elif type(default) is bool:
            value_type = "boolean"
        elif type(default) is int or key == "tuning_frame_index":
            value_type = "integer"
        elif type(default) is str:
            value_type = "string"
        else:
            value_type = "number"
        if default is None:
            value_type += " or null"
        parameters[key] = {"type": value_type, "description": label, "default": default}
        if choices:
            parameters[key]["choices"] = choices
    parameters["h5_pattern"]["description"] = "Filename-only glob in the HDF5 directory"
    parameters["h5_dir"]["description"] = "Required directory containing camera HDF5 files"
    parameters["output_dir"]["description"] = "Output directory. Defaults to centroiding/vmi_output in the workspace."
    parameters["gmd_map_path"]["description"] = "Existing timestamp map required for Agent batch GMD weighting"
    parameters["reuse_checkpoint"] = {
        "type": "boolean", "default": False,
        "description": "Allow a compatible saved checkpoint to skip detection. Applies only to batch runs.",
    }
    description = {
        "panel": JOB_NAME,
        "summary": "Locate electron spots with Picasso. Use tuning_only for a single-frame check or run "
                   "a batch to produce a localization checkpoint, VMI image and optional Abel spectrum.",
        "parameters": parameters,
        "manual": JOB_NAME,
        "outputs": [*TUNING_FILES, "checkpoint", *BATCH_FILES, *ABEL_FILES],
    }
    return (PanelOperation("centroiding", description, _prepare_agent, (JOB_NAME,), artifacts=_artifacts),)


def _prepare_agent(_jobs, params):
    snapshot = load_settings()
    for group, key, default in options():
        if key not in params:
            continue
        value = params[key]
        if type(default) is bool:
            boolean(value)
        elif type(default) is str:
            if not isinstance(value, str) or "\x00" in value:
                raise ValueError(f"Enter text for {key}.")
        elif value is not None or default is not None:
            integer = type(default) is int or key == "tuning_frame_index"
            if ((integer and type(value) is not int) or (not integer and type(value) not in (float, int))
                    or (type(value) is float and not isfinite(value))):
                raise ValueError(f"Enter a valid numeric value for {key}.")
        _section(snapshot, group)[key] = value

    vmi = snapshot["vmi_picasso"]
    proc = vmi["processing"]
    vmi["h5_dir"] = str(resolve_path(vmi["h5_dir"]))
    if not Path(vmi["h5_dir"]).is_dir():
        raise ValueError("Choose an existing HDF5 directory.")
    validate_h5_pattern(vmi["h5_pattern"])
    if "output_dir" in params:
        vmi["output_dir"] = str(resolve_path(vmi["output_dir"], write=True))
    checkpoint = vmi["checkpoint_file"]
    if not checkpoint.strip() or checkpoint in (".", "..") or any(char in checkpoint for char in "/\\:"):
        raise ValueError("Enter a checkpoint filename inside the output directory.")
    for key, choices in (("fitting_method", ("mle", "lq")), ("mle_method", ("sigma", "sigmaxy"))):
        if vmi[key] not in choices:
            raise ValueError(f"Choose a valid {key.replace('_', ' ')}.")
    for key, group, needed, label in (
        ("dark_file", vmi, proc["enable_dark_subtraction"], "precomputed dark .npy file"),
        ("flat_field_file", proc, proc["enable_flat_field"], "flat field file"),
        ("gmd_map_path", vmi, not proc["tuning_only"] and vmi["gmd"]["enable_gmd_weighting"], "GMD map"),
    ):
        path = resolve_path(group[key]) if group[key].strip() else None
        if needed and (path is None or not path.is_file() or (key == "dark_file" and path.suffix.lower() != ".npy")):
            raise ValueError(f"Choose an existing {label}.")
        group[key] = str(path) if path is not None else ""

    reuse = boolean(params.get("reuse_checkpoint", False))
    try:
        job = centroid_job(snapshot, reuse_checkpoint=reuse)
    except OverflowError as exc:
        raise ValueError("Enter finite Centroiding settings.") from exc
    summary = {"Mode": "Single-frame tuning" if proc["tuning_only"] else "Batch",
               "Output directory": job.extra["output_dir"]}
    if not proc["tuning_only"]:
        summary["Checkpoint"] = job.extra["checkpoint"]
        summary["Checkpoint policy"] = ("Reuse a compatible checkpoint if present and skip localization"
                                        if reuse else "Calculate new localizations")
    labels = {key: label for _, key, _, label in FIELDS}
    for group, key, _ in options():
        if key in ("output_dir", "tuning_only", "checkpoint_file", "meta_column_gmd"):
            continue
        value = _section(snapshot, group)[key]
        if (key in labels and value not in (None, "")) or key in params:
            summary[labels.get(key, key.replace("_", " ").capitalize())] = value
    return job, summary


def build(jobs: JobManager) -> None:
    def restore_form(job):
        form = adopt_agent_form(job, "centroid")
        for (group, key, _), widget in widgets.items():
            if f"{group}.{key}" in form:
                widget.value = form[f"{group}.{key}"]

    view = JobView(jobs, job_name=JOB_NAME, results_builder=_results, attach_builder=restore_form)
    theme.panel_header(
        "Picasso Centroiding",
        "Locate electron spots and build a VMI image. Use a single frame to tune detection.",
        view=view)

    doc = centroid_settings()
    widgets = {}

    def render_fields(group, caption=None, *, advanced=False):
        if caption:
            theme.form_section(caption)
        with field_grid():
            for section, key, kind, label in FIELDS:
                if section != group or (key in ADVANCED_KEYS) != advanced:
                    continue
                value = _section(doc, section).get(key, "")
                wid = persist(_make_widget(kind, label, value), storage_key(section, key))
                widgets[(section, key, kind)] = wid
                if key == "enable_median_subtraction":
                    wid.bind_enabled_from(widgets[(section, "enable_dark_subtraction", "bool")], "value", not_)
                    wid.tooltip("Subtract the 3σ-clipped median of each frame.")
                if key == "frame_limit":
                    wid.min = 0
                    wid.props("step=1")
                    wid.bind_enabled_from(widgets[(section, "tuning_only", "bool")], "value", not_)
                if key == "abel_rotation_deg":
                    wid.bind_enabled_from(widgets[(section, "enable_abel", "bool")], "value")
                    wid.props('hint="Align the polarization vertically. Use 0° if it is already vertical. '
                              'Positive angles rotate counterclockwise."')

    section_toggle_props = 'header-class="vmi-section-toggle text-primary" expand-icon-class="text-primary"'
    with theme.panel_card():
        render_fields("vmi_picasso", "Paths and detection")
        with ui.expansion("Advanced Options", icon="tune").classes("w-full").props(section_toggle_props):
            render_fields("vmi_picasso", advanced=True)
        render_fields("processing", "Processing")
        render_fields("histogram", "Histogram")
        with ui.expansion("Energy Calibration", icon="show_chart") \
                .classes("w-full").props(section_toggle_props):
            ui.label("E = α × r². Intensity is unchanged. Leave α empty for pixel². Xe fitting reports "
                     "a suggested α without changing your input. Fit limits "
                     "always use pixel².") \
                .classes("vmi-subtitle")
            render_fields("energy_calibration", "Energy calibration / Xe 4d fit")
        render_fields("gmd", "GMD normalization")

        def run():
            if not _save(widgets):
                return
            snapshot = centroid_settings()
            try:
                view.attach(jobs.submit(centroid_job(snapshot)))
            except (OSError, ValueError) as exc:
                ui.notify(str(exc), type="negative")

        def show_saved():
            if view.job is not None and view.job.status in ("running", "queued"):
                ui.notify("Wait for the current job to finish before loading saved results.", type="info")
                return
            view.results.clear()
            with view.results:
                try:
                    _result_files(_output_dir(centroid_settings()))
                except (OSError, ValueError) as exc:
                    ui.notify(str(exc), type="warning")
            view.open_console()
        with action_bar(view, on_click=run):
            ui.button("View saved results", icon="insights",
                      on_click=show_saved).props("flat color=secondary")
            ui.button("Clear outputs", icon="delete",
                      on_click=lambda: _clear_outputs(view)) \
                .props("flat color=negative").classes("vmi-action-danger")

    view.place_console()
    with ui.expansion("Checkpoint contents", icon="description", value=False).classes("w-full"):
        ui.label("The batch checkpoint stores one row per detected event in the HDF5 table /localizations. "
                 "Centroiding can resume from this table, and Binning uses it as input.")
        ui.label(
            "x, y          Fitted spot coordinates in detector pixels\n"
            "mass          Fitted photon count\n"
            "size          Mean fitted spot width (σx and σy) in pixels\n"
            "ecc           Eccentricity, 1 − smaller width / larger width\n"
            "signal        Photon count divided by π × size²\n"
            "frame         Zero-based frame index across the input files\n"
            "net_gradient  Picasso’s spot-detection score"
        ).classes("font-mono text-xs whitespace-pre-wrap select-text")
        ui.label("File attributes record the background subtraction mode. With a frame limit, "
                 "they also record the requested limit and the number of frames processed.")


def _make_widget(kind: str, label: str, value):
    if kind == "bool":
        return ui.checkbox(label, value=bool(value))
    if kind.startswith("select:"):
        options = kind.split(":", 1)[1].split(",")
        return ui.select(options, value=str(value) if value != "" else options[0],
                         label=label).props("outlined dense")
    if kind in ("int", "float"):
        return ui.number(label, value=value if value != "" else None) \
            .props("outlined dense")
    if kind == "dir":
        return path_input(label, str(value), browse="dir")
    if kind.startswith("file:"):
        filters = kind.split(":", 1)[1].split(",")
        return path_input(label, str(value), browse="file", filters=filters)
    return ui.input(label, value=str(value)).props("outlined dense")


def _save(widgets) -> bool:
    """Validate and snapshot the form before submitting; never write TOML."""
    doc = centroid_settings()
    for (section, key, kind), wid in widgets.items():
        sec = _section(doc, section)
        v = wid.value
        if kind == "bool":
            sec[key] = bool(v)
        elif kind in ("int", "float"):
            if key == "frame_limit":
                sec[key] = v or 0
            elif (v is None or v == "") and section == "energy_calibration":
                sec[key] = None
            elif (v is None or v == "") and key in ("tuning_frame_index", "mask_center_x",
                                                   "mask_center_y", "mask_radius"):
                sec[key] = None
            else:
                sec[key] = (int(v or 0) if kind == "int"
                            else float(v or 0.0))
        else:
            sec[key] = str(v)
    try:
        validate_settings(doc)
    except ValueError as exc:
        ui.notify(str(exc), type="negative")
        return False
    for section, key, _ in widgets:
        settings_store()[storage_key(section, key)] = _section(doc, section)[key]
    return True


def _clear_outputs(view: JobView) -> None:
    """Delete centroiding outputs so the next Run starts fresh.

    The script skips the whole batch when the checkpoint exists
    ("[Resume] Found checkpoint"), so a stale checkpoint must be removed
    explicitly to force re-localization.
    """
    doc = centroid_settings()
    out_dir = _output_dir(doc)
    vmi = doc["vmi_picasso"]
    inputs = (("HDF5 directory", vmi["h5_dir"]), ("GMD map", vmi["gmd_map_path"]),
              ("dark frame", vmi["dark_file"]), ("flat field", vmi["processing"]["flat_field_file"]))
    try:
        out_dir = checked_path(out_dir, write=True)
        checkpoint = validate_distinct_output(out_dir / vmi["checkpoint_file"], inputs,
                                              cwd=centroid_dir(CENTROID_DIR), output_label="checkpoint")
        checkpoint = checked_path(checkpoint, write=True)
        for name in OUTPUT_NAMES:
            validate_distinct_output(out_dir / name, inputs, cwd=centroid_dir(CENTROID_DIR))
    except (OSError, ValueError) as exc:
        ui.notify(str(exc), type="warning")
        return
    existing = [out_dir / name for name in OUTPUT_NAMES if (out_dir / name).is_file()]
    has_checkpoint = checkpoint.is_file()
    if not existing and not has_checkpoint:
        ui.notify(f"Nothing to delete in {out_dir}", type="info")
        return
    expected = {path: path.lstat() for path in existing}
    if has_checkpoint:
        expected[checkpoint] = checkpoint.lstat()

    def _size(p: Path) -> str:
        n = p.stat().st_size
        return (f"{n / 2**30:.2f} GiB" if n >= 2**30 else
                f"{n / 2**20:.1f} MiB" if n >= 2**20 else f"{n / 2**10:.1f} KiB")

    with ui.dialog() as dlg, ui.card():
        ui.label("Delete centroiding outputs?").classes("vmi-caption")
        for p in existing:
            ui.label(f"{p.name}  ({_size(p)})").style("font-family: monospace")
        with ui.row().classes("items-center gap-2"):
            chk = ui.checkbox(value=has_checkpoint)
            warn = (" — regenerating it takes a full batch run"
                    if has_checkpoint else "")
            ui.label(f"Also delete checkpoint {checkpoint.name} "
                     f"({_size(checkpoint) if has_checkpoint else 'missing'})"
                     f"{warn}").style("font-family: monospace")
        with ui.row().classes("gap-2"):
            ui.button("Delete", color="negative",
                      on_click=lambda: do_delete(chk.value))
            ui.button("Cancel", on_click=dlg.close).props("flat")
    dlg.open()

    def do_delete(with_checkpoint: bool) -> None:
        doomed = list(existing)
        if with_checkpoint and has_checkpoint:
            doomed.append(checkpoint)
        failed = delete_outputs(doomed, expected=expected)
        if failed is None:
            return
        dlg.close()
        view.clear_results()
        if failed:
            ui.notify("Could not delete: " + "; ".join(failed),
                      type="negative")
        else:
            ui.notify(f"Deleted {len(doomed)} file(s) from {out_dir}",
                      type="positive")


def _results(job: Job):
    if job.status != "done":
        return
    out_dir = Path(job.extra["output_dir"])
    _result_files(out_dir, tuning_only=job.extra.get("tuning_only") if job.extra.get("agent_panel") else None)


def _interactive_plot(figure, *, tuning=False):
    controls = []
    with ui.row().classes("gap-4 flex-wrap"):
        for index, trace in enumerate(figure.data):
            if trace.type != "heatmap":
                continue
            name = "Tuning" if tuning else "Raw VMI" if index == 0 else "Abel VMI"
            kind = "tuning" if tuning else "raw" if index == 0 else "abel"
            key = f"centroid.{kind}_colormap"
            colors = persist(ui.select(list(COLORMAPS), value=RESULT_COLORMAP_DEFAULTS[key],
                                       label=f"{name} colormap").classes("w-52").props("outlined dense"),
                             key)
            trace.colorscale = get_colorscale(colors.value)
            controls.append((index, colors))

    if not tuning and any(trace.xaxis == "x3" for trace in figure.data):
        axes = _section(centroid_settings(), "spectrum_plot")
        with ui.expansion("Summary spectrum plot axes", icon="show_chart").classes("w-full"):
            with field_grid():
                limits = {
                    (axis, bound): persist(
                        num_input(f"Spectrum {axis} {bound} (-1 = auto)", axes[f"{axis}_{bound}"], step="any"),
                        storage_key("spectrum_plot", f"{axis}_{bound}"),
                    )
                    for axis in ("x", "y") for bound in ("min", "max")
                }
            ui.button("Apply spectrum axes", on_click=lambda: apply_limits()).props("flat")

        def axis_layout():
            return {
                f"{axis}axis3": linear_axis_limits(*(
                    None if limits[axis, bound].value == -1 else limits[axis, bound].value
                    for bound in ("min", "max")
                ))
                for axis in ("x", "y")
            }

        def apply_limits():
            try:
                settings = axis_layout()
            except ValueError as exc:
                ui.notify(str(exc), type="warning")
                return
            plot.run_plot_method("relayout", {f"{axis}.{key}": value
                                            for axis, options in settings.items() for key, value in options.items()})

        try:
            for axis, options in axis_layout().items():
                figure.layout[axis].update(options)
        except ValueError as exc:
            ui.notify(str(exc), type="warning")

    plot = ui.plotly(plot_options(figure)).classes("w-full").style(f"height: {PLOT_HEIGHT}px")

    def recolor(index, event):
        scale = get_colorscale(event.value)
        plot.figure["data"][index]["colorscale"] = scale
        plot.run_plot_method("restyle", {"colorscale": [scale]}, [index])

    for index, colors in controls:
        colors.on_value_change(lambda event, index=index: recolor(index, event))
    return plot


def _result_files(out_dir: Path, *, tuning_only=None):
    out_dir = checked_path(out_dir)
    open_folder_button(out_dir, "open output folder")
    found = False
    for title, data_file, png_name, builder in (
        ("Tuning check", TUNING_DATA, "01_tuning_check_picasso.png", tuning_figure),
        ("Final summary", SUMMARY_DATA, "04_final_summary.png", summary_figure),
    ):
        if tuning_only is not None and (data_file == TUNING_DATA) != tuning_only:
            continue
        data, png = checked_path(out_dir / data_file), checked_path(out_dir / png_name)
        if not data.exists() and not png.exists():
            continue
        found = True
        ui.label(title).classes("vmi-caption")
        interactive = False
        if data.exists():
            try:
                figure = builder(data)
                ui.label("Drag inside a plot to zoom; wheel to zoom, Shift-drag to pan, double-click to reset. Hover for values.") \
                    .classes("vmi-subtitle")
                _interactive_plot(figure, tuning=data_file == TUNING_DATA)
                interactive = True
            except (ValueError, OSError, KeyError) as exc:
                ui.label(f"Interactive data could not be loaded: {exc}").classes("text-negative")
        elif png.exists():
            ui.label("Saved PNG only. Re-run this stage once to generate interactive data.").classes("vmi-subtitle")
        if png.exists():
            if interactive:
                with ui.expansion("Saved PNG", icon="image").classes("w-full"):
                    zoomable_image(file_url(png)).classes("w-full max-w-4xl")
            else:
                zoomable_image(file_url(png)).classes("w-full max-w-4xl")
    if not found:
        ui.label("No saved Centroiding plots found in this output directory.").classes("vmi-subtitle")
