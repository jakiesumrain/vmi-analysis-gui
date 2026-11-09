"""Panel: Picasso centroiding with saved fields and explicit CLI inputs."""

import sys
from pathlib import Path

from nicegui import app, ui
from plotly.colors import get_colorscale
from vmi_analysis.centroiding.options import centroid_cli_args, section as _section, validate_settings
from vmi_analysis.centroiding.plot_data import TUNING_DATA, SUMMARY_DATA

from ..jobs import Job, JobManager
from .. import theme
from ..centroid_plots import COLORMAPS, PLOT_HEIGHT, tuning_figure, summary_figure, plot_options
from ..settings import centroid_settings, storage_key, output_dir as _output_dir
from .common import (JobView, REPO_ROOT, file_url, open_folder_button,
                     path_input, persist, zoomable_image, delete_outputs)

CENTROID_DIR = REPO_ROOT / "centroiding"

# (section path, key, kind, label) — kind: str/int/float/bool/select:opts
FIELDS = [
    ("vmi_picasso", "h5_dir", "dir", "HDF5 directory"),
    ("vmi_picasso", "h5_pattern", "str", "HDF5 pattern"),
    ("vmi_picasso", "h5_dataset", "str", "Frame dataset"),
    ("vmi_picasso", "h5_meta", "str", "Metadata dataset"),
    ("vmi_picasso", "gmd_map_path", "file:.txt", "GMD map file (empty = raw/meta)"),
    ("vmi_picasso", "dark_data_dir", "dir", "Dark HDF5 directory"),
    ("vmi_picasso", "dark_h5_pattern", "str", "Dark pattern"),
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
    ("vmi_picasso", "tuning_frame_index", "int", "Tuning frame index"),
    ("processing", "enable_parallel", "bool", "Parallel processing"),
    ("processing", "num_cores", "int", "CPU cores"),
    ("processing", "enable_dark_subtraction", "bool", "Dark subtraction"),
    ("processing", "dark_method", "select:median,mean", "Dark method"),
    ("processing", "dark_frame_count", "int", "Dark frame count"),
    ("processing", "enable_flat_field", "bool", "Flat field"),
    ("processing", "flat_field_file", "file:.txt,.npy", "Flat field file"),
    ("processing", "enable_circular_mask", "bool", "Circular mask"),
    ("processing", "mask_center_x", "float", "Mask center X (empty = image center)"),
    ("processing", "mask_center_y", "float", "Mask center Y (empty = image center)"),
    ("processing", "mask_radius", "float", "Mask radius (empty = automatic)"),
    ("processing", "tuning_only", "bool", "Tuning mode only"),
    ("histogram", "enable_abel", "bool", "Abel inversion"),
    ("histogram", "abel_rotation_deg", "float", "Abel rotation (° CCW)"),
    ("histogram", "resolution_factor", "int", "Histogram resolution factor"),
    ("energy_calibration", "alpha", "float", "α (eV/pixel²; empty = uncalibrated)"),
    ("energy_calibration", "fit_xe_4d", "bool", "Fit Xe 4d doublet → hν and α in console"),
    ("energy_calibration", "fit_min_px2", "float", "Xe fit min (pixel²; empty = auto)"),
    ("energy_calibration", "fit_max_px2", "float", "Xe fit max (pixel²; empty = auto)"),
    ("spectrum_plot", "x_min", "float", "Spectrum x min (-1 = auto)"),
    ("spectrum_plot", "x_max", "float", "Spectrum x max (-1 = auto)"),
    ("spectrum_plot", "y_min", "float", "Spectrum y min (-1 = auto)"),
    ("spectrum_plot", "y_max", "float", "Spectrum y max (-1 = auto)"),
    ("gmd", "enable_gmd_weighting", "bool", "GMD weighting"),
    ("gmd", "gmd_weight_min", "float", "GMD weight min"),
    ("gmd", "gmd_weight_max", "float", "GMD weight max"),
]

GROUPS = {
    "vmi_picasso": "Paths, detection and fitting",
    "processing": "Processing",
    "histogram": "Histogram",
    "energy_calibration": "Energy calibration / Xe 4d fit",
    "spectrum_plot": "Summary spectrum plot axes",
    "gmd": "GMD normalization",
}


def build(jobs: JobManager) -> None:
    theme.section_title(
        "Picasso Centroiding",
        "Settings are saved automatically. Run the Picasso localization "
        "(process_andor_vmi_PICASSO_v5_h5_batch.py). "
        "Set 'Tuning mode only' to optimize detection on a single frame first.")

    view = JobView(jobs, job_name='Centroiding', results_builder=_results)

    doc = centroid_settings()
    widgets = {}

    with theme.panel_card():
        for group, caption in GROUPS.items():
            ui.label(caption).classes("vmi-caption")
            if group == "energy_calibration":
                ui.label("E = α × r²; intensity is unchanged. Leave α empty for pixel². Xe fitting reports "
                         "a suggested α without changing your input. Fit limits "
                         "always use pixel²; spectrum axis limits use the displayed units.") \
                    .classes("vmi-subtitle")
            fields = [f for f in FIELDS if f[0] == group]
            with ui.grid(columns=2).classes("w-full gap-x-4"):
                for section, key, kind, label in fields:
                    sec = _section(doc, section)
                    value = sec.get(key, "")
                    wid = persist(_make_widget(kind, label, value), storage_key(section, key))
                    widgets[(section, key, kind)] = wid
                    if key == "abel_rotation_deg":
                        wid.bind_enabled_from(widgets[(section, "enable_abel", "bool")], "value")

        def run():
            if not _save(widgets):
                return
            snapshot = centroid_settings()
            try:
                cmd = [sys.executable, "-m", "vmi_analysis.centroiding.process_andor_vmi_PICASSO_v5_h5_batch", *centroid_cli_args(snapshot)]
            except ValueError as exc:
                ui.notify(str(exc), type="negative")
                return
            job = jobs.submit(Job(name="Centroiding", cmd=cmd,
                                  cwd=str(CENTROID_DIR),
                                  extra={"output_dir": str(_output_dir(snapshot))}))
            view.attach(job)

        with ui.row().classes("gap-2"):
            view.run_button(on_click=run)
            def show_saved():
                if view.job is not None and view.job.status in ("running", "queued"):
                    ui.notify("Wait for the current job to finish before loading saved results.", type="info")
                    return
                view.results.clear()
                with view.results:
                    _result_files(_output_dir(centroid_settings()))
            ui.button("View saved results", icon="insights", on_click=show_saved).props("flat color=secondary")
            ui.button("Clear outputs", icon="delete",
                      on_click=_clear_outputs).props("flat color=negative")

    view.place_console()


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
            if (v is None or v == "") and section in ("spectrum_plot", "energy_calibration"):
                # Empty optional values are omitted so the script uses defaults.
                sec[key] = None if section == "energy_calibration" else -1.0
            elif v is None and key in ("mask_center_x", "mask_center_y", "mask_radius"):
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
        app.storage.general[storage_key(section, key)] = _section(doc, section)[key]
    return True


def _clear_outputs() -> None:
    """Delete centroiding outputs so the next Run starts fresh.

    The script skips the whole batch when the checkpoint exists
    ("[Resume] Found checkpoint"), so a stale checkpoint must be removed
    explicitly to force re-localization.
    """
    doc = centroid_settings()
    out_dir = _output_dir(doc)
    checkpoint = out_dir / str(doc["vmi_picasso"].get("checkpoint_file", ""))
    targets = [out_dir / name for name in (
        "01_tuning_check_picasso.png", "04_final_summary.png",
        TUNING_DATA, SUMMARY_DATA, "vmi_2d_raw.npy", "vmi_3d_abel.npy", "energy_spectrum.csv")]
    existing = [p for p in targets if p.exists()]
    has_checkpoint = checkpoint.exists()
    if not existing and not has_checkpoint:
        ui.notify(f"Nothing to delete in {out_dir}", type="info")
        return

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
        failed = delete_outputs(doomed)
        if failed is None:
            return
        dlg.close()
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
    _result_files(out_dir)


def _interactive_plot(figure, *, tuning=False):
    controls = []
    with ui.row().classes("gap-4 flex-wrap"):
        for index, trace in enumerate(figure.data):
            if trace.type != "heatmap":
                continue
            name = "Tuning" if tuning else "Raw VMI" if index == 0 else "Abel VMI"
            key = "tuning" if tuning else "raw" if index == 0 else "abel"
            colors = persist(ui.select(list(COLORMAPS), value="Gray" if tuning else "Turbo",
                                       label=f"{name} colormap").classes("w-52").props("outlined dense"),
                             f"centroid.{key}_colormap")
            trace.colorscale = get_colorscale(colors.value)
            controls.append((index, colors))
    plot = ui.plotly(plot_options(figure)).classes("w-full").style(f"height: {PLOT_HEIGHT}px")

    def recolor(index, event):
        scale = get_colorscale(event.value)
        plot.figure["data"][index]["colorscale"] = scale
        plot.run_plot_method("restyle", {"colorscale": [scale]}, [index])

    for index, colors in controls:
        colors.on_value_change(lambda event, index=index: recolor(index, event))
    return plot


def _result_files(out_dir: Path):
    open_folder_button(out_dir, "open output folder")
    found = False
    for title, data_file, png_name, builder in (
        ("Tuning check", TUNING_DATA, "01_tuning_check_picasso.png", tuning_figure),
        ("Final summary", SUMMARY_DATA, "04_final_summary.png", summary_figure),
    ):
        data, png = out_dir / data_file, out_dir / png_name
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
