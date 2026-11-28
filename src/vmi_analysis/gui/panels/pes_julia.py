"""Generate spectra, fit peaks, and estimate FWHM-window uncertainties."""

import csv
from importlib.resources import files
import io
import json
from math import isclose
from pathlib import Path
import sys

import h5py
from nicegui import ui
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vmi_analysis.paths import WORKSPACE_ROOT
from vmi_analysis.pes import (BASELINE_PARAMETERS, anisotropy_values, default_uniform_bin_width,
                              evaluate_fit_for_plot, integrate_spectral_bins, uniform_bin_edges)
from vmi_analysis.pes_julia import (
    DEFAULT_CONFIDENCE, DEFAULT_REPLICATES, DEFAULT_SEED, HDF5_SUFFIXES, MODES, ROTATIONS,
    analysis_options, input_file, julia_executable, legacy_energy_grid, pbasex_repo_path,
)
from vmi_analysis.pes_julia_fit import PROFILES, fit_settings, load_fit, load_spectrum

from .. import shared, theme
from ..agent_panels import PanelOperation, require_completed_job
from ..centroid_plots import SERIES_COLORS, _style, linear_axis_limits, plot_options
from ..jobs import Job, ResourceBusyError
from .common import JobView, action_bar, adopt_agent_form, field_grid, num_input, path_input, persist
from .pes import BETA_AXIS_SPACING, PES_SAMPLE_MARKER, _aligned_beta_ranges, _csv_download, _style_comparison_plot

JOB_NAME = "PES (Julia)"
FIT_JOB_NAME = "PES (Julia fit)"
ERROR_JOB_NAME = "PES (Julia uncertainty)"
JOB_NAMES = (JOB_NAME, FIT_JOB_NAME, ERROR_JOB_NAME)
OPERATION = "pes_julia"
FIT_OPERATION = "pes_julia_fit"
ERROR_OPERATION = "pes_julia_uncertainties"
SPECTRUM = "pes_julia_spectrum"
FIT = "pes_julia_peak_fit"
RESULT = "pes_julia_csv"
DEFAULT_WINDOW_BATCH = 8
LOCAL_MESSAGE = "Julia pBASEX requires a local server with a configured Julia environment."
_PLOT_SCRIPT = (files("vmi_analysis.gui.assets") / "pes_julia.js").read_text(encoding="utf-8")
_INVERSION = ("alpha", "resolution", "rotation", "regularization", "center_x", "center_y")
_BOOTSTRAP = ("replicates", "confidence", "block_length", "seed", "window_batch")


def init_plots():
    ui.add_body_html(f"<script>{_PLOT_SCRIPT}</script>", shared=True)


def _local():
    if shared.SERVER:
        raise ValueError(LOCAL_MESSAGE)


def _path(value, *, cwd, label):
    if not isinstance(value, (str, Path)) or not value:
        raise ValueError(f"Choose the {label}.")
    return shared.checked_path(value, cwd=cwd)


def _job(name, module, arguments, reads, artifact, suffix, *, cwd):
    job = Job(name, cmd=[sys.executable, "-m", module, *map(str, arguments)], cwd=str(cwd),
              reads=list(map(str, reads)))
    path = shared.generated_dir() / f"{artifact.replace('_', '-')}-{job.id}{suffix}"
    job.cmd.extend(("--output", str(path)))
    job.writes.append(str(path))
    job.extra[artifact] = str(path)
    return job


def _julia_job(name, mode, source, basis, repo, options, arguments, artifact, *, cwd, reads=()):
    julia_executable()
    args = [mode, source, basis, "--pbasex-repo", repo, *arguments]
    for key, value in options.items():
        if value is not None and key != "windows":
            args.extend(("--" + key.replace("_", "-"), value))
    return _job(name, "vmi_analysis.pes_julia", args, (source, basis, repo, *reads), artifact, ".csv", cwd=cwd)


def spectrum_job(source, basis, pbasex_repo, **settings):
    _local()
    cwd = shared.workspace_root()
    options = analysis_options("counts", **settings)
    source = input_file(_path(source, cwd=cwd, label="integer count image"), (".npy",), "integer count image")
    basis = input_file(_path(basis, cwd=cwd, label="pBASEX basis"), HDF5_SUFFIXES, "pBASEX basis")
    repo = pbasex_repo_path(_path(pbasex_repo, cwd=cwd, label="PBASEX-Julia checkout"))
    counts = np.load(source, mmap_mode="r", allow_pickle=False)
    if not isinstance(counts, np.ndarray) or counts.ndim != 2 or not counts.size or counts.dtype.kind not in "iu":
        raise ValueError("Choose a nonempty 2D integer count image.")
    rows, columns = counts.shape
    if options["center_x"] is None:
        options.update(center_x=columns // 2 + 1, center_y=rows // 2 + 1)
    if not (1 <= options["center_x"] <= columns and 1 <= options["center_y"] <= rows):
        raise ValueError("The center is outside the count image.")
    job = _julia_job(JOB_NAME, "counts", source, basis, repo, options, ("--spectrum-only",), SPECTRUM, cwd=cwd)
    job.extra["image_shape"] = [rows, columns]
    job.extra["panel_form"] = {**{key: options[key] for key in _INVERSION}, "counts": str(source),
                               "basis": str(basis), "pbasex_repo": str(repo)}
    return job


def peak_fit_job(parent, region, peaks=(), baseline="Constant", *, auto_profile="Gaussian"):
    _local()
    spectrum = input_file(shared.checked_path(parent.extra[SPECTRUM]), (".csv",), "native spectrum")
    region, peaks, baseline = fit_settings(region, peaks, baseline, auto_profile=auto_profile)
    job = _job(FIT_JOB_NAME, "vmi_analysis.pes_julia_fit",
               (spectrum, "--region", *region, "--peaks", json.dumps(peaks), "--baseline", baseline,
                "--auto-profile", auto_profile),
               (spectrum,), FIT, ".json", cwd=shared.workspace_root())
    job.extra.update(spectrum_job_id=parent.id,
                     panel_form=dict(region=region, peaks=peaks, baseline=baseline, auto_profile=auto_profile))
    return job


def uncertainty_job(jobs, fit_job, mode="counts", shots="", *, window_batch=DEFAULT_WINDOW_BATCH, **settings):
    _local()
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError("Choose an integer count image or shot-resolved checkpoint.")
    parent = require_completed_job(jobs, fit_job.extra["spectrum_job_id"], name=JOB_NAME)
    cwd = shared.workspace_root()
    record_path = input_file(shared.checked_path(fit_job.extra[FIT]), (".json",), "peak fit record")
    spectrum = parent.extra[SPECTRUM]
    with jobs.preview_read_access([record_path]):
        record = load_fit(record_path)
    if Path(record["spectrum"]) != Path(spectrum):
        raise ValueError("The peak fit record belongs to a different spectrum.")
    form = parent.extra["panel_form"]
    source = form["counts"]
    if mode == "shots":
        source = input_file(_path(shots, cwd=cwd, label="matching shot-resolved checkpoint"),
                            HDF5_SUFFIXES, "shot-resolved checkpoint")
        with jobs.preview_read_access([source]), h5py.File(source, "r") as checkpoint:
            geometry = checkpoint.get("geometry")
            if (geometry is None or "x_edges" not in geometry or "y_edges" not in geometry
                    or [len(geometry["y_edges"]) - 1, len(geometry["x_edges"]) - 1] != parent.extra["image_shape"]
                    or not isclose(geometry.attrs.get("resolution_factor", 0), form["resolution"])):
                raise ValueError("The checkpoint must match the count image's histogram shape and resolution.")
    options = analysis_options(mode, **{key: form[key] for key in _INVERSION},
                               windows=[peak["window"] for peak in record["peaks"]],
                               window_batch=window_batch, **settings)
    job = _julia_job(ERROR_JOB_NAME, mode, source, form["basis"], form["pbasex_repo"], options,
                     ("--fit-record", record_path), RESULT, cwd=cwd, reads=(record_path,))
    panel_form = dict(mode=mode)
    if mode == "shots":
        panel_form.update(shots=str(source), **{key: options[key] for key in _BOOTSTRAP})
    job.extra.update(spectrum_job_id=parent.id, fit_job_id=fit_job.id, peak_windows=record["peaks"],
                     alpha=form["alpha"], panel_form=panel_form)
    return job


def _prepare_spectrum(jobs, params):
    settings = dict(params)
    source = settings.pop("input", "")
    if "source_job_id" in settings:
        if source:
            raise ValueError("Supply a count image or a completed Circularize checkpoint job.")
        parent = require_completed_job(jobs, settings.pop("source_job_id"), name="Circularize")
        source = parent.extra.get("counts_output", "")
    job = spectrum_job(source, settings.pop("basis", ""), settings.pop("pbasex_repo", ""), **settings)
    form = job.extra["panel_form"]
    return job, {"Count image": form["counts"], "Basis": form["basis"], "PBASEX-Julia checkout": form["pbasex_repo"],
                 "Inversion": {key: form[key] for key in _INVERSION}}


def _prepare_fit(jobs, params):
    parent = require_completed_job(jobs, params.get("spectrum_job_id"), name=JOB_NAME)
    job = peak_fit_job(parent, params.get("region"), params.get("peaks", []), params.get("baseline", "Constant"),
                       auto_profile=params.get("auto_profile", "Gaussian"))
    return job, {"Spectrum": parent.extra[SPECTRUM], **job.extra["panel_form"]}


def _prepare_uncertainty(jobs, params):
    settings = dict(params)
    fit_job = require_completed_job(jobs, settings.pop("fit_job_id", None), name=FIT_JOB_NAME)
    job = uncertainty_job(jobs, fit_job, **settings)
    form = job.extra["panel_form"]
    return job, {"Estimator": MODES[form["mode"]], "Input": job.reads[0], "Fit": fit_job.extra[FIT],
                 "FWHM windows": [peak["window"] for peak in job.extra["peak_windows"]],
                 **{key: form[key] for key in _BOOTSTRAP if key in form}}


def agent_operations():
    def operation(key, name, summary, parameters, prepare, output):
        return PanelOperation(key, dict(panel=name, manual="PES → PBASEX", summary=summary,
                                        parameters=parameters), prepare, (name,), outputs=(output,))

    return (
        operation(OPERATION, JOB_NAME, "Generate a native-grid pBASEX spectrum and β from integer counts.", {
            "input": {"type": "path", "description": "Integer count image (.npy)."},
            "source_job_id": {"type": "job ID", "description": "Completed Circularize checkpoint job, alternative to input."},
            "basis": {"type": "path", "required": True},
            "pbasex_repo": {"type": "path", "description": "Local PBASEX-Julia checkout.", "required": True},
            "alpha": {"type": "number or null", "unit": "eV/detector pixel²"},
            "resolution": {"type": "number", "unit": "bins/detector pixel", "default": 1},
            "rotation": {"type": "integer", "choices": list(ROTATIONS), "unit": "degrees counterclockwise"},
            "regularization": {"type": "number", "default": 0},
            "center_x": {"type": "integer or null", "description": "One-based column before rotation."},
            "center_y": {"type": "integer or null", "description": "One-based row before rotation."},
        }, _prepare_spectrum, SPECTRUM),
        operation(FIT_OPERATION, FIT_JOB_NAME, "Fit selected peaks to a completed native pBASEX spectrum.", {
            "spectrum_job_id": {"type": "job ID", "required": True},
            "region": {"type": "pair of numbers", "description": "Fit limits in the spectrum's units."},
            "peaks": {"type": "list of objects", "description": "Omit for automatic estimates. Each manual peak uses profile, center, gaussian_fwhm, and optional lorentzian_fwhm."},
            "auto_profile": {"type": "string", "choices": list(PROFILES), "default": "Gaussian"},
            "baseline": {"type": "string", "choices": list(BASELINE_PARAMETERS)},
        }, _prepare_fit, FIT),
        operation(ERROR_OPERATION, ERROR_JOB_NAME, "Calculate yield and β uncertainties in saved fitted FWHM windows.", {
            "fit_job_id": {"type": "job ID", "required": True},
            "mode": {"type": "string", "choices": list(MODES), "default": "counts"},
            "shots": {"type": "path", "description": "Matching HDF5 checkpoint, required in shot mode."},
            "replicates": {"type": "integer", "default": DEFAULT_REPLICATES},
            "confidence": {"type": "number", "default": DEFAULT_CONFIDENCE},
            "block_length": {"type": "integer", "default": 1},
            "seed": {"type": "integer", "default": DEFAULT_SEED},
            "window_batch": {"type": "integer", "default": DEFAULT_WINDOW_BATCH},
        }, _prepare_uncertainty, RESULT),
    )


def _spectrum_trace(x, y, name):
    return dict(x=x.tolist(), y=y.tolist(), name=name, mode="lines+markers", marker=PES_SAMPLE_MARKER,
                hovertemplate="%{x:.6g}, %{y:.6g}<extra>%{fullData.name}</extra>")


def _add_beta_axes(fig, x, beta, limits, *, subplot=1):
    ranges = _aligned_beta_ranges(beta.values(), limits)
    suffix = str(subplot) if subplot > 1 else ""
    for index, ((order, values), axis_range) in enumerate(zip(beta.items(), ranges)):
        axis_suffix = str(subplot + index) if index else suffix
        color = SERIES_COLORS[1 + index % (len(SERIES_COLORS) - 1)]
        label = f"β{order}"
        fig.add_scatter(x=x.tolist(), y=np.where(np.isfinite(values), values, None).tolist(),
                        xaxis=f"x{suffix}", yaxis=f"y{axis_suffix}", name=label,
                        mode="lines+markers", marker=PES_SAMPLE_MARKER, line_color=color)
        axis = dict(title_text=f"{label} (dimensionless)", title_font_color=color, tickfont_color=color,
                    linecolor=color, showline=True, showgrid=index == 0, zeroline=index == 0,
                    range=axis_range, autorange=False, fixedrange=True)
        if index:
            axis.update(overlaying=f"y{suffix}", anchor="free", side="left", position=0,
                        shift=-BETA_AXIS_SPACING * index, automargin=False)
        fig.update_layout({f"yaxis{axis_suffix}": axis})


def _plot_data(fig):
    options = plot_options(fig)
    options["config"]["responsive"] = False
    return options


def _chart():
    chart = ui.plotly(_plot_data(go.Figure())).classes("w-full")
    chart.on("plotly_afterplot", js_handler=f"""() => requestAnimationFrame(() => {{
        const chart = getElement({chart.id});
        if (!chart?.$el?.isConnected) return;
        vmiObserveJuliaPesPlot(chart);
        vmiUpdatePesAxes(chart);
    }})""")
    return chart


def _fit_figure(record, energy, intensity, axis_title):
    x = np.asarray(record["fit_x"])
    grid, total, baseline, components = evaluate_fit_for_plot(x, np.asarray(record["background"]), record["peaks"])
    # The saved fit coordinates are native samples and are also included in the model grid.
    measured = intensity[np.searchsorted(energy, x)]
    residual = measured - total[np.searchsorted(grid, x)]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.74, 0.26], vertical_spacing=0.09)
    fig.add_scatter(x=x.tolist(), y=measured.tolist(), mode="markers", name="Spectrum",
                    marker={"size": 8}, row=1, col=1)
    fig.add_scatter(x=grid.tolist(), y=total.tolist(), mode="lines", name="Total fit", row=1, col=1)
    has_baseline = record["baseline"] != "None"
    if has_baseline:
        fig.add_scatter(x=grid.tolist(), y=baseline.tolist(), mode="lines", name="Baseline",
                        line={"dash": "dot"}, row=1, col=1)
    suffix = " + baseline" if has_baseline else ""
    for index, peak in enumerate(components, 1):
        fig.add_scatter(x=grid.tolist(), y=(baseline + peak).tolist(), mode="lines", name=f"Peak {index}{suffix}",
                        line={"dash": "dash"}, row=1, col=1)
    fig.add_scatter(x=x.tolist(), y=residual.tolist(), name="Residual", showlegend=False, row=2, col=1)
    _style(fig, "Peak fit and residuals")
    fig.update_layout(height=620, showlegend=True)
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(title_text=axis_title, row=2, col=1)
    fig.update_yaxes(title_text="Intensity", row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=2, col=1)
    return fig


def _axis_controls(title, button_label, apply_label, apply, build_fields, *, selector=None, note=None):
    def toggle():
        panel.set_visibility(not panel.visible)
        button.props(f'aria-expanded="{str(panel.visible).lower()}"')

    with ui.row().classes("w-full items-center gap-x-3 gap-y-1"):
        ui.label(title).classes("font-mono text-lg text-primary")
        if selector:
            selector()
        button = ui.button(button_label, icon="show_chart", on_click=toggle).props('flat aria-expanded="false"')
    with ui.column().classes("w-full gap-2") as panel:
        panel.set_visibility(False)
        if note:
            ui.label(note).classes("vmi-subtitle")
        fields = build_fields()
        ui.button(apply_label, on_click=apply).props("flat")
    return fields


def _axis_fields(label, prefix, *, auto_label=False):
    with field_grid():
        return tuple(persist(num_input(f"{label} {bound}" + (" (blank = auto)" if auto_label else ""),
                                       None, step="any").props('placeholder="Auto"'),
                             f"{prefix}_{bound}") for bound in ("min", "max"))


def _axis_values(fields):
    low, high = (field.value for field in fields)
    linear_axis_limits(low, high)
    return low, high


def _uniform_data(energy, intensity, beta, width):
    edges = uniform_bin_edges(energy[0], energy[-1], width)
    moments = [intensity, *(np.where(intensity == 0, 0, intensity * values) for values in beta.values())]
    integrated = integrate_spectral_bins(energy, moments, edges)
    return (edges[:-1] + np.diff(edges) / 2, integrated[0] / np.diff(edges),
            dict(zip(beta, anisotropy_values(integrated))))


def _show_uniform(energy, intensity, beta, unit, x_label, name):
    prefix = f"{OPERATION}.uniform_{'ev' if unit == 'eV' else 'pixel2'}_"
    key = prefix + "bin_width"
    store = shared.settings_store()
    configured_unit = "eV" if store.get(f"{OPERATION}.alpha") is not None else "pixel²"
    prior_width = store.get(f"{OPERATION}.delta_e") or legacy_energy_grid(store.get(f"{OPERATION}.windows")).get("delta_e")
    default_width = default_uniform_bin_width(energy[0], energy[-1])
    initial_width = store.get(key) or (prior_width if unit == configured_unit else None) or default_width
    expansion_props = 'header-class="vmi-section-toggle text-primary" hide-expand-icon'
    settings = {}
    chart = None

    def field(label, setting):
        settings[setting] = num_input(label, store.get(prefix + setting), step="any").props('placeholder="Auto"')
        return settings[setting]

    def range_fields(label, axis):
        with field_grid():
            return tuple(field(f"{label} {bound}", f"{axis}_{bound}") for bound in ("min", "max"))

    with ui.expansion("Uniform-grid spectra", icon="show_chart", value=False).classes("w-full").props(
            expansion_props) as section:
        with ui.expansion("Plot settings", icon="tune", value=False).classes("w-full").props(expansion_props):
            with ui.column().classes("w-full"):
                with ui.row().classes("items-center gap-2"):
                    width = num_input(f"Bin width ({unit})", initial_width, step="any")
                    x_fields = tuple(field(f"{x_label} {bound} ({unit})", f"x_{bound}") for bound in ("min", "max"))
                intensity_fields = range_fields("Bin-average intensity Y", "intensity_y")
                beta_fields = [range_fields(f"Resampled β{order} Y", f"beta_{order}_y") for order in beta]
                ui.button("Apply plot settings", on_click=lambda: redraw(save=True)).props("flat")
                ui.label("Intensity is averaged over each bin. Edge bins include only the measured yield."
                         + (" β ranges can extend to align zero." if beta else "")).classes("vmi-subtitle")
        plots = ui.column().classes("w-full gap-0")

    def redraw(*, save=False):
        nonlocal chart
        try:
            x, y, resampled_beta = _uniform_data(energy, intensity, beta, width.value)
            x_axis = linear_axis_limits(*_axis_values(x_fields))
            y_axis = linear_axis_limits(*_axis_values(intensity_fields))
            beta_limits = [_axis_values(pair) for pair in beta_fields]
            if save:
                store.update({key: width.value, **{prefix + name: field.value for name, field in settings.items()}})
        except (ValueError, OSError) as exc:
            ui.notify(str(exc), type="warning")
            if save:
                return
            x, y, resampled_beta = _uniform_data(energy, intensity, beta, default_width)
            x_axis = y_axis = linear_axis_limits(None, None)
            beta_limits = [(None, None)] * len(beta)
        rows = 2 if beta else 1
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.04)
        fig.add_scatter(**_spectrum_trace(x, y, name), row=1, col=1)
        _style_comparison_plot(fig, 610 if beta else 347, beta_orders=beta)
        fig.update_yaxes(title_text="Bin-average intensity", row=1, col=1, **y_axis)
        if beta:
            _add_beta_axes(fig, x, resampled_beta, beta_limits, subplot=2)
        fig.update_xaxes(**x_axis)
        fig.update_xaxes(title_text=f"{x_label} ({unit})", row=rows, col=1)
        if chart is None:
            with plots:
                chart = _chart()
        chart.update_figure(_plot_data(fig))

    section.on_value_change(lambda event: redraw() if event.value and chart is None else None)


def _table(headers, rows):
    ui.table(columns=[dict(name=key, field=key, label=label) for key, label in headers.items()],
             rows=[{"id": index, **row} for index, row in enumerate(rows)],
             row_key="id", pagination=12).classes("w-full")


def _display(value):
    value = float(value) if value != "" else np.nan
    return f"{value:.6g}" if np.isfinite(value) else "—"


def _uncertainty_figure(points, title, x_title, y_title, *, orders=None):
    x, values, errors = zip(*points) if points else ((), (), ())
    fig = go.Figure(go.Scatter(
        x=x, y=values, mode="markers", marker={"size": 8},
        error_y={"type": "data", "array": errors},
        hovertemplate="%{x:.6g}, %{y:.6g}<extra></extra>" if orders is None else "β%{x}: %{y:.6g}<extra></extra>",
    ))
    _style(fig, title)
    fig.update_layout(height=340, margin={"l": 80, "r": 30, "t": 60, "b": 60})
    fig.update_xaxes(title_text=x_title, automargin=True)
    fig.update_yaxes(title_text=y_title, automargin=True)
    if orders is not None:
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=orders,
                         tickmode="array", tickvals=orders, ticktext=[f"β{order}" for order in orders])
    return fig


def _show_uncertainties(jobs, job):
    if job.status != "done":
        return
    path = shared.checked_path(job.extra[RESULT])
    with jobs.preview_read_access([path]):
        content = path.read_text(encoding="utf-8")
    bootstrap = job.extra["panel_form"]["mode"] == "shots"
    unit = "eV" if job.extra["alpha"] is not None else "detector pixel²"
    theme.form_section("Shot-bootstrap errors" if bootstrap else "Poisson counting errors")
    rows = []
    yields = []
    beta_by_peak = {}
    peak_labels = {}
    peak_index = -1
    for row in csv.DictReader(io.StringIO(content)):
        if row["kind"] != "window" or row["quantity"] == "spectrum":
            continue
        if row["quantity"] == "yield":
            peak_index += 1
        peak = job.extra["peak_windows"][peak_index]
        peak_number, center = peak["peak"], peak["center"]
        value = float(row["value"] or "nan")
        error = float(row["se"] or "nan")
        estimate = (value if np.isfinite(value) else None, error if np.isfinite(error) else None)
        if row["quantity"] == "yield":
            yields.append((center, *estimate))
        else:
            beta_by_peak.setdefault(peak_number, []).append((int(row["order"]), *estimate))
            peak_labels[peak_number] = f"Peak {peak_number} ({center:.6g} {unit})"
        rows.append(dict(peak=peak_number,
                         window=f"{_display(row['lower'])} to {_display(row['upper'])}",
                         quantity=f"β{row['order']}" if row["quantity"] == "beta" else "Yield",
                         value=_display(row["value"]), se=_display(row["se"]),
                         ci=f"{_display(row['ci_lower'])} to {_display(row['ci_upper'])}",
                         valid=row["valid_replicates"]))
    headers = {"peak": "Peak", "window": f"FWHM window ({unit})", "quantity": "Quantity",
               "value": "Estimate", "se": "Standard error"}
    if bootstrap:
        confidence = job.extra["panel_form"]["confidence"]
        headers.update(ci=f"{confidence:.0%} percentile interval", valid="Valid β draws")
    if yields:
        ui.label("Error bars show ±1 standard error.").classes("vmi-caption")
        _chart().update_figure(_plot_data(_uncertainty_figure(
            yields, "Peak yield", f"Fitted peak center ({unit})", "Integrated yield (electrons)")))
    if beta_by_peak:
        selector = ui.select(peak_labels, value=next(iter(peak_labels)), label="Peak for β plot").props("outlined dense")
        chart = _chart()

        def show_beta():
            points = sorted(beta_by_peak[selector.value], key=lambda point: point[0])
            fig = _uncertainty_figure(points, "Peak angular anisotropy", "Legendre order (L)", "β (dimensionless)",
                                      orders=[order for order, _, _ in points])
            chart.update_figure(_plot_data(fig))

        selector.on_value_change(lambda _: show_beta())
        show_beta()
    _table(headers, rows)
    ui.button("Download uncertainty CSV", icon="download",
              on_click=lambda: ui.download.content(content, path.name, "text/csv")).props("flat")


def build(jobs, *, status_slot):
    if shared.SERVER:
        ui.label(LOCAL_MESSAGE).classes("vmi-subtitle")
        return

    state = {}
    peaks = []
    generation_fields = {}
    error_fields = {}

    def form_field(fields, key, element):
        fields[key] = persist(element, f"{OPERATION}.{key}")
        return fields[key]

    def report(label, submit, view):
        try:
            view.attach(jobs.submit(submit()))
        except (ValueError, OSError, EOFError, ResourceBusyError) as exc:
            label.text = str(exc)
            label.set_visibility(True)
        else:
            label.set_visibility(False)

    def error_label():
        label = ui.label().classes("text-negative").props('role="alert"')
        label.set_visibility(False)
        return label

    def belongs_to(job, key, parent):
        selected = state.get(parent)
        return selected is not None and job.extra.get(key) == selected.id

    def latest(view, predicate):
        candidates = [j for j in jobs.jobs() if j.name == view.job_name and predicate(j)]
        active = [j for j in candidates if j.status in ("running", "queued")]
        view.attach(next(iter(active), candidates[-1] if candidates else None))

    def restore_fields(job, fields, namespace=OPERATION):
        form = adopt_agent_form(job, namespace)
        for key, value in form.items():
            if key in fields:
                fields[key].value = value

    def clear_guesses():
        peaks.clear()
        peak_inputs.clear()

    def peak_values():
        return [{key: field.value for key, field in row.items()} for row in peaks]

    def save_peaks():
        peak_store.value = json.dumps(peak_values())

    def add_peak(guess=None):
        guess = guess or {}
        with peak_inputs, ui.row().classes("w-full gap-3 items-center") as row:
            profile = ui.select(list(PROFILES), value=guess.get("profile", "Gaussian"), label="Profile") \
                .props("outlined dense").classes("w-28")
            center = num_input("Peak center", guess.get("center"), step="any")
            gaussian = num_input("Gaussian FWHM", guess.get("gaussian_fwhm"), step="any")
            lorentzian = num_input("Lorentzian FWHM", guess.get("lorentzian_fwhm", 0), step="any")
            lorentzian.bind_visibility_from(profile, "value", lambda value: value == "Voigt")
            fields = dict(profile=profile, center=center, gaussian_fwhm=gaussian, lorentzian_fwhm=lorentzian)
            peaks.append(fields)

            def remove():
                peaks.remove(fields)
                row.delete()
                save_peaks()

            ui.button(icon="remove_circle_outline", on_click=remove).props('flat round aria-label="Remove peak"')
            for field in fields.values():
                field.on_value_change(save_peaks)

    def native_figure(axis_title, name, beta, x_axis, y_axis, revision, selecting):
        fig = go.Figure()
        fig.add_scatter(**_spectrum_trace(state["energy"], state["intensity"], name))
        low, high = fit_low.value, fit_high.value
        if low is not None and high is not None and low < high:
            fig.add_vrect(x0=low, x1=high, fillcolor=SERIES_COLORS[0], opacity=0.14, line_width=0, layer="below")
        record = state.get("fit_record")
        if record:
            x, total, _, _ = evaluate_fit_for_plot(np.asarray(record["fit_x"]),
                                                  np.asarray(record["background"]), record["peaks"])
            fig.add_scatter(x=x.tolist(), y=total.tolist(), name="Peak fit", mode="lines",
                            line_color=SERIES_COLORS[1])
            for peak in record["peaks"]:
                fig.add_vrect(x0=peak["window"][0], x1=peak["window"][1],
                              fillcolor=SERIES_COLORS[1], line_width=0, opacity=0.10)
        _style_comparison_plot(fig, 402, beta_orders=beta)
        fig.update_layout(uirevision=revision, selectdirection="h", selections=[], selectionrevision=f"{low}:{high}",
                          dragmode="select" if selecting else "zoom")
        fig.update_xaxes(title_text=axis_title, **x_axis)
        fig.update_yaxes(title_text=f"Electrons / {state['unit']}", **y_axis)
        return fig

    def refresh_spectrum():
        if redraw := state.get("redraw"):
            redraw()

    def reset_fit(job):
        state.pop("fit", None)
        state.pop("fit_record", None)
        uncertainty_card.set_visibility(False)
        uncertainty_view.attach(None)
        refresh_spectrum()
        form = (adopt_agent_form(job, FIT_OPERATION) if job.extra.get("agent_panel")
                else job.extra["panel_form"])
        if form:
            fit_low.value, fit_high.value = form["region"]
            baseline.value = form["baseline"]
            clear_guesses()
            for guess in form["peaks"] or [{"profile": form.get("auto_profile", "Gaussian")}]:
                add_peak(guess)
            save_peaks()

    def show_fit(job):
        if job.status != "done" or not belongs_to(job, "spectrum_job_id", "spectrum"):
            return
        path = shared.checked_path(job.extra[FIT])
        with jobs.preview_read_access([path]):
            record = load_fit(path)
            content = path.read_text(encoding="utf-8")
        state.update(fit=job, fit_record=record)
        refresh_spectrum()
        unit = state["unit"]
        if not record["guesses"]:
            ui.label(f"Peak estimates detected automatically ({record.get('auto_profile', 'Gaussian')}).") \
                .classes("vmi-subtitle")
        _table({"peak": "Peak", "profile": "Profile", "center": f"Center ({unit})",
                "fwhm": f"FWHM ({unit})", "window": "Integration window", "area": "Fit area"},
               [dict(peak=peak["peak"], profile=peak["profile"], center=_display(peak["center"]),
                     fwhm=_display(peak["fwhm"]), window=" to ".join(map(_display, peak["window"])),
                     area=_display(peak["area"])) for peak in record["peaks"]])
        ui.button("Download peak fit JSON", icon="download",
                  on_click=lambda: ui.download.content(content, path.name, "application/json")).props("flat")
        name = state["spectrum_name"]
        ui.select([name], value=name, label="Inspect fit").props("outlined dense")
        _chart().update_figure(_plot_data(_fit_figure(record, state["energy"], state["intensity"], state["axis_title"])))
        uncertainty_card.set_visibility(True)
        latest(uncertainty_view, lambda candidate: belongs_to(candidate, "fit_job_id", "fit"))

    def reset_spectrum(job):
        state.clear()
        fit_card.set_visibility(False)
        uncertainty_card.set_visibility(False)
        fit_view.attach(None)
        uncertainty_view.attach(None)
        restore_fields(job, generation_fields)

    def show_spectrum(job):
        if job.status != "done":
            return
        path = shared.checked_path(job.extra[SPECTRUM])
        with jobs.preview_read_access([path]):
            energy, intensity, beta = load_spectrum(path)
        form = job.extra["panel_form"]
        alpha = form["alpha"]
        unit, x_label = ("eV", "Kinetic energy") if alpha is not None else ("pixel²", "Squared detector radius")
        name = f"1. {Path(form['counts']).name}"
        axis_title = f"{x_label} ({unit})"
        fit_low.label, fit_high.label = f"Fit X min ({unit})", f"Fit X max ({unit})"
        state.update(spectrum=job, energy=energy, intensity=intensity, unit=unit, spectrum_name=name, axis_title=axis_title)
        padding = (energy[-1] - energy[0]) / 16
        start, end = float(energy[0] - padding), float(energy[-1] + padding)
        x_axis = linear_axis_limits(start, end)
        y_axis = linear_axis_limits(None, None)
        beta_limits = [(None, None)] * len(beta)
        revision = beta_revision = 0
        beta_chart = None

        def redraw():
            chart.update_figure(_plot_data(native_figure(axis_title, name, beta, x_axis, y_axis, revision,
                                                         drag_mode.value == "Select fit range")))

        def redraw_beta():
            if beta_chart is not None:
                fig = go.Figure()
                _style_comparison_plot(fig, 402, beta_orders=beta)
                _add_beta_axes(fig, energy, beta, beta_limits)
                fig.update_layout(uirevision=beta_revision)
                fig.update_xaxes(title_text=axis_title, uirevision=revision, **x_axis)
                beta_chart.update_figure(_plot_data(fig))

        def apply_spectrum_axes(*, initialize=False):
            nonlocal x_axis, y_axis, revision
            try:
                low, high = _axis_values(x_fields)
                new_y = linear_axis_limits(*_axis_values(y_fields))
            except ValueError as exc:
                ui.notify(str(exc), type="warning")
                if initialize:
                    redraw()
                    redraw_beta()
                return
            span = end - start
            if low is None:
                low = start if high is None or high > start else high - span
            if high is None:
                high = end if low < end else low + span
            x_axis, y_axis = linear_axis_limits(low, high), new_y
            revision += 1
            redraw()
            redraw_beta()

        def apply_beta_axes(*, initialize=False):
            nonlocal beta_limits, beta_revision
            try:
                beta_limits = [_axis_values(pair) for pair in beta_fields]
            except ValueError as exc:
                ui.notify(str(exc), type="warning")
                return
            beta_revision += 1
            if not initialize:
                redraw_beta()

        def select_range(event):
            if drag_mode.value == "Select fit range" and event.args:
                limits = event.args.get("x")
                if limits and len(limits) == 2:
                    fit_low.value, fit_high.value = sorted(limits)

        def download_spectrum():
            radius_squared = energy if alpha is None else energy / alpha
            _csv_download(((name, r2, x, y) for r2, x, y in zip(radius_squared, energy, intensity)),
                          ("spectrum", "r2_detector_px2", f"x ({unit})", "intensity"), "pes-pbasex-spectra.csv")

        def download_beta():
            _csv_download(((name, point, *(value if np.isfinite(value) else "" for value in values))
                           for point, *values in zip(energy, *beta.values())),
                          ("spectrum", f"x ({unit})", *(f"beta_{order}" for order in beta)),
                          "pes-pbasex-anisotropy.csv")

        ui.label(f"{name} · {form['counts']}").classes("vmi-subtitle break-all")
        ui.label(f"α = {alpha:g} eV/pixel²" if alpha is not None else "Uncalibrated").classes("vmi-caption")
        with ui.row().classes("items-center gap-2"):
            drag_mode = ui.toggle(["Zoom", "Select fit range"], value="Zoom", on_change=lambda _: redraw())
            ui.button("Clear fit range", on_click=clear_fit_range).props("flat")
        with ui.column().classes("w-full gap-0"):
            x_fields, y_fields = _axis_controls(
                "Photoelectron spectra", "Spectrum axes", "Apply axes", apply_spectrum_axes,
                lambda: [_axis_fields(f"Spectrum {axis}", f"{OPERATION}.{axis.lower()}", auto_label=True)
                         for axis in ("X", "Y")],
                note="The spectrum and β plots share the X limits." if beta else None,
            )
            chart = _chart()
            chart.on("plotly_selected", select_range, js_handler="event => emit(event?.range ?? null)")
            if beta:
                beta_fields = _axis_controls(
                    "Angular anisotropy (β)", "β plot axes", "Apply β axes", apply_beta_axes,
                    lambda: [_axis_fields(f"β{order} Y", f"{OPERATION}.beta_{order}_y", auto_label=True)
                             for order in beta],
                    selector=lambda: ui.select([name], value=name, label="β spectrum")
                    .props("outlined dense").classes("max-w-full"),
                    note="Leave limits blank to use the data range. β ranges can extend to align zero. "
                         "β can be unreliable where the signal is weak.",
                )
                beta_chart = _chart()
                for source, target in ((chart, beta_chart), (beta_chart, chart)):
                    source.on("plotly_relayout", js_handler=(
                        f"event => vmiSyncPesX(event, getElement({source.id}), getElement({target.id}))"
                    ))
                apply_beta_axes(initialize=True)
            apply_spectrum_axes(initialize=True)
            _show_uniform(energy, intensity, beta, unit, x_label, name)
        with ui.row().classes("gap-2"):
            ui.button("Download spectra CSV", icon="download", on_click=download_spectrum).props("flat")
            if beta:
                ui.button("Download β CSV", icon="download", on_click=download_beta).props("flat")
        state["redraw"] = redraw
        fit_targets.set_options([name], value=[name])
        fit_card.set_visibility(True)
        latest(fit_view, lambda candidate: belongs_to(candidate, "spectrum_job_id", "spectrum"))

    def generate():
        report(generation_error, lambda: spectrum_job(
            generation_fields["counts"].value, generation_fields["basis"].value,
            generation_fields["pbasex_repo"].value, **{key: generation_fields[key].value for key in _INVERSION}),
            spectrum_view)

    def fit():
        values = peak_values()
        entered = [item for item in values if item["center"] is not None or item["gaussian_fwhm"] is not None
                   or (item["profile"] == "Voigt" and item["lorentzian_fwhm"])]
        if not fit_targets.value:
            fit_error.text = "Select at least one of the displayed spectra to fit."
            fit_error.set_visibility(True)
            return
        report(fit_error, lambda: peak_fit_job(
            state["spectrum"], [fit_low.value, fit_high.value], entered, baseline.value,
            auto_profile=values[0]["profile"] if values else "Gaussian"), fit_view)

    def clear_fit_range():
        fit_low.value = fit_high.value = None

    def calculate_errors():
        report(uncertainty_error, lambda: uncertainty_job(
            jobs, state["fit"], **{key: field.value for key, field in error_fields.items()}), uncertainty_view)

    spectrum_view = JobView(jobs, job_name=JOB_NAME, results_builder=show_spectrum, attach_builder=reset_spectrum)
    fit_view = JobView(jobs, job_name=FIT_JOB_NAME, results_builder=show_fit, attach_builder=reset_fit,
                       job_filter=lambda job: belongs_to(job, "spectrum_job_id", "spectrum"))
    uncertainty_view = JobView(jobs, job_name=ERROR_JOB_NAME, results_builder=lambda job: _show_uncertainties(jobs, job),
                               attach_builder=lambda job: restore_fields(job, error_fields),
                               job_filter=lambda job: belongs_to(job, "fit_job_id", "fit"))
    with status_slot:
        spectrum_view.place_status()
    with theme.panel_card().classes("vmi-pes-julia"):
        target = shared.circularization_dir()
        form_field(generation_fields, "counts", path_input("Integer count image (.npy)",
                   str(target / "image_circularized_counts.npy"), browse="file", filters=[".npy"]))
        form_field(generation_fields, "basis", path_input("Existing pBASEX basis (.h5)", browse="file",
                   filters=list(HDF5_SUFFIXES)))
        sibling = WORKSPACE_ROOT.parent / "PBASEX-Julia"
        form_field(generation_fields, "pbasex_repo", path_input("PBASEX-Julia checkout",
                   str(sibling) if sibling.is_dir() else "", browse="dir"))
        theme.form_section("Inversion")
        with field_grid():
            for key, label, value in (
                ("resolution", "Bins per detector pixel", 1), ("alpha", "Alpha (eV/detector pixel²)", None),
                ("center_x", "Center X (one-based column)", None), ("center_y", "Center Y (one-based row)", None),
                ("regularization", "Regularization", 0),
            ):
                form_field(generation_fields, key, num_input(label, value, step="any"))
            form_field(generation_fields, "rotation", ui.select(list(ROTATIONS), value=0,
                       label="Rotation (° counterclockwise)").props("outlined dense"))
        ui.label("Leave the center blank to use the image midpoint.").classes("vmi-caption")
        generation_error = error_label()
        with action_bar(spectrum_view, "Generate pBASEX spectrum", on_click=generate):
            pass
        console = ui.column().classes("w-full")

    with theme.panel_card().classes("vmi-pes-julia") as fit_card:
        theme.form_section("Peak fitting")
        ui.label("Drag an interval on the spectrum in Select fit range mode, or enter the limits below.") \
            .classes("vmi-subtitle")
        with field_grid():
            fit_low = persist(num_input("Fit X min", None, step="any"), f"{FIT_OPERATION}.fit_xmin")
            fit_high = persist(num_input("Fit X max", None, step="any"), f"{FIT_OPERATION}.fit_xmax")
            baseline = persist(ui.select(list(BASELINE_PARAMETERS), value="Linear", label="Baseline")
                               .props("outlined dense"), f"{FIT_OPERATION}.baseline")
            fit_targets = ui.select([], multiple=True, label="Spectra to fit").props("outlined dense use-chips")
        for field in (fit_low, fit_high):
            field.on_value_change(lambda _: refresh_spectrum())
        peak_store = persist(ui.input(value="[{}]"), f"{FIT_OPERATION}.peak_guesses")
        peak_store.set_visibility(False)
        ui.label("Leave peak estimates blank to detect peaks automatically. Use the first row to choose the profile.") \
            .classes("vmi-subtitle")
        peak_inputs = ui.column().classes("w-full")
        try:
            for guess in json.loads(peak_store.value):
                add_peak(guess)
        except (AttributeError, TypeError, ValueError):
            clear_guesses()
            add_peak()
        fit_error = error_label()
        with ui.row().classes("gap-2"):
            ui.button("Add peak", icon="add", on_click=lambda: (add_peak(), save_peaks())).props("flat")
            fit_view.run_button("Fit peaks", on_click=fit).props("icon=show_chart")
        fit_view.place_console()
    fit_card.set_visibility(False)

    with theme.panel_card().classes("vmi-pes-julia") as uncertainty_card:
        theme.panel_header("Peak uncertainties", view=uncertainty_view)
        ui.label("Uses each saved FWHM window. Overlapping windows include shared signal.").classes("vmi-caption")
        mode = form_field(error_fields, "mode", ui.select(
            MODES, value="counts", label="Uncertainty input").props("outlined dense").classes("w-full"))
        with ui.column().classes("w-full").bind_visibility_from(mode, "value", lambda value: value == "shots"):
            form_field(error_fields, "shots", path_input("Matching shot-resolved checkpoint (.h5)",
                       str(target / "localizations_circularized.h5"), browse="file", filters=list(HDF5_SUFFIXES)))
            with field_grid():
                for key, label, value in (
                    ("replicates", "Replicates", DEFAULT_REPLICATES), ("confidence", "Confidence level", DEFAULT_CONFIDENCE),
                    ("block_length", "Block length (shots)", 1), ("seed", "Random seed", DEFAULT_SEED),
                    ("window_batch", "Windows per batch", DEFAULT_WINDOW_BATCH),
                ):
                    form_field(error_fields, key, num_input(label, value, step="any" if key == "confidence" else 1))
        uncertainty_error = error_label()
        with action_bar(uncertainty_view, "Calculate peak uncertainties", on_click=calculate_errors):
            pass
        uncertainty_view.place_console()
    uncertainty_card.set_visibility(False)

    with console:
        spectrum_view.place_console()
    if spectrum_view.job is None:
        latest(spectrum_view, lambda job: job.status == "done")
