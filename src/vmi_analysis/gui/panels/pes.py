"""Workbench panel for photoelectron spectra extracted from 2D VMI images."""

import asyncio
import csv
import io
import json
import re
import sys
from importlib.resources import files
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from nicegui import ui
from plotly.subplots import make_subplots

from vmi_analysis.centroiding.plot_data import load_plot_data
from vmi_analysis.pes import (ABEL_METHODS, BASELINE_PARAMETERS, BETA_ORDERS, QUADRANT_LABELS,
                              QUADRANT_ORDER, SYMMETRY_AXES, Geometry, PeakGuess, anisotropy_values,
                              average_anisotropy, evaluate_fit_for_plot, fit_peaks)

from ..agent_panels import PanelOperation, PanelView, number as finite_number, require_completed_job, resolve_path
from ..shared import checked_path, generated_dir, workspace_root
from .. import theme
from ..centroid_plots import _style, linear_axis_limits, plot_options
from ..jobs import Job, JobManager
from ..resources import canonical_path
from .common import (GENERATED, REPO_ROOT, JobView, action_bar, adopt_agent_form, check_grid,
                     confirm_clear_outputs, field_grid, num_input, persist)
from .picker import LocalFilePicker

JOB_NAME = "PES"
FIT_OPERATION = "pes_fit"
PROFILES = ("Gaussian", "Voigt")
_AXIS_SCRIPT = (files("vmi_analysis.gui.assets") / "pes_axes.js").read_text(encoding="utf-8")


def _csv_download(rows, header, name):
    data = io.StringIO(newline="")
    writer = csv.writer(data)
    writer.writerow(header)
    writer.writerows(rows)
    ui.download.content(data.getvalue(), name, "text/csv")


def init_axes() -> None:
    ui.add_body_html(f"<script>{_AXIS_SCRIPT}</script>", shared=True)


def _scientific_plot():
    chart = ui.plotly(go.Figure()).classes("w-full")
    chart.on("plotly_afterplot", js_handler=f"() => requestAnimationFrame(() => vmiUpdatePesAxes(getElement({chart.id})))")
    return chart


def _style_comparison_plot(fig, height):
    _style(fig, "")
    fig.update_layout(
        height=height, showlegend=True,
        margin={"l": 82, "r": 24, "t": 48, "b": 57, "autoexpand": False},
        legend={"orientation": "h", "x": 1, "xanchor": "right", "y": 1.01, "yanchor": "bottom"},
    )
    fig.update_xaxes(automargin=False)
    fig.update_yaxes(automargin=False)


def spectra_job(paths, geometry, overrides, *, alpha=None, abel_method="hansenlaw", beta_order=0,
                quadrants=QUADRANT_ORDER, symmetry_axis="none"):
    cwd = workspace_root(REPO_ROOT)
    paths = list(dict.fromkeys(str(canonical_path(checked_path(path, cwd=cwd))) for path in paths))
    if not paths:
        raise ValueError("Add one or more VMI images.")
    if alpha is not None and finite_number(alpha) <= 0:
        raise ValueError("Alpha must be positive.")
    if (abel_method not in tuple(ABEL_METHODS) or type(beta_order) is not int or beta_order not in BETA_ORDERS
            or symmetry_axis not in tuple(SYMMETRY_AXES)):
        raise ValueError("Choose a valid Abel method, maximum β order, and symmetry axis.")
    if not quadrants or any(name not in QUADRANT_ORDER for name in quadrants):
        raise ValueError("Choose detector quadrants from ur, ul, ll, and lr.")
    chosen = [name for name in QUADRANT_ORDER if name in quadrants]
    if len(chosen) < len(QUADRANT_ORDER) and symmetry_axis == "none":
        raise ValueError("Select a symmetry axis when using only some quadrants.")
    cmd = [sys.executable, "-m", "vmi_analysis.pes", "--abel-method", abel_method, "--beta-order", str(beta_order),
           "--quadrants", ",".join(chosen), "--symmetry-axis", symmetry_axis]
    for path in paths:
        spec = Geometry(**overrides[path]) if path in overrides else geometry
        if spec is None:
            raise ValueError("Enter shared geometry or an override for every image.")
        x, y = spec.center or ("auto", "auto")
        cmd += ["--image", path, str(x), str(y), str(spec.angle), str(spec.resolution)]
    if alpha is not None:
        cmd += ["--alpha", str(alpha)]
    job = Job(JOB_NAME, cmd=cmd, cwd=str(cwd), reads=paths)
    result = generated_dir(GENERATED) / f"pes-spectra-{job.id}.npz"
    job.cmd += ["--output", str(result)]
    job.writes.append(str(result))
    job.extra["pes_spectra"] = str(result)
    return job


def _geometry(values, defaults=Geometry()):
    if not isinstance(values, dict) or values.keys() - {"center", "angle", "resolution"}:
        raise ValueError("Use center, angle, and resolution for image geometry.")
    center = values.get("center", defaults.center)
    if center is not None:
        if not isinstance(center, (list, tuple)) or len(center) != 2:
            raise ValueError("Enter a center as [X, Y] or null for the brightest pixel.")
        center = tuple(map(finite_number, center))
    return Geometry(center, finite_number(values.get("angle", defaults.angle)),
                    finite_number(values.get("resolution", defaults.resolution)))


def _prepare_spectra(jobs, params):
    requested = params.get("images", [])
    if not isinstance(requested, list):
        raise ValueError("Provide a list of VMI image paths.")
    paths = [resolve_path(path) for path in requested]
    if "source_job_id" in params:
        job_id = params["source_job_id"]
        parent = jobs.get(job_id) if isinstance(job_id, str) else None
        defaults = {"Centroiding": "raw_image", "GMD binning": "all_bins_image", "Circularize": "output"}
        if parent is None or parent.name not in defaults:
            raise ValueError("Choose a completed Centroiding, Binning, or Circularize job from your workspace.")
        parent = require_completed_job(jobs, job_id, name=parent.name)
        key = params.get("image_artifact", defaults[parent.name])
        allowed_bin = parent.name == "GMD binning" and isinstance(key, str) and re.fullmatch(r"bin_\d+", key)
        if key != defaults[parent.name] and not allowed_bin:
            raise ValueError("Choose a VMI image artifact from the completed job.")
        if key not in parent.extra:
            raise ValueError("The completed job has no image for the requested artifact.")
        paths.append(resolve_path(parent.extra[key]))
    elif "image_artifact" in params:
        raise ValueError("An image artifact requires a source job ID.")
    if any(not path.is_file() or path.suffix.lower() not in (".npy", ".txt") for path in paths):
        raise ValueError("Choose existing VMI images (.npy or .txt).")
    geometry = _geometry(params.get("geometry", {}))
    requested_overrides = params.get("image_geometry", {})
    if not isinstance(requested_overrides, dict):
        raise ValueError("Use an object mapping image paths to their geometry overrides.")
    overrides = {}
    for image, values in requested_overrides.items():
        path = resolve_path(image)
        if path not in paths:
            raise ValueError("An image geometry override must refer to a selected image.")
        spec = _geometry(values, geometry)
        overrides[str(canonical_path(path))] = {"center": spec.center, "angle": spec.angle, "resolution": spec.resolution}
    quadrants = params.get("quadrants", list(QUADRANT_ORDER))
    if not isinstance(quadrants, list):
        raise ValueError("Provide the selected quadrants as a list.")
    options = {key: params[key] for key in ("alpha", "abel_method", "beta_order", "symmetry_axis") if key in params}
    try:
        job = spectra_job(paths, geometry, overrides, quadrants=quadrants, **options)
    except OverflowError as exc:
        raise ValueError("Enter finite PES settings.") from exc
    form = dict(images="\n".join(job.reads), image_geometry=json.dumps(overrides),
                brightest_center=geometry.center is None, center_x=geometry.center[0] if geometry.center else None,
                center_y=geometry.center[1] if geometry.center else None, rotation=geometry.angle,
                resolution=geometry.resolution, alpha=options.get("alpha"),
                abel_method=options.get("abel_method", "hansenlaw"), beta_order=options.get("beta_order", 0),
                symmetry_axis=options.get("symmetry_axis", "none"))
    form.update({f"quadrant_{name}": name in quadrants for name in QUADRANT_ORDER})
    job.extra["panel_form"] = form
    return job, {
        "VMI images": job.reads,
        "Shared geometry": {"center": geometry.center or "Brightest pixel",
                            "rotation (degrees)": geometry.angle, "resolution": geometry.resolution},
        "Image geometry overrides": overrides or None,
        "Energy calibration": form["alpha"] if form["alpha"] is not None else "Uncalibrated (pixel²)",
        "Abel method": ABEL_METHODS[form["abel_method"]], "Maximum β order": form["beta_order"],
        "Quadrants": quadrants, "Symmetry axis": form["symmetry_axis"],
    }


def agent_operations():
    spectra = {
        "panel": JOB_NAME, "manual": JOB_NAME,
        "summary": "Generate spectra and optional angular anisotropy from one or more VMI images.",
        "parameters": {
            "images": {"type": "list of paths", "description": "VMI images (.npy or .txt)"},
            "source_job_id": {"type": "job ID", "description": "Optional completed Centroiding, Binning, or Circularize job"},
            "image_artifact": {"type": "string", "description": "Optional Binning artifact such as bin_0. "
                               "Defaults to the combined image."},
            "geometry": {"type": "object", "description": "Shared center [X, Y] or null for brightest pixel, "
                         "angle in degrees, and resolution factor. Defaults to auto center, 0°, and factor 1."},
            "image_geometry": {"type": "object", "description": "Optional geometry overrides keyed by image path"},
            "alpha": {"type": "number or null", "unit": "eV/detector pixel²", "default": None},
            "abel_method": {"type": "string", "choices": list(ABEL_METHODS), "default": "hansenlaw"},
            "beta_order": {"type": "integer", "choices": list(BETA_ORDERS), "default": 0},
            "quadrants": {"type": "list of strings", "default": list(QUADRANT_ORDER)},
            "symmetry_axis": {"type": "string", "choices": list(SYMMETRY_AXES), "default": "none"},
        },
    }
    fit = {
        "panel": JOB_NAME, "manual": JOB_NAME,
        "summary": "Fit Gaussian or Voigt peaks in the current spectrum selection. Open Ask Agent in PES first.",
        "parameters": {
            "region": {"type": "pair of numbers", "description": "Fit range in the displayed axis units"},
            "targets": {"type": "list of strings", "description": "Spectrum names from the current selection"},
            "baseline": {"type": "string", "choices": list(BASELINE_PARAMETERS)},
            "peaks": {"type": "list of objects", "description": "Each peak uses profile, center, gaussian_fwhm, "
                      "and optional lorentzian_fwhm. Empty means automatic."},
            "auto_profile": {"type": "string", "choices": list(PROFILES)},
        },
    }
    return (PanelOperation("pes", spectra, _prepare_spectra, (JOB_NAME,), outputs=("pes_spectra",)),
            PanelOperation(FIT_OPERATION, fit, None, ()))


def _fit_settings(params, spectra):
    if params.keys() - {"region", "targets", "baseline", "peaks", "auto_profile"}:
        raise ValueError("Use the peak fit settings listed for PES.")
    region = params["region"]
    if not isinstance(region, (list, tuple)) or len(region) != 2 or None in region:
        raise ValueError("Select a fit range or enter both endpoints.")
    low, high = map(finite_number, region)
    if not low < high or not np.isfinite(high - low):
        raise ValueError("Enter an increasing, finite fit range.")
    targets = params["targets"]
    if not isinstance(targets, list) or not targets or any(type(name) is not str or name not in spectra for name in targets):
        raise ValueError("Select at least one of the displayed spectra to fit.")
    if params["baseline"] not in tuple(BASELINE_PARAMETERS) or params["auto_profile"] not in PROFILES:
        raise ValueError("Choose a valid baseline and peak profile.")
    peaks = params["peaks"]
    if not isinstance(peaks, list):
        raise ValueError("Provide one object per peak estimate.")
    guesses = []
    for item in peaks:
        if not isinstance(item, dict) or item.keys() - {"profile", "center", "gaussian_fwhm", "lorentzian_fwhm"}:
            raise ValueError("Enter a peak profile, center, and full widths at half maximum.")
        profile = item.get("profile", "Gaussian")
        if profile not in PROFILES or item.get("center") is None or item.get("gaussian_fwhm") is None:
            raise ValueError("Enter a center and Gaussian FWHM for each peak.")
        center = finite_number(item["center"])
        width = finite_number(item["gaussian_fwhm"])
        wing = finite_number(item.get("lorentzian_fwhm") or 0) if profile == "Voigt" else 0
        if not low <= center <= high or not 0 < width <= high - low or not 0 <= wing <= high - low:
            raise ValueError("Keep peak centers inside the fit range and widths within its span.")
        guesses.append(dict(profile=profile, center=center, gaussian_fwhm=width, lorentzian_fwhm=wing))
    return dict(region=[low, high], targets=list(dict.fromkeys(targets)), baseline=params["baseline"],
                auto_profile=params["auto_profile"], peaks=guesses)


def build(jobs: JobManager, *, ask_agent=None) -> None:
    state = {"spectra": {}, "angular": {}, "beta_orders": (), "plot": None, "units": "pixel²",
             "revision": 0, "selecting": False, "fit_busy": False}

    def restore_form(job):
        reset()
        for key, value in adopt_agent_form(job, "pes").items():
            if key in fields:
                fields[key].value = value

    view = JobView(jobs, job_name=JOB_NAME, results_builder=lambda job: show_job(job), attach_builder=restore_form)
    theme.panel_header(JOB_NAME, "Generate photoelectron spectra from VMI images and fit selected peaks.", view=view)

    def number(label, key, value=None):
        return persist(num_input(label, value, step="any"), f"pes.{key}")

    def paths():
        return list(dict.fromkeys(str(canonical_path(line.strip(), workspace_root(REPO_ROOT)))
                                  for line in (images.value or "").splitlines() if line.strip()))

    def geometry_fields(auto, center_x, center_y, rotation, resolution):
        if rotation.value is None or resolution.value is None:
            raise ValueError("Enter a rotation and resolution factor.")
        if not auto.value and (center_x.value is None or center_y.value is None):
            raise ValueError("Enter both center coordinates or use the brightest pixel.")
        center = None if auto.value else (float(center_x.value), float(center_y.value))
        return Geometry(center, float(rotation.value), float(resolution.value))

    def shared_geometry():
        return geometry_fields(automatic, cx, cy, angle, scale)

    def overrides():
        data = json.loads(override_store.value)
        if not isinstance(data, dict):
            raise ValueError("Invalid image overrides in the saved workflow.")
        return data

    async def add_image():
        existing = paths()
        chosen = await LocalFilePicker(start=existing[-1] if existing else workspace_root(REPO_ROOT), filters=[".npy", ".txt"])
        if chosen:
            images.value = "\n".join(dict.fromkeys([*paths(), str(canonical_path(chosen))]))

    async def edit_geometry():
        files = paths()
        if not files:
            ui.notify("Add an image first.", type="warning")
            return
        try:
            saved = overrides()
        except (TypeError, ValueError) as exc:
            ui.notify(str(exc), type="warning")
            return
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-xl"):
            ui.label("Image geometry overrides").classes("text-lg font-semibold")
            selected = ui.select(files, value=files[0], label="Image").classes("w-full").props("outlined dense")
            per_auto = ui.checkbox("Use brightest pixel as center")
            with field_grid():
                per_x = num_input("Center X", None, step="any").bind_enabled_from(per_auto, "value", lambda v: not v)
                per_y = num_input("Center Y", None, step="any").bind_enabled_from(per_auto, "value", lambda v: not v)
                per_rotation = num_input("Rotation (°)", None, step="any")
                per_scale = num_input("Resolution factor", None, step="any")
            details = ui.label().classes("vmi-subtitle")

            def reload_geometry():
                data = saved.get(selected.value)
                if data:
                    per_auto.value = data["center"] is None
                    per_x.value, per_y.value = data["center"] or (None, None)
                    per_rotation.value, per_scale.value = data["angle"], data["resolution"]
                else:
                    per_auto.value = automatic.value
                    per_x.value, per_y.value = cx.value, cy.value
                    per_rotation.value, per_scale.value = angle.value, scale.value
                details.text = "Using an image override." if data else "Using shared geometry."

            def save_geometry():
                try:
                    geometry = geometry_fields(per_auto, per_x, per_y, per_rotation, per_scale)
                except ValueError as exc:
                    ui.notify(str(exc), type="warning")
                    return
                saved[selected.value] = dict(center=geometry.center, angle=geometry.angle, resolution=geometry.resolution)
                override_store.value = json.dumps(saved)
                reload_geometry()

            def reset_geometry():
                saved.pop(selected.value, None)
                override_store.value = json.dumps(saved)
                reload_geometry()

            selected.on_value_change(lambda _: reload_geometry())
            reload_geometry()
            with ui.row().classes("gap-2"):
                ui.button("Save override", on_click=save_geometry)
                ui.button("Use shared geometry", on_click=reset_geometry).props("flat")
                ui.button("Close", on_click=dialog.close).props("flat")
        await dialog

    def run():
        try:
            files = paths()
            saved = overrides()
            geometry = shared_geometry() if any(path not in saved for path in files) else None
            job = spectra_job(files, geometry, saved, alpha=alpha.value, abel_method=abel_method.value,
                              beta_order=beta_order.value, symmetry_axis=symmetry_axis.value,
                              quadrants=[name for name, field in quadrant_fields.items() if field.value])
            view.attach(jobs.submit(job))
        except (OSError, TypeError, ValueError, OverflowError) as exc:
            ui.notify(str(exc), type="warning")

    def latest():
        return next((job for job in reversed(jobs.jobs()) if job.name == JOB_NAME and job.status == "done"
                     and (path := job.extra.get("pes_spectra")) and Path(path).is_file()), None)

    def show_latest():
        if view.job and view.job.status in ("running", "queued"):
            ui.notify("Wait for the current job to finish.", type="warning")
            return
        if (job := latest()) is None:
            ui.notify("No saved spectra in this server session.", type="info")
            return
        view.results.clear()
        with view.results:
            show_job(job)
        view.open_console()

    def clear():
        results = [job.extra["pes_spectra"] for job in jobs.jobs()
                   if job.name == JOB_NAME and "pes_spectra" in job.extra]
        confirm_clear_outputs(results, "Delete generated PES spectra from this server session?",
                              on_cleared=clear_display)

    def clear_display():
        reset()
        view.clear_results()

    def reset():
        state.update(spectra={}, angular={}, beta_orders=(), plot=None, spectrum_axes=(), selecting=False)
        fit_card.set_visibility(False)
        fit_display.clear()

    def axis_controls(title, label, prefix, apply_label, apply, *, selector=None, note=None):
        def toggle():
            panel.set_visibility(not panel.visible)
            button.props(f'aria-expanded="{str(panel.visible).lower()}"')

        with ui.row().classes("w-full items-center gap-x-3 gap-y-1"):
            ui.label(title).classes("font-mono text-lg text-primary")
            selected = selector() if selector else None
            button = ui.button(f"{label} axes", icon="show_chart", on_click=toggle).props('flat aria-expanded="false"')
        with ui.column().classes("w-full gap-2") as panel:
            panel.set_visibility(False)
            if note:
                ui.label(note).classes("vmi-subtitle")
            with field_grid():
                fields = [number(f"{label} {axis} {bound} (blank = auto)", f"{prefix}{axis.lower()}{bound}")
                          for axis in ("X", "Y") for bound in ("min", "max")]
            ui.button(apply_label, on_click=apply).props("flat")
        return fields, selected

    def axis_bounds(fields):
        xmin, xmax, ymin, ymax = (field.value for field in fields)
        return linear_axis_limits(xmin, xmax), linear_axis_limits(ymin, ymax)

    def refresh_plot(reset_axes=False):
        if state["plot"] is None:
            return
        try:
            x_axis, y_axis = axis_bounds(state["spectrum_axes"])
        except ValueError as exc:
            ui.notify(str(exc), type="warning")
            x_axis = y_axis = {"autorange": True}
        if reset_axes:
            state["revision"] += 1
        selecting = state["selecting"]
        low, high = fit_low.value, fit_high.value
        fig = go.Figure()
        for name, (_, x, y) in state["spectra"].items():
            fig.add_scatter(x=x, y=y, name=name, mode="lines+markers" if selecting else "lines",
                            marker={"opacity": 0}, hovertemplate="%{x:.6g}, %{y:.6g}<extra>%{fullData.name}</extra>")
        if low is not None and high is not None and low < high:
            fig.add_vrect(x0=low, x1=high, fillcolor="#1e40af",
                          opacity=0.14, line_width=0, layer="below")
        _style_comparison_plot(fig, 402)
        fig.update_layout(uirevision=state["revision"], selectdirection="h",
                          selections=[], selectionrevision=f"{low}:{high}",
                          dragmode="select" if selecting else "zoom")
        fig.update_xaxes(title_text=f"{state['x_label']} ({state['units']})", **x_axis)
        fig.update_yaxes(title_text="Intensity (arb. units)", **y_axis)
        state["plot"].update_figure(plot_options(fig))

    def select_range(event):
        if not state["selecting"] or not event.args:
            return
        limits = event.args.get("x")
        if limits and len(limits) == 2:
            fit_low.value, fit_high.value = sorted(limits)

    def clear_fit_range():
        fit_low.value = fit_high.value = None

    def set_drag_mode(event):
        state["selecting"] = event.value == "Select fit range"
        refresh_plot()

    def show_job(job):
        if job.status != "done" or not (path := job.extra.get("pes_spectra")):
            return
        try:
            meta, arrays = load_plot_data(path)
        except (OSError, ValueError, KeyError) as exc:
            ui.label(f"Could not open the generated spectra. {exc}").classes("text-negative")
            return
        reset()
        state["beta_orders"] = tuple(range(2, meta.get("beta_order", 0) + 1, 2))
        for i, image in enumerate(meta["images"]):
            name = f"{i + 1}. {Path(image['path']).name}"
            state["spectra"][name] = arrays[f"r2_{i}"], arrays[f"x_{i}"], arrays[f"y_{i}"]
            if state["beta_orders"]:
                state["angular"][name] = arrays[f"beta_x_{i}"], arrays[f"moments_{i}"]
            ui.label(f"{name} · {image['path']}").classes("vmi-subtitle break-all")
        state["units"] = "eV" if meta["alpha"] is not None else "pixel²"
        state["x_label"] = "Kinetic energy" if meta["alpha"] is not None else "Squared detector radius"
        ui.label(f"α = {meta['alpha']:g} eV/pixel²" if meta["alpha"] is not None else "Uncalibrated") \
            .classes("vmi-caption")
        ui.label(f"Abel inversion: {ABEL_METHODS[meta.get('abel_method', 'hansenlaw')]}").classes("vmi-caption")
        quadrants = meta.get("quadrants", list(QUADRANT_ORDER))
        ui.label(f"Quadrants: {', '.join(quadrants)} · symmetry: {meta.get('symmetry_axis', 'none')}") \
            .classes("vmi-caption")
        with ui.row().classes("items-center gap-2"):
            ui.toggle(["Zoom", "Select fit range"], value="Zoom", on_change=set_drag_mode)
            ui.button("Clear fit range", on_click=clear_fit_range).props("flat")
        with ui.column().classes("w-full gap-0"):
            state["spectrum_axes"], _ = axis_controls("Photoelectron spectra", "Spectrum", "", "Apply axes",
                                                      lambda: refresh_plot(reset_axes=True))
            state["plot"] = _scientific_plot()
            state["plot"].on("plotly_selected", select_range,
                               js_handler="event => emit(event?.range ?? null)")
            if state["angular"]:
                show_anisotropy()
        with ui.row().classes("gap-2"):
            ui.button("Download spectra CSV", icon="download", on_click=download_spectra).props("flat")
            if state["angular"]:
                ui.button("Download β CSV", icon="download", on_click=download_anisotropy).props("flat")
        fit_targets.set_options(list(state["spectra"]), value=list(state["spectra"]))
        fit_low.label = f"Fit X min ({state['units']})"
        fit_high.label = f"Fit X max ({state['units']})"
        fit_card.set_visibility(True)
        refresh_plot(reset_axes=True)

    def download_spectra():
        _csv_download(((name, q, x, y) for name, arrays in state["spectra"].items()
                       for q, x, y in zip(*arrays)),
                      ("spectrum", "r2_detector_px2", f"x ({state['units']})", "intensity"), "pes-spectra.csv")

    def download_anisotropy():
        _csv_download(
            ((name, point, *(value if np.isfinite(value) else "" for value in column))
             for name, (x, moments) in state["angular"].items()
             for point, column in zip(x, anisotropy_values(moments).T)),
            ("spectrum", f"x ({state['units']})", *(f"beta_{order}" for order in state["beta_orders"])),
            "pes-anisotropy.csv",
        )

    def show_anisotropy():
        names = list(state["angular"])
        fields, spectrum = axis_controls(
            "Angular anisotropy (β)", "β plot", "beta_", "Apply β axes", lambda: redraw(update_axes=True),
            selector=lambda: ui.select(names, value=names[0], label="β spectrum").props("outlined dense").classes("max-w-full"),
            note="Each β order is shown in its own panel with an independent vertical scale. "
                 "Leave the Y limits blank to auto-range each panel; explicit Y limits apply to every panel. "
                 "β can be unreliable where the signal is weak; bins with nonpositive isotropic intensity are omitted.",
        )
        chart = _scientific_plot()
        revision = 0
        x_axis = y_axis = {"autorange": True}

        def redraw(update_axes=False):
            nonlocal x_axis, y_axis, revision
            if update_axes:
                try:
                    x_axis, y_axis = axis_bounds(fields)
                except ValueError as exc:
                    ui.notify(str(exc), type="warning")
                else:
                    revision += 1
            x, moments = state["angular"][spectrum.value]
            orders = state["beta_orders"]
            beta_values = anisotropy_values(moments)
            rows = max(1, len(orders))
            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                                vertical_spacing=0.04 if rows > 1 else 0.0)
            for row, (order, values) in enumerate(zip(orders, beta_values), start=1):
                fig.add_scatter(x=x, y=values, mode="lines", name=f"β{order}",
                                row=row, col=1)
                fig.update_yaxes(title_text=f"β{order} (dimensionless)",
                                 row=row, col=1, **y_axis)
            _style_comparison_plot(fig, max(347, 210 * rows + 100))
            fig.update_layout(uirevision=f"{spectrum.value}:{revision}", showlegend=False)
            for row in range(1, rows + 1):
                fig.update_xaxes(**x_axis, row=row, col=1)
                if row < rows:
                    fig.update_xaxes(showticklabels=False, title_text="", row=row, col=1)
            fig.update_xaxes(title_text=f"{state['x_label']} ({state['units']})",
                             row=rows, col=1)
            chart.update_figure(plot_options(fig))

        spectrum.on_value_change(lambda _: redraw())
        redraw(update_axes=True)

    with theme.panel_card():
        theme.form_section("VMI images")
        images = persist(ui.textarea("Image paths, one per line").classes("w-full").props("outlined dense"), "pes.images")
        override_store = persist(ui.input(value="{}"), "pes.image_geometry")
        override_store.set_visibility(False)
        ui.button("Add image", icon="folder_open", on_click=add_image).props("flat")
        theme.form_section("Shared geometry and energy calibration")
        automatic = persist(ui.checkbox("Use brightest pixel as center", value=True), "pes.brightest_center")
        with field_grid():
            cx = number("Center X", "center_x").bind_enabled_from(automatic, "value", lambda v: not v)
            cy = number("Center Y", "center_y").bind_enabled_from(automatic, "value", lambda v: not v)
            angle = number("Rotation before Abel inversion (°)", "rotation", 0)
            angle.tooltip("Align the polarization vertically. Positive angles rotate counterclockwise.")
            scale = number("Image resolution factor", "resolution", 1)
            scale.tooltip("Image pixels per detector pixel. Enter the resolution factor used to make the image.")
            alpha = number("α (eV/detector pixel², blank = uncalibrated)", "alpha")
        ui.label("Center X and Y are zero-based column and row. Geometry applies to every image unless overridden. "
                 "α applies to all images.") \
            .classes("vmi-subtitle")
        ui.button("Image geometry overrides", icon="tune", on_click=edit_geometry).props("flat")
        theme.form_section("Abel inversion and anisotropy")
        with field_grid():
            abel_method = persist(ui.select(ABEL_METHODS, value="hansenlaw", label="Abel method")
                                  .props("outlined dense"), "pes.abel_method")
            beta_order = persist(ui.select({order: str(order) if order else "Off" for order in BETA_ORDERS},
                                           value=0, label="Maximum β order")
                                 .props("outlined dense"), "pes.beta_order")
        with ui.expansion("Abel inversion options", icon="tune", value=False).classes(
                "w-full vmi-abel-options").props(
                'header-class="vmi-section-toggle text-primary" expand-icon-class="text-primary"'):
            ui.label("Choose the detector quadrants used by the inversion and optional symmetry averaging. "
                     "All four quadrants with no symmetry is the default full-image inversion.") \
                .classes("vmi-subtitle")
            with check_grid():
                quadrant_fields = {
                    name: persist(ui.checkbox(f"{QUADRANT_LABELS[name]} (Q{index})", value=True),
                                  f"pes.quadrant_{name}")
                    for index, name in enumerate(QUADRANT_ORDER)
                }
            with field_grid():
                symmetry_axis = persist(ui.select(
                    {"none": "None (independent quadrants)", "vertical": "Vertical axis",
                     "horizontal": "Horizontal axis", "both": "Both axes"},
                    value="none", label="Symmetry axis").props("outlined dense"), "pes.symmetry_axis")
            ui.label("Selecting a subset of quadrants requires a symmetry axis. "
                     "Symmetry averaging combines the selected quadrants before inversion.") \
                .classes("vmi-subtitle")
        ui.label("Align the symmetry axis vertically before inversion. β includes even orders up to the selected maximum.") \
            .classes("vmi-subtitle")
        fields = {"images": images, "image_geometry": override_store, "brightest_center": automatic,
                  "center_x": cx, "center_y": cy, "rotation": angle, "resolution": scale,
                  "alpha": alpha, "abel_method": abel_method, "beta_order": beta_order,
                  "symmetry_axis": symmetry_axis,
                  **{f"quadrant_{key}": field for key, field in quadrant_fields.items()}}
        with action_bar(view, "Generate spectra", on_click=run):
            ui.button("View latest spectra", icon="insights", on_click=show_latest).props("flat")
            ui.button("Clear outputs", icon="delete", on_click=clear).props("flat color=negative").classes("vmi-action-danger")
    console_area = ui.column().classes("w-full")

    def peak_values():
        return [dict(profile=profile.value, center=center.value, gaussian_fwhm=width.value,
                     lorentzian_fwhm=wing.value) for _, profile, center, width, wing in peak_rows]

    def save_peaks():
        peak_store.value = json.dumps(peak_values())

    def add_peak(guess=None):
        guess = guess or {}
        with peak_list, ui.row().classes("w-full gap-3 items-center") as row:
            profile = ui.select(["Gaussian", "Voigt"], value=guess.get("profile", "Gaussian"), label="Profile") \
                .props("outlined dense").classes("w-28")
            center = num_input("Peak center", guess.get("center"), step="any")
            width = num_input("Gaussian FWHM", guess.get("gaussian_fwhm"), step="any")
            wing = num_input("Lorentzian FWHM", guess.get("lorentzian_fwhm", 0), step="any")
            wing.bind_visibility_from(profile, "value", lambda value: value == "Voigt")

            def remove():
                peak_rows.remove(entry)
                row.delete()
                save_peaks()

            ui.button(icon="remove_circle_outline", on_click=remove).props('flat round aria-label="Remove peak"')
        entry = (row, profile, center, width, wing)
        peak_rows.append(entry)
        for field in entry[1:]:
            field.on_value_change(save_peaks)

    def fit_form():
        values = peak_values()
        entered = [item for item in values if item["center"] is not None or item["gaussian_fwhm"] is not None
                   or (item["profile"] == "Voigt" and item["lorentzian_fwhm"])]
        return {"region": [fit_low.value, fit_high.value], "targets": list(fit_targets.value or []),
                "baseline": baseline.value, "peaks": entered,
                "auto_profile": values[0]["profile"] if values else "Gaussian"}

    async def fit():
        try:
            await run_fit(_fit_settings(fit_form(), state["spectra"]))
        except (TypeError, ValueError, OverflowError) as exc:
            ui.notify(str(exc), type="warning")

    async def run_fit(settings):
        if state["fit_busy"]:
            raise ValueError("Wait for the current PES fit to finish.")
        spectra = state["spectra"]
        guesses = [PeakGuess(**item) for item in settings["peaks"]]
        region, model = settings["region"], settings["baseline"]
        auto_profile = settings["auto_profile"]

        def fit_selected():
            fitted, errors = {}, []
            for name in settings["targets"]:
                _, x, y = spectra[name]
                try:
                    fitted[name] = fit_peaks(x, y, region, guesses, model, auto_profile=auto_profile)
                except (RuntimeError, ValueError, FloatingPointError) as exc:
                    errors.append(f"{name}: {exc}")
            return fitted, errors

        state["fit_busy"] = True
        fit_button.disable()
        try:
            results, errors = await asyncio.to_thread(fit_selected)
        finally:
            state["fit_busy"] = False
            fit_button.enable()
        if state["spectra"] is not spectra:
            return {"status": "cancelled", "note": "The displayed spectra changed while the fit was running."}
        fit_display.clear()
        rows = []
        with fit_display:
            for error in errors:
                ui.label(error).classes("text-negative")
            if results:
                if not guesses:
                    ui.label(f"Peak estimates detected automatically for each spectrum ({auto_profile}).") \
                        .classes("vmi-caption")
                rows = show_fits(results, model)
        return {"status": "done" if results else "failed", "panel": JOB_NAME, "units": state["units"],
                "peak_table": rows, "errors": errors}

    def open_fit_agent():
        spectra = state["spectra"]
        try:
            initial = _fit_settings(fit_form(), spectra)
            if state["fit_busy"]:
                raise ValueError("Wait for the current PES fit to finish.")
        except (TypeError, ValueError, OverflowError) as exc:
            ui.notify(str(exc), type="warning")
            return

        def bind(parameters):
            if (state["spectra"] is not spectra or state["fit_busy"]
                    or fit_form()["region"] != initial["region"] or fit_form()["targets"] != initial["targets"]):
                raise ValueError("The displayed spectra or fit selection changed. Ask the Agent again from PES.")
            try:
                return _fit_settings(initial | parameters, spectra)
            except OverflowError as exc:
                raise ValueError("Enter finite peak fit settings.") from exc

        def review(_job, draft, approve, decline):
            with fit_review, theme.panel_card():
                theme.form_section("Agent peak fit")
                with field_grid():
                    low = num_input(f"Draft minimum ({state['units']})", draft["region"][0], step="any")
                    high = num_input(f"Draft maximum ({state['units']})", draft["region"][1], step="any")
                    base = ui.select(list(BASELINE_PARAMETERS), value=draft["baseline"],
                                     label="Draft baseline").props("outlined dense")
                    targets = ui.select(list(spectra), value=draft["targets"], multiple=True,
                                        label="Draft spectra").props("outlined dense use-chips")
                if draft["peaks"]:
                    for peak in draft["peaks"]:
                        width = f"{peak['gaussian_fwhm']:g} Gaussian FWHM"
                        if peak["profile"] == "Voigt":
                            width += f", {peak['lorentzian_fwhm']:g} Lorentzian FWHM"
                        ui.label(f"{peak['profile']} at {peak['center']:g} · {width}").classes("vmi-caption")
                else:
                    ui.label(f"Detect peaks automatically with a {draft['auto_profile']} profile.").classes("vmi-caption")

                async def confirm():
                    edits = {"region": [low.value, high.value], "baseline": base.value, "targets": targets.value}
                    try:
                        bind(draft | edits)
                    except (ValueError, TypeError) as exc:
                        ui.notify(str(exc), type="warning")
                        return
                    await approve(edits)

                with ui.row().classes("items-center gap-2"):
                    ui.button("Approve Agent peak fit", on_click=confirm)
                    ui.button("Decline", on_click=decline).props("flat")

        async def run(draft):
            fit_low.value, fit_high.value = draft["region"]
            baseline.value = draft["baseline"]
            fit_targets.value = draft["targets"]
            peak_rows.clear()
            peak_list.clear()
            for guess in draft["peaks"] or [{"profile": draft["auto_profile"]}]:
                add_peak(guess)
            save_peaks()
            return await run_fit(draft)

        sample = {}
        low, high = initial["region"]
        for name in initial["targets"][:4]:
            _, x, y = spectra[name]
            indices = np.flatnonzero(np.isfinite(x) & np.isfinite(y) & (low <= x) & (x <= high))
            if indices.size > 32:
                indices = indices[np.linspace(0, len(indices) - 1, 32, dtype=int)]
            sample[name] = np.column_stack((x[indices], y[indices])).tolist()
        context = {"units": state["units"], "selected_region_samples_xy": sample,
                   "spectra": {name: {"minimum": float(x[0]), "maximum": float(x[-1]), "samples": len(x)}
                               for name, (_, x, _) in spectra.items() if len(x)},
                   "beta_orders": state["beta_orders"]}
        ask_agent(PanelView(panel=JOB_NAME, operation=FIT_OPERATION, label="PES peak fit selection",
                            instruction="Use the displayed axis units and selected spectra and fit range unless the user changes them.",
                            settings=initial, bind=bind, render_review=review, review_container=fit_review,
                            run_interactive=run, context=context))

    def show_fits(results, model):
        ui.label(f"Fit coordinates in {state['units']} · unweighted least squares").classes("vmi-caption")
        display = ui.select(list(results), value=next(iter(results)), label="Inspect fit").props("outlined dense")
        chart = _scientific_plot()
        rows = [dict(spectrum=name, **row) for name, result in results.items() for row in result[-1]]
        fields = [("spectrum", "Spectrum"), ("peak", "Peak"), ("profile", "Profile"), ("center", "Center"),
                  ("fwhm", "FWHM"), ("area", "Area"), ("gaussian_fwhm", "Gaussian FWHM"),
                  ("lorentzian_fwhm", "Lorentzian FWHM"), ("bound", "At bound")]
        if state["beta_orders"]:
            ui.label("β values summarize the measured signal within each fitted FWHM, including any overlapping peaks "
                     "and background. Blank values indicate insufficient data or nonpositive integrated intensity.") \
                .classes("vmi-subtitle")
            beta_fields = [(f"beta_{order}", f"β{order} (FWHM)") for order in state["beta_orders"]]
            fields.extend(beta_fields)
            for row in rows:
                x, moments = state["angular"][row["spectrum"]]
                for (key, _), value in zip(beta_fields, average_anisotropy(x, moments, row["center"], row["fwhm"])):
                    row[key] = "" if value is None else value
        ui.table(columns=[dict(name=k, field=k, label=label) for k, label in fields],
                 rows=[dict(id=index, **{key: (f"{value:.6g}" if isinstance(value, float) else value)
                                         for key, value in row.items()}) for index, row in enumerate(rows)],
                 row_key="id").classes("w-full")
        ui.button("Download peak table CSV", icon="download", on_click=lambda: _csv_download(
            ([row[key] for key, _ in fields] for row in rows), [label for _, label in fields], "pes-peaks.csv")).props("flat")

        def redraw():
            x, fitted, background, _, residual, peaks = results[display.value]
            grid, total, baseline_curve, components = evaluate_fit_for_plot(x, background, peaks)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.74, 0.26], vertical_spacing=0.09)
            fig.add_scatter(x=x, y=fitted + residual, mode="markers", name="Spectrum", marker={"size": 8}, row=1, col=1)
            fig.add_scatter(x=grid, y=total, mode="lines", name="Total fit", row=1, col=1)
            if model != "None":
                fig.add_scatter(x=grid, y=baseline_curve, mode="lines", name="Baseline",
                                line={"dash": "dot"}, row=1, col=1)
            suffix = " + baseline" if model != "None" else ""
            for i, peak in enumerate(components, 1):
                fig.add_scatter(x=grid, y=baseline_curve + peak, mode="lines", name=f"Peak {i}{suffix}",
                                line={"dash": "dash"}, row=1, col=1)
            fig.add_scatter(x=x, y=residual, name="Residual", showlegend=False, row=2, col=1)
            _style(fig, "Peak fit and residuals")
            fig.update_layout(height=620, showlegend=True)
            fig.update_xaxes(automargin=True)
            fig.update_yaxes(automargin=True)
            fig.update_xaxes(title_text=f"{state['x_label']} ({state['units']})", row=2, col=1)
            fig.update_yaxes(title_text="Intensity", row=1, col=1)
            fig.update_yaxes(title_text="Residual", row=2, col=1)
            chart.update_figure(plot_options(fig))

        display.on_value_change(lambda _: redraw())
        redraw()
        return rows

    with theme.panel_card() as fit_card:
        fit_card.set_visibility(False)
        theme.form_section("Peak fitting")
        ui.label("Drag an interval on the spectrum in Select fit range mode, or enter the limits below.") \
            .classes("vmi-subtitle")
        with field_grid():
            fit_low = number("Fit X min", "fit_xmin")
            fit_high = number("Fit X max", "fit_xmax")
            baseline = persist(ui.select(list(BASELINE_PARAMETERS), value="Linear", label="Baseline")
                               .props("outlined dense"), "pes.baseline")
            fit_targets = ui.select([], multiple=True, label="Spectra to fit").props("outlined dense use-chips")
        for field in (fit_low, fit_high):
            field.on_value_change(lambda _: refresh_plot())
        peak_rows = []
        peak_store = persist(ui.input(value="[{}]"), "pes.peak_guesses")
        peak_store.set_visibility(False)
        ui.label("Leave peak estimates blank to detect peaks automatically. Use the first row to choose the profile.") \
            .classes("vmi-subtitle")
        peak_list = ui.column().classes("w-full")
        try:
            guesses = json.loads(peak_store.value)
            for guess in guesses:
                add_peak(guess)
        except (AttributeError, TypeError, ValueError):
            ui.notify("Could not restore saved peak guesses.", type="warning")
            add_peak()
        with ui.row().classes("gap-2"):
            ui.button("Add peak", icon="add", on_click=lambda: (add_peak(), save_peaks())).props("flat")
            fit_button = ui.button("Fit peaks", icon="show_chart", on_click=fit)
            if ask_agent:
                ui.button("Ask Agent about fit range", icon="smart_toy", on_click=open_fit_agent).props("flat")
        fit_review = ui.column().classes("w-full")
        fit_display = ui.column().classes("w-full")

    with console_area:
        view.place_console()
