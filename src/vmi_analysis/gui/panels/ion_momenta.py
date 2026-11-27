"""Detector-position and TOF workbench for ion momentum reconstruction."""

import asyncio
from importlib.resources import files
import json
from math import isfinite, log, sqrt
from numbers import Real
from pathlib import Path
import re
import sys
from tempfile import NamedTemporaryFile

from nicegui import ui
import numpy as np
import plotly.graph_objects as go
from plotly.colors import get_colorscale

from vmi_analysis.ion_acquisition import collection_reads
from vmi_analysis.ion_momenta import (DEFAULT_MM_PER_PIXEL, LONGITUDINAL_CONVERSION,
                                     OXYGEN_MASS_U, OXYGEN_TOF_NS, POLYNOMIAL_LIMIT_AU,
                                     TRANSVERSE_CONVERSION, SpeciesMomentum, gated_preview,
                                     momentum_projections, oxygen_scaled_t0)
from vmi_analysis.ion_tof import DEFAULT_REFERENCE
from vmi_analysis.pes import evaluate_fit_for_plot, fit_peaks
from vmi_analysis.tof_bins import DEFAULT_BIN_SIZE_NS

from .. import shared, theme
from ..agent_panels import PanelOperation, PanelView, number, require_completed_job, resolve_path
from ..centroid_plots import COLORMAPS, _style, plot_options
from ..jobs import Job, ResourceBusyError
from . import ion_luna
from .common import (JobView, action_bar, adopt_agent_form, confirm_clear_outputs, field_grid, num_input,
                     path_input, persist)
from .ion_tof import REFERENCE_LABELS, SOURCES, _drag_interval, _mark_interval, _tof_figure


JOB_NAME = "Ion momenta"
GATE_OPERATION = "ion_momenta_gate"
FIT_OPERATION = "ion_momenta_fit"
SPECIES_OPERATION = "ion_momenta_species"
MOMENTUM_OPERATION = "ion_momenta_reconstruct"
PLOT_HEIGHT = 580
MOMENTUM_MODELS = {"simple": "Simple formula", "oxygen": "O⁺ reference polynomial"}
MAX_SPECIES = 64
SPECIES_SCHEMA = "vmi-ion-species/1"


def init_area_selection():
    script = (files("vmi_analysis.gui.assets") / "ion_area_selection.js").read_text(encoding="utf-8")
    ui.add_body_html(f"<script>{script}</script>", shared=True)


def _position_figure(image, title="Detector positions"):
    fig = _style(go.Figure(), title)
    fig.add_heatmap(z=image, colorscale="Viridis",
                    colorbar=dict(title="Counts", x=1, xanchor="left", thickness=14, xpad=5),
                    hovertemplate="X %{x} px<br>Y %{y} px<br>%{z:,} counts<extra></extra>")
    fig.update_layout(height=PLOT_HEIGHT, margin=dict(l=56, r=70, t=90, b=65, autoexpand=False))
    fig.update_xaxes(title_text="X (pixels)", range=[-0.5, image.shape[1] - 0.5],
                     constrain="domain", constraintoward="right")
    fig.update_yaxes(title_text="Y (pixels)", range=[-0.5, image.shape[0] - 0.5],
                     constrain="domain", scaleanchor="x", scaleratio=1)
    return fig


def _image_color_controls(plot, image):
    data_min, data_max = float(image.min()), float(image.max())
    applied = dict(colorscale=get_colorscale("Viridis"), zauto=True, zmin=None, zmax=None)

    def apply():
        low, high = lower.value, upper.value
        automatic = low is None and high is None
        if not automatic:
            low = data_min if low is None else low
            high = data_max if high is None else high
            if not isfinite(low) or not isfinite(high) or low >= high:
                ui.notify("Enter finite color limits with vmin below vmax.", type="warning")
                return
        applied.update(colorscale=get_colorscale(colors.value), zauto=automatic, zmin=low, zmax=high)
        plot.figure["data"][0].update(applied)
        plot.run_plot_method("restyle", {key: [value] for key, value in applied.items()}, [0])

    with ui.expansion("Image colors", icon="palette", value=False).classes("w-full vmi-hide-expand-icon"):
        with field_grid(3):
            colors = ui.select(list(COLORMAPS), value="Viridis", label="Colormap").props("outlined dense")
            lower = num_input("vmin (counts)", None, step="any")
            upper = num_input("vmax (counts)", None, step="any")
        ui.label("Blank limits use the image minimum or maximum.").classes("vmi-caption")
        ui.button("Apply colors", on_click=apply).props("flat")

    def draw(figure):
        figure.update_traces(**applied)
        plot.update_figure(plot_options(figure))

    return draw


def _image_plot(figure, image):
    plot = ui.plotly(plot_options(figure)).classes("w-full min-w-0")
    _image_color_controls(plot, image)


def _interval(values, defaults=None):
    if defaults is not None:
        values = [default if value is None else value for value, default in zip(values, defaults, strict=True)]
    if (None in values or not np.isfinite(values).all() or values[0] > values[1]
            or (defaults is None and values[0] == values[1])):
        raise ValueError("Enter increasing, finite gate limits.")
    return tuple(float(value) for value in values)


def _gate_fields(labels):
    with field_grid(2):
        return [num_input(label, None, step="any") for label in labels]


def _fit_controls(plot, source, edges, counts, context, save_peak, register_peak, species_file, ask_agent=None):
    selected_region = model = None
    fit_busy = False
    with ui.row().classes("items-center gap-2"):
        mode = ui.toggle(["Zoom", "Select fit range"], value="Zoom")
        ui.button("Clear fit", on_click=lambda: clear()).props("flat")
    fields = _gate_fields(("Fit TOF min (ns)", "Fit TOF max (ns)"))

    def redraw():
        fig = _tof_figure(edges, {source: counts}, "Position-filtered TOF",
                          selecting=mode.value == "Select fit range")
        fig.update_layout(uirevision="position-fit", showlegend=model is not None)
        _mark_interval(fig, *(selected_region or (None, None)), mode.value == "Select fit range")
        if model is not None:
            x, fitted, background, _, residual, peaks = model
            grid, total, _, _ = evaluate_fit_for_plot(x, background, peaks)
            fig.add_scatter(x=grid, y=total, mode="lines", name="Gaussian fit")
            fig.add_scatter(x=x, y=fitted + residual, mode="markers", name="Fit samples")
        plot.update_figure(plot_options(fig))

    def clear():
        nonlocal selected_region, model
        selected_region = model = None
        for field in fields:
            field.value = None
        output.clear()
        redraw()

    def selected(event):
        nonlocal selected_region, model
        if mode.value == "Select fit range" and (limits := _drag_interval(event)):
            for field, limit in zip(fields, limits, strict=True):
                field.value = limit
            output.clear()
            selected_region, model = limits, None
            redraw()

    async def fit():
        nonlocal selected_region, model, fit_busy
        if fit_busy:
            return {"status": "failed", "error": "Wait for the current Gaussian fit to finish."}
        fit_busy = True
        try:
            region = _interval([field.value for field in fields])
            fit_button.disable()
            output.clear()
            centers = (edges[:-1] + edges[1:]) / 2
            fitted_model = await asyncio.to_thread(fit_peaks, centers, counts, region)
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            ui.notify(str(exc), type="warning")
            return {"status": "failed", "error": str(exc)}
        finally:
            fit_busy = False
            fit_button.enable()
        if output.is_deleted or region != tuple(field.value for field in fields):
            return {"status": "cancelled", "note": "The fit selection changed while fitting."}
        selected_region, model = region, fitted_model
        redraw()
        error = float(np.sqrt(np.mean(model[4] ** 2)))
        records = []
        with output:
            ui.label(f"Constant baseline · Residual RMS {error:.4g} counts/bin").classes("vmi-caption")
            ui.label("The fit supplies initial values. Edit the zero-momentum reference and TOF gate before saving."
                     ).classes("vmi-subtitle")
            for peak in sorted(model[-1], key=lambda item: item["center"]):
                center = peak["center"]
                sigma = peak["gaussian_fwhm"] / (2 * sqrt(2 * log(2)))
                low, high = center - 3 * sigma, center + 3 * sigma
                record = dict(context, fit_range_ns=list(region), center_ns=center, sigma_ns=sigma,
                              fwhm_ns=peak["fwhm"], fit_window_ns=[low, high], profile="Gaussian",
                              baseline="Constant", residual_rms_counts_per_bin=error, at_bound=peak["bound"])
                records.append(record)
                register_peak(*_species_fields(record, save_peak))
        return {"status": "done", "panel": JOB_NAME, "fit_range_ns": list(region),
                "residual_rms_counts_per_bin": error, "peaks": records}

    def open_agent():
        nonlocal fit_busy
        try:
            if fit_busy:
                raise ValueError("Wait for the current Gaussian fit to finish.")
            values = [field.value for field in fields]
            region = selected_region or (_interval(values) if any(value is not None for value in values)
                                         else (float(edges[0]), float(edges[-1])))
            initial = {"region": list(region)}
        except ValueError as exc:
            ui.notify(str(exc), type="warning")
            return
        selected = selected_region
        field_values = tuple(field.value for field in fields)

        def bind(parameters):
            if (fit_busy or plot.is_deleted or output.is_deleted or selected_region != selected
                    or tuple(field.value for field in fields) != field_values):
                raise ValueError("The position-filtered TOF or fit selection changed. Ask the Agent again.")
            if not isinstance(parameters, dict) or parameters.keys() - {"region"}:
                raise ValueError("Use only the fit region for this operation.")
            return {"region": list(_interval(parameters.get("region", initial["region"])))}

        def review(_job, draft, approve, decline):
            with fit_review, theme.panel_card():
                theme.form_section("Agent position-filtered TOF fit")
                limits = _gate_fields(("Fit TOF min (ns)", "Fit TOF max (ns)"))
                for field, value in zip(limits, draft["region"], strict=True):
                    field.value = value

                async def confirm():
                    edits = {"region": [field.value for field in limits]}
                    try:
                        bind(draft | edits)
                    except ValueError as exc:
                        ui.notify(str(exc), type="warning")
                        return
                    await approve(edits)

                with ui.row().classes("items-center gap-2"):
                    ui.button("Approve Gaussian fit", on_click=confirm)
                    ui.button("Decline", on_click=decline).props("flat")

        async def run(draft):
            for field, value in zip(fields, draft["region"], strict=True):
                field.value = value
            result = await fit()
            if result["status"] != "done":
                raise ValueError(result.get("error") or result.get("note") or "The Gaussian fit failed.")
            return result

        ask_agent(PanelView(panel=JOB_NAME, operation=FIT_OPERATION, label=f"{SOURCES[source]} position-filtered TOF",
                            instruction="Fit Gaussian peaks in the currently position-filtered TOF spectrum.",
                            settings=initial, bind=bind, render_review=review, review_container=fit_review,
                            run_interactive=run,
                            context={**context, "samples": int(counts.sum()), "tof_range_ns": [float(edges[0]), float(edges[-1])]}))

    mode.on_value_change(lambda _: redraw())
    plot.on("plotly_selected", selected, js_handler="event => emit(event?.range ?? null)")
    fit_button = ui.button("Fit Gaussian peaks", on_click=fit)
    if ask_agent:
        ui.button("Ask Agent about fit", icon="smart_toy", on_click=open_agent).props("flat")
    with ui.row().classes("w-full items-center gap-2"):
        path_input("Species JSON file (append assignments)", browse="save", filters=[".json"],
                   default_name="ion-species.json").bind_value(species_file).classes("grow basis-72")
        ui.button("Delete species JSON", icon="delete",
                  on_click=lambda: _delete_species_file(species_file)).props("flat color=negative")
    fit_review = ui.column().classes("w-full")
    output = ui.column().classes("w-full")
    redraw()


def _species_fields(record, save_peak):
    low, high = record["fit_window_ns"]
    with ui.column().classes("w-full gap-1") as row:
        warning = " · Fit at parameter bound" if record["at_bound"] else ""
        ui.label(f'μ {record["center_ns"]:.6g} ns · σ {record["sigma_ns"]:.6g} ns · '
                 f'μ ± 3σ [{low:.6g}, {high:.6g}] ns{warning}').classes("vmi-caption")
        with field_grid(3):
            reference_tof = num_input("Zero-momentum TOF t₀ (ns)", record["center_ns"], step="any")
            minimum_tof = num_input("Species TOF min (ns)", low, step="any")
            maximum_tof = num_input("Species TOF max (ns)", high, step="any")
        with ui.row().classes("items-center gap-2"):
            formula = ui.input("Chemical formula", placeholder="Xe").props("outlined dense")
            charge = num_input("Charge (e)", 1, step=1)

            def read_peak(*, allow_blank=False):
                name = (formula.value or "").strip()
                number = charge.value
                if allow_blank and not name:
                    return None
                if not re.fullmatch(r"(?:[A-Z][a-z]?|[0-9]|[()\[\]]|[·.])+", name) or not re.search("[A-Z]", name):
                    raise ValueError("Enter a chemical formula such as Xe or N2.")
                if number is None or not isfinite(number) or not float(number).is_integer() or not number:
                    raise ValueError(f"Enter a nonzero integer charge for {name}, such as 1 or 2.")
                t0 = reference_tof.value
                if t0 is None or not isfinite(t0):
                    raise ValueError(f"Enter a finite zero-momentum TOF for {name}.")
                window = _interval([minimum_tof.value, maximum_tof.value])
                return dict(record, formula=name, charge=int(number), t0_ns=float(t0), window_ns=list(window))

            def add():
                try:
                    peak = read_peak()
                except ValueError as exc:
                    ui.notify(str(exc), type="warning")
                    return
                save_peak(peak)

            ui.button("Append to species file", on_click=add).props("flat")
    return row, read_peak


def _linked_plots(job, jobs, source, result, save_peak, register_peak, species_file, ask_agent=None):
    image, edges = result[f"{source}_image"], result["edges"]
    fallback_limits = ((-0.5, image.shape[1] - 0.5), (-0.5, image.shape[0] - 0.5), (edges[0], edges[-1]))
    x_limits, y_limits, tof_limits = result.get(f"{source}_data_limits", fallback_limits)
    triggers, offsets = result["trigger_times"], result["segment_offsets"]
    reference = job.extra["reference"]
    revisions = {"position": 0, "tof": 0}
    gates = dict.fromkeys(revisions)
    redraw = {"position": lambda: redraw_image(), "tof": lambda: redraw_tof()}

    async def apply(kind, bounds):
        revisions[kind] += 1
        revision = revisions[kind]
        target = filtered_tof if kind == "position" else filtered_image
        target.clear()
        if bounds is None:
            gates[kind] = None
            redraw[kind]()
            return {"status": "done", "kind": kind, "bounds": None}
        with target:
            ui.label("Applying gate…").classes("vmi-caption")
        try:
            path = shared.checked_path(job.extra["source_file"])
            with jobs.preview_read_access(collection_reads(path)) as stopping:
                filtered = await asyncio.to_thread(
                    gated_preview, path, reference, source, edges, triggers, offsets, image.shape,
                    position_gate=bounds if kind == "position" else None,
                    tof_gate=bounds if kind == "tof" else None,
                    stopped=lambda: stopping() or revisions[kind] != revision or target.is_deleted,
                )
        except (OSError, ValueError, KeyError, ResourceBusyError) as exc:
            if revisions[kind] == revision and not target.is_deleted:
                target.clear()
                with target:
                    ui.label(str(exc)).classes("text-negative")
                gates[kind] = None
                redraw[kind]()
            return {"status": "failed", "kind": kind, "error": str(exc)}
        if revisions[kind] != revision or target.is_deleted:
            return {"status": "cancelled", "kind": kind,
                    "note": "The result view changed while the gate was being applied."}
        gates[kind] = bounds
        redraw[kind]()
        target.clear()
        with target:
            if kind == "tof":
                ui.label(f"{bounds[0]:.6g}–{bounds[1]:.6g} ns · {filtered.sum():,} mapped positions").classes("vmi-caption")
                _image_plot(_position_figure(filtered, "TOF-filtered detector positions"), filtered)
            else:
                ui.label(f"{filtered.sum():,} arrivals inside the X/Y gate").classes("vmi-caption")
                chart = ui.plotly(go.Figure()).classes("w-full")
                context = dict(source=source, input_file=job.extra["input_name"], trigger_reference=reference,
                               position_gate_px=dict(zip(("xmin", "xmax", "ymin", "ymax"), bounds, strict=True)),
                               histogram_range_ns=[float(edges[0]), float(edges[-1])],
                               bin_size_ns=job.extra["bin_size_ns"])
                _fit_controls(chart, source, edges, filtered, context, save_peak, register_peak, species_file,
                              ask_agent=ask_agent)
        return {"status": "done", "kind": kind, "bounds": list(bounds), "arrivals": int(filtered.sum())}

    def open_gate_agent(kind):
        limits = (*x_limits, *y_limits) if kind == "position" else tof_limits
        initial = {"kind": kind, "bounds": list(gates[kind] or limits)}
        opened_revision = revisions[kind]
        target = filtered_tof if kind == "position" else filtered_image

        def bind(parameters):
            if revisions[kind] != opened_revision or target.is_deleted:
                raise ValueError("The linked result or gate changed. Ask the Agent again from this view.")
            if not isinstance(parameters, dict) or parameters.keys() - {"kind", "bounds"}:
                raise ValueError("Use only the gate kind and bounds for this operation.")
            draft = initial | parameters
            if draft["kind"] != kind:
                raise ValueError("Keep the gate kind from the originating control.")
            bounds = draft["bounds"]
            if bounds is None:
                return {"kind": kind, "bounds": None}
            if not isinstance(bounds, (list, tuple)) or len(bounds) != len(limits):
                raise ValueError("Enter four X/Y limits or two TOF limits, or null to clear the gate.")
            values = (_interval(bounds[:2], x_limits) + _interval(bounds[2:], y_limits)
                      if kind == "position" else _interval(bounds, tof_limits))
            return {"kind": kind, "bounds": list(values)}

        def review(_job, draft, approve, decline):
            with gate_review, theme.panel_card():
                theme.form_section(f"Agent {kind} gate")
                labels = ("X min (px)", "X max (px)", "Y min (px)", "Y max (px)") if kind == "position" else (
                    "Gate TOF min (ns)", "Gate TOF max (ns)")
                clear_gate = ui.checkbox("Clear this gate", value=draft["bounds"] is None)
                fields = _gate_fields(labels)
                for field, value in zip(fields, draft["bounds"] or limits, strict=True):
                    field.value = value
                    field.bind_enabled_from(clear_gate, "value", lambda enabled: not enabled)

                async def confirm():
                    edits = {"bounds": None if clear_gate.value else [field.value for field in fields]}
                    try:
                        bind(draft | edits)
                    except ValueError as exc:
                        ui.notify(str(exc), type="warning")
                        return
                    await approve(edits)

                with ui.row().classes("items-center gap-2"):
                    ui.button("Approve gate", on_click=confirm)
                    ui.button("Decline", on_click=decline).props("flat")

        async def run(draft):
            outcome = await apply(kind, None if draft["bounds"] is None else tuple(draft["bounds"]))
            if outcome["status"] != "done":
                raise ValueError(outcome.get("error") or outcome.get("note") or "The gate could not be applied.")
            return {**outcome, "panel": JOB_NAME, "job_id": job.id, "source": source}

        ask_agent(PanelView(panel=JOB_NAME, operation=GATE_OPERATION, label=f"{SOURCES[source]} {kind} gate",
                            instruction="Apply or clear this gate on the originating linked histogram result.",
                            settings=initial, bind=bind, render_review=review, review_container=gate_review,
                            run_interactive=run,
                            context={"job_id": job.id, "source": source, "input": job.extra["input_name"],
                                     "current_gates": gates.copy()}))

    async def enter(kind, fields):
        try:
            values = [field.value for field in fields]
            bounds = (_interval(values[:2], x_limits) + _interval(values[2:], y_limits)
                      if kind == "position" else _interval(values, tof_limits))
        except ValueError as exc:
            ui.notify(str(exc), type="warning")
            return
        await apply(kind, bounds)

    async def select(kind, fields, bounds):
        for field, value in zip(fields, bounds, strict=True):
            field.value = value
        await enter(kind, fields)

    async def clear(kind, fields):
        for field in fields:
            field.value = None
        await apply(kind, None)

    async def image_selected(event):
        ranges = event.args or {}
        if image_mode.value == "Select area" and "x" in ranges and "y" in ranges:
            await select("position", xy, sorted(ranges["x"]) + sorted(ranges["y"]))

    async def tof_selected(event):
        if tof_mode.value == "Select TOF range" and (limits := _drag_interval(event)):
            await select("tof", times, limits)

    def redraw_image():
        fig = _position_figure(image)
        if gates["position"] is not None:
            xmin, xmax, ymin, ymax = gates["position"]
            fig.add_shape(type="rect", x0=xmin, x1=xmax, y0=ymin, y1=ymax,
                          line=dict(color="#38bdf8", width=2))
        fig.update_layout(uirevision="positions")
        draw_image(fig)

    def redraw_tof():
        selecting = tof_mode.value == "Select TOF range"
        fig = _tof_figure(edges, {source: result[source]}, "Time-of-flight spectrum", selecting=selecting)
        fig.update_layout(height=PLOT_HEIGHT, margin=dict(l=65, r=24, t=90, b=65), uirevision="tof")
        _mark_interval(fig, *(gates["tof"] or (None, None)), selecting)
        tof_plot.update_figure(plot_options(fig))

    with ui.element("div").classes("vmi-ion-detector-row"):
        with ui.column().classes("min-w-0 w-full"):
            with ui.row().classes("items-center gap-2"):
                image_mode = ui.toggle(["Zoom", "Select area"], value="Zoom")
            image_plot = ui.plotly(go.Figure()).classes("w-full min-w-0")
            draw_image = _image_color_controls(image_plot, image)
            with ui.expansion("X/Y gate limits", icon="filter_alt").classes("w-full vmi-hide-expand-icon"):
                ui.label("Blank limits use the data range.").classes("vmi-caption")
                xy = _gate_fields(("X min (px)", "X max (px)", "Y min (px)", "Y max (px)"))
                ui.button("Apply X/Y gate", on_click=lambda: enter("position", xy)).props("flat")
                ui.button("Clear X/Y gate", on_click=lambda: clear("position", xy)).props("flat")
                if ask_agent:
                    ui.button("Ask Agent about X/Y gate", icon="smart_toy",
                              on_click=lambda: open_gate_agent("position")).props("flat")
        with ui.column().classes("min-w-0 w-full"):
            with ui.row().classes("items-center gap-2"):
                tof_mode = ui.toggle(["Zoom", "Select TOF range"], value="Zoom")
            tof_plot = ui.plotly(go.Figure()).classes("w-full min-w-0")
            with ui.expansion("TOF gate limits", icon="filter_alt").classes("w-full vmi-hide-expand-icon"):
                ui.label("Blank limits use the data range.").classes("vmi-caption")
                times = _gate_fields(("Gate TOF min (ns)", "Gate TOF max (ns)"))
                ui.button("Apply TOF gate", on_click=lambda: enter("tof", times)).props("flat")
                ui.button("Clear TOF gate", on_click=lambda: clear("tof", times)).props("flat")
                if ask_agent:
                    ui.button("Ask Agent about TOF gate", icon="smart_toy",
                              on_click=lambda: open_gate_agent("tof")).props("flat")
    with ui.element("div").classes("vmi-ion-detector-row"):
        filtered_image = ui.column().classes("w-full min-w-0")
        filtered_tof = ui.column().classes("w-full min-w-0")
    gate_review = ui.column().classes("w-full")
    image_mode.on_value_change(lambda event: image_plot.props(
        'data-ion-area="on"' if event.value == "Select area" else 'data-ion-area="off"'))
    tof_mode.on_value_change(lambda _: redraw_tof())
    image_plot.on("ion-area-selected", image_selected, js_handler="event => emit(event.detail)")
    tof_plot.on("plotly_selected", tof_selected, js_handler="event => emit(event?.range ?? null)")
    redraw_image()
    redraw_tof()


def _momentum_figure(edges, image, horizontal, vertical):
    centers = (edges[:-1] + edges[1:]) / 2
    fig = _style(go.Figure(), f"{horizontal}–{vertical}")
    fig.add_heatmap(x=centers, y=centers, z=image, colorscale="Viridis",
                    colorbar=dict(title="Counts", thickness=14),
                    hovertemplate="%{x:.4g}, %{y:.4g} a.u.<br>%{z:,} counts<extra></extra>")
    fig.update_layout(height=460, margin=dict(l=65, r=65, t=55, b=55))
    axis = dict(range=[edges[0], edges[-1]], constrain="domain", zeroline=False)
    fig.update_xaxes(title_text=f"{horizontal} (a.u.)", **axis)
    fig.update_yaxes(title_text=f"{vertical} (a.u.)", scaleanchor="x", scaleratio=1, **axis)
    return fig


def _species_entries(document, sources, reference):
    if not isinstance(document, dict) or document.get("schema") != SPECIES_SCHEMA:
        raise ValueError("Choose an ion-species.json file exported by this panel.")
    items = document.get("species")
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_SPECIES:
        raise ValueError(f"The species file must contain 1–{MAX_SPECIES} assignments.")
    entries = []
    for item in items:
        if (not isinstance(item, dict) or not isinstance(item.get("formula"), str) or not item["formula"].strip()
                or type(item.get("charge")) is not int or not item["charge"]
                or abs(item["charge"]) > 2**53):
            raise ValueError("Each assignment needs a chemical formula and a nonzero integer charge.")
        if item.get("source") not in sources or item.get("trigger_reference") != reference:
            raise ValueError("Plot positions and TOF with the count sources and trigger reference in the species file.")
        window = item.get("window_ns")
        t0 = item.get("t0_ns", item.get("center_ns"))
        if (not isinstance(window, list) or len(window) != 2
                or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                       or not -1e100 < value < 1e100 for value in (*window, t0))
                or window[0] >= window[1]):
            raise ValueError("Each assignment needs finite TOF limits and a zero-momentum TOF.")
        entries.append((item["formula"], item["charge"], item["source"], tuple(map(float, window)), float(t0)))
    return entries


def _momentum_parameters(parameters, sources, reference):
    allowed = {"model", "center_px", "mm_per_pixel", "assignments"}
    if not isinstance(parameters, dict) or parameters.keys() - allowed:
        raise ValueError("Use the momentum model, detector calibration, and species assignments.")
    model = parameters["model"]
    if model not in MOMENTUM_MODELS:
        raise ValueError("Choose a valid momentum model.")
    center = parameters["center_px"]
    if not isinstance(center, (list, tuple)) or len(center) != 2:
        raise ValueError("Enter the detector center as [X, Y].")
    center = [number(value) for value in center]
    pixel_scale = number(parameters["mm_per_pixel"])
    if pixel_scale <= 0:
        raise ValueError("Enter a positive detector scale.")
    requested = parameters["assignments"]
    if not isinstance(requested, list) or not 1 <= len(requested) <= MAX_SPECIES:
        raise ValueError(f"Provide 1–{MAX_SPECIES} species assignments.")
    assignments, species = [], []
    fields = {"formula", "charge", "source", "window_ns", "t0_ns", "mass_u", "pz_scale"}
    for item in requested:
        if not isinstance(item, dict) or item.keys() - fields or not {"formula", "charge", "source", "window_ns", "mass_u"} <= item.keys():
            raise ValueError("Each momentum assignment needs formula, charge, source, TOF window, and mass.")
        entry = _species_entries(dict(schema=SPECIES_SCHEMA, species=[
            {**item, "trigger_reference": reference,
             "t0_ns": 0 if item.get("t0_ns") is None and model == "oxygen" else item.get("t0_ns")}
        ]), sources, reference)[0]
        formula, charge, source, window, _ = entry
        mass = number(item["mass_u"])
        if mass <= 0:
            raise ValueError(f"Enter a positive ion mass for {formula}.")
        if model == "oxygen" and charge <= 0:
            raise ValueError(f"The O⁺ reference model requires positive ion charge for {formula}.")
        t0 = item.get("t0_ns")
        if t0 is None and model == "oxygen":
            t0 = float(oxygen_scaled_t0(mass, charge))
        if t0 is None:
            raise ValueError(f"Enter a finite t₀ for {formula}.")
        t0 = number(t0)
        factor = number(item.get("pz_scale", 1)) if model == "simple" else 1
        if factor <= 0:
            raise ValueError(f"Enter a positive p_z scale for {formula}.")
        normalized = {"formula": formula, "charge": charge, "source": source, "window_ns": list(window),
                      "mass_u": mass, "t0_ns": t0, "pz_scale": factor}
        assignments.append(normalized)
        species.append(SpeciesMomentum(source, window, mass, charge, t0, factor))
    return {"model": model, "center_px": center, "mm_per_pixel": pixel_scale,
            "assignments": assignments}, species


def _momentum_controls(job, jobs, result, species_file, read_current, ask_agent=None):
    sources = tuple(source for source in ("clusters", "pixels") if source in result)
    detector = result[f"{sources[0]}_image"]
    if not detector.any() and len(sources) > 1:
        detector = result[f"{sources[1]}_image"]
    y, x = np.unravel_index(detector.argmax(), detector.shape) if detector.any() else (None, None)
    rows = []
    assignment_source = "appended"
    plotting = False

    def add_row(entry):
        formula, charge, source, window, t0 = entry
        with calibrations, ui.column().classes("w-full gap-1"):
            ui.label(f"{formula} · charge {charge:+d} · {SOURCES[source]} · "
                     f"{window[0]:.6g}–{window[1]:.6g} ns").classes("vmi-caption")
            with field_grid(3):
                mass = num_input("Ion mass (u)", 131.293 if formula == "Xe" else None, step="any")
                reference = num_input("Momentum t₀ (ns)", t0, step="any")
                pz_scale = num_input("p_z scale factor", 1, step="any").bind_visibility_from(
                    model, "value", lambda value: value == "simple")
        rows.append((formula, charge, source, window, mass, reference, pz_scale))
        plot_button.set_enabled(not plotting)

    def show_assignments(entries, source, message):
        nonlocal assignment_source
        assignment_source = source
        calibrations.clear()
        for panel in projections.values():
            panel.clear()
        rows.clear()
        plot_button.disable()
        for entry in entries:
            add_row(entry)
        status.set_text(message)

    def append_assignment(item):
        if assignment_source == "appended" and len(rows) < MAX_SPECIES:
            entries = _species_entries(dict(schema=SPECIES_SCHEMA, species=[item]),
                                       sources, job.extra["reference"])
            add_row(entries[0])
            status.set_text(f"{len(rows)} assignments from this run.")

    def use_current():
        try:
            items = read_current()
            if not items:
                show_assignments([], "current", "Enter a chemical formula on a fitted peak to use it here.")
                return
            entries = _species_entries(dict(schema=SPECIES_SCHEMA, species=items), sources, job.extra["reference"])
        except ValueError as exc:
            ui.notify(str(exc), type="warning")
            return
        show_assignments(entries, "current", f"{len(entries)} assignments from the fitted peaks.")

    def load():
        try:
            destination = species_file.value.strip()
            if not destination:
                raise ValueError("Choose a Species JSON file in the Ion momenta form.")
            path = shared.checked_path(destination, cwd=shared.workspace_root())
            with jobs.preview_read_access([path]):
                with path.open("rb") as stream:
                    raw = stream.read(256 * 1024 + 1)
            if len(raw) > 256 * 1024:
                raise ValueError("The species JSON file exceeds 256 KiB.")
            entries = _species_entries(json.loads(raw.decode("utf-8-sig")), sources, job.extra["reference"])
        except (OSError, ValueError, RecursionError, ResourceBusyError) as exc:
            ui.notify(str(exc), type="warning")
            return
        show_assignments(entries, "file", f"{len(entries)} assignments from {path.name}.")

    def current_parameters():
        return {
            "model": model.value, "center_px": [center_x.value, center_y.value],
            "mm_per_pixel": pixel_scale.value,
            "assignments": [{"formula": formula, "charge": charge, "source": source,
                             "window_ns": list(window), "mass_u": mass.value, "t0_ns": reference.value,
                             "pz_scale": scale.value}
                            for formula, charge, source, window, mass, reference, scale in rows],
        }

    async def plot():
        nonlocal plotting
        if plotting:
            return {"status": "failed", "error": "Wait for the current momentum reconstruction to finish."}
        plotting = True
        method = model.value
        simple = method == "simple"
        plots = projections[method]
        try:
            settings, species = _momentum_parameters(current_parameters(), sources, job.extra["reference"])
            labels = [(item["formula"], item["charge"], item["source"]) for item in settings["assignments"]]
            center = settings["center_px"]
            outside_domain = np.zeros(len(species), dtype=np.int64)
            plot_button.disable()
            model.disable()
            load_button.disable()
            current_button.disable()
            plots.clear()
            with plots:
                ui.label("Calculating momentum projections…").classes("vmi-caption")
            path = shared.checked_path(job.extra["source_file"])
            with jobs.preview_read_access(collection_reads(path)) as stopping:
                edges, images, totals = await asyncio.to_thread(
                    momentum_projections, path, job.extra["reference"], species, tuple(center),
                    settings["mm_per_pixel"], result["trigger_times"], result["segment_offsets"],
                    model=method, outside_domain=outside_domain,
                    stopped=lambda: stopping() or plots.is_deleted,
                )
        except (ValueError, OSError, KeyError, ResourceBusyError) as exc:
            if not plots.is_deleted:
                plots.clear()
                ui.notify(str(exc), type="warning")
            return {"status": "failed", "error": str(exc)}
        finally:
            plotting = False
            if not plots.is_deleted:
                plot_button.set_enabled(bool(rows))
                model.enable()
                load_button.enable()
                current_button.enable()
        if plots.is_deleted:
            return {"status": "cancelled", "note": "The momentum result view was closed."}
        records = []
        plots.clear()
        with plots:
            for label, bins, maps, count, calibration, omitted in zip(
                    labels, edges, images, totals, species, outside_domain, strict=True):
                formula, charge, source = label
                records.append({"formula": formula, "charge": charge, "source": source,
                                "arrivals": int(count), "outside_model_domain": int(omitted),
                                "t0_ns": float(calibration.t0_ns), "pz_scale": float(calibration.pz_scale)})
                detail = (f"p_z scale {calibration.pz_scale:.6g}" if simple
                          else f"{omitted:,} arrivals outside the {POLYNOMIAL_LIMIT_AU:g} a.u. model limit")
                ui.label(f"{formula} · charge {charge:+d} · {SOURCES[source]} · {count:,} arrivals · "
                         f"t₀ {calibration.t0_ns:.6g} ns · {detail}"
                         ).classes("vmi-caption")
                if count:
                    with ui.element("div").classes("grid grid-cols-1 md:grid-cols-2 gap-4 w-full"):
                        for image, horizontal, vertical in zip(maps, ("pₓ", "pᵧ"), ("pᵧ", "p_z"), strict=True):
                            with ui.column().classes("w-full min-w-0"):
                                _image_plot(_momentum_figure(bins, image, horizontal, vertical), image)
                else:
                    message = ("No arrivals inside the polynomial model’s momentum limit." if omitted
                               else "No arrivals with positive TOF inside this species window.")
                    ui.label(message).classes("vmi-subtitle")
        return {"status": "done", "panel": JOB_NAME, "job_id": job.id, "settings": settings, "results": records}

    def open_agent():
        try:
            if plotting:
                raise ValueError("Wait for the current momentum reconstruction to finish.")
            initial, _ = _momentum_parameters(current_parameters(), sources, job.extra["reference"])
        except ValueError as exc:
            ui.notify(str(exc), type="warning")
            return
        snapshot = current_parameters()

        def bind(parameters):
            if plotting or current_parameters() != snapshot or any(panel.is_deleted for panel in projections.values()):
                raise ValueError("The assignments or detector calibration changed. Ask the Agent again.")
            draft, _ = _momentum_parameters(initial | parameters, sources, job.extra["reference"])
            return draft

        def review(_job, draft, approve, decline):
            with momentum_review, theme.panel_card():
                theme.form_section("Agent momentum reconstruction")
                method = ui.select(MOMENTUM_MODELS, value=draft["model"], label="Momentum model").props("outlined dense")
                with field_grid(3):
                    cx = num_input("Detector center X (px)", draft["center_px"][0], step="any")
                    cy = num_input("Detector center Y (px)", draft["center_px"][1], step="any")
                    scale = num_input("Detector scale (mm/px)", draft["mm_per_pixel"], step="any")
                editors = []
                for item in draft["assignments"]:
                    ui.label(f"{item['formula']} · charge {item['charge']:+d} · {SOURCES[item['source']]} · "
                             f"{item['window_ns'][0]:.6g}–{item['window_ns'][1]:.6g} ns").classes("vmi-caption")
                    with field_grid(3):
                        mass = num_input("Ion mass (u)", item["mass_u"], step="any")
                        t0 = num_input("Momentum t₀ (ns)", item["t0_ns"], step="any")
                        pz = num_input("p_z scale factor", item["pz_scale"], step="any")
                    editors.append((mass, t0, pz))

                async def confirm():
                    assignments = []
                    for item, (mass, t0, pz) in zip(draft["assignments"], editors, strict=True):
                        assignments.append({**item, "mass_u": mass.value, "t0_ns": t0.value, "pz_scale": pz.value})
                    edits = {"model": method.value, "center_px": [cx.value, cy.value],
                             "mm_per_pixel": scale.value, "assignments": assignments}
                    try:
                        bind(draft | edits)
                    except ValueError as exc:
                        ui.notify(str(exc), type="warning")
                        return
                    await approve(edits)

                with ui.row().classes("items-center gap-2"):
                    ui.button("Approve momentum reconstruction", on_click=confirm)
                    ui.button("Decline", on_click=decline).props("flat")

        async def run(draft):
            model.value = draft["model"]
            center_x.value, center_y.value = draft["center_px"]
            pixel_scale.value = draft["mm_per_pixel"]
            entries = [(item["formula"], item["charge"], item["source"], tuple(item["window_ns"]), item["t0_ns"])
                       for item in draft["assignments"]]
            show_assignments(entries, "agent", f"{len(entries)} assignments from the approved Agent draft.")
            for row, item in zip(rows, draft["assignments"], strict=True):
                row[4].value = item["mass_u"]
                row[5].value = item["t0_ns"]
                row[6].value = item["pz_scale"]
            outcome = await plot()
            if outcome["status"] != "done":
                raise ValueError(outcome.get("error") or outcome.get("note") or "Momentum reconstruction failed.")
            return outcome

        ask_agent(PanelView(panel=JOB_NAME, operation=MOMENTUM_OPERATION, label="Ion momentum reconstruction",
                            instruction="Reconstruct momenta from the current assignments and source run. Preserve species identities unless asked to change them.",
                            settings=initial, bind=bind, render_review=review, review_container=momentum_review,
                            run_interactive=run,
                            context={"job_id": job.id, "input": job.extra["input_name"],
                                     "trigger_reference": job.extra["reference"], "available_sources": sources}))

    theme.form_section("Momentum projections")
    model = ui.toggle(MOMENTUM_MODELS, value="simple")
    ui.label("Uses the species TOF window across the whole detector. TOF is used directly. "
             "The initial center comes from the brightest detector bin.").classes("vmi-subtitle")
    with ui.expansion("Momentum formulas and units", icon="functions", value=False).classes(
        "w-full vmi-hide-expand-icon"
    ):
        with ui.column().classes("w-full gap-1").bind_visibility_from(model, "value", lambda value: value == "simple"):
            ui.label(f"p_x = {TRANSVERSE_CONVERSION:.9f} × M × s × (x − x₀) / t").classes("font-mono")
            ui.label(f"p_y = {TRANSVERSE_CONVERSION:.9f} × M × s × (y − y₀) / t").classes("font-mono")
            ui.label(f"p_z = {LONGITUDINAL_CONVERSION:.6f} × Z × f_z × (t₀ − t)").classes("font-mono")
            ui.label("All momenta are in atomic units. M is the ion mass in u and Z is the charge number in units of e."
                     ).classes("vmi-subtitle")
            ui.label("x and y are the hit or cluster coordinates in pixels. x₀ and y₀ are the detector center in pixels. "
                     f"s is the detector scale in mm/pixel and defaults to {DEFAULT_MM_PER_PIXEL:.6f} (75 mm / 180 pixels)."
                     ).classes("vmi-subtitle")
            ui.label("t is the measured TOF in ns relative to the selected trigger, used directly. "
                     "t₀ is the species’ editable zero-momentum TOF in ns on the same time reference. "
                     "f_z is the dimensionless p_z scale factor and defaults to 1.").classes("vmi-subtitle")
            ui.label("The transverse model assumes negligible transverse forces. The longitudinal formula is linear "
                     "near t₀ and uses the first extraction field of 22.2 V/mm. For positive ions, an earlier arrival "
                     "gives positive p_z toward the detector.").classes("vmi-subtitle")
        with ui.column().classes("w-full gap-1").bind_visibility_from(model, "value", lambda value: value == "oxygen"):
            ui.label("X = s(x − x₀), Y = s(y − y₀), R = √(X² + Y²)").classes("font-mono")
            ui.label(f"a = M / {OXYGEN_MASS_U:g}, b = Z, τ = √(b/a) × (t − t₀)/1000, S = √(ab)").classes("font-mono")
            ui.label("P_z = S F_z(τ, R), P_x = S (X/R) F_⊥(τ, R), P_y = S (Y/R) F_⊥(τ, R)").classes("font-mono")
            ui.label("F_z = 312.68τ + 0.011545R² + 6.8175τ² − 0.0016119R²τ − 0.61458τ³ "
                     "+ 3.2141×10⁻⁷R⁴ − 0.010424R²τ² − 5.2163τ⁴").classes("font-mono")
            ui.label("F_⊥ = 4.5586R − 0.097123Rτ + 1.1114×10⁻⁴R³ − 0.025513Rτ² "
                     "+ 1.7719×10⁻⁴R³τ − 0.25058Rτ³").classes("font-mono")
            ui.label("X, Y and R are in mm. M is the ion mass in u and Z is the positive charge number. "
                     "The panel takes t and t₀ in ns and converts τ to μs. Momenta are in atomic units. "
                     "At R = 0, P_x = P_y = 0. Near the reference on axis, later arrivals give positive P_z."
                     ).classes("vmi-subtitle")
            ui.label(f"A blank t₀ uses {OXYGEN_TOF_NS:g} × √(a/b) ns. The O⁺ reference uses the nominal mass "
                     f"{OXYGEN_MASS_U:g} u and charge +1.").classes("vmi-subtitle")
    ui.label(f"Use the O⁺ coefficients with their calibrated electrode geometry and voltages. "
             f"Arrivals with |P| above {POLYNOMIAL_LIMIT_AU:g} a.u. are omitted. Blank t₀ uses the scaled O⁺ reference."
             ).classes("vmi-subtitle").bind_visibility_from(model, "value", lambda value: value == "oxygen")
    with field_grid(3):
        center_x = num_input("Detector center X (px)", None if x is None else int(x), step="any")
        center_y = num_input("Detector center Y (px)", None if y is None else int(y), step="any")
        pixel_scale = num_input("Detector scale (mm/px)", DEFAULT_MM_PER_PIXEL, step="any")
    with ui.row().classes("items-center gap-2"):
        load_button = ui.button("Load species JSON", on_click=load, icon="folder_open").props("flat")
        current_button = ui.button("Use current assignments", on_click=use_current).props("flat")
    status = ui.label("Add fitted peaks or load a species file.").classes("vmi-caption")
    calibrations = ui.column().classes("w-full")
    plot_button = ui.button("Plot momenta", on_click=plot, icon="scatter_plot").disable()
    if ask_agent:
        ui.button("Ask Agent about reconstruction", icon="smart_toy", on_click=open_agent).props("flat")
    momentum_review = ui.column().classes("w-full")
    projections = {method: ui.column().classes("w-full").bind_visibility_from(
        model, "value", lambda value, method=method: value == method) for method in MOMENTUM_MODELS}
    return append_assignment


def _read_species_file(path):
    if path.suffix.lower() != ".json":
        raise ValueError("Choose a species file ending in .json.")
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    if (not isinstance(document, dict) or document.get("schema") != SPECIES_SCHEMA
            or not isinstance(document.get("species"), list)):
        raise ValueError("Choose a valid ion-species JSON file.")
    return document


def _delete_species_file(field):
    selection = field.value
    if not selection.strip():
        ui.notify("Choose a Species JSON file first.", type="warning")
        return
    try:
        path = shared.checked_path(selection.strip(), write=True, cwd=shared.workspace_root())
        _read_species_file(path)
    except FileNotFoundError:
        field.set_value("")
        ui.notify("The selected species file is already absent.", type="info")
        return
    except (OSError, ValueError, RecursionError) as exc:
        ui.notify(str(exc), type="warning")
        return

    def clear_path():
        if field.value == selection and not path.exists():
            field.set_value("")

    confirm_clear_outputs([path], "Delete the selected species JSON file?", on_cleared=clear_path)


def _append_species_file(path, peak):
    try:
        document = _read_species_file(path)
    except FileNotFoundError:
        document = dict(schema=SPECIES_SCHEMA, species=[])
    assignments = document["species"]
    if peak in assignments:
        return
    if len(assignments) >= MAX_SPECIES:
        raise ValueError(f"Choose a file with fewer than {MAX_SPECIES} species assignments.")
    assignments.append(peak)
    data = json.dumps(document, indent=2, allow_nan=False) + "\n"
    temporary = NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                   suffix=".tmp", delete=False)
    temp_path = Path(temporary.name)
    try:
        with temporary:
            temporary.write(data)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def _results(job, jobs, species_file, ask_agent=None):
    if job.status != "done":
        return
    with np.load(shared.checked_path(job.extra["histogram"]), allow_pickle=False) as data:
        result = {key: data[key] for key in data.files if not key.endswith("_shot_counts")}
    saved = []
    fitted_rows = []

    def register_peak(row, read):
        fitted_rows[:] = [(element, reader) for element, reader in fitted_rows if not element.is_deleted]
        fitted_rows.append((row, read))

    def read_current():
        return [peak for row, read in fitted_rows
                if not row.is_deleted and (peak := read(allow_blank=True)) is not None]

    def write_peak(peak):
        _species_entries(dict(schema=SPECIES_SCHEMA, species=[peak]), tuple(SOURCES), job.extra["reference"])
        destination = species_file.value.strip()
        if not destination:
            raise ValueError("Choose a Species JSON file in the Ion momenta form.")
        path = shared.checked_path(destination, write=True, cwd=shared.workspace_root())
        with jobs.write_access([path]):
            _append_species_file(path, peak)
        ui.notify(f"Assignment saved to {path.name}.", type="positive")
        if peak not in saved:
            if len(saved) < MAX_SPECIES:
                saved.append(peak)
                summary.set_text(f"{len(saved)} species assignments ready to download.")
                download.enable()
                if "source_file" in job.extra:
                    append_assignment(peak)
        return {"status": "done", "panel": JOB_NAME, "job_id": job.id,
                "output": str(path), "assignment": peak}

    def save_peak(peak):
        try:
            return write_peak(peak)
        except (ValueError, OSError, RecursionError, ResourceBusyError) as exc:
            ui.notify(str(exc), type="warning")
            return None

    def open_species_agent():
        try:
            candidates = read_current()
            if not candidates:
                raise ValueError("Enter a chemical formula on at least one fitted peak first.")
            _species_entries(dict(schema=SPECIES_SCHEMA, species=candidates), tuple(SOURCES), job.extra["reference"])
        except ValueError as exc:
            ui.notify(str(exc), type="warning")
            return
        initial = {"output": species_file.value.strip() or str(shared.workspace_root() / "ion-species.json"),
                   "assignment": candidates[0]}

        def bind(parameters):
            try:
                current = read_current()
            except ValueError as exc:
                raise ValueError(f"A fitted species assignment is incomplete. {exc}") from exc
            if current != candidates:
                raise ValueError("The fitted species assignments changed. Ask the Agent again.")
            if not isinstance(parameters, dict) or parameters.keys() - {"output", "assignment"}:
                raise ValueError("Use only the output and one fitted assignment.")
            draft = initial | parameters
            requested = draft["assignment"]
            if not isinstance(requested, dict):
                raise ValueError("Choose one of the current fitted assignments.")
            identity = ("source", "center_ns", "fit_range_ns")
            matches = [item for item in candidates
                       if all(requested.get(key) == item.get(key) for key in identity)]
            if len(matches) != 1:
                raise ValueError("Choose a peak from the originating fit without changing its identity.")
            record = dict(matches[0])
            for key in ("formula", "charge", "t0_ns", "window_ns"):
                if key in requested:
                    record[key] = requested[key]
            formula, charge, _source, window, t0 = _species_entries(
                dict(schema=SPECIES_SCHEMA, species=[record]), tuple(SOURCES), job.extra["reference"])[0]
            record.update(formula=formula, charge=charge, window_ns=list(window), t0_ns=t0)
            output = resolve_path(draft["output"], write=True)
            if output.suffix.lower() != ".json":
                raise ValueError("Choose a species file ending in .json.")
            return {"output": str(output), "assignment": record}

        def review(_job, draft, approve, decline):
            with species_review, theme.panel_card():
                theme.form_section("Agent species assignment")
                selected = next(index for index, item in enumerate(candidates)
                                if item["source"] == draft["assignment"]["source"]
                                and item["center_ns"] == draft["assignment"]["center_ns"])
                choice = ui.select({index: f"{item['source']} · μ {item['center_ns']:.6g} ns"
                                    for index, item in enumerate(candidates)}, value=selected,
                                   label="Fitted peak").props("outlined dense")
                output = path_input("Species JSON file", draft["output"], browse="save", filters=[".json"],
                                    default_name="ion-species.json")
                with field_grid(3):
                    formula = ui.input("Chemical formula", value=draft["assignment"].get("formula", "")) \
                        .props("outlined dense")
                    charge = num_input("Charge (e)", draft["assignment"].get("charge", 1), step=1)
                    t0 = num_input("Zero-momentum TOF t₀ (ns)", draft["assignment"]["t0_ns"], step="any")
                    minimum = num_input("Species TOF min (ns)", draft["assignment"]["window_ns"][0], step="any")
                    maximum = num_input("Species TOF max (ns)", draft["assignment"]["window_ns"][1], step="any")

                def reload_assignment():
                    item = candidates[choice.value]
                    formula.value = item.get("formula", "")
                    charge.value = item.get("charge", 1)
                    t0.value = item["t0_ns"]
                    minimum.value, maximum.value = item["window_ns"]

                choice.on_value_change(lambda _: reload_assignment())

                async def confirm():
                    record = dict(candidates[choice.value])
                    record.update(formula=formula.value, charge=charge.value, t0_ns=t0.value,
                                  window_ns=[minimum.value, maximum.value])
                    edits = {"output": output.value, "assignment": record}
                    try:
                        bind(draft | edits)
                    except (OSError, TypeError, ValueError) as exc:
                        ui.notify(str(exc), type="warning")
                        return
                    await approve(edits)

                with ui.row().classes("items-center gap-2"):
                    ui.button("Approve and append assignment", on_click=confirm)
                    ui.button("Decline", on_click=decline).props("flat")

        async def run(draft):
            species_file.value = draft["output"]
            try:
                return write_peak(draft["assignment"])
            except (OSError, RecursionError, ResourceBusyError) as exc:
                raise ValueError(str(exc)) from exc

        ask_agent(PanelView(panel=JOB_NAME, operation=SPECIES_OPERATION, label="Ion species assignment",
                            instruction="Append one reviewed assignment from the current fitted peaks to a workspace species JSON file.",
                            settings=initial, bind=bind, render_review=review, review_container=species_review,
                            run_interactive=run,
                            context={"job_id": job.id, "input": job.extra["input_name"], "fitted_peaks": candidates}))

    def download_species():
        document = dict(schema=SPECIES_SCHEMA, species=saved)
        ui.download.content(json.dumps(document, indent=2, allow_nan=False) + "\n",
                            "ion-species.json", "application/json")

    with ui.column().classes("w-full"):
        ui.label(f'{job.extra["input_name"]} · {REFERENCE_LABELS[job.extra["reference"]]} '
                 f'· {int(result["triggers"]):,} triggers').classes("vmi-caption break-all")
        for source in ("pixels", "clusters"):
            if source not in result:
                continue
            ui.label(SOURCES[source]).classes("font-mono text-lg text-primary")
            image = result[f"{source}_image"]
            ui.label(f'{int(result[source].sum()):,} arrivals in the TOF interval · '
                     f'{int(image.sum()):,} mapped positions · '
                     f'{int(result[f"{source}_unmatched"]):,} with no preceding trigger · '
                     f'{int(result[f"{source}_outside"]):,} outside the TOF interval · '
                     f'{int(result[f"{source}_invalid_positions"]):,} invalid positions').classes("vmi-caption")
            if "source_file" in job.extra:
                _linked_plots(job, jobs, source, result, save_peak, register_peak, species_file, ask_agent=ask_agent)
            else:
                ui.label("Plot positions and TOF again to enable gating.").classes("vmi-subtitle")
                with ui.element("div").classes("vmi-ion-detector-row"):
                    with ui.column().classes("w-full min-w-0"):
                        _image_plot(_position_figure(image), image)
                    ui.plotly(plot_options(_tof_figure(result["edges"], {source: result[source]},
                                                       "Time-of-flight spectrum"))).classes("w-full min-w-0")
        with ui.row().classes("items-center gap-2"):
            summary = ui.label("Add fitted peaks to download a species list.").classes("vmi-caption")
            download = ui.button("Download species JSON", on_click=download_species, icon="download")
            download.disable()
            if ask_agent:
                ui.button("Ask Agent about species", icon="smart_toy", on_click=open_species_agent).props("flat")
        species_review = ui.column().classes("w-full")
        if "source_file" in job.extra:
            append_assignment = _momentum_controls(job, jobs, result, species_file, read_current, ask_agent=ask_agent)


def positions_job(path, *, source="pixels", reference=DEFAULT_REFERENCE, width=DEFAULT_BIN_SIZE_NS,
                  low=0, high=None):
    if not path:
        raise ValueError("Choose a LUNA HDF5 file or acquisition collection.")
    source_file = shared.checked_path(path)
    if not source_file.is_file():
        raise ValueError("The selected input file does not exist.")
    for member in collection_reads(source_file):
        shared.checked_path(member)
    if source not in SOURCES or reference not in REFERENCE_LABELS:
        raise ValueError("Choose a valid count source and trigger reference.")
    if width is None or number(width) <= 0:
        raise ValueError("TOF bin size must be finite and greater than zero.")
    if low is None or number(low) < 0:
        raise ValueError("TOF minimum must be finite and at least zero.")
    if high is not None and number(high) <= low:
        raise ValueError("TOF maximum must be greater than the minimum.")

    job = Job(JOB_NAME, cwd=str(shared.workspace_root()))
    result = shared.generated_dir() / f"ion-momenta-{job.id}.npz"
    job.cmd = [sys.executable, "-m", "vmi_analysis.ion_momenta",
               "--input", str(source_file), "--output", str(result), "--source", source,
               "--reference", reference, "--bin-size", str(width), "--minimum", str(low)]
    if high is not None:
        job.cmd += ["--maximum", str(high)]
    job.extra.update(
        histogram=str(result), reference=reference, input_name=source_file.name,
        source_file=str(source_file), bin_size_ns=width,
        panel_form={"input": str(source_file), "source": source, "reference": reference,
                    "bin_size_ns": width, "minimum": low, "maximum": high},
    )
    return job


def _prepare_agent(jobs, params):
    dump_id, path = params.get("dump_job_id"), params.get("input")
    if bool(dump_id) == bool(path):
        raise ValueError("Supply one TPX3 Dump job ID or one input file.")
    if dump_id:
        dump = require_completed_job(jobs, dump_id, name=ion_luna.JOB_NAME)
        path, _ = ion_luna.completed_dump_input(dump)
    job = positions_job(resolve_path(path), source=params.get("source", "pixels"),
                        reference=params.get("reference", DEFAULT_REFERENCE),
                        width=params.get("bin_size_ns", DEFAULT_BIN_SIZE_NS),
                        low=params.get("minimum_ns", 0), high=params.get("maximum_ns"))
    form = job.extra["panel_form"]
    return job, {"LUNA input": form["input"], "Count": SOURCES[form["source"]],
                 "Trigger": REFERENCE_LABELS[form["reference"]], "Bin size (ns)": form["bin_size_ns"],
                 "TOF minimum (ns)": form["minimum"],
                 "TOF maximum (ns)": form["maximum"] if form["maximum"] is not None else "Automatic"}


def agent_operations():
    histogram = PanelOperation("ion_momenta", {
        "panel": JOB_NAME,
        "summary": "Build linked detector-position and TOF histograms from a LUNA dump. "
                   "The Ion momenta panel opens the results for interactive gating, species assignment and reconstruction.",
        "parameters": {
            "input": {"type": "path", "description": "LUNA HDF5 file or acquisition collection"},
            "dump_job_id": {"type": "job ID", "description": "Completed TPX3 Dump job ID in place of input"},
            "source": {"type": "string", "default": "pixels", "choices": list(SOURCES)},
            "reference": {"type": "string", "default": DEFAULT_REFERENCE, "choices": list(REFERENCE_LABELS)},
            "bin_size_ns": {"type": "number", "unit": "ns", "default": DEFAULT_BIN_SIZE_NS},
            "minimum_ns": {"type": "number", "unit": "ns", "default": 0},
            "maximum_ns": {"type": "number or null", "unit": "ns", "default": None},
        },
    }, _prepare_agent, (JOB_NAME,), outputs=("histogram",))
    gate = PanelOperation(GATE_OPERATION, {
        "panel": JOB_NAME, "manual": JOB_NAME,
        "summary": "Apply or clear an X/Y position gate or TOF gate on the originating linked result. Open its Ask Agent gate control.",
        "parameters": {
            "kind": {"type": "string", "choices": ["position", "tof"]},
            "bounds": {"type": "list of numbers or null",
                       "description": "[xmin, xmax, ymin, ymax] in pixels or [tof_min, tof_max] in ns; null clears"},
        },
    }, None, ())
    fit = PanelOperation(FIT_OPERATION, {
        "panel": JOB_NAME, "manual": JOB_NAME,
        "summary": "Fit Gaussian peaks in the current position-filtered TOF spectrum. Open Ask Agent about fit after applying an X/Y gate.",
        "parameters": {"region": {"type": "pair of numbers", "unit": "ns"}},
    }, None, ())
    species = PanelOperation(SPECIES_OPERATION, {
        "panel": JOB_NAME, "manual": JOB_NAME,
        "summary": "Append one reviewed assignment from the current fitted peaks to a workspace species JSON file.",
        "parameters": {
            "output": {"type": "path", "description": "Workspace .json species file"},
            "assignment": {"type": "object", "description": "A current fitted peak with formula, charge, t0_ns, and window_ns edits"},
        },
    }, None, ())
    momentum = PanelOperation(MOMENTUM_OPERATION, {
        "panel": JOB_NAME, "manual": JOB_NAME,
        "summary": "Reconstruct and display momentum projections from the current species assignments and detector calibration.",
        "parameters": {
            "model": {"type": "string", "choices": list(MOMENTUM_MODELS)},
            "center_px": {"type": "pair of numbers", "description": "Detector center [X, Y] in pixels"},
            "mm_per_pixel": {"type": "number", "unit": "mm/pixel"},
            "assignments": {"type": "list of objects", "description": "Each uses formula, charge, source, window_ns, "
                            "mass_u, t0_ns, and pz_scale. Blank t0_ns uses the O+ reference scaling in oxygen mode."},
        },
    }, None, ())
    return histogram, gate, fit, species, momentum


def build(jobs, *, ask_agent=None):
    def restore_form(job):
        values = adopt_agent_form(job, "ion_momenta")
        for key, widget in fields.items():
            if key in values:
                widget.value = values[key]

    view = JobView(jobs, job_name=JOB_NAME,
                   results_builder=lambda job: _results(job, jobs, species_file, ask_agent=ask_agent),
                   attach_builder=restore_form)
    theme.panel_header(JOB_NAME, "Inspect detector positions and time of flight for ion momentum analysis.", view=view)

    def save(widget, key):
        return persist(widget, f"ion_momenta.{key}")

    def use_latest():
        candidates = [job for job in jobs.jobs() if job.name == ion_luna.JOB_NAME and job.status == "done"]
        if not candidates:
            ui.notify("Complete a TPX3 dump first or choose a saved collection.", type="warning")
            return
        latest = max(candidates, key=lambda job: job.finished or job.created)
        input_file.value, _ = ion_luna.completed_dump_input(latest)

    def run():
        try:
            job = positions_job(input_file.value.strip(), source=source.value, reference=reference.value,
                                width=bin_size.value, low=minimum.value, high=maximum.value)
            view.attach(jobs.submit(job))
        except (OSError, ValueError) as exc:
            ui.notify(str(exc), type="warning")

    def clear():
        outputs = [job.extra["histogram"] for job in jobs.jobs()
                   if job.name == JOB_NAME and "histogram" in job.extra]
        selection = species_file.value
        optional = None
        if selection.strip():
            try:
                path = shared.checked_path(selection.strip(), write=True, cwd=shared.workspace_root())
                _read_species_file(path)
                optional = (path, "Also delete the selected species JSON")
            except (OSError, ValueError, RecursionError):
                pass

        def clear_species_path():
            if species_file.value == selection:
                species_file.set_value("")

        confirm_clear_outputs(outputs, "Delete the generated Ion momenta position and TOF results?",
                              on_cleared=view.clear_results, optional=optional,
                              on_optional_cleared=clear_species_path)

    with theme.panel_card():
        input_file = save(path_input("LUNA HDF5 file or acquisition collection", browse="file",
                                    filters=[".h5", ".hdf5", ".json"]), "input")
        ui.button("Use latest dump from TPX3 Dump", icon="history", on_click=use_latest).props("flat")
        ui.label("Choose either an uncorrected dump or a timewalk-corrected file.").classes("vmi-subtitle")
        with field_grid():
            source = save(ui.select(SOURCES, value="pixels", label="Count").props("outlined dense"), "source")
            reference = save(ui.select(REFERENCE_LABELS, value=DEFAULT_REFERENCE, label="Trigger reference")
                             .props("outlined dense"), "reference")
        with field_grid(3):
            bin_size = save(num_input("TOF bin size (ns)", DEFAULT_BIN_SIZE_NS, step="any"), "bin_size_ns")
            minimum = save(num_input("TOF minimum (ns)", 0, min=0, step="any"), "minimum")
            maximum = save(num_input("TOF maximum (ns, blank = auto)", None, step="any"), "maximum")
        fields = {"input": input_file, "source": source, "reference": reference,
                  "bin_size_ns": bin_size, "minimum": minimum, "maximum": maximum}
        species_file = save(ui.input(), "species_file").set_visibility(False)
        ui.label("Momentum reconstruction requires the ion mass and charge, detector center, "
                 "and spectrometer calibration or flight-field model.").classes("vmi-subtitle")
        with action_bar(view, "Plot positions and TOF", on_click=run):
            ui.button("Clear outputs", icon="delete", on_click=clear).props("flat color=negative") \
                .classes("vmi-action-danger")
    view.place_console()
