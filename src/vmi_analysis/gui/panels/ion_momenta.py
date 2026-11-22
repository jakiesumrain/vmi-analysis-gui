"""Detector-position and TOF workbench for ion momentum reconstruction."""

import asyncio
from importlib.resources import files
import json
from math import isfinite, log, sqrt
from pathlib import Path
import re
import sys
from tempfile import NamedTemporaryFile

from nicegui import ui
import numpy as np
import plotly.graph_objects as go

from vmi_analysis.ion_acquisition import collection_reads
from vmi_analysis.ion_momenta import (DEFAULT_MM_PER_PIXEL, LONGITUDINAL_CONVERSION,
                                     TRANSVERSE_CONVERSION, SpeciesMomentum, gated_preview,
                                     momentum_projections)
from vmi_analysis.ion_tof import DEFAULT_REFERENCE
from vmi_analysis.pes import evaluate_fit_for_plot, fit_peaks
from vmi_analysis.tof_bins import DEFAULT_BIN_SIZE_NS

from .. import shared, theme
from ..centroid_plots import _style, plot_options
from ..jobs import Job, ResourceBusyError
from .common import (JobView, action_bar, confirm_clear_outputs, field_grid, num_input,
                     path_input, persist)
from .ion_tof import REFERENCE_LABELS, SOURCES, _drag_interval, _mark_interval, _tof_figure


JOB_NAME = "Ion momenta"
PLOT_HEIGHT = 580
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


def _interval(values):
    if None in values or not np.isfinite(values).all() or values[0] >= values[1]:
        raise ValueError("Enter increasing, finite gate limits.")
    return tuple(values)


def _gate_fields(labels):
    with field_grid(2):
        return [num_input(label, None, step="any") for label in labels]


def _fit_controls(plot, source, edges, counts, context, save_peak, register_peak):
    selected_region = model = None
    with ui.row().classes("items-center gap-2"):
        mode = ui.toggle(["Zoom", "Select fit range"], value="Zoom")
        ui.button("Clear fit", on_click=lambda: clear()).props("flat")
    fields = _gate_fields(("Fit TOF min (ns)", "Fit TOF max (ns)"))
    output = ui.column().classes("w-full")

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
        nonlocal selected_region, model
        try:
            region = _interval([field.value for field in fields])
            fit_button.disable()
            output.clear()
            centers = (edges[:-1] + edges[1:]) / 2
            fitted_model = await asyncio.to_thread(fit_peaks, centers, counts, region)
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            ui.notify(str(exc), type="warning")
            return
        finally:
            fit_button.enable()
        if output.is_deleted or region != tuple(field.value for field in fields):
            return
        selected_region, model = region, fitted_model
        redraw()
        error = float(np.sqrt(np.mean(model[4] ** 2)))
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
                register_peak(*_species_fields(record, save_peak))

    mode.on_value_change(lambda _: redraw())
    plot.on("plotly_selected", selected, js_handler="event => emit(event?.range ?? null)")
    fit_button = ui.button("Fit Gaussian peaks", on_click=fit)
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
                return dict(record, formula=name, charge=int(number), t0_ns=t0, window_ns=list(window))

            def add():
                try:
                    peak = read_peak()
                except ValueError as exc:
                    ui.notify(str(exc), type="warning")
                    return
                save_peak(peak)

            ui.button("Append to species file", on_click=add).props("flat")
    return row, read_peak


def _linked_plots(job, jobs, source, result, save_peak, register_peak):
    image, edges = result[f"{source}_image"], result["edges"]
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
            return
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
            return
        if revisions[kind] != revision or target.is_deleted:
            return
        gates[kind] = bounds
        redraw[kind]()
        target.clear()
        with target:
            if kind == "tof":
                ui.label(f"{bounds[0]:.6g}–{bounds[1]:.6g} ns · {filtered.sum():,} mapped positions").classes("vmi-caption")
                ui.plotly(plot_options(_position_figure(filtered, "TOF-filtered detector positions"))).classes("w-full")
            else:
                ui.label(f"{filtered.sum():,} arrivals inside the X/Y gate").classes("vmi-caption")
                chart = ui.plotly(go.Figure()).classes("w-full")
                context = dict(source=source, input_file=job.extra["input_name"], trigger_reference=reference,
                               position_gate_px=dict(zip(("xmin", "xmax", "ymin", "ymax"), bounds, strict=True)),
                               histogram_range_ns=[float(edges[0]), float(edges[-1])],
                               bin_size_ns=job.extra["bin_size_ns"])
                _fit_controls(chart, source, edges, filtered, context, save_peak, register_peak)

    async def enter(kind, fields):
        try:
            values = [field.value for field in fields]
            bounds = _interval(values[:2])
            if kind == "position":
                bounds += _interval(values[2:])
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
        image_plot.update_figure(plot_options(fig))

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
            with ui.expansion("X/Y gate limits", icon="filter_alt").classes("w-full"):
                xy = _gate_fields(("X min (px)", "X max (px)", "Y min (px)", "Y max (px)"))
                ui.button("Apply X/Y gate", on_click=lambda: enter("position", xy)).props("flat")
                ui.button("Clear X/Y gate", on_click=lambda: clear("position", xy)).props("flat")
        with ui.column().classes("min-w-0 w-full"):
            with ui.row().classes("items-center gap-2"):
                tof_mode = ui.toggle(["Zoom", "Select TOF range"], value="Zoom")
            tof_plot = ui.plotly(go.Figure()).classes("w-full min-w-0")
            with ui.expansion("TOF gate limits", icon="filter_alt").classes("w-full"):
                times = _gate_fields(("Gate TOF min (ns)", "Gate TOF max (ns)"))
                ui.button("Apply TOF gate", on_click=lambda: enter("tof", times)).props("flat")
                ui.button("Clear TOF gate", on_click=lambda: clear("tof", times)).props("flat")
    with ui.element("div").classes("vmi-ion-detector-row"):
        filtered_image = ui.column().classes("w-full min-w-0")
        filtered_tof = ui.column().classes("w-full min-w-0")
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
                or any(type(value) not in (int, float) or not -1e100 < value < 1e100 for value in (*window, t0))
                or window[0] >= window[1]):
            raise ValueError("Each assignment needs finite TOF limits and a zero-momentum TOF.")
        entries.append((item["formula"], item["charge"], item["source"], tuple(window), t0))
    return entries


def _momentum_controls(job, jobs, result, species_file, read_current):
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
                pz_scale = num_input("p_z scale factor", 1, step="any")
        rows.append((formula, charge, source, window, mass, reference, pz_scale))
        plot_button.set_enabled(not plotting)

    def show_assignments(entries, source, message):
        nonlocal assignment_source
        assignment_source = source
        calibrations.clear()
        plots.clear()
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

    async def plot():
        nonlocal plotting
        if plotting:
            return
        plotting = True
        try:
            calibration = [center_x.value, center_y.value, pixel_scale.value]
            if not all(value is not None and isfinite(value) for value in calibration) or calibration[-1] <= 0:
                raise ValueError("Enter a finite detector center and a positive pixel scale.")
            species, labels = [], []
            for formula, charge, source, window, mass, reference, scale in rows:
                values = [mass.value, reference.value, scale.value]
                if not all(value is not None and isfinite(value) for value in values) or min(values[0], values[2]) <= 0:
                    raise ValueError(f"Enter a positive mass and p_z scale and a finite t₀ for {formula} {charge:+d}.")
                species.append(SpeciesMomentum(source, window, values[0], charge, values[1], values[2]))
                labels.append((formula, charge, source))
            plot_button.disable()
            load_button.disable()
            current_button.disable()
            plots.clear()
            with plots:
                ui.label("Calculating momentum projections…").classes("vmi-caption")
            path = shared.checked_path(job.extra["source_file"])
            with jobs.preview_read_access(collection_reads(path)) as stopping:
                edges, images, totals = await asyncio.to_thread(
                    momentum_projections, path, job.extra["reference"], species, tuple(calibration[:2]),
                    calibration[2], result["trigger_times"], result["segment_offsets"],
                    stopped=lambda: stopping() or plots.is_deleted,
                )
        except (ValueError, OSError, KeyError, ResourceBusyError) as exc:
            if not plots.is_deleted:
                plots.clear()
                ui.notify(str(exc), type="warning")
            return
        finally:
            plotting = False
            if not plots.is_deleted:
                plot_button.set_enabled(bool(rows))
                load_button.enable()
                current_button.enable()
        if plots.is_deleted:
            return
        plots.clear()
        with plots:
            for label, bins, maps, count, calibration in zip(labels, edges, images, totals, species, strict=True):
                formula, charge, source = label
                ui.label(f"{formula} · charge {charge:+d} · {SOURCES[source]} · {count:,} arrivals · "
                         f"t₀ {calibration.t0_ns:.6g} ns · p_z scale {calibration.pz_scale:.6g}"
                         ).classes("vmi-caption")
                if count:
                    with ui.element("div").classes("grid grid-cols-1 md:grid-cols-2 gap-4 w-full"):
                        for image, horizontal, vertical in zip(maps, ("pₓ", "pᵧ"), ("pᵧ", "p_z"), strict=True):
                            ui.plotly(plot_options(_momentum_figure(bins, image, horizontal, vertical))
                                      ).classes("w-full min-w-0")
                else:
                    ui.label("No arrivals with positive TOF inside this species window.").classes("vmi-subtitle")

    theme.form_section("Momentum projections")
    ui.label("Uses the species TOF window across the whole detector. TOF is used directly. "
             "The initial center comes from the brightest detector bin.").classes("vmi-subtitle")
    with ui.expansion("Momentum formulas and units", icon="functions", value=True).classes("w-full"):
        with ui.column().classes("w-full gap-1"):
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
    plots = ui.column().classes("w-full")
    return append_assignment


def _append_species_file(path, peak):
    if path.suffix.lower() != ".json":
        raise ValueError("Choose a species file ending in .json.")
    document = dict(schema=SPECIES_SCHEMA, species=[])
    if path.exists():
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        if (not isinstance(document, dict) or document.get("schema") != SPECIES_SCHEMA
                or not isinstance(document.get("species"), list)):
            raise ValueError("Choose an ion-species JSON file or a new filename.")
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


def _results(job, jobs, species_file):
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

    def save_peak(peak):
        try:
            destination = species_file.value.strip()
            if not destination:
                raise ValueError("Choose a Species JSON file in the Ion momenta form.")
            path = shared.checked_path(destination, write=True, cwd=shared.workspace_root())
            with jobs.write_access([path]):
                _append_species_file(path, peak)
        except (ValueError, OSError, RecursionError, ResourceBusyError) as exc:
            ui.notify(str(exc), type="warning")
            return
        ui.notify(f"Assignment saved to {path.name}.", type="positive")
        if peak not in saved:
            if len(saved) >= MAX_SPECIES:
                return
            saved.append(peak)
            summary.set_text(f"{len(saved)} species assignments ready to download.")
            download.enable()
            if "source_file" in job.extra:
                append_assignment(peak)

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
                _linked_plots(job, jobs, source, result, save_peak, register_peak)
            else:
                ui.label("Plot positions and TOF again to enable gating.").classes("vmi-subtitle")
                with ui.element("div").classes("vmi-ion-detector-row"):
                    ui.plotly(plot_options(_position_figure(image))).classes("w-full min-w-0")
                    ui.plotly(plot_options(_tof_figure(result["edges"], {source: result[source]},
                                                       "Time-of-flight spectrum"))).classes("w-full min-w-0")
        with ui.row().classes("items-center gap-2"):
            summary = ui.label("Add fitted peaks to download a species list.").classes("vmi-caption")
            download = ui.button("Download species JSON", on_click=download_species, icon="download")
            download.disable()
        if "source_file" in job.extra:
            append_assignment = _momentum_controls(job, jobs, result, species_file, read_current)


def build(jobs):
    view = JobView(jobs, job_name=JOB_NAME, results_builder=lambda job: _results(job, jobs, species_file))
    theme.panel_header(JOB_NAME, "Inspect detector positions and time of flight for ion momentum analysis.", view=view)

    def save(widget, key):
        return persist(widget, f"ion_momenta.{key}")

    def use_latest():
        candidates = [job for job in jobs.jobs() if job.name == "TPX3 Dump" and job.status == "done"]
        if not candidates:
            ui.notify("Complete a TPX3 dump first or choose a saved collection.", type="warning")
            return
        latest = max(candidates, key=lambda job: job.finished or job.created)
        input_file.value = latest.extra.get("collection") or latest.extra["output"]

    def run():
        try:
            if not input_file.value.strip():
                raise ValueError("Choose a LUNA HDF5 file or acquisition collection.")
            source_file = shared.checked_path(input_file.value.strip())
            if not source_file.is_file():
                raise ValueError("The selected input file does not exist.")
            for path in collection_reads(source_file):
                shared.checked_path(path)
            width, low, high = bin_size.value, minimum.value, maximum.value
            if width is None or not isfinite(width) or width <= 0:
                raise ValueError("TOF bin size must be finite and greater than zero.")
            if low is None or not isfinite(low) or low < 0:
                raise ValueError("TOF minimum must be finite and at least zero.")
            if high is not None and (not isfinite(high) or high <= low):
                raise ValueError("TOF maximum must be greater than the minimum.")
            job = Job(JOB_NAME, cwd=str(shared.workspace_root()))
            result = shared.generated_dir() / f"ion-momenta-{job.id}.npz"
            job.cmd = [sys.executable, "-m", "vmi_analysis.ion_momenta",
                       "--input", str(source_file), "--output", str(result),
                       "--source", source.value, "--reference", reference.value,
                       "--bin-size", str(width), "--minimum", str(low)]
            if high is not None:
                job.cmd += ["--maximum", str(high)]
            job.extra.update(histogram=str(result), reference=reference.value, input_name=source_file.name,
                             source_file=str(source_file), bin_size_ns=width)
            view.attach(jobs.submit(job))
        except (OSError, ValueError) as exc:
            ui.notify(str(exc), type="warning")

    def clear():
        outputs = [job.extra["histogram"] for job in jobs.jobs()
                   if job.name == JOB_NAME and "histogram" in job.extra]
        confirm_clear_outputs(outputs, "Delete the generated Ion momenta position and TOF results?",
                              on_cleared=view.clear_results)

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
        species_file = save(path_input("Species JSON file (append assignments)", browse="save",
                                       filters=[".json"], default_name="ion-species.json"), "species_file")
        ui.label("Momentum reconstruction requires the ion mass and charge, detector center, "
                 "and spectrometer calibration or flight-field model.").classes("vmi-subtitle")
        with action_bar(view, "Plot positions and TOF", on_click=run):
            ui.button("Clear outputs", icon="delete", on_click=clear).props("flat color=negative")
    view.place_console()
