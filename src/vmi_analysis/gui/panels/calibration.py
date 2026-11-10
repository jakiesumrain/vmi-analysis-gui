"""Panel: interactive clean-room trig-fit calibration.

The panel remaps a VMI image to polar coordinates and prepares the fit
image with optional inverse Abel. The user selects a ring band and masks
with the mouse. The panel fits the ridge and appends a block to the
extracted trig-fit file.

The numeric work lives in ``vmi_analysis.circularization.polar_fit`` (a
library module with unit tests); this module only renders and handles input.
"""

from __future__ import annotations

import asyncio
from base64 import b64encode
from io import BytesIO
from pathlib import Path
import time

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LogNorm, Normalize
from nicegui import ui
from PIL import Image
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

from vmi_analysis.circularization import polar_fit as pf
from vmi_analysis.imaging.centering import brightest_pixel

from .. import theme
from ..centroid_plots import COLORMAPS, _style, plot_options
from ..jobs import MANAGER
from ..settings import centroid_settings, output_dir
from .common import REPO_ROOT, delete_outputs, num_input, path_input, persist

CIRC_DIR = REPO_ROOT / "circularization"
POLAR_HEIGHT = "calc(clamp(220px, 38vh, 360px) * 2 / 3)"

FILTER_FIELDS = [
    ("smooth_theta", "Smooth θ", 1.0, 0.1),
    ("smooth_r", "Smooth r", 0.8, 0.1),
    ("low_r", "Bandpass low r", 2.0, 0.5),
    ("high_r", "Bandpass high r", 20.0, 1.0),
    ("background_theta", "Ridge bg θ", 8.0, 0.5),
    ("background_r", "Ridge bg r", 25.0, 1.0),
    ("ridge_sigma", "Ridge σ", 1.2, 0.1),
    ("blur_theta", "Unsharp blur θ", 6.0, 0.5),
    ("blur_r", "Unsharp blur r", 18.0, 1.0),
    ("clip_limit", "CLAHE clip", 0.02, 0.005),
    ("kernel_theta", "CLAHE kernel θ", 24, 4),
    ("kernel_r", "CLAHE kernel r", 96, 8),
]


def _render_polar_image(polar, cmap="viridis", *, log_scale=False,
                        vmin_pct=None, vmax_pct=None):
    """Render the polar image at exactly one pixel per bin (x=r, y=theta)."""
    polar = np.asarray(polar, dtype=float)
    lower = 1.0 if vmin_pct is None else float(vmin_pct)
    upper = 99.5 if vmax_pct is None else float(vmax_pct)
    if not 0 <= lower < upper <= 100:
        raise ValueError("Enter percentiles from 0 to 100 with vmin below vmax.")
    valid = np.isfinite(polar)
    if log_scale:
        valid &= polar > 0
        if not valid.any():
            raise ValueError("Log scale needs positive intensities.")
    values = polar[valid]
    scaled = np.zeros(polar.shape)
    if values.size:
        low, high = np.percentile(values, [lower, upper])
        if high > low:
            norm = LogNorm if log_scale else Normalize
            scaled[valid] = norm(vmin=low, vmax=high, clip=True)(values)
    rgb = (colormaps[cmap.lower()](scaled)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def _px_to_data(state, x, y):
    width, height = state["img_size"]
    view = state["view"]
    radial_fraction = min(1.0, max(0.0, float(x) / max(width, 1)))
    angular_fraction = min(1.0, max(0.0, float(y) / max(height, 1)))
    r = view["r0"] + radial_fraction * (view["r1"] - view["r0"])
    theta = view["theta0"] + angular_fraction * (view["theta1"] - view["theta0"])
    return r, theta


def _polar_sample(state, key, x, y):
    indices = state["view_indices"]
    if indices is None or state[key] is None:
        return None
    ti, ri = indices
    if not (0 <= x < ri.size and 0 <= y < ti.size):
        return None
    row, col = ti[int(y)], ri[int(x)]
    return state["r"][col], np.degrees(state["theta"][row]), state[key][row, col]


def _polar_view_indices(theta, r, view):
    ti = np.flatnonzero((theta >= view["theta0"] - 1e-12)
                        & (theta <= view["theta1"] + 1e-12))
    ri = np.flatnonzero((r >= view["r0"] - 1e-12)
                        & (r <= view["r1"] + 1e-12))
    return ti, ri


def _data_to_px(state, r, theta):
    width, height = state["img_size"]
    view = state["view"]
    x = (r - view["r0"]) / max(view["r1"] - view["r0"], 1e-12) * width
    y = (theta - view["theta0"]) / max(view["theta1"] - view["theta0"], 1e-12) * height
    return x, y


def _clipped_rect(state, rect):
    view = state["view"]
    r_lo, r_hi = sorted((rect.r0, rect.r1))
    t_lo, t_hi = rect.theta0, rect.theta1
    if t_hi - t_lo >= pf.TWO_PI - 1e-9:
        t_lo, t_hi = view["theta0"], view["theta1"]
    r_lo = max(r_lo, view["r0"])
    r_hi = min(r_hi, view["r1"])
    t_lo = max(t_lo, view["theta0"])
    t_hi = min(t_hi, view["theta1"])
    if r_hi <= r_lo or t_hi <= t_lo:
        return None
    x0, y0 = _data_to_px(state, r_lo, t_lo)
    x1, y1 = _data_to_px(state, r_hi, t_hi)
    return x0, y0, x1 - x0, y1 - y0


def _overlay_svg(state):
    width, height = state["img_size"]
    if not width or not height:
        return ""
    parts = []
    if state.get("band") is not None:
        box = _clipped_rect(state, state["band"])
        if box:
            x, y, w, h = box
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                f'fill="rgba(33,150,243,0.14)" stroke="#2196f3" stroke-width="2" '
                f'vector-effect="non-scaling-stroke"/>')
    for rect in state.get("masks", []):
        box = _clipped_rect(state, rect)
        if box:
            x, y, w, h = box
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                f'fill="rgba(244,67,54,0.28)" stroke="#f44336" stroke-width="2" '
                f'vector-effect="non-scaling-stroke"/>')
    drag = state.get("drag_rect")
    if drag is not None:
        box = _clipped_rect(state, drag)
        if box:
            x, y, w, h = box
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                f'fill="rgba(255,235,59,0.15)" stroke="#fbc02d" '
                f'stroke-width="2" stroke-dasharray="6 4" vector-effect="non-scaling-stroke"/>')
    fit = state.get("last_fit")
    if fit is not None:
        theta = np.linspace(state["view"]["theta0"], state["view"]["theta1"],
                            max(2, int(height)))
        rg = fit.ridge_centers(theta)
        segments = []
        current = []
        for t, radius in zip(theta, rg):
            if state["view"]["r0"] <= radius <= state["view"]["r1"]:
                current.append(_data_to_px(state, radius, t))
            elif current:
                segments.append(current)
                current = []
        if current:
            segments.append(current)
        for segment in segments:
            if len(segment) < 2:
                continue
            path = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in segment)
            parts.append(
                f'<path d="{path}" fill="none" stroke="#ffd54f" stroke-width="2" '
                f'vector-effect="non-scaling-stroke"/>')
    return "".join(parts)


def _parameter_rows(fit):
    rows = []
    names = pf.parameter_names(fit.n_terms, use_beta=fit.use_beta)
    for name in names:
        if name not in fit.params:
            continue
        value = fit.params[name]
        sigma = fit.sigmas.get(name, float("nan"))
        rows.append({"parameter": name, "value": f"{value:.6g}",
                     "sigma": f"{sigma:.3g}" if np.isfinite(sigma) else "—"})
    return rows


def _fit_comparison_figure(polar, theta, r, fit):
    angles, radii, data, model = pf.fit_comparison(polar, theta, r, fit)
    finite = np.concatenate([data[np.isfinite(data)], model[np.isfinite(model)]])
    if not finite.size:
        return None
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Processed band used for fit", "Fitted model"),
                        horizontal_spacing=0.1, shared_yaxes="all")
    degrees = np.degrees(angles)
    for col, values in enumerate((data, model), 1):
        fig.add_trace(go.Heatmap(
            x=radii, y=degrees, z=values, coloraxis="coloraxis", hoverongaps=False,
            hovertemplate="Radius %{x:.2f} px<br>Angle %{y:.1f}°<br>Intensity %{z:.4g}<extra></extra>",
        ), row=1, col=col)
    ridge = np.where(np.isfinite(data).any(axis=1), fit.ridge_centers(angles), np.nan)
    fig.add_trace(go.Scatter(x=ridge, y=degrees, mode="lines",
                             line={"color": "#ffd54f", "width": 2},
                             hoverinfo="skip"), row=1, col=1)
    half_r = (r[1] - r[0]) / 2
    half_angle = np.degrees(theta[1] - theta[0]) / 2
    fig.update_xaxes(title_text="Radius (pixels)", range=[radii[0] - half_r, radii[-1] + half_r], row=1)
    fig.update_xaxes(matches="x", row=1, col=2)
    fig.update_yaxes(range=[degrees[-1] + half_angle, degrees[0] - half_angle])
    fig.update_yaxes(title_text="Angle (degrees)", row=1, col=1)
    _style(fig, "Fit comparison")
    fig.update_layout(height=675, plot_bgcolor="#d8d2ca",
                      coloraxis={"colorscale": "Viridis", "cmin": float(finite.min()),
                                 "cmax": float(finite.max()), "colorbar": {"title": "Intensity"}})
    return fig


def _radial_model_figures(reduced):
    dataset = reduced["dataset"]
    radii = dataset["radii"]
    sample_r, curves = pf.radial_model_curves(reduced)
    model_label = "PCHIP" if reduced["model"] == "interp" else f"Polynomial order {reduced['order']}"
    for prefix in ("A", "B"):
        figure = _style(go.Figure(), f"{prefix}ₙ(r) · {model_label}")
        for key, curve in curves.items():
            if not key.startswith(prefix):
                continue
            n = int(key[1:])
            color = qualitative.Plotly[(n - 1) % len(qualitative.Plotly)]
            figure.add_scatter(x=sample_r, y=curve, name=f"{key} model",
                               mode="lines", line={"color": color}, legendgroup=key)
            figure.add_scatter(x=radii, y=dataset[key], name=f"{key} rings",
                               mode="markers", marker={"color": color}, legendgroup=key,
                               error_y={"type": "data", "array": dataset[f"{key}_sigma"]})
        figure.update_layout(height=410, showlegend=True)
        figure.update_xaxes(title_text="Radius (pixels)")
        figure.update_yaxes(title_text=f"{prefix}ₙ (pixels)")
        yield figure


def build(jobs=None, *, use_calibration=None) -> None:
    theme.section_title(
        "Trig Calibration",
        "Extract circularisation trig terms from a polar remap with optional "
        "inverse Abel. Select a ring, mask interference and fit its ridge. Accepted "
        "rings are appended as Gascooke-compatible blocks.")
    ui.label(
        "Use the mouse: drag a rectangle to select a ring band (blue) or to "
        "mask a region (red). Masked pixels are excluded from the fit."
    ).classes("vmi-caption")

    state = {
        "raw": None, "processed": None, "theta": None, "r": None, "dr": None,
        "options": None, "view": None, "view_indices": None, "img_size": (0, 0),
        "tool": "band", "drag_start": None, "drag_now": None,
        "drag_rect": None, "band": None, "masks": [], "last_fit": None,
        "selection_history": [],
        "rings": [], "ready": False,
        "last_overlay": 0.0,
        "image": None,
        "inspection": None,
    }
    widgets = {}

    def _polar_image(key):
        image = ui.interactive_image(
            "", on_mouse=lambda event: _on_mouse(event, key),
            events=["mousedown", "mousemove", "mouseup", "mouseleave"], sanitize=False,
        ).classes("w-full").style(f"height: {POLAR_HEIGHT}; border: 1px solid #e0ddd6")
        image.on("pointerdown", js_handler="""(event) => {
            if (event.button === 0 && event.target instanceof HTMLImageElement) {
                event.target.setPointerCapture(event.pointerId);
            }
        }""")
        return image

    # ---------------------------------------------------------------- inputs
    with theme.panel_card():
        with ui.row().classes("gap-4 flex-wrap items-end w-full"):
            widgets["image"] = persist(path_input(
                "VMI image (.npy)", str(output_dir(centroid_settings()) / "vmi_2d_raw.npy"),
                browse="file", filters=[".npy"]), "calibration.image")
            widgets["brightest_center"] = persist(
                ui.checkbox("Use brightest pixel as center", value=True),
                "calibration.brightest_center")
            widgets["cx"] = persist(num_input("Centre X", 511.0, step="any"),
                                    "calibration.center_x")
            widgets["cy"] = persist(num_input("Centre Y", 511.0, step="any"),
                                    "calibration.center_y")
        with ui.row().classes("gap-4 flex-wrap items-end w-full"):
            widgets["radial_bin_size"] = persist(num_input("Radial bin size (px)", 1.0, step="any"),
                                                "calibration.radial_bin_size")
            widgets["radial_bin_size"].tooltip("Spacing between samples in detector pixels.")
            widgets["n_theta"] = persist(num_input("Angular bins", 720, step=1),
                                         "calibration.n_theta")
            widgets["r_max"] = persist(num_input("Max radius (0 = auto)", 0.0,
                                                 step=1.0), "calibration.r_max")
        with ui.row().classes("gap-2 items-center"):
            load_button = ui.button("Load image & remap", icon="refresh")
            export_button = ui.button("Export to Circularize", icon="output").props("flat")
            remap_status = ui.label("").classes("vmi-caption")

        ui.label("Centre X and Y are the zero-based detector column and row.") \
            .classes("vmi-subtitle")

        def update_geometry():
            for key in ("cx", "cy"):
                widgets[key].set_enabled(not widgets["brightest_center"].value)

        widgets["brightest_center"].on_value_change(lambda _: update_geometry())
        update_geometry()

    # ------------------------------------------------------------ processing
    with theme.panel_card():
        ui.label("Processing chain").classes("vmi-subtitle")
        with ui.row().classes("gap-4 flex-wrap items-center"):
            widgets["abel"] = persist(
                ui.checkbox("Inverse Abel per θ", value=True), "calibration.abel")
            widgets["abel_method"] = persist(ui.select(
                ["hansenlaw", "dasch_three_point", "dasch_two_point",
                 "onion_peeling"], value="hansenlaw",
                label="Abel method").props("outlined dense"),
                "calibration.abel_method")
        with ui.row().classes("gap-4 flex-wrap items-center"):
            widgets["mode"] = persist(ui.select(
                list(pf.ENHANCE_MODES), value="bandpass",
                label="Enhancement").props("outlined dense").classes("w-44"),
                "calibration.enhance_mode")
            widgets["placement"] = persist(ui.select(
                {"raw": "Raw polar", "pre_abel": "Before Abel inversion",
                 "post_abel": "After Abel inversion", "off": "Off"},
                value="post_abel", label="Filter placement")
                .props("outlined dense").classes("w-56"), "calibration.placement")

            def _update_filter_placement(preferred):
                abel = widgets["abel"].value
                active = "post_abel" if abel else "raw"
                label = "After Abel inversion" if abel else "Raw polar"
                widgets["placement"].set_options(
                    {active: label, "off": "Off"},
                    value="off" if preferred == "off" else active,
                )

            widgets["abel"].on_value_change(lambda _: _update_filter_placement(widgets["placement"].value))
            _update_filter_placement(widgets["placement"].value)
            apply_button = ui.button("Apply processing", icon="tune")
        with ui.expansion("Filter parameters", icon="tune").classes("w-full"):
            with ui.row().classes("gap-3 flex-wrap"):
                for key, label, value, step in FILTER_FIELDS:
                    widgets[f"param.{key}"] = persist(
                        num_input(label, value, step=step),
                        f"calibration.param.{key}")

    # --------------------------------------------------------- image + fit
    with theme.panel_card():
        with ui.column().classes("w-full gap-4"):
            with ui.column().classes("w-full gap-2"):
                with ui.row().classes("gap-3 items-center flex-wrap"):
                    color_scale = persist(
                        ui.select(["Linear", "Log"], value="Linear", label="Color scale")
                        .props("outlined dense").classes("w-32"), "calibration.color_scale")
                    colormap = persist(
                        ui.select(list(COLORMAPS), value="Viridis", label="Colormap")
                        .props("outlined dense").classes("w-36"), "calibration.colormap")
                with ui.row().classes("w-full gap-4 flex-wrap"):
                    with ui.column().classes("w-64 max-w-full gap-0"):
                        lower_label = ui.label().classes("vmi-subtitle")
                        color_min = persist(
                            ui.slider(min=0, max=99.9, step=0.1, value=1)
                            .props('aria-label="Lower percentile (vmin)"').classes("w-full"),
                            "calibration.color_min_pct")
                    with ui.column().classes("w-64 max-w-full gap-0"):
                        upper_label = ui.label().classes("vmi-subtitle")
                        color_max = persist(
                            ui.slider(min=0.1, max=100, step=0.1, value=99.5)
                            .props('aria-label="Upper percentile (vmax)"').classes("w-full"),
                            "calibration.color_max_pct")
                lower = 1 if color_min.value is None else round(color_min.value, 1)
                upper = 99.5 if color_max.value is None else round(color_max.value, 1)
                if not 0 <= lower < upper <= 100:
                    lower, upper = 1, 99.5
                color_min.value, color_max.value = lower, upper
                lower_label.bind_text_from(color_min, "value", lambda value: f"Lower percentile (vmin) {value:g}%")
                upper_label.bind_text_from(color_max, "value", lambda value: f"Upper percentile (vmax) {value:g}%")
                color_status = ui.label("").classes("vmi-subtitle")
                color_status.visible = False
                ui.label("Percentiles use each visible image separately. Log scale uses positive pixels.") \
                    .classes("vmi-subtitle")
                polar_images = {}
                pointer_readouts = {}
                for key, label in (("raw", "Raw polar"), ("processed", "Processed polar")):
                    ui.label(label).classes("vmi-subtitle")
                    polar_images[key] = _polar_image(key)
                    pointer_readouts[key] = ui.label().classes(
                        "polar-pointer-readout w-full text-right font-mono text-xs min-h-4")
                widgets["tool"] = ui.toggle(
                    {"band": "Select band", "mask": "Mask"}, value="band")
                with ui.row().classes("gap-2 items-center"):
                    widgets["full_angle"] = persist(
                        ui.checkbox("Full angular range", value=True),
                        "calibration.full_angle")
                    undo_button = ui.button("Undo selection", icon="undo",
                                            on_click=lambda: _undo_selection()).disable()
                    clear_band_button = ui.button(
                        "Clear band", on_click=lambda: _set_selection(None, state["masks"])).disable()
                    clear_masks_button = ui.button(
                        "Clear masks", on_click=lambda: _set_selection(state["band"], [])).disable()
                    ui.button("Zoom to band", on_click=lambda: _zoom_band()).tooltip("Zoom along radius only")
                    inspect_button = ui.button("Inspect band", on_click=lambda: _inspect_band()).disable()
                    ui.button("Reset view", on_click=lambda: _reset_view())
                band_label = ui.label("").classes("vmi-caption")
                mask_label = ui.label("").classes("vmi-caption")
            with ui.dialog() as inspection_dialog, ui.card().style("width: 94vw; max-width: 1400px"):
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label("Inspect processed band").classes("text-lg")
                    ui.button("Close", on_click=inspection_dialog.close).props("flat")
                with ui.row().classes("items-center gap-4"):
                    zoom_label = ui.label()
                    horizontal_zoom = ui.slider(min=0.05, max=1, step=0.05, value=1).classes("w-52").props(
                        'aria-label="Horizontal zoom"')
                    zoom_label.bind_text_from(horizontal_zoom, "value", lambda value: f"Horizontal zoom {value:g}×")
                    inspection_undo = ui.button("Undo selection", icon="undo",
                                                on_click=lambda: _undo_selection()).disable()
                ui.label("Drag to mask. Adjust horizontal zoom to narrow the band.").classes("vmi-subtitle")
                with ui.element("div").classes("w-full overflow-x-auto"):
                    inspection_image = _polar_image("inspection")
                inspection_status = ui.label().classes("vmi-subtitle")
                pointer_readouts["inspection"] = ui.label().classes(
                    "w-full text-right font-mono text-xs min-h-4")
                horizontal_zoom.on_value_change(lambda e: inspection_image.style(f"width: {e.value * 100}%"))
            inspection_dialog.on("before-hide", lambda: _close_inspection())
            with ui.column().classes("w-full gap-3"):
                with ui.row().classes("gap-3 items-center flex-wrap"):
                    widgets["n_terms"] = persist(
                        num_input("Trig terms N", 3, step=1),
                        "calibration.n_terms")
                    widgets["beta"] = persist(
                        ui.checkbox("Fit beta", value=True), "calibration.beta")
                    widgets["iterative"] = persist(
                        ui.checkbox("Iterative terms", value=True),
                        "calibration.iterative")
                    fit_button = ui.button("Fit selection", icon="timeline")
                fit_status = ui.label("No fit yet.").classes("vmi-caption")
                param_table = ui.table(
                    columns=[{"name": "parameter", "label": "Parameter",
                              "field": "parameter", "align": "left"},
                             {"name": "value", "label": "Value", "field": "value"},
                             {"name": "sigma", "label": "1σ", "field": "sigma"}],
                    rows=[], row_key="parameter").classes("w-full")
                accept_button = ui.button("Accept ring (append to file)",
                                           icon="save").disable()
        ui.label(
            "The fit uses the selected band from Processed polar, including any applied filter."
        ).classes("vmi-caption")
        fit_preview = ui.column().classes("w-full")

    # ----------------------------------------------------- rings, file, session
    with theme.panel_card():
        ui.label("Accepted rings").classes("vmi-subtitle")
        rings_table = ui.table(
            columns=[{"name": "radius", "label": "r0 (px)", "field": "radius"},
                     {"name": "terms", "label": "N", "field": "terms"},
                     {"name": "rms", "label": "RMS", "field": "rms"}],
            rows=[], row_key="radius").classes("w-full")
        with ui.row().classes("gap-4 flex-wrap items-end w-full"):
            widgets["output"] = persist(path_input(
                "Extracted trig-fit file",
                str(CIRC_DIR / f"extracted-{time.strftime('%Y%m%d')}.txt"),
                browse="save", filters=[".txt"],
                default_name="extracted.txt"), "calibration.output")
            widgets["provenance"] = persist(
                ui.checkbox("Provenance comments", value=True),
                "calibration.provenance")
            ui.button("Rebuild file from rings", on_click=lambda: _rebuild_file())
            refresh_rings_button = ui.button("Refresh rings from file", icon="refresh").props("flat")
            clear_file_button = ui.button("Clear calibration file", icon="delete").props("flat color=negative")
        with ui.row().classes("gap-4 items-end flex-wrap"):
            radial_model = persist(ui.select({"interp": "Interp", "poly": "Poly"},
                                             label="Radial model", value="interp")
                                   .props("outlined dense").classes("w-36"), "calibration.radial_model")
            poly_order = persist(num_input("Poly order", 1, min=1, step=1),
                                 "calibration.slope_order")
            ui.button("Update radial model plots",
                      on_click=lambda: _update_reduction())
        ui.label("Interp uses shape-preserving cubic interpolation (PCHIP).") \
            .classes("vmi-subtitle")
        model_plots = ui.column().classes("w-full")
        with ui.row().classes("gap-4 items-end flex-wrap w-full"):
            widgets["session"] = persist(path_input(
                "Session file (.json)",
                str(CIRC_DIR / "trig-calibration.session.json"),
                browse="save", filters=[".json"],
                default_name="trig-calibration.session.json"),
                "calibration.session")
            ui.button("Save session", on_click=lambda: _save_session())
            ui.button("Load session", on_click=lambda: _load_session())

    console = theme.log_console()

    # --------------------------------------------------------------- helpers
    def _options():
        params = {}
        for key, *_ in FILTER_FIELDS:
            value = widgets[f"param.{key}"].value
            params[key] = (pf.DEFAULT_ENHANCE_PARAMS[key] if value in (None, "")
                           else float(value))
        return pf.process_options(
            abel=bool(widgets["abel"].value),
            abel_method=widgets["abel_method"].value,
            fit_filtered=True,
            enhancement={"placement": widgets["placement"].value,
                         "mode": widgets["mode"].value,
                         "params": params},
        )

    def _notify_error(exc):
        console.push(f"[Trig Calibration] {type(exc).__name__}: {exc}")
        ui.notify(f"{type(exc).__name__}: {exc}", type="negative", timeout=6000)

    def _show_fit(fit):
        state["last_fit"] = fit
        fit_preview.clear()
        param_table.rows = [] if fit is None else _parameter_rows(fit)
        issue = fit.export_issue() if fit is not None else None
        accept_button.set_enabled(fit is not None and issue is None)
        if fit is None:
            fit_status.text = "No fit yet."
            return
        fit_status.text = (f"r0 = {fit.params['r0']:.3f} px, "
                           f"σ = {fit.params['sigma']:.3f} px, RMS = {fit.rms:.4g}, "
                           f"pixels = {fit.n_pixels}")
        with fit_preview:
            masked_fraction = fit.masked_ridge_fraction(state["theta"])
            if masked_fraction:
                ui.label(f"The fitted ridge crosses a masked region at {masked_fraction:.1%} of the "
                         "sampled angles. The fit excluded the masked intensities.").classes("text-warning")
            if issue:
                ui.label(issue).classes("text-negative")
            if not fit.converged:
                ui.label(f"Optimizer did not converge. {fit.message}").classes("text-warning")
            figure = _fit_comparison_figure(state["processed"], state["theta"], state["r"], fit)
            if figure is not None:
                ui.label("Grey areas are excluded. The yellow line marks the fitted ridge.").classes("vmi-subtitle")
                ui.plotly(plot_options(figure)).classes("self-center").style(
                    f"width: calc(100% * 2 / 3); height: {figure.layout.height}px")

    def _update_labels():
        if state["band"] is not None:
            band = state["band"]
            band_label.text = (
                f"Band r = {band.r0:.2f} – {band.r1:.2f} px, "
                f"θ = {np.degrees(band.theta0):.1f}° – {np.degrees(band.theta1):.1f}°")
        else:
            band_label.text = "No band selected."
        mask_label.text = f"Masks: {len(state['masks'])}"

    def _update_overlay():
        overlay = _overlay_svg(state)
        for image in polar_images.values():
            image.content = overlay
        inspection = state["inspection"]
        if inspection is not None:
            inspection_image.content = _overlay_svg(state | inspection | {"band": None})
        inspect_button.set_enabled(state["band"] is not None and state["processed"] is not None)
        history = state["selection_history"]
        undo_button.set_enabled(bool(history))
        inspection_undo.set_enabled(bool(history) and history[-1][0] == state["band"])
        clear_band_button.set_enabled(state["band"] is not None)
        clear_masks_button.set_enabled(bool(state["masks"]))
        _update_labels()

    def _reset_view():
        state["view"] = {"r0": 0.0, "r1": float(state["r"][-1]),
                         "theta0": 0.0, "theta1": float(pf.TWO_PI)}
        _refresh_view()

    def _refresh_view():
        if not state["ready"] or state["processed"] is None:
            return
        ti, ri = _polar_view_indices(state["theta"], state["r"], state["view"])
        if ti.size < 2 or ri.size < 2:
            return
        indices = np.ix_(ti, ri)
        warnings = []
        for key, label in (("raw", "Raw polar"), ("processed", "Processed polar")):
            warning = _set_polar_source(polar_images[key], state[key][indices])
            if warning:
                warnings.append(f"{label} uses automatic linear colors. {warning}")
            pointer_readouts[key].text = ""
        state["view_indices"] = (ti, ri)
        color_status.text = " ".join(warnings)
        color_status.visible = bool(warnings)
        state["img_size"] = (int(ri.size), int(ti.size))
        _update_overlay()

    def _set_polar_source(element, data):
        warning = ""
        try:
            image = _render_polar_image(data, cmap=colormap.value, log_scale=color_scale.value == "Log",
                                        vmin_pct=color_min.value, vmax_pct=color_max.value)
        except ValueError as exc:
            warning = str(exc)
            image = _render_polar_image(data, cmap=colormap.value)
        # Give each update its own bytes so in-flight requests cannot read a changing file.
        with BytesIO() as buffer:
            image.save(buffer, format="PNG")
            encoded = b64encode(buffer.getbuffer()).decode("ascii")
        element.set_source(f"data:image/png;base64,{encoded}")
        return warning

    def _inspect_band():
        band = state["band"]
        if band is None or state["processed"] is None:
            return
        view = {"r0": band.r0, "r1": band.r1, "theta0": band.theta0, "theta1": band.theta1}
        ti, ri = _polar_view_indices(state["theta"], state["r"], view)
        if not ti.size or not ri.size:
            ui.notify("The selected band contains no polar samples.", type="warning")
            return
        state["inspection"] = {"view": view, "view_indices": (ti, ri), "img_size": (ri.size, ti.size)}
        warning = _set_polar_source(inspection_image, state["processed"][np.ix_(ti, ri)])
        inspection_status.text = f"Showing automatic linear colors. {warning}" if warning else ""
        horizontal_zoom.value = 1
        _update_overlay()
        inspection_dialog.open()

    def _close_inspection():
        state.update(inspection=None, drag_start=None, drag_now=None, drag_rect=None)
        pointer_readouts["inspection"].text = ""
        _update_overlay()

    def _zoom_band():
        if state["band"] is None:
            ui.notify("Select a band first.", type="warning")
            return
        band = state["band"]
        pad = max((band.r1 - band.r0) * 0.25, 2.0)
        state["view"].update(r0=max(band.r0 - pad, 0.0),
                             r1=min(band.r1 + pad, float(state["r"][-1])))
        _refresh_view()

    def _change_color_limit(slider):
        if color_min.value >= color_max.value:
            if slider is color_min:
                color_min.value = round(color_max.value - 0.1, 1)
            else:
                color_max.value = round(color_min.value + 0.1, 1)
            return
        _refresh_view()

    def _set_selection(band, masks):
        state["selection_history"].append((state["band"], state["masks"]))
        _restore_selection(band, masks)

    def _undo_selection():
        if state["selection_history"]:
            _restore_selection(*state["selection_history"].pop())

    def _restore_selection(band, masks):
        state["band"], state["masks"] = band, masks
        _show_fit(None)
        _update_overlay()

    def _on_mouse(event, key):
        if event.type == "mouseleave":
            pointer_readouts[key].text = ""
        inspection = key == "inspection"
        if not state["ready"] or (inspection and state["inspection"] is None):
            return
        viewport = state | state["inspection"] if inspection else state
        if event.type == "mousemove":
            sample = _polar_sample(viewport, "processed" if inspection else key, event.image_x, event.image_y)
            if sample is None:
                pointer_readouts[key].text = ""
            else:
                radius, angle, value = sample
                intensity = f"{value:.6g}" if np.isfinite(value) else "—"
                pointer_readouts[key].text = f"r {radius:.3f} px · θ {angle:.2f}° · Intensity {intensity}"
        if event.type == "mousedown" and event.button == 0:
            state["drag_start"] = _px_to_data(viewport, event.image_x, event.image_y)
            state["drag_now"] = state["drag_start"]
            return
        if event.type == "mousemove" and state["drag_start"] is not None:
            state["drag_now"] = _px_to_data(viewport, event.image_x, event.image_y)
            now = time.monotonic()
            if now - state["last_overlay"] > 0.06:
                state["last_overlay"] = now
                state["drag_rect"] = _drag_rect()
                _update_overlay()
            return
        if event.type in ("mouseup", "mouseleave"):
            if state["drag_start"] is None:
                return
            state["drag_now"] = _px_to_data(viewport, event.image_x, event.image_y)
            rect = _drag_rect()
            state["drag_start"] = None
            state["drag_now"] = None
            state["drag_rect"] = None
            if rect is None:
                _update_overlay()
                return
            if event.type == "mouseleave":
                _update_overlay()
                return
            if not inspection and state["tool"] == "band":
                full = bool(widgets["full_angle"].value)
                band = pf.Rect(
                    rect.r0, rect.r1, 0.0 if full else rect.theta0,
                    pf.TWO_PI if full else rect.theta1)
                _set_selection(band, state["masks"])
            else:
                _set_selection(state["band"], [*state["masks"], rect])
            return

    def _drag_rect():
        if state["drag_start"] is None or state["drag_now"] is None:
            return None
        r0, t0 = state["drag_start"]
        r1, t1 = state["drag_now"]
        if abs(r1 - r0) < 1e-6 or abs(t1 - t0) < 1e-6:
            return None
        return pf.Rect(min(r0, r1), max(r0, r1), min(t0, t1), max(t0, t1))

    async def _apply_processing():
        if not state["ready"]:
            ui.notify("Load an image first.", type="warning")
            return
        state["processed"] = None
        inspect_button.disable()
        _show_fit(None)
        try:
            options = _options()
            processed, _, options = await asyncio.to_thread(
                pf.process_polar, state["raw"], state["dr"], **options)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            _notify_error(exc)
            return
        state["processed"] = processed
        state["options"] = options
        _refresh_view()
        remap_status.text = (f"{processed.shape[1]} radial × {processed.shape[0]} angular bins")

    async def _load_image(saved_center=None, saved_radial_count=None):
        path = widgets["image"].value
        if not path:
            ui.notify("Choose an image.", type="warning")
            return False
        remap_status.text = "loading and remapping…"
        try:
            path = Path(path).expanduser().resolve()
            auto_center = bool(widgets["brightest_center"].value)
            center = saved_center
            if center is None and not auto_center:
                center = (widgets["cx"].value, widgets["cy"].value)
                if None in center:
                    raise ValueError("Enter both Centre X and Centre Y.")
            bin_size = widgets["radial_bin_size"].value
            if saved_radial_count is None and bin_size is None:
                raise ValueError("Enter a radial bin size in pixels.")
            n_theta = int(widgets["n_theta"].value)
            r_max = float(widgets["r_max"].value or 0.0) or None
            image = await asyncio.to_thread(pf.load_image, path)
            if auto_center:
                row, col = await asyncio.to_thread(brightest_pixel, image)
                console.push(f"[Trig Calibration] Brightest pixel in {path.name} "
                             f"(zero-based): row={row}, col={col}.")
                if saved_center is not None and saved_center != (col, row):
                    raise ValueError("The saved fit center differs from the raw detector origin. Refit the rings.")
                center = (col, row)
            radial_grid = ({"radial_bin_size": bin_size} if saved_radial_count is None else
                           {"n_r": saved_radial_count})
            polar, theta, r, dr, _valid = await asyncio.to_thread(
                pf.remap_polar, image, center, n_theta=n_theta, r_max=r_max, **radial_grid)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            remap_status.text = "remap failed"
            _notify_error(exc)
            return False
        widgets["cx"].value, widgets["cy"].value = center
        if saved_radial_count is not None:
            widgets["radial_bin_size"].value = dr
        console.push(f"[Trig Calibration] Polar fit center for Circularize on the matching grid: "
                     f"Center X={center[0]:g}, Center Y={center[1]:g}.")
        console.push(f"[Trig Calibration] {r.size} radial samples from 0 to {r[-1]:g} px "
                     f"at {dr:.6g} px spacing.")
        state.update(raw=polar, theta=theta, r=r, dr=dr, ready=True,
                     image={"path": str(path), "center": list(center),
                            "brightest_center": auto_center,
                            "radial_bin_size": dr, "n_theta": n_theta, "r_max": r_max},
                     band=None, masks=[], selection_history=[], last_fit=None, view_indices=None,
                     processed=None,
                     rings=[], view={"r0": 0.0, "r1": float(r[-1]),
                                     "theta0": 0.0, "theta1": float(pf.TWO_PI)})
        rings_table.rows = []
        model_plots.clear()
        remap_status.text = "processing…"
        await _apply_processing()
        if state["processed"] is None:
            state["ready"] = False
        param_table.rows = []
        fit_status.text = "No fit yet."
        return state["ready"]

    async def _fit_selection():
        if not state["ready"]:
            ui.notify("Load an image first.", type="warning")
            return
        if state["processed"] is None:
            ui.notify("Apply processing before fitting.", type="warning")
            return
        if state["band"] is None:
            ui.notify("Drag a band around one ring first.", type="warning")
            return
        fit_button.disable()
        _show_fit(None)
        fit_status.text = "fitting…"
        image, theta, radii = state["processed"], state["theta"], state["r"]
        band, masks = state["band"], tuple(state["masks"])
        try:
            fit = await asyncio.to_thread(
                pf.fit_ring, image, theta, radii, band, masks,
                n_terms=int(widgets["n_terms"].value if widgets["n_terms"].value is not None else 3),
                use_beta=bool(widgets["beta"].value),
                iterative=bool(widgets["iterative"].value),
                processing=state["options"])
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            fit_status.text = "fit failed"
            _notify_error(exc)
            return
        finally:
            fit_button.enable()
        if image is not state["processed"] or band != state["band"] or masks != tuple(state["masks"]):
            return
        _show_fit(fit)
        _update_overlay()

    def _ring_rows():
        return [{"radius": f"{fit.params['r0']:.4f}", "terms": fit.n_terms,
                 "rms": f"{fit.rms:.4g}" if np.isfinite(fit.rms) else "—"} for fit in state["rings"]]

    def _session_payload():
        return {
            "schema": pf.SESSION_SCHEMA,
            "image": state["image"],
            "processing": state["options"],
            "view": state["view"],
            "band": None if state["band"] is None else
            [state["band"].r0, state["band"].r1, state["band"].theta0,
             state["band"].theta1],
            "masks": [[m.r0, m.r1, m.theta0, m.theta1] for m in state["masks"]],
            "rings": [fit.to_dict() for fit in state["rings"]],
            "fit": None if state["last_fit"] is None else state["last_fit"].to_dict(),
            "output": widgets["output"].value,
        }

    def _save_session():
        if state["options"] is None:
            ui.notify("Nothing to save yet.", type="warning")
            return
        try:
            pf.save_session(widgets["session"].value, _session_payload())
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            _notify_error(exc)
            return
        ui.notify(f"Session saved to {widgets['session'].value}")

    async def _load_session():
        try:
            payload = pf.load_session(widgets["session"].value)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            _notify_error(exc)
            return
        image = payload.get("image", {})
        if Path(image.get("path") or "").suffix.lower() != ".npy" or image.get("rotation", 0):
            ui.notify("Choose the original NumPy detector image and refit the rings.",
                      type="warning")
            return
        widgets["image"].value = image.get("path", widgets["image"].value)
        center = image.get("center", [widgets["cx"].value, widgets["cy"].value])
        widgets["brightest_center"].value = image.get("brightest_center", False)
        widgets["cx"].value = center[0]
        widgets["cy"].value = center[1]
        if "radial_bin_size" in image:
            widgets["radial_bin_size"].value = image["radial_bin_size"]
        widgets["n_theta"].value = image.get("n_theta", widgets["n_theta"].value)
        widgets["r_max"].value = image.get("r_max", 0.0)
        processing = payload.get("processing") or {}
        if processing:
            widgets["abel"].value = processing.get("abel", True)
            widgets["abel_method"].value = processing.get("abel_method",
                                                          "hansenlaw")
            enh = processing.get("enhancement", {})
            _update_filter_placement(enh.get("placement", "post_abel"))
            widgets["mode"].value = enh.get("mode", "bandpass")
            for key, value in (enh.get("params") or {}).items():
                if f"param.{key}" in widgets:
                    widgets[f"param.{key}"].value = value
        widgets["output"].value = payload.get("output", widgets["output"].value)
        saved_radial_count = None if "radial_bin_size" in image else image.get("n_r")
        if await _load_image(saved_center=tuple(center), saved_radial_count=saved_radial_count):
            view = payload.get("view")
            if view:
                state["view"] = view
            band = payload.get("band")
            state["band"] = None if band is None else pf.Rect(*band)
            state["masks"] = [pf.Rect(*item) for item in payload.get("masks", [])]
            state["rings"] = pf.session_rings(payload)
            fit_data = payload.get("fit")
            if fit_data and fit_data.get("processing") != state["options"]:
                ui.notify("Processing changed since this fit was saved. Fit the selected band again.",
                          type="warning")
                fit_data = None
            _show_fit(None if fit_data is None else pf.RingFit.from_dict(fit_data))
            rings_table.rows = _ring_rows()
            _refresh_view()
            await _update_reduction()
        ui.notify("Session loaded.")

    def _rebuild_file():
        if not state["rings"]:
            ui.notify("No accepted rings to write.", type="warning")
            return
        try:
            path = _calibration_path()
            blocks = [fit.source_block if fit.source_block is not None else
                      pf.ring_block_text(fit, provenance=bool(widgets["provenance"].value))
                      for fit in state["rings"]]
            with MANAGER.write_access([path]):
                pf.write_ring_blocks(path, blocks)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            _notify_error(exc)
            return
        ui.notify(f"Wrote {len(blocks)} ring block(s) to {widgets['output'].value}")

    def _calibration_path():
        value = (widgets["output"].value or "").strip()
        if not value or Path(value).suffix.lower() != ".txt":
            raise ValueError("Choose an extracted trig-fit .txt file.")
        return Path(value).expanduser().absolute()

    async def _refresh_rings_from_file():
        try:
            path = _calibration_path()
            rings = pf.load_accepted_rings(path)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            _notify_error(exc)
            return
        state["rings"] = rings
        rings_table.rows = _ring_rows()
        await _update_reduction()
        ui.notify(f"Loaded {len(rings)} ring(s) from {path.name}.")

    def _confirm_clear_file():
        try:
            path = _calibration_path()
        except ValueError as exc:
            _notify_error(exc)
            return

        def clear():
            failed = delete_outputs([path])
            if failed is None:
                return
            if failed:
                _notify_error(OSError(failed[0]))
                return
            if widgets["output"].value and Path(widgets["output"].value.strip()).expanduser().absolute() == path:
                state["rings"] = []
                state["band"] = None
                state["masks"] = []
                state["selection_history"].clear()
                rings_table.rows = []
                model_plots.clear()
                _show_fit(None)
                _update_overlay()
            dialog.close()
            ui.notify(f"Cleared {path.name}. Accept a ring to create it again.")

        with ui.dialog() as dialog, ui.card():
            ui.label("Clear the calibration file?").classes("text-lg")
            ui.label(str(path)).classes("break-all")
            with ui.row():
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button("Clear file", on_click=clear).props("color=negative")
        dialog.open()

    async def _accept():
        if state["last_fit"] is None:
            ui.notify("Fit a ring first.", type="warning")
            return
        fit = state["last_fit"]
        if issue := fit.export_issue():
            ui.notify(issue, type="warning")
            return
        try:
            path = _calibration_path()
            block = pf.ring_block_text(
                fit, provenance=bool(widgets["provenance"].value))
            with MANAGER.write_access([path]):
                if any(abs(radius - fit.params["r0"]) < 0.5 for radius in pf.existing_radii(path)):
                    ui.notify("A ring at this radius is already in the file. Adjust the band or output path.",
                              type="warning")
                    return
                pf.append_ring_block(path, block)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            _notify_error(exc)
            return
        state["rings"].append(fit)
        _show_fit(None)
        state["band"] = None
        state["masks"] = []
        state["selection_history"].clear()
        fit_status.text = "Accepted. Select the next ring."
        rings_table.rows = _ring_rows()
        _update_overlay()
        if widgets["session"].value:
            _save_session()
        await _update_reduction()
        ui.notify(f"Appended ring r0 = {fit.params['r0']:.3f} px")

    async def _update_reduction():
        model_plots.clear()
        rings = tuple(state["rings"])
        if not rings:
            return
        model = radial_model.value
        order = int(poly_order.value or 1)
        if model == "poly" and len(rings) < order:
            with model_plots:
                ui.label(f"Accept at least {order} rings for polynomial order {order}.")
            return
        try:
            reduced = await asyncio.to_thread(
                pf.reduce_rings, rings, model=model, order=order)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            _notify_error(exc)
            return
        if rings != tuple(state["rings"]) or model != radial_model.value or order != int(poly_order.value or 1):
            return
        model_plots.clear()
        with model_plots:
            for figure in _radial_model_figures(reduced):
                ui.plotly(plot_options(figure)).classes("w-full")

    def _export_to_circularize():
        if not widgets["output"].value:
            ui.notify("Set the extracted file first.", type="warning")
            return
        image = state["image"]
        raw_path = image["path"] if image else None
        manual_center = image["center"] if image and not image["brightest_center"] else None
        if use_calibration is None:
            return
        use_calibration(str(Path(widgets["output"].value).expanduser().resolve()), raw_path, manual_center,
                        radial_model_value=radial_model.value,
                        poly_order_value=int(poly_order.value or 1) if radial_model.value == "poly" else None)
        ui.notify("Set the Circularize calibration file and NumPy input." if raw_path else
                  "Set the Circularize calibration file.")

    # ------------------------------------------------------------- wiring
    async def _load_clicked():
        await _load_image()

    load_button.on_click(_load_clicked)
    apply_button.on_click(_apply_processing)
    fit_button.on_click(_fit_selection)
    accept_button.on_click(_accept)
    widgets["tool"].on_value_change(
        lambda e: state.__setitem__("tool", e.value or "band"))
    for control in (color_scale, colormap):
        control.on_value_change(lambda _: _refresh_view())
    color_min.on_value_change(lambda _: _change_color_limit(color_min))
    color_max.on_value_change(lambda _: _change_color_limit(color_max))
    export_button.on_click(_export_to_circularize)
    refresh_rings_button.on_click(_refresh_rings_from_file)
    clear_file_button.on_click(_confirm_clear_file)

    async def _change_radial_model():
        poly_order.set_enabled(radial_model.value == "poly")
        await _update_reduction()

    poly_order.set_enabled(radial_model.value == "poly")
    radial_model.on_value_change(_change_radial_model)
    poly_order.on_value_change(_update_reduction)
