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
from pathlib import Path
import time

import numpy as np
from matplotlib import colormaps
from nicegui import app, ui
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vmi_analysis.circularization import polar_fit as pf
from vmi_analysis.imaging.centering import brightest_pixel

from .. import theme
from ..centroid_plots import _style, plot_options
from ..settings import centroid_settings, output_dir
from .common import (GENERATED, REPO_ROOT, file_url, num_input, path_input,
                     persist)

CIRC_DIR = REPO_ROOT / "circularization"
VIEW_PNG = GENERATED / "polarfit_view.png"

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


def _render_polar_png(polar, out_path, cmap="viridis"):
    """Write the polar image at exactly one pixel per bin (x=r, y=theta)."""
    polar = np.asarray(polar, dtype=float)
    finite = np.isfinite(polar)
    data = np.where(finite, polar, 0.0)
    scaled = pf.robust_normalize(data, 1.0, 99.5)
    rgba = (colormaps[cmap](np.clip(scaled, 0.0, 1.0))[..., :3] * 255).astype(
        np.uint8)
    Image.fromarray(rgba).save(out_path, format="PNG")
    return out_path


def _px_to_data(state, x, y):
    width, height = state["img_size"]
    view = state["view"]
    r = view["r0"] + (float(x) / max(width, 1)) * (view["r1"] - view["r0"])
    theta = view["theta0"] + (float(y) / max(height, 1)) * (
        view["theta1"] - view["theta0"])
    return r, theta


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
                f'fill="rgba(33,150,243,0.14)" stroke="#2196f3" stroke-width="2"/>')
    for rect in state.get("masks", []):
        box = _clipped_rect(state, rect)
        if box:
            x, y, w, h = box
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                f'fill="rgba(244,67,54,0.28)" stroke="#f44336" stroke-width="2"/>')
    drag = state.get("drag_rect")
    if drag is not None:
        box = _clipped_rect(state, drag)
        if box:
            x, y, w, h = box
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                f'fill="rgba(255,235,59,0.15)" stroke="#fbc02d" '
                f'stroke-width="2" stroke-dasharray="6 4"/>')
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
                f'<path d="{path}" fill="none" stroke="#ffd54f" stroke-width="2"/>')
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
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Data used for fit", "Fitted model"),
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
    fig.update_layout(height=450, plot_bgcolor="#d8d2ca",
                      coloraxis={"colorscale": "Viridis", "cmin": float(finite.min()),
                                 "cmax": float(finite.max()), "colorbar": {"title": "Intensity"}})
    return fig


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
        "raw": None, "fit_image": None, "display": None, "theta": None, "r": None, "dr": None,
        "options": None, "view": None, "img_size": (0, 0),
        "tool": "band", "drag_start": None, "drag_now": None,
        "drag_rect": None, "band": None, "masks": [], "last_fit": None,
        "rings": [], "ready": False,
        "last_overlay": 0.0,
        "image": None,
    }
    widgets = {}

    # ---------------------------------------------------------------- inputs
    with theme.panel_card():
        with ui.row().classes("gap-4 flex-wrap items-end w-full"):
            widgets["image"] = persist(path_input(
                "VMI image (.npy or .txt)", str(output_dir(centroid_settings()) / "vmi_2d_raw.npy"),
                browse="file", filters=[".npy", ".txt"]), "calibration.image")
            widgets["rotation"] = persist(ui.select(
                [0, 90, 180, 270], value=0,
                label="Rotation (ccw)").props("outlined dense"), "calibration.rotation")
            widgets["brightest_center"] = persist(
                ui.checkbox("Use brightest pixel as center", value=True),
                "calibration.brightest_center")
            widgets["cx"] = persist(num_input("Centre X", 511.0, step=1.0),
                                    "calibration.center_x")
            widgets["cy"] = persist(num_input("Centre Y", 511.0, step=1.0),
                                    "calibration.center_y")
            widgets["n_r"] = persist(num_input("Radial bins", 512, step=1),
                                     "calibration.n_r")
            widgets["n_theta"] = persist(num_input("Angular bins", 720, step=1),
                                         "calibration.n_theta")
            widgets["r_max"] = persist(num_input("Max radius (0 = auto)", 0.0,
                                                 step=1.0), "calibration.r_max")
        with ui.row().classes("gap-2 items-center"):
            load_button = ui.button("Load image & remap", icon="refresh")
            use_circ_button = ui.button("Use Circularize input",
                                        icon="content_copy").props("flat")
            remap_status = ui.label("").classes("vmi-caption")

        ui.label("NumPy input uses the original detector grid and its brightest pixel.") \
            .classes("vmi-subtitle")

        def update_geometry():
            raw_grid = Path(widgets["image"].value or "").suffix.lower() == ".npy"
            widgets["rotation"].set_enabled(not raw_grid)
            widgets["brightest_center"].set_enabled(not raw_grid)
            for key in ("cx", "cy"):
                widgets[key].set_enabled(not raw_grid and not widgets["brightest_center"].value)

        widgets["image"].on_value_change(lambda _: update_geometry())
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
            widgets["placement"] = persist(ui.select(
                ["post_abel", "pre_abel", "raw", "off"], value="post_abel",
                label="Display filter placement").props("outlined dense"),
                "calibration.placement")
            widgets["mode"] = persist(ui.select(
                list(pf.ENHANCE_MODES), value="bandpass",
                label="Display enhancement").props("outlined dense"),
                "calibration.enhance_mode")
            apply_button = ui.button("Apply processing", icon="tune")
            widgets["show_raw"] = persist(
                ui.checkbox("Show raw polar", value=False),
                "calibration.show_raw")
        with ui.expansion("Filter parameters", icon="tune").classes("w-full"):
            with ui.row().classes("gap-3 flex-wrap"):
                for key, label, value, step in FILTER_FIELDS:
                    widgets[f"param.{key}"] = persist(
                        num_input(label, value, step=step),
                        f"calibration.param.{key}")

    # --------------------------------------------------------- image + fit
    with theme.panel_card():
        with ui.row().classes("w-full gap-6 items-start no-wrap"):
            with ui.column().classes("gap-2"):
                polar_image = ui.interactive_image(
                    "", on_mouse=lambda e: _on_mouse(e),
                    events=["mousedown", "mousemove", "mouseup", "mouseleave"],
                    sanitize=False,
                ).style("width: 100%; max-width: 560px; border: 1px solid #e0ddd6")
                widgets["tool"] = ui.toggle(
                    {"band": "Select band", "mask": "Mask"}, value="band")
                with ui.row().classes("gap-2 items-center"):
                    widgets["full_angle"] = persist(
                        ui.checkbox("Full angular range", value=True),
                        "calibration.full_angle")
                    ui.button("Clear masks", on_click=lambda: _clear_masks())
                    ui.button("Zoom to band", on_click=lambda: _zoom_band())
                    ui.button("Reset view", on_click=lambda: _reset_view())
                band_label = ui.label("").classes("vmi-caption")
                mask_label = ui.label("").classes("vmi-caption")
            with ui.column().classes("grow gap-3"):
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
            "Enhancement helps you select a band and masks. The fit uses "
            "unenhanced data with inverse Abel applied if enabled."
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
            ui.button("Use in Circularize", on_click=lambda: _use_in_circularize())
        with ui.row().classes("gap-4 items-end flex-wrap"):
            widgets["order"] = persist(num_input("Slope order", 1, step=1),
                                       "calibration.slope_order")
            ui.button("Update a_n/b_n preview",
                      on_click=lambda: asyncio.create_task(_update_reduction()))
        coefficients_table = ui.table(
            columns=[{"name": "coefficient", "label": "Coefficient",
                      "field": "coefficient"},
                     {"name": "value", "label": "Value", "field": "value"}],
            rows=[], row_key="coefficient").classes("w-full")
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
            if issue:
                ui.label(issue).classes("text-negative")
            if not fit.converged:
                ui.label(f"Optimizer did not converge. {fit.message}").classes("text-warning")
            figure = _fit_comparison_figure(state["fit_image"], state["theta"], state["r"], fit)
            if figure is not None:
                ui.label("Grey areas are excluded. The yellow line marks the fitted ridge.").classes("vmi-subtitle")
                ui.plotly(plot_options(figure)).classes("w-full").style("height: 450px")

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
        polar_image.content = _overlay_svg(state)
        _update_labels()

    def _reset_view():
        state["view"] = {"r0": 0.0, "r1": float(state["r"][-1]),
                         "theta0": 0.0, "theta1": float(pf.TWO_PI)}
        _refresh_view()

    def _refresh_view():
        if not state["ready"]:
            return
        view = state["view"]
        theta, r = state["theta"], state["r"]
        ti = np.flatnonzero((theta >= view["theta0"] - 1e-12)
                            & (theta <= view["theta1"] + 1e-12))
        ri = np.flatnonzero((r >= view["r0"] - 1e-12)
                            & (r <= view["r1"] + 1e-12))
        if ti.size < 2 or ri.size < 2:
            return
        source = state["raw"] if widgets["show_raw"].value else state["display"]
        if source is None:
            return
        patch = source[np.ix_(ti, ri)]
        _render_polar_png(patch, VIEW_PNG)
        state["img_size"] = (int(ri.size), int(ti.size))
        polar_image.set_source(file_url(VIEW_PNG))
        _update_overlay()

    def _zoom_band():
        if state["band"] is None:
            ui.notify("Select a band first.", type="warning")
            return
        band = state["band"]
        pad = max((band.r1 - band.r0) * 0.25, 2.0)
        state["view"] = {
            "r0": max(band.r0 - pad, 0.0),
            "r1": min(band.r1 + pad, float(state["r"][-1])),
            "theta0": band.theta0 if band.theta1 - band.theta0 < pf.TWO_PI - 1e-9 else 0.0,
            "theta1": band.theta1 if band.theta1 - band.theta0 < pf.TWO_PI - 1e-9 else float(pf.TWO_PI),
        }
        _refresh_view()

    def _clear_masks():
        state["masks"] = []
        _show_fit(None)
        _update_overlay()

    def _on_mouse(event):
        if not state["ready"]:
            return
        if event.type == "mousedown" and event.button == 0:
            state["drag_start"] = _px_to_data(state, event.image_x, event.image_y)
            state["drag_now"] = state["drag_start"]
            return
        if event.type == "mousemove" and state["drag_start"] is not None:
            state["drag_now"] = _px_to_data(state, event.image_x, event.image_y)
            now = time.monotonic()
            if now - state["last_overlay"] > 0.06:
                state["last_overlay"] = now
                state["drag_rect"] = _drag_rect()
                _update_overlay()
            return
        if event.type in ("mouseup", "mouseleave"):
            if state["drag_start"] is None:
                return
            state["drag_now"] = _px_to_data(state, event.image_x, event.image_y)
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
            if state["tool"] == "band":
                full = bool(widgets["full_angle"].value)
                state["band"] = pf.Rect(
                    rect.r0, rect.r1, 0.0 if full else rect.theta0,
                    pf.TWO_PI if full else rect.theta1)
            else:
                state["masks"].append(rect)
            _show_fit(None)
            _update_overlay()
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
        state["fit_image"] = state["display"] = None
        _show_fit(None)
        try:
            options = _options()
            fit_image, display, options = await asyncio.to_thread(
                pf.process_polar, state["raw"], state["dr"], **options)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            _notify_error(exc)
            return
        state["fit_image"] = fit_image
        state["display"] = display
        state["options"] = options
        _refresh_view()
        remap_status.text = (f"{display.shape[1]} radial × {display.shape[0]} angular bins")

    async def _load_image(saved_center=None):
        path = widgets["image"].value
        if not path:
            ui.notify("Choose an image.", type="warning")
            return False
        remap_status.text = "loading and remapping…"
        try:
            path = Path(path).expanduser().resolve()
            raw_grid = path.suffix.lower() == ".npy"
            rotation = 0 if raw_grid else int(widgets["rotation"].value or 0)
            auto_center = raw_grid or bool(widgets["brightest_center"].value)
            center = saved_center
            if center is None and not auto_center:
                center = (float(widgets["cx"].value), float(widgets["cy"].value))
            n_r = int(widgets["n_r"].value)
            n_theta = int(widgets["n_theta"].value)
            r_max = float(widgets["r_max"].value or 0.0) or None
            image = await asyncio.to_thread(pf.load_image, path)
            row, col = await asyncio.to_thread(brightest_pixel, image)
            console.push(f"[Trig Calibration] Brightest pixel in {Path(path).name} "
                         f"(zero-based): row={row}, col={col}.")
            if raw_grid:
                if saved_center is not None and saved_center != (col, row):
                    raise ValueError("The saved fit center differs from the raw detector origin. Refit the rings.")
                center = (col, row)
            rotated = pf.rotate_image(image, rotation) if rotation else image
            if center is None:
                fit_row, fit_col = ((row, col) if rotation == 0 else
                                    await asyncio.to_thread(brightest_pixel, rotated))
                center = (fit_col, fit_row)
            polar, theta, r, dr, _valid = await asyncio.to_thread(
                pf.remap_polar, rotated, center, n_r=n_r, n_theta=n_theta, r_max=r_max)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            remap_status.text = "remap failed"
            _notify_error(exc)
            return False
        widgets["cx"].value, widgets["cy"].value = center
        console.push(f"[Trig Calibration] Polar fit center for Circularize on the matching grid: "
                     f"Center X={center[0]:g}, Center Y={center[1]:g}.")
        if rotation:
            console.push("[Trig Calibration] The fit uses the remapped orientation. "
                         "Apply its calibration to a grid with that same orientation and center.")
        state.update(raw=polar, theta=theta, r=r, dr=dr, ready=True,
                     image={"path": str(path), "center": list(center), "rotation": rotation,
                            "brightest_center": auto_center,
                            "n_r": n_r, "n_theta": n_theta, "r_max": r_max},
                     band=None, masks=[], last_fit=None, fit_image=None, display=None,
                     rings=[], view={"r0": 0.0, "r1": float(r[-1]),
                                     "theta0": 0.0, "theta1": float(pf.TWO_PI)})
        rings_table.rows = []
        coefficients_table.rows = []
        remap_status.text = "processing…"
        await _apply_processing()
        if state["fit_image"] is None:
            state["ready"] = False
        param_table.rows = []
        fit_status.text = "No fit yet."
        return state["ready"]

    async def _fit_selection():
        if not state["ready"]:
            ui.notify("Load an image first.", type="warning")
            return
        if state["fit_image"] is None:
            ui.notify("Apply processing before fitting.", type="warning")
            return
        if state["band"] is None:
            ui.notify("Drag a band around one ring first.", type="warning")
            return
        fit_button.disable()
        _show_fit(None)
        fit_status.text = "fitting…"
        image, theta, radii = state["fit_image"], state["theta"], state["r"]
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
        if image is not state["fit_image"] or band != state["band"] or masks != tuple(state["masks"]):
            return
        _show_fit(fit)
        _update_overlay()

    def _ring_rows():
        return [{"radius": f"{fit.params['r0']:.4f}", "terms": fit.n_terms,
                 "rms": f"{fit.rms:.4g}"} for fit in state["rings"]]

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
        if (Path(image.get("path") or "").suffix.lower() == ".npy"
                and image.get("rotation", 0)):
            ui.notify("This session rotated the NumPy image. Refit its rings on the original detector grid.",
                      type="warning")
            return
        widgets["image"].value = image.get("path", widgets["image"].value)
        center = image.get("center", [widgets["cx"].value, widgets["cy"].value])
        widgets["brightest_center"].value = image.get("brightest_center", False)
        widgets["cx"].value = center[0]
        widgets["cy"].value = center[1]
        widgets["rotation"].value = image.get("rotation", 0)
        widgets["n_r"].value = image.get("n_r", widgets["n_r"].value)
        widgets["n_theta"].value = image.get("n_theta", widgets["n_theta"].value)
        widgets["r_max"].value = image.get("r_max", 0.0)
        processing = payload.get("processing") or {}
        if processing:
            widgets["abel"].value = processing.get("abel", True)
            widgets["abel_method"].value = processing.get("abel_method",
                                                          "hansenlaw")
            enh = processing.get("enhancement", {})
            widgets["placement"].value = enh.get("placement", "post_abel")
            widgets["mode"].value = enh.get("mode", "bandpass")
            for key, value in (enh.get("params") or {}).items():
                if f"param.{key}" in widgets:
                    widgets[f"param.{key}"].value = value
        widgets["output"].value = payload.get("output", widgets["output"].value)
        if await _load_image(saved_center=tuple(center)):
            view = payload.get("view")
            if view:
                state["view"] = view
            band = payload.get("band")
            state["band"] = None if band is None else pf.Rect(*band)
            state["masks"] = [pf.Rect(*item) for item in payload.get("masks", [])]
            state["rings"] = pf.session_rings(payload)
            fit_data = payload.get("fit")
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
            blocks = [pf.ring_block_text(fit,
                                         provenance=bool(widgets["provenance"].value))
                      for fit in state["rings"]]
            pf.write_ring_blocks(widgets["output"].value, blocks)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            _notify_error(exc)
            return
        ui.notify(f"Wrote {len(blocks)} ring block(s) to {widgets['output'].value}")

    async def _accept():
        if state["last_fit"] is None:
            ui.notify("Fit a ring first.", type="warning")
            return
        fit = state["last_fit"]
        if issue := fit.export_issue():
            ui.notify(issue, type="warning")
            return
        try:
            radii = await asyncio.to_thread(pf.existing_radii,
                                            widgets["output"].value)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            _notify_error(exc)
            return
        if any(abs(radius - fit.params["r0"]) < 0.5 for radius in radii):
            ui.notify("A ring at this radius is already in the file; adjust the "
                      "band or output path.", type="warning")
            return
        try:
            block = pf.ring_block_text(
                fit, provenance=bool(widgets["provenance"].value))
            await asyncio.to_thread(pf.append_ring_block,
                                    widgets["output"].value, block)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            _notify_error(exc)
            return
        state["rings"].append(fit)
        _show_fit(None)
        state["band"] = None
        state["masks"] = []
        fit_status.text = "Accepted. Select the next ring."
        rings_table.rows = _ring_rows()
        _update_overlay()
        if widgets["session"].value:
            _save_session()
        await _update_reduction()
        ui.notify(f"Appended ring r0 = {fit.params['r0']:.3f} px")

    async def _update_reduction():
        if len(state["rings"]) < 2:
            coefficients_table.rows = []
            return
        try:
            reduced = await asyncio.to_thread(
                pf.reduce_rings, state["rings"], model="poly",
                order=int(widgets["order"].value or 1))
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            _notify_error(exc)
            return
        rows = []
        for key, value in sorted(reduced["a"].items()):
            if key.endswith("_sigma"):
                continue
            sigma = reduced["a"].get(f"{key}_sigma", float("nan"))
            rows.append({"coefficient": key,
                         "value": f"{value:.6g} ± {sigma:.3g}"})
        coefficients_table.rows = rows

    def _use_in_circularize():
        if not widgets["output"].value:
            ui.notify("Set the extracted file first.", type="warning")
            return
        image = state["image"]
        raw_path = image["path"] if image and Path(image["path"]).suffix.lower() == ".npy" else None
        if use_calibration is None:
            return
        use_calibration(str(Path(widgets["output"].value).expanduser().resolve()), raw_path)
        ui.notify("Set the Circularize calibration file and NumPy input." if raw_path else
                  "Set the Circularize calibration file.")

    def _use_circularize_input():
        path = app.storage.general.get("circularize.inp")
        if path:
            widgets["image"].value = str(CIRC_DIR / Path(path).expanduser())

    # ------------------------------------------------------------- wiring
    async def _load_clicked():
        await _load_image()

    def _apply_clicked():
        asyncio.create_task(_apply_processing())

    def _accept_clicked():
        asyncio.create_task(_accept())

    def _reduction_clicked():
        asyncio.create_task(_update_reduction())

    def _load_session_clicked():
        asyncio.create_task(_load_session())

    load_button.on_click(_load_clicked)
    apply_button.on_click(_apply_clicked)
    fit_button.on_click(_fit_selection)
    accept_button.on_click(_accept_clicked)
    widgets["tool"].on_value_change(
        lambda e: state.__setitem__("tool", e.value or "band"))
    widgets["show_raw"].on_value_change(lambda e: _refresh_view())
    use_circ_button.on_click(_use_circularize_input)
