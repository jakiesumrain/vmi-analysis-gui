"""Shared display controls for polar detector images."""

from base64 import b64encode
from importlib.resources import files
from io import BytesIO

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LogNorm, Normalize
from nicegui import app, ui
from PIL import Image

from ..centroid_plots import COLORMAPS
from .common import persist

POLAR_HEIGHT = "calc(clamp(220px, 38vh, 360px) * 2 / 3)"
_GUIDE_SCRIPT = (files("vmi_analysis.gui.assets") / "polar_guides.js").read_text(encoding="utf-8")
_guides_installed = False


def polar_controls(prefix, *, radius_only=False):
    global _guides_installed
    if not _guides_installed:
        ui.add_body_html(f"<script>{_GUIDE_SCRIPT}</script>", shared=True)
        _guides_installed = True

    with ui.row().classes("gap-3 items-center flex-wrap"):
        scale = persist(ui.select(["Linear", "Log"], value="Linear", label="Color scale")
                        .props("outlined dense").classes("w-32"), f"{prefix}.color_scale")
        cmap = persist(ui.select(list(COLORMAPS), value="Viridis", label="Colormap")
                       .props("outlined dense").classes("w-36"), f"{prefix}.colormap")
        label = "Radius guide" if radius_only else "Crosshair guides"
        guides = persist(ui.checkbox(label), f"{prefix}.crosshair_guides")
        guides.tooltip("Drag the vertical line in either polar preview." if radius_only else
                       "Drag either line or the center handle in either polar preview.")
        axes = "x" if radius_only else "xy"
        ui.element("span").props(f'data-polar-guide-readout="{axes}"') \
            .classes("font-mono text-xs").bind_visibility_from(guides, "value")
    ui.label("Log scale uses positive pixels.").classes("vmi-subtitle")
    return scale, cmap, guides


def polar_preview(prefix, key, label, guides, make_image, *, legacy_shared_limits=False,
                  radius_only=False):
    ui.label(label).classes("vmi-subtitle")
    sliders = []
    with ui.row().classes("w-full gap-4 flex-wrap"):
        for bound, caption, minimum, maximum, default in (
            ("min", "Lower percentile (vmin)", 0, 99.9, 1),
            ("max", "Upper percentile (vmax)", 0.1, 100, 99.5),
        ):
            legacy = app.storage.general.get(f"{prefix}.color_{bound}_pct") if legacy_shared_limits else None
            with ui.column().classes("w-64 max-w-full gap-0"):
                value_label = ui.label().classes("vmi-subtitle")
                slider = persist(
                    ui.slider(min=minimum, max=maximum, step=0.1,
                              value=default if legacy is None else legacy)
                    .props(f'aria-label="{label} {caption.lower()}"').classes("w-full"),
                    f"{prefix}.{key}_color_{bound}_pct")
                value_label.bind_text_from(
                    slider, "value", lambda value, caption=caption: f"{caption} {value:g}%")
                sliders.append(slider)
    color_min, color_max = sliders
    lower = 1 if color_min.value is None else round(color_min.value, 1)
    upper = 99.5 if color_max.value is None else round(color_max.value, 1)
    color_min.value, color_max.value = (lower, upper) if 0 <= lower < upper <= 100 else (1, 99.5)
    status = ui.label("").classes("vmi-subtitle").set_visibility(False)
    with ui.element("div").classes("relative w-full"):
        image = make_image().classes("w-full").style(f"height: {POLAR_HEIGHT}; border: 1px solid #e0ddd6")
        with ui.element("div").classes("polar-guide-layer").bind_visibility_from(guides, "value"):
            handles = [("x", "vertical guide")]
            if not radius_only:
                handles.extend((("y", "horizontal guide"), ("xy", "crosshair handle")))
            for axis, name in handles:
                ui.element("button").props(
                    f'type="button" data-polar-guide="{axis}" aria-label="{label} {name}"')
    return image, (color_min, color_max), status


def bind_percentiles(limits, refresh):
    color_min, color_max = limits

    def change(slider):
        if color_min.value >= color_max.value:
            if slider is color_min:
                color_min.value = round(color_max.value - 0.1, 1)
            else:
                color_max.value = round(color_min.value + 0.1, 1)
            return
        refresh()

    for slider in limits:
        slider.on_value_change(lambda _, slider=slider: change(slider))


def set_polar_view(group, ri, ti, dr, angular_step):
    group.props(f'data-polar-view="{ri[0]} {ri[-1] + 1} {ti[0]} {ti[-1] + 1} {dr} {angular_step}"')


def render_polar_image(polar, cmap="viridis", *, log_scale=False,
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


def set_polar_source(element, data, colormap, scale, limits, *, renderer=render_polar_image):
    warning = ""
    try:
        image = renderer(data, cmap=colormap.value, log_scale=scale.value == "Log",
                         vmin_pct=limits[0].value, vmax_pct=limits[1].value)
    except ValueError as exc:
        warning = str(exc)
        image = renderer(data, cmap=colormap.value)
    with BytesIO() as buffer:
        image.save(buffer, format="PNG")
        encoded = b64encode(buffer.getbuffer()).decode("ascii")
    element.set_source(f"data:image/png;base64,{encoded}")
    return warning
