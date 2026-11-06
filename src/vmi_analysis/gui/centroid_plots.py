"""Plotly views of exported Centroiding data, without rerunning analysis."""

from html import escape

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vmi_analysis.centroiding.plot_data import load_plot_data

PLOT_HEIGHT = 1200
COLORMAPS = ("Viridis", "Cividis", "Plasma", "Inferno", "Magma", "Turbo", "Gray", "Hot")

PLOT_CONFIG = {"scrollZoom": True, "displaylogo": False, "responsive": True,
               "doubleClick": "reset", "toImageButtonOptions": {"format": "png", "scale": 2}}


def _compact_values(values):
    """Smaller typed arrays only where every original value is preserved exactly."""
    array = np.asarray(values)
    finite = array[np.isfinite(array)]
    if finite.size == array.size and finite.size and finite.min() >= -32768 and finite.max() <= 32767:
        if np.equal(finite, np.trunc(finite)).all():
            return array.astype(np.int16)
    if finite.size and np.abs(finite).max() <= np.finfo(np.float32).max:
        reduced = array.astype(np.float32)
        if np.array_equal(array, reduced, equal_nan=True):
            return reduced
    return array


def _style(fig, title):
    fig.update_layout(title={"text": title, "font": {"size": 20}},
                      template="plotly_white", height=PLOT_HEIGHT, showlegend=False,
                      margin={"l": 65, "r": 70, "t": 90, "b": 55},
                      font={"family": "Inter, Arial, sans-serif", "color": "#3d3d3a", "size": 14},
                      paper_bgcolor="#faf9f5", dragmode="zoom", hovermode="closest")
    return fig


def _heatmap(data, *, logarithmic=False, colorbar=None):
    original = np.asarray(data, dtype=float)
    if original.ndim != 2 or original.size == 0:
        raise ValueError("Image data must be a nonempty 2D array")
    if logarithmic:
        finite = original[np.isfinite(original) & (original > 0)]
        maximum = float(finite.max()) if finite.size else 1.0
        high = np.log10(maximum)
        z = np.full(original.shape, np.nan, dtype=np.float32)
        valid = np.isfinite(original) & (original > 0)
        z[valid] = np.log10(original[valid])
        ticks = list(range(int(np.ceil(high - 4)), int(np.floor(high)) + 1))
        return go.Heatmap(z=z, customdata=_compact_values(original), zmin=high - 4, zmax=high,
                          colorscale="Turbo", zsmooth=False, hoverongaps=False,
                          hovertemplate="X: %{x:.0f} px<br>Y: %{y:.0f} px<br>Counts: %{customdata:.6g}<extra></extra>",
                          colorbar={"title": {"text": "Counts"}, "tickvals": ticks,
                                    "ticktext": [f"10<sup>{tick}</sup>" for tick in ticks], **(colorbar or {})})
    finite = original[np.isfinite(original)]
    maximum = float(np.percentile(finite, 99.9)) if finite.size else 1.0
    return go.Heatmap(z=_compact_values(original), zmin=0, zmax=max(maximum, 1e-12), colorscale="Gray",
                      showscale=False, zsmooth=False, hoverongaps=False,
                      hovertemplate="X: %{x:.0f} px<br>Y: %{y:.0f} px<br>Corrected signal: %{z:.6g}<extra></extra>")


def _image_axes(fig, data, row, col, anchor):
    ny, nx = data.shape
    fig.add_shape(type="rect", xref=f"{anchor} domain", yref=f"{anchor.replace('x', 'y')} domain",
                  x0=0, y0=0, x1=1, y1=1, fillcolor="black", line_width=0, layer="below")
    fig.update_xaxes(title_text="X (pixels)", range=[-.5, nx - .5],
                     showgrid=False, zeroline=False, constrain="domain", row=row, col=col)
    fig.update_yaxes(title_text="Y (pixels)", range=[-.5, ny - .5],
                     showgrid=False, zeroline=False, scaleanchor=anchor, scaleratio=1,
                     constrain="domain", row=row, col=col)


def _histogram(fig, values, bins, title, row, col, color, reference=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    counts, edges = np.histogram(values, bins=int(bins))
    fig.add_trace(go.Bar(x=(edges[1:] + edges[:-1]) / 2, y=counts, width=np.diff(edges),
                         customdata=np.column_stack([edges[:-1], edges[1:]]),
                         marker_color=color, marker_line_width=.5,
                         hovertemplate="Bin: %{customdata[0]:.5g}–%{customdata[1]:.5g}<br>Count: %{y}<extra></extra>"),
                  row=row, col=col)
    if reference is not None:
        fig.add_vline(x=reference, line_dash="dash", line_color="#c64545", row=row, col=col)
    fig.update_xaxes(title_text=title, row=row, col=col)
    fig.update_yaxes(title_text="Count", rangemode="tozero", row=row, col=col)


def _message(fig, axis_number, text):
    suffix = str(axis_number) if axis_number > 1 else ""
    xdomain = fig.layout[f"xaxis{suffix}"].domain
    ydomain = fig.layout[f"yaxis{suffix}"].domain
    fig.add_annotation(x=sum(xdomain) / 2, y=sum(ydomain) / 2, xref="paper", yref="paper",
                       text=text, showarrow=False)


def tuning_figure(path):
    meta, arrays = load_plot_data(path)
    frame = arrays["frame"]
    precision = (arrays["lpx"] + arrays["lpy"]) / 2
    fig = make_subplots(rows=2, cols=2, horizontal_spacing=.12, vertical_spacing=.12,
                        subplot_titles=(f"Corrected frame · {meta['n_events']} localizations",
                                        "Photon distribution", "Net gradient distribution", "Localization precision"))
    fig.add_trace(_heatmap(frame), row=1, col=1)
    fig.add_trace(go.Scattergl(x=arrays["x"], y=arrays["y"], mode="markers",
                               customdata=np.column_stack([arrays["photons"], arrays["net_gradient"], precision]),
                               marker={"symbol": "circle-open", "color": "red", "size": 10, "line": {"width": 1}},
                               hovertemplate="Localization<br>X: %{x:.3f} px<br>Y: %{y:.3f} px"
                                             "<br>Photons: %{customdata[0]:.6g}<br>Net gradient: %{customdata[1]:.6g}"
                                             "<br>Precision: %{customdata[2]:.4g} px<extra></extra>"), row=1, col=1)
    _image_axes(fig, frame, 1, 1, "x")
    photons, gradient = arrays["photons"], arrays["net_gradient"]
    precision = precision[np.isfinite(precision)]
    _histogram(fig, photons, meta["bins_photons"], "Photons (integrated intensity)", 1, 2, "#cc785c",
               np.median(photons) if photons.size else None)
    _histogram(fig, gradient, meta["bins_net_gradient"], "Net gradient", 2, 1, "#5db872", meta["min_net_gradient"])
    _histogram(fig, precision, meta["bins_precision"], "Precision (pixels)", 2, 2, "#e8a55a",
               np.median(precision) if precision.size else None)
    return _style(fig, f"Tuning check · frame {meta['frame_index']} · {escape(meta['fitting_method'].upper())}")


def summary_figure(path):
    meta, arrays = load_plot_data(path)
    raw, abel = arrays["raw"], arrays["abel"]
    fig = make_subplots(rows=2, cols=2, horizontal_spacing=.14, vertical_spacing=.12,
                        subplot_titles=("2D VMI image (raw, log)", "3D VMI (Abel inverted, log)",
                                        "Photoelectron spectrum", "Analysis summary"))
    fig.add_trace(_heatmap(raw, logarithmic=True, colorbar={"x": .44, "y": .78, "len": .44, "thickness": 12}), row=1, col=1)
    _image_axes(fig, raw, 1, 1, "x")
    if abel.size:
        fig.add_trace(_heatmap(abel, logarithmic=True, colorbar={"x": 1.01, "y": .78, "len": .44, "thickness": 12}), row=1, col=2)
        _image_axes(fig, abel, 1, 2, "x2")
    else:
        _message(fig, 2, "Abel inversion not available for this run")
    energy, intensity = arrays["energy"], arrays["intensity"]
    if energy.size and energy.shape == intensity.shape:
        positive = intensity > 0
        fig.add_trace(go.Scatter(x=energy[positive], y=intensity[positive], mode="lines",
                                 line={"color": "#c64545", "width": 1},
                                 hovertemplate="X: %{x:.6g}<br>Intensity: %{y:.6g}<extra></extra>"), row=2, col=1)
        xlo, xhi = meta.get("x_min", -1), meta.get("x_max", -1)
        ylo, yhi = meta.get("y_min", -1), meta.get("y_max", -1)
        fig.update_xaxes(range=[xlo if xlo >= 0 else 0, xhi if xhi > 0 else float(energy.max())], row=2, col=1)
        fig.update_yaxes(range=[ylo if ylo >= 0 else 0, yhi] if yhi > 0 else None, rangemode="tozero", row=2, col=1)
    else:
        _message(fig, 3, "Energy spectrum not available for this run")
    fig.update_xaxes(title_text="Kinetic energy (eV)" if meta.get("alpha") is not None else "Squared radius (pixel²)", row=2, col=1)
    fig.update_yaxes(title_text="Intensity (arb. units)", row=2, col=1)
    fig.update_xaxes(visible=False, row=2, col=2)
    fig.update_yaxes(visible=False, row=2, col=2)
    lines = [f"Frames: {meta['n_frames']:,}", f"Events: {meta['n_events']:,}",
             f"Events/frame: {meta['n_events'] / max(meta['n_frames'], 1):.1f}",
             f"Box size: {meta['box_size']} pixels", f"Min net gradient: {meta['min_net_gradient']}",
             f"α: {meta['alpha'] if meta.get('alpha') is not None else 'uncalibrated'}",
             *(f"{label}: {'on' if meta.get(key) else 'off'}" for key, label in (
                 ("dark_subtraction", "Dark subtraction"), ("flat_field", "Flat field"), ("circular_mask", "Circular mask")))]
    fig.add_annotation(x=fig.layout.xaxis4.domain[0], y=fig.layout.yaxis4.domain[1],
                       xref="paper", yref="paper", text="<br>".join(escape(line) for line in lines),
                       align="left", xanchor="left", yanchor="top", showarrow=False)
    return _style(fig, "Final summary · interactive VMI and spectrum")


def plot_options(figure):
    return {**figure.to_plotly_json(), "config": PLOT_CONFIG.copy()}
