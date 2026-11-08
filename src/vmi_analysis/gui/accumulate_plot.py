"""Interactive view of the accumulation script's unchanged numerical output."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .centroid_plots import COLORMAPS, _heatmap, _image_axes, _style

ACCUMULATE_HEIGHT = 800
FRAME_HISTOGRAM_HEIGHT = 380


def accumulation_figure(values, *, title="Accumulated image", revision=None, colormap="Viridis"):
    values = np.asarray(values, dtype=float)
    trace = _heatmap(values, logarithmic=True)
    # Match the standalone script's Viridis colors and two-decade log range.
    positive = values[np.isfinite(values) & (values > 0)]
    maximum = float(np.percentile(values[np.isfinite(values)], 99.999999)) if positive.size else 1.0
    maximum = max(maximum, np.finfo(float).tiny)
    high = np.log10(maximum)
    ticks = list(range(int(np.ceil(high - 2)), int(np.floor(high)) + 1))
    trace.update(colorscale=colormap, zmin=high - 2, zmax=high,
                 colorbar={"title": {"text": "Weighted counts"}, "tickvals": ticks,
                           "ticktext": [f"10<sup>{tick}</sup>" for tick in ticks]})
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(trace, row=1, col=1)
    _image_axes(fig, values, 1, 1, "x")
    _style(fig, title + " · log scale")
    fig.update_layout(height=ACCUMULATE_HEIGHT, uirevision=revision)
    return fig


def frame_histogram_figure(values, threshold, *, title="Frame histogram",
                           bins=160, log_y=True):
    """Pixel-value histogram of one raw frame with the current threshold marked."""
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Frame contains no finite pixels")

    counts, edges = np.histogram(values, bins=int(bins))
    centers = (edges[1:] + edges[:-1]) / 2
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Bar(
        x=centers, y=counts, width=np.diff(edges),
        marker_color="#cc785c", marker_line_width=0,
        customdata=np.column_stack([edges[:-1], edges[1:]]),
        hovertemplate="Pixel value %{customdata[0]:.4g}–%{customdata[1]:.4g}"
                      "<br>%{y} pixels<extra></extra>"), row=1, col=1)
    fig.add_vline(x=float(threshold), line_dash="dash", line_color="#181715",
                  line_width=1.5, annotation_text=f"threshold {threshold:g}",
                  annotation_position="top right",
                  annotation_font={"color": "#181715", "size": 12})
    fig.update_xaxes(title_text="Pixel value (counts)", showgrid=False,
                     zeroline=False)
    fig.update_yaxes(title_text="Pixels", type="log" if log_y else "linear")
    _style(fig, title)
    fig.update_layout(height=FRAME_HISTOGRAM_HEIGHT,
                      margin={"l": 70, "r": 30, "t": 60, "b": 50})
    return fig
