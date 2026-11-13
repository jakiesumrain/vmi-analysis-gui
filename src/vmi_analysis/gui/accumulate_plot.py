"""Interactive view of the accumulation script's unchanged numerical output."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .centroid_plots import COLORMAPS, _heatmap, _image_axes, _style

ACCUMULATE_HEIGHT = 800
FRAME_HISTOGRAM_HEIGHT = 380
MAX_FRAME_HISTOGRAM_BINS = 65_536


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


def frame_histogram_figure(values, *, title="Raw frame histogram",
                           bins=160, log_y=True, x_min=None, x_max=None,
                           bin_width=None, x_label="Raw pixel value (counts)",
                           y_label="Pixels", count_label="pixels",
                           auto_bins=False, x_tickformat=None):
    """Plot a one-dimensional distribution (pixels or per-frame samples)."""
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Frame contains no finite pixels")

    lower, upper = float(values.min()), float(values.max())
    if lower == upper:
        lower -= 0.5
        upper += 0.5
    lower = lower if x_min is None else float(x_min)
    upper = upper if x_max is None else float(x_max)
    span = upper - lower
    if not np.isfinite(span) or span <= 0:
        raise ValueError("Histogram X min must be less than X max, with a finite range.")

    if bin_width is not None:
        width = float(bin_width)
        if not np.isfinite(width) or width <= 0:
            raise ValueError("Histogram bin width must be positive and finite.")
        bin_count = span / width
        if bin_count > MAX_FRAME_HISTOGRAM_BINS:
            raise ValueError(f"Increase the bin width or narrow the X range "
                             f"to use at most {MAX_FRAME_HISTOGRAM_BINS:,} bins.")
        left_edges = lower + width * np.arange(max(1, int(np.ceil(bin_count))))
        bins = np.append(left_edges[left_edges < upper], upper)
        if np.any(np.diff(bins) <= 0):
            raise ValueError("Histogram bin width is too small for the selected X range.")
    elif auto_bins:
        # No user-supplied size: let NumPy choose the bin edges.
        bins = np.histogram_bin_edges(values, bins="auto", range=(lower, upper))

    counts, edges = np.histogram(values, bins=bins, range=(lower, upper))
    widths = np.diff(edges)
    centers = edges[:-1] + widths / 2
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Bar(
        x=centers, y=counts, width=widths,
        marker_color="#cc785c", marker_line_width=0,
        customdata=np.column_stack([edges[:-1], edges[1:]]),
        hovertemplate="Bin %{customdata[0]:.4g}–%{customdata[1]:.4g}"
                      f"<br>%{{y}} {count_label}<extra></extra>"), row=1, col=1)
    fig.update_xaxes(title_text=x_label, showgrid=False, zeroline=False)
    if x_tickformat is not None:
        fig.update_xaxes(tickformat=x_tickformat)
    if x_min is not None or x_max is not None:
        fig.update_xaxes(range=[lower, upper])
    fig.update_yaxes(title_text=y_label, type="log" if log_y else "linear")
    _style(fig, title)
    fig.update_layout(height=FRAME_HISTOGRAM_HEIGHT,
                      margin={"l": 70, "r": 30, "t": 60, "b": 50})
    return fig
