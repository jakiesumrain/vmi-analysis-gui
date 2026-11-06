"""Interactive view of the accumulation script's unchanged numerical output."""

import numpy as np
from plotly.subplots import make_subplots

from .centroid_plots import COLORMAPS, _heatmap, _image_axes, _style

ACCUMULATE_HEIGHT = 800


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
