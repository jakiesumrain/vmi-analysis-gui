"""Live plots for the acquisition publisher."""

from importlib.util import find_spec
from math import isfinite
import os
from urllib.parse import urlsplit

from nicegui import ui
import numpy as np
import plotly.graph_objects as go

from vmi_analysis.ion_live import GMD_HISTORY, LivePreview, gmd_histogram
from vmi_analysis.tof_bins import DEFAULT_BIN_SIZE_NS, tof_bin_edges

from .. import shared, theme
from ..centroid_plots import SERIES_COLORS, _style, plot_options
from .common import field_grid, num_input


JOB_NAME = "Live TPX3"
DETECTOR_ROW_HEIGHT = 580
GMD_TIMELINE_SECONDS = 60


def _fig(title, xlabel, ylabel, *, height=350):
    fig = _style(go.Figure(), title)
    fig.update_layout(height=height, margin=dict(l=64, r=42, t=62, b=60))
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=ylabel)
    return fig


def _endpoint(value):
    target = urlsplit(value.strip())
    if (target.scheme != "tcp" or not target.hostname or target.hostname == "*" or not target.port
            or target.username or target.password or target.path or target.query or target.fragment):
        raise ValueError("Enter a TCP endpoint such as tcp://acquisition-host:5556.")
    return value.strip()


def build(_jobs):
    theme.panel_header(JOB_NAME, "Watch detector hits, time of flight, and GMD readings during acquisition.")
    configured_endpoint = os.environ.get("VMI_TPX3_LIVE_ENDPOINT", "")
    if shared.SERVER and not configured_endpoint:
        ui.label("Ask the server administrator to configure VMI_TPX3_LIVE_ENDPOINT.").classes("vmi-subtitle")
        return

    receiver = None
    previous = None
    revision = 0
    charts = None
    gmd_limits = {}

    def refresh():
        nonlocal previous
        if receiver is None:
            return
        image, counts, readings, frames, trigger_issues, invalid, error = receiver.snapshot()
        key = frames, trigger_issues, invalid, error
        if key == previous:
            return
        previous = key
        if error:
            stop()
            status.set_text(f"Stream error: {error}")
            return
        status.set_text(f"{frames:,} frames received · {trigger_issues:,} without exactly one TDC1 rising edge "
                        f"· {invalid:,} invalid messages" if frames or invalid else "Waiting for TPX3 frames…")
        hit_map = _fig("Detector hit map", "X (pixels)", "Y (pixels)", height=DETECTOR_ROW_HEIGHT)
        hit_map.update_layout(margin=dict(l=50, r=52, autoexpand=False))
        hit_map.add_heatmap(z=image, colorscale="Viridis",
                            colorbar=dict(title="Hits", x=1, xanchor="left", thickness=14, xpad=5))
        detector_axis = dict(range=[-0.5, 255.5], constrain="domain", nticks=6)
        hit_map.update_xaxes(**detector_axis, constraintoward="right")
        hit_map.update_yaxes(**detector_axis, scaleanchor="x", scaleratio=1)
        spectrum = _fig("Time of flight", "TOF (ns)", "Hits / bin", height=DETECTOR_ROW_HEIGHT)
        spectrum.update_layout(margin=dict(l=56, r=14))
        spectrum.add_scatter(x=receiver.edges, y=np.r_[counts, 0], mode="lines", line_shape="hv")
        timeline = _fig("GMD readings versus time", "Elapsed acquisition time (s)", "GMD (nA)")
        timeline.update_layout(showlegend=True)
        times = readings[:, 0]
        right = max(GMD_TIMELINE_SECONDS, float(times.max())) if times.size else GMD_TIMELINE_SECONDS
        left = right - GMD_TIMELINE_SECONDS
        visible = times >= left
        timeline.update_xaxes(range=[left, right], fixedrange=True)
        histograms = []
        for index, name in enumerate(("GMD1", "GMD2")):
            values = readings[:, index + 1].copy()
            values[~np.isfinite(values)] = np.nan
            color = SERIES_COLORS[index]
            timeline.add_scatter(x=times[visible], y=values[visible], name=name, mode="lines+markers",
                                 line=dict(color=color), marker=dict(size=4), connectgaps=False)
            histogram = _fig(f"{name} histogram", f"{name} (nA)", "Frames / bin")
            try:
                bins, edges = gmd_histogram(values, *gmd_limits[name], name)
                histogram.add_bar(x=(edges[:-1] + edges[1:]) / 2, y=bins, width=np.diff(edges), marker_color=color)
            except ValueError as exc:
                histogram.add_annotation(text=str(exc), x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            histograms.append(histogram)
        for chart, fig in zip(charts, (hit_map, spectrum, timeline, *histograms), strict=True):
            fig.update_layout(uirevision=revision)
            chart.update_figure(plot_options(fig))

    def disconnect():
        if receiver is not None:
            receiver.stop()

    def stop():
        nonlocal receiver
        disconnect()
        receiver = None
        start_button.enable()
        stop_button.disable()
        reset_button.disable()
        for control in bin_controls:
            control.enable()
        status.set_text("Preview stopped.")

    def reset():
        nonlocal previous, revision
        if receiver is None:
            return
        receiver.reset()
        previous = None
        revision += 1
        refresh()

    def start():
        nonlocal receiver, charts, previous, revision
        if receiver is not None:
            return
        try:
            if find_spec("zmq") is None:
                raise ValueError("Install live monitoring with uv sync --extra live, then restart the server.")
            address = _endpoint(configured_endpoint if shared.SERVER else endpoint.value)
            low, high = minimum.value, maximum.value
            if low is None or high is None or not isfinite(low) or not isfinite(high) or low >= high:
                raise ValueError("Enter increasing, finite TOF limits.")
            if tof_width.value is None:
                raise ValueError("Enter a TOF bin size.")
            edges = tof_bin_edges(low, high, None, bin_size_ns=tof_width.value)
            limits = {name: tuple(control.value for control in fields) for name, fields in gmd_fields.items()}
            for name, settings in limits.items():
                gmd_histogram(np.empty(0), *settings, name)
        except ValueError as exc:
            ui.notify(str(exc), type="warning")
            return
        if charts is None:
            with output:
                with ui.element("div").classes("vmi-live-detector-row"):
                    plots = [ui.plotly(plot_options(go.Figure())).classes("w-full min-w-0") for _ in range(2)]
                timeline = ui.plotly(plot_options(go.Figure())).classes("w-full min-w-0")
                with field_grid():
                    histograms = [ui.plotly(plot_options(go.Figure())).classes("w-full min-w-0") for _ in range(2)]
                charts = (*plots, timeline, *histograms)
        gmd_limits.update(limits)
        receiver = LivePreview(address, edges)
        receiver.start()
        start_button.disable()
        stop_button.enable()
        reset_button.enable()
        for control in bin_controls:
            control.disable()
        previous = None
        revision += 1
        refresh()

    with theme.panel_card():
        endpoint = ui.input("Acquisition publisher", value=configured_endpoint,
                            placeholder="tcp://acquisition-host:5556").props("outlined dense").classes("w-full")
        if shared.SERVER:
            endpoint.disable()
        with field_grid(3):
            minimum = num_input("TOF minimum (ns)", 0, step="any")
            maximum = num_input("TOF maximum (ns)", 15_000, step="any")
            tof_width = num_input("TOF bin size (ns)", DEFAULT_BIN_SIZE_NS, step="any")
        gmd_fields = {}
        with ui.expansion("GMD histogram settings", icon="bar_chart").classes("w-full"):
            ui.label("Leave a limit blank to use the data range. Leave bin size blank for 40 bins. "
                     "These settings apply only to the GMD histograms.").classes("vmi-subtitle")
            for name in ("GMD1", "GMD2"):
                with field_grid(3):
                    gmd_fields[name] = tuple(num_input(f"{name} {label} (nA)", None, step="any")
                                             for label in ("minimum", "maximum", "bin size"))
        bin_controls = [minimum, maximum, tof_width, *(control for fields in gmd_fields.values() for control in fields)]
        with ui.row():
            start_button = ui.button("Start preview", icon="play_arrow", on_click=start)
            stop_button = ui.button("Stop", icon="stop", on_click=stop).props("flat")
            reset_button = ui.button("Reset counts", icon="restart_alt", on_click=reset).props("flat")
            stop_button.disable()
            reset_button.disable()
        status = ui.label("Preview stopped.").classes("vmi-caption")
        ui.label("Stop the preview to change binning. Starting again clears the accumulated counts."
                 ).classes("vmi-subtitle")
        ui.label(f"The hit map and TOF accumulate until reset. The GMD timeline follows the last "
                 f"{GMD_TIMELINE_SECONDS} s. GMD plots retain up to {GMD_HISTORY:,} recent frames."
                 ).classes("vmi-subtitle")
        ui.label("Live TOF uses Go4 pixel timing. For the same raw hit and trigger, the live TOF is expected "
                 "to be about 23.4 ns later than LUNA. Account for this offset when comparing peak positions "
                 "or transferring TOF windows.").classes("vmi-subtitle")
    output = ui.column().classes("w-full")
    shared.workspace_timer(0.5, refresh)
    ui.context.client.on_delete(disconnect)
