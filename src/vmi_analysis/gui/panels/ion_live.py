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
from ..agent_panels import PanelInspection, PanelOperation
from ..centroid_plots import SERIES_COLORS, _style, plot_options
from ..workflow import register_value
from .common import field_grid, num_input, persist


JOB_NAME = "Live TPX3"
INSPECT_OPERATION = "live_tpx3"
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


def _valid_endpoint_setting(value):
    if not isinstance(value, str):
        return False
    try:
        if value.strip():
            _endpoint(value)
    except ValueError:
        return False
    return True


def agent_operations():
    return (PanelOperation(INSPECT_OPERATION, {
        "panel": JOB_NAME, "manual": JOB_NAME, "parameters": {}, "action": "inspect",
        "summary": "Read the current tab's live preview counters, accumulated hit map and TOF summaries, "
                   "and recent GMD readings. Open Ask Agent about preview in Live TPX3.",
    }, None, ()),)


def _preview_summary(receiver):
    image, counts, readings, frames, trigger_issues, invalid, error = receiver.snapshot()
    pixel_hits = int(image.sum())
    peak_pixel = None
    if pixel_hits:
        y, x = np.unravel_index(image.argmax(), image.shape)
        peak_pixel = {"x_px": int(x), "y_px": int(y), "hits": int(image[y, x])}
    bins = np.flatnonzero(counts)
    if len(bins) > 3:
        bins = bins[np.argpartition(counts[bins], -3)[-3:]]
    busiest = [{"minimum_ns": float(receiver.edges[i]), "maximum_ns": float(receiver.edges[i + 1]),
                "hits": int(counts[i])} for i in sorted(bins, key=lambda i: (-counts[i], i))]
    times = readings[:, 0]
    right = max(GMD_TIMELINE_SECONDS, float(times.max())) if times.size else GMD_TIMELINE_SECONDS
    recent = readings[times >= right - GMD_TIMELINE_SECONDS]
    gmd = {}
    for column, name in enumerate(("GMD1", "GMD2"), 1):
        values = recent[:, column]
        finite = values[np.isfinite(values)]
        gmd[name] = {
            "valid_recent_frames": len(finite),
            "latest_na": float(values[-1]) if len(values) and np.isfinite(values[-1]) else None,
        }
        if len(finite):
            gmd[name].update(minimum_na=float(finite.min()), maximum_na=float(finite.max()), mean_na=float(finite.mean()))
    return {
        "status": "stream error" if error else "preview active" if frames else "waiting for frames",
        "error": error,
        "frames_received": int(frames), "frames_without_exactly_one_trigger": int(trigger_issues),
        "invalid_messages": int(invalid),
        "detector": {"pixel_hits": pixel_hits, "occupied_pixels": int(np.count_nonzero(image)),
                     "busiest_pixel": peak_pixel},
        "tof": {"hits_in_range": int(counts.sum()), "minimum_ns": float(receiver.edges[0]),
                "maximum_ns": float(receiver.edges[-1]), "bins": len(counts), "busiest_bins": busiest,
                "timing": "Go4 convention. About 23.4 ns later than LUNA for the same hit and trigger."},
        "gmd": {"window_s": GMD_TIMELINE_SECONDS, "recent_frames": len(recent),
                "latest_elapsed_s": float(readings[-1, 0]) if len(readings) else None, **gmd},
    }


def build(_jobs, *, ask_agent=None):
    theme.panel_header(JOB_NAME, "Watch detector hits, time of flight, and GMD readings during acquisition.")
    configured_endpoint = os.environ.get("VMI_TPX3_LIVE_ENDPOINT", "")
    if shared.SERVER and not configured_endpoint:
        ui.label("Ask the server administrator to configure VMI_TPX3_LIVE_ENDPOINT.").classes("vmi-subtitle")
        return

    def number(label, key, value=None):
        return persist(num_input(label, value, step="any"), f"ion_live.{key}")

    receiver = None
    previous = None
    revision = 0
    charts = None
    gmd_limits = {}
    last_error = None

    def read_preview():
        if receiver is None:
            return {"status": "stream error" if last_error else "stopped", "error": last_error}
        return _preview_summary(receiver)

    def open_agent():
        ask_agent(PanelInspection(panel=JOB_NAME, operation=INSPECT_OPERATION,
                                  label="Live TPX3 preview", read_snapshot=read_preview))

    def refresh():
        nonlocal previous, last_error
        if receiver is None:
            return
        image, counts, readings, frames, trigger_issues, invalid, error = receiver.snapshot()
        key = frames, trigger_issues, invalid, error
        if key == previous:
            return
        previous = key
        if error:
            stop()
            last_error = error
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
        nonlocal receiver, charts, previous, revision, last_error
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
        last_error = None
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
        else:
            store = shared.settings_store()
            endpoint_key = "ion_live.endpoint"
            default_endpoint = configured_endpoint if _valid_endpoint_setting(configured_endpoint) else ""

            def saved_endpoint():
                value = store.get(endpoint_key, default_endpoint)
                return value if _valid_endpoint_setting(value) else default_endpoint

            def save_endpoint(event):
                if _valid_endpoint_setting(event.value):
                    store[endpoint_key] = event.value.strip()

            endpoint.value = saved_endpoint()
            endpoint.on_value_change(save_endpoint)
            register_value(endpoint_key, saved_endpoint, _valid_endpoint_setting)
        with field_grid(3):
            minimum = number("TOF minimum (ns)", "tof_min", 0)
            maximum = number("TOF maximum (ns)", "tof_max", 15_000)
            tof_width = number("TOF bin size (ns)", "tof_bin_width", DEFAULT_BIN_SIZE_NS)
        gmd_fields = {}
        with ui.expansion("GMD histogram settings", icon="bar_chart").classes("w-full"):
            ui.label("Leave a limit blank to use the data range. Leave bin size blank for 40 bins. "
                     "These settings apply only to the GMD histograms.").classes("vmi-subtitle")
            for name in ("GMD1", "GMD2"):
                with field_grid(3):
                    gmd_fields[name] = tuple(number(f"{name} {label} (nA)", f"{name.lower()}_{key}")
                                             for label, key in (("minimum", "min"), ("maximum", "max"),
                                                                ("bin size", "bin_width")))
        bin_controls = [minimum, maximum, tof_width, *(control for fields in gmd_fields.values() for control in fields)]
        with ui.row():
            start_button = ui.button("Start preview", icon="play_arrow", on_click=start)
            stop_button = ui.button("Stop", icon="stop", on_click=stop).props("flat")
            reset_button = ui.button("Reset counts", icon="restart_alt", on_click=reset).props("flat")
            if ask_agent:
                ui.button("Ask Agent about preview", icon="smart_toy", on_click=open_agent).props("flat")
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
