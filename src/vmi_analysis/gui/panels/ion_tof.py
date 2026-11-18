"""Inspect TOF spectra and reproduce the standalone Xe yield calculation."""

import json
from math import isfinite
from pathlib import Path
import sys

from nicegui import ui
import numpy as np
import plotly.graph_objects as go

from vmi_analysis.tpx3_analysis import FRAME_SHIFT, GMD2_DEFAULT
from vmi_analysis.tof_bins import DEFAULT_BIN_SIZE_NS

from .. import shared, theme
from ..centroid_plots import _style, plot_options
from ..jobs import Job
from . import ion_unpack
from .common import JobView, action_bar, field_grid, num_input, path_input, persist


JOB_NAME = "Ion TOF"
FILTER_JOB_NAME = "GMD-filtered TOF"


def _gmd_histogram(channel, values):
    counts, edges = np.histogram(values[np.isfinite(values)], bins=80)
    fig = go.Figure(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, width=np.diff(edges)))
    _style(fig, f"{channel} histogram")
    fig.update_layout(height=340)
    fig.update_xaxes(title_text=f"{channel} (nA)")
    fig.update_yaxes(title_text="Frames / bin", rangemode="tozero")
    return fig


def _diagnostics(gmd1, gmd2, correlations):
    scatter = go.Figure(go.Scattergl(x=gmd2, y=gmd1, mode="markers", marker=dict(size=4, opacity=0.6)))
    _style(scatter, "GMD1 versus GMD2")
    scatter.update_layout(height=340)
    scatter.update_xaxes(title_text="GMD2 (nA)")
    scatter.update_yaxes(title_text="GMD1 (nA)")
    correlation = go.Figure()
    for column, name, dash in ((1, "Pearson", "solid"), (2, "Spearman", "dash")):
        correlation.add_scatter(x=correlations[:, 0], y=correlations[:, column], mode="lines+markers", name=name,
                                line=dict(dash=dash), customdata=correlations[:, 3],
                                hovertemplate="d = %{x}<br>r = %{y:.3f}<br>%{customdata:,.0f} pairs")
    _style(correlation, "GMD2 versus pixel-packet count")
    correlation.update_layout(height=340, showlegend=True, legend=dict(orientation="h", y=-0.24))
    correlation.update_xaxes(title_text="Diagnostic lag d (frames)", dtick=1)
    correlation.update_yaxes(title_text="Correlation coefficient", range=[-1.05, 1.05])
    with ui.expansion("GMD diagnostic plots", icon="insights", value=False).classes("w-full"):
        with ui.element("div").classes("grid grid-cols-1 md:grid-cols-2 gap-4 w-full"):
            for fig in (scatter, _gmd_histogram("GMD1", gmd1), _gmd_histogram("GMD2", gmd2), correlation):
                ui.plotly(plot_options(fig)).classes("w-full min-w-0")
        ui.label("The standalone diagnostic compares GMD2 row j with frame j + d. "
                 "The standalone yield filter pairs row j with frame j − shift.").classes("vmi-subtitle")


def _show_spectrum(data, *, filtered):
    summary = json.loads(data["yield_summary"].item())
    edges = data["edges"]
    fig = go.Figure()
    fig.add_scatter(x=edges, y=np.r_[data["unfiltered"], 0], mode="lines", line_shape="hv",
                    name="All decoded pixel hits")
    if filtered:
        fig.add_scatter(x=edges, y=np.r_[data["filtered"], 0], mode="lines", line_shape="hv",
                        name="GMD2-selected hits", fill="tozeroy")
    if summary["valid"]:
        for window, color in (("win1", "#1e40af"), ("win2", "#c2410c")):
            low, high = summary[window]
            fig.add_vrect(x0=low, x1=high, fillcolor=color, opacity=0.14, line_width=0, layer="below")
    _style(fig, "GMD2-filtered time-of-flight spectrum" if filtered else "Time-of-flight spectrum")
    fig.update_layout(height=420, showlegend=filtered, legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(title_text="Time of flight (ns)", range=[edges[0], edges[-1]])
    fig.update_yaxes(title_text="Pixel hits / bin", rangemode="tozero")
    ui.plotly(plot_options(fig)).classes("w-full")
    ui.label(f"{int(data['selected_frames']):,} of {int(data['total_frames']):,} camera frames included"
             ).classes("vmi-caption")
    if not summary["valid"]:
        reason = summary.get("reason", "No frames passed the GMD2 filter.")
        ui.label(f"Yield unavailable. {reason}").classes("vmi-subtitle")
        return
    rows = []
    for ion, i in (("Xe⁺", 1), ("Xe²⁺", 2)):
        prefix = f"xe{i}"
        low, high = summary[f"win{i}"]
        mu = summary[f"{prefix}_mu"]
        sigma = summary[f"{prefix}_sig"]
        rows.append(dict(ion=ion, center=f"{mu:.8g}" if isfinite(mu) else "—",
                         sigma=f"{sigma:.8g}" if isfinite(sigma) else "—",
                         window=f"[{low:.8g}, {high:.8g})", count=summary[prefix],
                         yield_value=f"{summary[f'y{i}']:.8g}", error=f"{summary[f'e{i}']:.8g}"))
    columns = [("ion", "Ion"), ("center", "Center (ns)"), ("sigma", "σ (ns)"), ("window", "TOF window (ns)"),
               ("count", "Observed hits"), ("yield_value", "Hits per frame"), ("error", "Yield uncertainty")]
    ui.table(columns=[dict(name=key, field=key, label=label) for key, label in columns],
             rows=rows, row_key="ion").classes("w-full")
    ui.label(f"Xe⁺/Xe²⁺ · {summary['ratio']:.8g} ± {summary['ratio_err']:.8g}").classes("vmi-caption")
    if summary["xe2"] == 0:
        ui.label("Xe²⁺ has zero hits. The standalone script uses a small denominator in the reported ratio."
                 ).classes("vmi-subtitle")


def build(jobs):
    readings = None

    def use_dump():
        completed = [job for job in jobs.jobs() if job.name == ion_unpack.JOB_NAME and job.status == "done"]
        if not completed:
            ui.notify("Complete a TPX3 dump first, or browse for a saved verification dump.", type="warning")
            return
        dump.value = max(completed, key=lambda job: job.finished).extra["output"]

    def submit(filtering=False):
        nonlocal readings
        try:
            parent = view.job
            if filtering:
                if readings is None or parent is None or parent.status != "done":
                    raise ValueError("Plot TOF from a verification dump first.")
                if filtered_view.job and filtered_view.job.status in ("queued", "running"):
                    raise ValueError("Wait for the current filter or stop it before applying a new range.")
                source, width, fixed = (parent.extra[key] for key in ("source", "bin_size_ns", "fixed"))
                lo, hi, offset = low.value, high.value, shift.value
                if lo is None or hi is None or not isfinite(lo) or not isfinite(hi) or lo > hi:
                    raise ValueError("Enter finite GMD2 limits with minimum at or below maximum.")
                if offset is None or not isfinite(offset) or not float(offset).is_integer():
                    raise ValueError("Enter a whole-number frame shift.")
            else:
                if not dump.value.strip():
                    raise ValueError("Choose an HDF5 file from TPX3 Dump.")
                source = shared.checked_path(dump.value.strip())
                width, fixed = bin_size.value, fixed_window.value
                if not source.is_file():
                    raise ValueError("Choose an existing verification dump.")
                if width is None or not isfinite(width) or width <= 0:
                    raise ValueError("TOF bin size must be finite and greater than zero.")
            active_view = filtered_view if filtering else view
            job = Job(FILTER_JOB_NAME if filtering else JOB_NAME, cwd=str(shared.workspace_root()))
            output = shared.generated_dir() / f"ion-verify-{job.id}.npz"
            job.cmd = [sys.executable, "-m", "vmi_analysis.ion_verification", "analyze",
                       "--input", str(source), "--output", str(output), "--bin-size", str(width)]
            if fixed:
                job.cmd.append("--fixed-window")
            job.extra.update(result=str(output), source=str(source), bin_size_ns=width, fixed=fixed)
            if filtering:
                job.cmd += [f"--gmd-min={lo}", f"--gmd-max={hi}", "--shift", str(int(offset)),
                            "--expected-dump-id", parent.extra["dump_id"]]
                job.extra["parent"] = parent.id
                job.extra["selection"] = f"GMD2 {lo:g}–{hi:g} nA · yield shift {int(offset):+d}"
            submitted = jobs.submit(job)
            if not filtering:
                readings = None
                diagnostics.clear()
                filter_card.set_visibility(False)
            active_view.attach(submitted)
        except (OSError, ValueError) as exc:
            ui.notify(str(exc), type="warning")

    def show_unfiltered(job):
        nonlocal readings
        if job.status != "done":
            return
        with np.load(shared.checked_path(job.extra["result"]), allow_pickle=False) as data:
            ui.label(Path(job.extra["source"]).name).classes("vmi-caption")
            _show_spectrum(data, filtered=False)
            job.extra["dump_id"] = data["dump_id"].item()
            readings = data["gmd2"]
            diagnostics.clear()
            with diagnostics:
                _diagnostics(data["gmd1"], readings, data["correlations"])
        filter_card.set_visibility(True)
        update_histogram()

    def show_filtered(job):
        if job.status != "done" or view.job is None or view.job.id != job.extra["parent"]:
            return
        with np.load(shared.checked_path(job.extra["result"]), allow_pickle=False) as data:
            ui.label(job.extra["selection"]).classes("vmi-caption")
            _show_spectrum(data, filtered=True)

    def update_histogram():
        if readings is None:
            return
        fig = _gmd_histogram("GMD2", readings)
        lo, hi = low.value, high.value
        if lo is not None and hi is not None and isfinite(lo) and isfinite(hi) and lo < hi:
            fig.add_vrect(x0=lo, x1=hi, opacity=0.15, fillcolor="#1e40af", line_width=0)
        fig.update_layout(dragmode="select" if mode.value == "Select GMD2 range" else "zoom",
                          selectdirection="h", uirevision=view.job.id, selections=[], selectionrevision=f"{lo}:{hi}")
        histogram.update_figure(plot_options(fig))

    def select_range(event):
        if mode.value == "Select GMD2 range" and (limits := (event.args or {}).get("x")) and len(limits) == 2:
            low.value, high.value = sorted(limits)
            submit(filtering=True)

    view = JobView(jobs, job_name=JOB_NAME, results_builder=show_unfiltered)
    filtered_view = JobView(jobs, job_name=FILTER_JOB_NAME, results_builder=show_filtered)
    theme.panel_header(JOB_NAME, "Compare TOF spectra and Xe yields with the standalone analysis.", view=view)
    with theme.panel_card():
        dump = persist(path_input("TPX3 verification HDF5", browse="file", filters=[".h5", ".hdf5"]), "ion_tof.dump")
        ui.button("Use latest dump", icon="history", on_click=use_dump).props("flat")
        with field_grid():
            bin_size = persist(num_input("TOF bin size (ns)", DEFAULT_BIN_SIZE_NS, step="any"),
                               "ion_tof.bin_size_ns")
            fixed_window = persist(ui.checkbox("Use standalone fixed Xe windows"), "ion_tof.fixed_window")
        ui.label("By default the standalone Gaussian fit uses all decoded hits to set the Xe windows. "
                 "GMD2 selects the frames used for counting.").classes("vmi-subtitle")
        with action_bar(view, "Plot TOF", on_click=submit):
            pass
    console_area = ui.column().classes("w-full")
    diagnostics = ui.column().classes("w-full")
    with theme.panel_card() as filter_card:
        filter_card.set_visibility(False)
        theme.form_section("GMD2 range")
        mode = ui.toggle(["Zoom", "Select GMD2 range"], value="Zoom", on_change=lambda _: update_histogram())
        histogram = ui.plotly(go.Figure()).classes("w-full")
        histogram.on("plotly_selected", select_range, js_handler="event => emit(event?.range ?? null)")
        with field_grid(3):
            low = persist(num_input("GMD2 min (nA)", GMD2_DEFAULT[0], step="any"), "ion_tof.gmd_min")
            high = persist(num_input("GMD2 max (nA)", GMD2_DEFAULT[1], step="any"), "ion_tof.gmd_max")
            shift = persist(num_input("Standalone yield shift (frames)", FRAME_SHIFT, step=1), "ion_tof.shift")
        for field in (low, high):
            field.on_value_change(lambda _: update_histogram())
        ui.label("Yield pairs GMD2 row j with camera frame j − shift. At shift −1 it uses frame j + 1. "
                 "The GMD2 limits are inclusive.").classes("vmi-subtitle")
        filtered_view.run_button("Apply GMD2 range", on_click=lambda: submit(filtering=True))
    with console_area:
        view.place_console()
    with filter_card:
        filtered_view.place_console()
