"""Unfiltered time-of-flight spectra from LUNA Timepix3 dumps."""

import asyncio
from math import isfinite
import sys

from nicegui import ui
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vmi_analysis.ion_tof import DEFAULT_BINS, DEFAULT_REFERENCE, MAX_BINS, REFERENCES, gmd_diagnostics

from .. import shared, theme
from ..centroid_plots import _style, plot_options
from ..jobs import Job, ResourceBusyError
from .common import JobView, action_bar, field_grid, num_input, path_input, persist


JOB_NAME = "Ion TOF"
SOURCES = {"pixels": "Raw pixel hits", "clusters": "Clustered events", "both": "Both"}
REFERENCE_LABELS = {name: name.replace("Rising", " rising").replace("Falling", " falling")
                    for name in REFERENCES}


def _gmd_results(result, sources):
    theme.form_section("GMD alignment diagnostic")
    ui.label("The diagnostic compares GMD2 from row j with counts from camera frame j + d. "
             "At d = −1, a camera frame uses the next row’s GMD2. At d = +1, it uses the previous row’s GMD2.")
    issue = str(result["gmd_alignment_issue"])
    if issue:
        ui.label(f"Alignment unavailable. {issue}").classes("vmi-subtitle")
    else:
        ui.label("Uses all hits assigned to each trigger, regardless of the TOF plot range. "
                 "Pairs require positive GMD2 and positive counts, with at least 10 pairs per shift."
                 ).classes("vmi-subtitle")
        shifts = result["gmd_shifts"]
        fig = go.Figure()
        colors = ("#1e40af", "#c2410c")
        for index, source in enumerate(sources):
            coefficients = result[f"{source}_gmd_correlation"]
            pairs = result[f"{source}_gmd_pairs"]
            for column, (statistic, dash) in enumerate((("Pearson", "solid"), ("Spearman", "dash"))):
                fig.add_trace(go.Scatter(
                    x=shifts, y=coefficients[:, column], customdata=pairs,
                    mode="lines+markers", line=dict(dash=dash, color=colors[index]),
                    name=f"{SOURCES[source]} · {statistic}",
                    hovertemplate="d = %{x:+d}<br>Correlation %{y:.3f}<br>%{customdata:,} pairs",
                ))
            pearson = coefficients[:, 0]
            if np.any(np.isfinite(pearson)):
                best = int(np.nanargmax(pearson))
                ties = np.count_nonzero(pearson == pearson[best])
                detail = (f"highest Pearson r = {pearson[best]:.3f} at d = {shifts[best]:+d}"
                          if ties == 1 else f"highest Pearson r = {pearson[best]:.3f}, tied at {ties} shifts")
            else:
                detail = "no correlation could be calculated"
            ui.label(f"{SOURCES[source]} · {detail}").classes("vmi-caption")
        _style(fig, "GMD2 versus counts per frame")
        fig.update_layout(height=420, showlegend=True, legend=dict(orientation="h", y=-0.22),
                          margin=dict(b=115))
        fig.update_xaxes(title_text="Shift d (frames)", tickmode="array", tickvals=shifts)
        fig.update_yaxes(title_text="Correlation coefficient", range=[-1.05, 1.05])
        fig.add_vline(x=0, line_dash="dot", line_color="#94a3b8")
        ui.plotly(plot_options(fig)).classes("w-full")

    theme.form_section("GMD readings")
    first, second = result["gmd1_na"], result["gmd2_na"]
    valid = np.isfinite(first) & np.isfinite(second)
    fig = go.Figure(go.Scattergl(x=second[valid], y=first[valid], mode="markers",
                                marker=dict(size=4, opacity=0.55)))
    _style(fig, "GMD1 versus GMD2")
    fig.update_layout(height=400)
    fig.update_xaxes(title_text="GMD2 (nA)")
    fig.update_yaxes(title_text="GMD1 (nA)")
    ui.plotly(plot_options(fig)).classes("w-full")
    with field_grid():
        for label, values in (("GMD1", first), ("GMD2", second)):
            counts, edges = np.histogram(values[np.isfinite(values)], bins=80)
            plot = go.Figure(go.Scatter(x=edges, y=np.r_[counts, 0], mode="lines", line_shape="hv"))
            _style(plot, f"{label} histogram")
            plot.update_layout(height=330)
            plot.update_xaxes(title_text=f"{label} (nA)")
            plot.update_yaxes(title_text="Frames / bin", rangemode="tozero")
            ui.plotly(plot_options(plot)).classes("w-full")


def _load_gmd_diagnostics(result_path, metadata):
    with np.load(result_path, allow_pickle=False) as result:
        counts = {source: result[f"{source}_shot_counts"] for source in ("pixels", "clusters")
                  if f"{source}_shot_counts" in result}
    if not counts:
        raise ValueError("Plot TOF again to save the counts per trigger for GMD diagnostics.")
    return gmd_diagnostics(metadata, counts), list(counts)


def _results(job):
    if job.status != "done":
        return
    result_path = shared.checked_path(job.extra["histogram"])
    with np.load(result_path, allow_pickle=False) as result:
        edges = result["edges"]
        sources = [name for name in ("pixels", "clusters") if name in result]
        ui.label(f'{job.extra["input_name"]} · {REFERENCE_LABELS[job.extra["reference"]]} '
                 f'· {int(result["triggers"]):,} triggers').classes("vmi-caption break-all")
        fig = make_subplots(rows=len(sources), cols=1, shared_xaxes=True,
                            subplot_titles=[SOURCES[source] for source in sources])
        for row, source in enumerate(sources, start=1):
            counts = result[source]
            ui.label(f'{SOURCES[source]} · {int(counts.sum()):,} plotted · '
                     f'{int(result[f"{source}_unmatched"]):,} with no preceding trigger · '
                     f'{int(result[f"{source}_outside"]):,} outside the TOF range').classes("vmi-subtitle")
            fig.add_trace(go.Scatter(
                x=edges, y=np.r_[counts, 0], mode="lines", line_shape="hv",
                name=SOURCES[source],
                hovertemplate="TOF %{x:.4~g} ns<br>Count %{y:,}<extra></extra>",
            ), row=row, col=1)
            fig.update_yaxes(title_text="Counts / bin", rangemode="tozero", row=row, col=1)
        _style(fig, "Time-of-flight spectrum")
        fig.update_layout(height=400 if len(sources) == 1 else 650)
        fig.update_xaxes(title_text="Time of flight (ns)", row=len(sources), col=1)
        ui.plotly(plot_options(fig)).classes("w-full")


def build(jobs):
    view = JobView(jobs, job_name=JOB_NAME, results_builder=_results)
    theme.panel_header(JOB_NAME, "Plot unfiltered time-of-flight spectra from a Timepix3 dump.", view=view)

    def save(widget, key):
        return persist(widget, f"ion_tof.{key}")

    def run():
        try:
            if not input_file.value.strip():
                raise ValueError("Choose a LUNA HDF5 file.")
            source_file = shared.checked_path(input_file.value.strip())
            if not source_file.is_file():
                raise ValueError("The selected HDF5 file does not exist.")
            count = bins.value
            if count is None or not float(count).is_integer() or not 1 <= count <= MAX_BINS:
                raise ValueError(f"Choose a whole number of bins between 1 and {MAX_BINS:,}.")
            low = minimum.value
            high = maximum.value
            if low is None or not isfinite(low) or low < 0:
                raise ValueError("TOF minimum must be finite and at least zero.")
            if high is not None and (not isfinite(high) or high <= low):
                raise ValueError("TOF maximum must be greater than the minimum.")
            job = Job(JOB_NAME, cwd=str(shared.workspace_root()))
            result = shared.generated_dir() / f"ion-tof-{job.id}.npz"
            job.cmd = [sys.executable, "-m", "vmi_analysis.ion_tof",
                       "--input", str(source_file), "--output", str(result),
                       "--source", source.value, "--reference", reference.value,
                       "--bins", str(int(count)), "--minimum", str(low)]
            if high is not None:
                job.cmd += ["--maximum", str(high)]
            job.extra.update(histogram=str(result), reference=reference.value, input_name=source_file.name)
            view.attach(jobs.submit(job))
            gmd_display.clear()
        except (OSError, ValueError) as exc:
            ui.notify(str(exc), type="warning")

    with theme.panel_card():
        input_file = save(path_input("LUNA HDF5 file", browse="file", filters=[".h5", ".hdf5"]), "input")
        ui.label("Choose either an uncorrected dump or a timewalk-corrected file.").classes("vmi-subtitle")
        with field_grid():
            source = save(ui.select(SOURCES, value="pixels", label="Count").props("outlined dense"), "source")
            reference = save(ui.select(REFERENCE_LABELS, value=DEFAULT_REFERENCE, label="Trigger reference")
                             .props("outlined dense"), "reference")
        with field_grid(3):
            bins = save(num_input("Bins", DEFAULT_BINS, min=1, max=MAX_BINS, step=1), "bins")
            minimum = save(num_input("TOF minimum (ns)", 0, min=0, step="any"), "minimum")
            maximum = save(num_input("TOF maximum (ns, blank = auto)", None, step="any"), "maximum")
        with action_bar(view, "Plot TOF", on_click=run):
            pass
    view.place_console()

    async def plot_gmd():
        job = view.job
        if job is None or job.status != "done":
            ui.notify("Complete a TOF plot first.", type="warning")
            return
        metadata_value = metadata_file.value.strip()
        if not metadata_value:
            ui.notify("Choose the matching acquisition metadata HDF5 file.", type="warning")
            return
        gmd_button.disable()
        try:
            metadata = shared.checked_path(metadata_value)
            result_path = shared.checked_path(job.extra["histogram"])
            with jobs.preview_read_access([result_path, metadata]):
                result, sources = await asyncio.to_thread(_load_gmd_diagnostics, result_path, metadata)
            if view.job is job:
                gmd_display.clear()
                with gmd_display:
                    ui.label(f'{job.extra["input_name"]} · {REFERENCE_LABELS[job.extra["reference"]]} '
                             f'· {metadata.name}').classes("vmi-caption break-all")
                    _gmd_results(result, sources)
        except (OSError, ValueError, ResourceBusyError) as exc:
            ui.notify(str(exc), type="warning")
        finally:
            gmd_button.enable()

    with theme.panel_card():
        theme.form_section("GMD diagnostics")
        ui.label("Uses the count source and trigger reference from the latest TOF plot. Choose the metadata "
                 "exported with that dump. Alignment assumes one selected trigger per metadata row, in "
                 "acquisition order.").classes("vmi-subtitle")
        metadata_file = save(path_input("Matching acquisition metadata HDF5",
                                        browse="file", filters=[".h5", ".hdf5"]), "metadata")
        gmd_button = ui.button("Plot GMD diagnostics", icon="show_chart", on_click=plot_gmd)
        gmd_display = ui.column().classes("w-full gap-4")
