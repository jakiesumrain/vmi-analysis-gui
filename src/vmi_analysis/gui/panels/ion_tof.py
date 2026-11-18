"""Unfiltered time-of-flight spectra from LUNA Timepix3 dumps."""

from math import isfinite
from pathlib import Path
import sys

from nicegui import ui
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vmi_analysis.ion_tof import DEFAULT_BINS, DEFAULT_REFERENCE, MAX_BINS, REFERENCES

from .. import shared, theme
from ..centroid_plots import _style, plot_options
from ..jobs import Job
from .common import JobView, action_bar, field_grid, num_input, path_input, persist


JOB_NAME = "Ion TOF"
SOURCES = {"pixels": "Raw pixel hits", "clusters": "Clustered events", "both": "Both"}
REFERENCE_LABELS = {name: name.replace("Rising", " rising").replace("Falling", " falling")
                    for name in REFERENCES}


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
