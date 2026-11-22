"""Detector-position and TOF workbench for ion momentum reconstruction."""

from math import isfinite
import sys

from nicegui import ui
import numpy as np
import plotly.graph_objects as go

from vmi_analysis.ion_acquisition import collection_reads
from vmi_analysis.ion_tof import DEFAULT_REFERENCE
from vmi_analysis.tof_bins import DEFAULT_BIN_SIZE_NS

from .. import shared, theme
from ..centroid_plots import _style, plot_options
from ..jobs import Job
from .common import (JobView, action_bar, confirm_clear_outputs, field_grid, num_input,
                     path_input, persist)
from .ion_tof import REFERENCE_LABELS, SOURCES, _tof_figure


JOB_NAME = "Ion momenta"
PLOT_HEIGHT = 580


def _position_figure(image):
    fig = _style(go.Figure(), "Detector positions")
    fig.add_heatmap(z=image, colorscale="Viridis",
                    colorbar=dict(title="Counts", x=1, xanchor="left", thickness=14, xpad=5),
                    hovertemplate="X %{x} px<br>Y %{y} px<br>%{z:,} counts<extra></extra>")
    fig.update_layout(height=PLOT_HEIGHT, margin=dict(l=56, r=70, t=90, b=65, autoexpand=False))
    fig.update_xaxes(title_text="X (pixels)", range=[-0.5, image.shape[1] - 0.5],
                     constrain="domain", constraintoward="right")
    fig.update_yaxes(title_text="Y (pixels)", range=[-0.5, image.shape[0] - 0.5],
                     constrain="domain", scaleanchor="x", scaleratio=1)
    return fig


def _results(job):
    if job.status != "done":
        return
    with np.load(shared.checked_path(job.extra["histogram"]), allow_pickle=False) as result:
        ui.label(f'{job.extra["input_name"]} · {REFERENCE_LABELS[job.extra["reference"]]} '
                 f'· {int(result["triggers"]):,} triggers').classes("vmi-caption break-all")
        for source in ("pixels", "clusters"):
            if source not in result:
                continue
            ui.label(SOURCES[source]).classes("font-mono text-lg text-primary")
            image = result[f"{source}_image"]
            ui.label(f'{int(result[source].sum()):,} arrivals in the TOF interval · '
                     f'{int(image.sum()):,} mapped positions · '
                     f'{int(result[f"{source}_unmatched"]):,} with no preceding trigger · '
                     f'{int(result[f"{source}_outside"]):,} outside the TOF interval · '
                     f'{int(result[f"{source}_invalid_positions"]):,} invalid positions').classes("vmi-caption")
            with ui.element("div").classes("vmi-ion-detector-row"):
                ui.plotly(plot_options(_position_figure(image))).classes("w-full min-w-0")
                fig = _tof_figure(result["edges"], {source: result[source]}, "Time-of-flight spectrum")
                fig.update_layout(height=PLOT_HEIGHT, margin=dict(l=65, r=24, t=90, b=65))
                ui.plotly(plot_options(fig)).classes("w-full min-w-0")
        ui.label("The detector image uses arrivals inside the displayed TOF interval. "
                 "Positions use one-pixel bins; fractional cluster positions are binned at pixel centers."
                 ).classes("vmi-subtitle")


def build(jobs):
    view = JobView(jobs, job_name=JOB_NAME, results_builder=_results)
    theme.panel_header(JOB_NAME, "Inspect detector positions and time of flight for ion momentum analysis.", view=view)

    def save(widget, key):
        return persist(widget, f"ion_momenta.{key}")

    def use_latest():
        candidates = [job for job in jobs.jobs() if job.name == "TPX3 Dump" and job.status == "done"]
        if not candidates:
            ui.notify("Complete a TPX3 dump first or choose a saved collection.", type="warning")
            return
        latest = max(candidates, key=lambda job: job.finished or job.created)
        input_file.value = latest.extra.get("collection") or latest.extra["output"]

    def run():
        try:
            if not input_file.value.strip():
                raise ValueError("Choose a LUNA HDF5 file or acquisition collection.")
            source_file = shared.checked_path(input_file.value.strip())
            if not source_file.is_file():
                raise ValueError("The selected input file does not exist.")
            for path in collection_reads(source_file):
                shared.checked_path(path)
            width, low, high = bin_size.value, minimum.value, maximum.value
            if width is None or not isfinite(width) or width <= 0:
                raise ValueError("TOF bin size must be finite and greater than zero.")
            if low is None or not isfinite(low) or low < 0:
                raise ValueError("TOF minimum must be finite and at least zero.")
            if high is not None and (not isfinite(high) or high <= low):
                raise ValueError("TOF maximum must be greater than the minimum.")
            job = Job(JOB_NAME, cwd=str(shared.workspace_root()))
            result = shared.generated_dir() / f"ion-momenta-{job.id}.npz"
            job.cmd = [sys.executable, "-m", "vmi_analysis.ion_momenta",
                       "--input", str(source_file), "--output", str(result),
                       "--source", source.value, "--reference", reference.value,
                       "--bin-size", str(width), "--minimum", str(low)]
            if high is not None:
                job.cmd += ["--maximum", str(high)]
            job.extra.update(histogram=str(result), reference=reference.value, input_name=source_file.name)
            view.attach(jobs.submit(job))
        except (OSError, ValueError) as exc:
            ui.notify(str(exc), type="warning")

    def clear():
        outputs = [job.extra["histogram"] for job in jobs.jobs()
                   if job.name == JOB_NAME and "histogram" in job.extra]
        confirm_clear_outputs(outputs, "Delete the generated Ion momenta position and TOF results?",
                              on_cleared=view.clear_results)

    with theme.panel_card():
        input_file = save(path_input("LUNA HDF5 file or acquisition collection", browse="file",
                                    filters=[".h5", ".hdf5", ".json"]), "input")
        ui.button("Use latest dump from TPX3 Dump", icon="history", on_click=use_latest).props("flat")
        ui.label("Choose either an uncorrected dump or a timewalk-corrected file.").classes("vmi-subtitle")
        with field_grid():
            source = save(ui.select(SOURCES, value="pixels", label="Count").props("outlined dense"), "source")
            reference = save(ui.select(REFERENCE_LABELS, value=DEFAULT_REFERENCE, label="Trigger reference")
                             .props("outlined dense"), "reference")
        with field_grid(3):
            bin_size = save(num_input("TOF bin size (ns)", DEFAULT_BIN_SIZE_NS, step="any"), "bin_size_ns")
            minimum = save(num_input("TOF minimum (ns)", 0, min=0, step="any"), "minimum")
            maximum = save(num_input("TOF maximum (ns, blank = auto)", None, step="any"), "maximum")
        ui.label("Momentum reconstruction requires the ion mass and charge, detector center, "
                 "and spectrometer calibration or flight-field model.").classes("vmi-subtitle")
        with action_bar(view, "Plot positions and TOF", on_click=run):
            ui.button("Clear outputs", icon="delete", on_click=clear).props("flat color=negative")
    view.place_console()
