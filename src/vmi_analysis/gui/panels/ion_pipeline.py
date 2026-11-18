"""Ion workbench for measuring peak yields across delay positions."""

import csv
import json
from pathlib import Path
import sys

from nicegui import ui
import plotly.graph_objects as go

from vmi_analysis.ion_pipeline import CSV_NAME, DEFAULT_PEAKS, FIGURE_NAME, positions, validate_settings
from vmi_analysis.ion_pipeline_backend import DEFAULT_SHIFT, NAME, SHIFT_HELP, y_label

from .. import shared, theme
from ..centroid_plots import _style, plot_options
from ..jobs import Job
from .common import JobView, action_bar, field_grid, file_url, num_input, open_folder_button, path_input, persist
from .ion_pipeline_options import options_form


JOB_NAME = "Ion Pipeline"


def show_results(job):
    output = shared.checked_path(job.extra["output"])
    csv_file = output / CSV_NAME
    if not csv_file.is_file():
        return
    with csv_file.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    options = job.extra["settings"]
    successful = [row for row in rows if row["status"] == "OK"]
    ui.label(f"{len(successful):,} of {len(rows):,} peak measurements complete").classes("vmi-caption")
    with ui.row().classes("items-center gap-2"):
        for path, label in ((csv_file, "Yield CSV"), (output / FIGURE_NAME, "Yield plot PNG")):
            if path.is_file():
                url = file_url(path)
                ui.button(label, icon="download", on_click=lambda path=path, url=url:
                          ui.download.from_url(url, filename=path.name)).props("flat")
        if not shared.SERVER:
            open_folder_button(output, "Open output folder")
    if successful:
        fig = go.Figure()
        for peak in options["peaks"]:
            points = sorted((row for row in rows if row["peak"] == peak["name"]),
                            key=lambda row: float(row["delay_mm"]))
            fig.add_scatter(x=[float(row["delay_mm"]) for row in points],
                            y=[float(row["hits_per_frame"]) if row["hits_per_frame"] else None for row in points],
                            name=peak["name"], mode="lines+markers", customdata=[
                                [row["folder"], row["counts"], row["selected_frames"]] for row in points],
                            error_y=dict(array=[float(row["poisson_error"]) if row["poisson_error"] else None
                                                for row in points]),
                            hovertemplate="%{customdata[0]}<br>%{x:.6g} mm · %{y:.5g} per frame"
                                          "<br>%{customdata[1]} counts / %{customdata[2]} frames")
        _style(fig, f"Ion yields · {NAME}")
        fig.update_layout(showlegend=True, height=460, legend=dict(orientation="h"))
        fig.update_xaxes(title_text="Delay-line position (mm)")
        fig.update_yaxes(title_text=y_label(options), rangemode="tozero")
        ui.plotly(plot_options(fig)).classes("w-full")
    with ui.expansion("Peak measurements", icon="table_chart").classes("w-full"):
        columns = [("folder", "Folder"), ("peak", "Peak"), ("delay_mm", "Delay (mm)"), ("counts", "Counts"),
                   ("selected_frames", "Selected frames"), ("hits_per_frame", "Yield per frame"), ("status", "Status")]
        ui.table(columns=[dict(name=key, field=key, label=label) for key, label in columns],
                 rows=[dict(row, id=i) for i, row in enumerate(rows)], row_key="id",
                 pagination=10).classes("w-full")


def build(jobs):
    peak_rows = []
    view = JobView(jobs, job_name=JOB_NAME, results_builder=show_results)
    theme.panel_header(JOB_NAME, "Decode a TPX3 delay scan and plot GMD-gated peak yields.", view=view)

    def save(field, key):
        return persist(field, f"ion_pipeline.{key}")

    def peak_values():
        return [dict(name=name.value.strip(), low=low.value, high=high.value, center=center.value)
                for name, low, high, center in peak_rows]

    def store_peaks():
        peak_store.value = json.dumps(peak_values())

    def add_peak(peak):
        with peak_list, ui.row().classes("w-full items-center gap-2") as row:
            fields = (
                ui.input("Peak name", value=peak.get("name", "")).props("outlined dense"),
                num_input("Fit minimum (ns)", peak.get("low"), step="any"),
                num_input("Fit maximum (ns)", peak.get("high"), step="any"),
                num_input("Initial center (ns)", peak.get("center"), step="any"),
            )
            peak_rows.append(fields)
            for field in fields:
                field.on_value_change(store_peaks)

            def remove():
                peak_rows.remove(fields)
                row.delete()
                store_peaks()

            ui.button(icon="remove_circle_outline", on_click=remove).props('flat round aria-label="Remove peak"')

    def run():
        try:
            if not root.value.strip() or not output.value.strip():
                raise ValueError("Choose the delay-scan root and an output directory.")
            source = shared.checked_path(root.value.strip())
            if not source.is_dir():
                raise ValueError("Choose an existing delay-scan root directory.")
            inputs = [shared.checked_path(path) for _, path in positions(source)]
            if not inputs:
                raise ValueError("No delay-position folders were found.")
            if any(field.value is None or not float(field.value).is_integer() for field in (fit_bins, shift, workers)):
                raise ValueError("Use whole numbers for fit bins, GMD shift, and parallel delay positions.")
            options = dict(**decode_options(), peaks=peak_values(), gate_gmd=gate.value, gmd_min=gmd_min.value,
                           gmd_max=gmd_max.value, gmd_shift=int(shift.value), fit_bins=int(fit_bins.value),
                           window_sigma=width.value, keep_dumps=keep.value, workers=int(workers.value))
            validate_settings(options)
            job = Job(JOB_NAME, cwd=str(shared.workspace_root()))
            parent = shared.checked_path(output.value.strip(), write=True)
            destination = shared.checked_path(parent / f"ion-pipeline-{job.id}", write=True)
            job.cmd = [sys.executable, "-m", "vmi_analysis.ion_pipeline", "--input", str(source),
                       "--output-dir", str(destination), "--settings", json.dumps(options, allow_nan=False)]
            job.reads = [str(path) for path in inputs]
            job.extra.update(output=str(destination), settings=options)
            view.attach(jobs.submit(job))
        except (OSError, ValueError, OverflowError) as exc:
            ui.notify(str(exc), type="warning")

    with theme.panel_card():
        theme.form_section("Delay scan")
        root = save(path_input("Dataset root (one subfolder per delay position)", browse="dir"), "input")
        output = save(path_input("Output directory", str(Path(shared.workspace_root()) / "ion-pipeline"),
                                 browse="dir"), "output")
        with field_grid():
            workers = save(num_input("Parallel delay positions", 1, min=1, max=8, step=1), "workers")
            keep = save(ui.checkbox("Keep intermediate HDF5 dumps"), "keep_dumps")
        decode_options = options_form(save)
        theme.form_section("GMD2 gate")
        gate = save(ui.checkbox("Filter by GMD2", value=True), "gate")
        with field_grid(3):
            gmd_min = save(num_input("GMD2 minimum (nA)", 2, step="any"), "gmd_min")
            gmd_max = save(num_input("GMD2 maximum (nA)", 6, step="any"), "gmd_max")
            shift = save(num_input("GMD pairing shift (frames)", DEFAULT_SHIFT, step=1), "gmd_shift")
        for field in (gmd_min, gmd_max, shift):
            field.bind_enabled_from(gate, "value")
        ui.label(SHIFT_HELP + " Both GMD2 limits are inclusive.").classes("vmi-subtitle")
        theme.form_section("Peak windows")
        ui.label("Fits each Gaussian separately on ungated TOFs. Counts individual arrivals in each window "
                 "and divides by the selected frame count. Overlapping windows can count the same arrival."
                 ).classes("vmi-subtitle")
        with field_grid():
            fit_bins = save(num_input("Histogram bins per peak fit", 60, min=4, step=1), "fit_bins")
            width = save(num_input("Count-window half-width (σ)", 3, step="any"), "window_sigma")
        peak_store = save(ui.input(value=json.dumps(DEFAULT_PEAKS)), "peaks")
        peak_store.set_visibility(False)
        peak_list = ui.column().classes("w-full")
        try:
            peaks = json.loads(peak_store.value)
            if not isinstance(peaks, list) or any(not isinstance(peak, dict) for peak in peaks):
                raise ValueError("Invalid peak settings.")
        except (TypeError, ValueError):
            peaks = DEFAULT_PEAKS
        for peak in peaks:
            add_peak(peak)
        with action_bar(view, "Run ion pipeline", on_click=run):
            ui.button("Add peak", icon="add", on_click=lambda: (add_peak({}), store_peaks())).props("flat")
    view.place_console()
