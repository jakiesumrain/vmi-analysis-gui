"""Ion workbench for measuring peak yields across delay positions."""

import csv
import json
from math import ceil, floor, isfinite, log10
from pathlib import Path
import sys

from nicegui import ui
import plotly.graph_objects as go

from vmi_analysis.ion_pipeline import CSV_NAME, DEFAULT_PEAKS, FIGURE_NAME, positions, validate_settings
from vmi_analysis.ion_pipeline_backend import DEFAULT_SHIFT, NAME, SHIFT_HELP, y_label

from .. import shared, theme
from ..centroid_plots import SERIES_COLORS, _style, plot_options
from ..jobs import Job
from .common import JobView, action_bar, field_grid, file_url, num_input, open_folder_button, path_input, persist
from .ion_pipeline_options import options_form


JOB_NAME = "Ion Pipeline"
MM_PER_PS = 0.15
Y_SCALES = {"linear": "Linear", "log": "Logarithmic"}


def delay_ticks(low, high):
    """Position evenly spaced picosecond ticks on the millimetre coordinate system."""
    low, high = sorted((low / MM_PER_PS, high / MM_PER_PS))
    target = (high - low) / 6
    if not isfinite(target) or target <= 0:
        return [], []
    magnitude = 10 ** floor(log10(target))
    step = next(factor * magnitude for factor in (1, 2, 5, 10) if factor * magnitude >= target)
    ticks = [index * step for index in range(ceil(low / step), floor(high / step) + 1)]
    return [tick * MM_PER_PS for tick in ticks], [f"{tick:.10g}" for tick in ticks]


def yield_figure(rows, options, names, scales=("linear", "linear")):
    fig = go.Figure()
    for index, (name, scale) in enumerate(zip(names, scales)):
        points = sorted((row for row in rows if row["peak"] == name),
                        key=lambda row: float(row["delay_mm"]))
        values = [float(row["hits_per_frame"]) if row["status"] == "OK" and row["hits_per_frame"]
                  else None for row in points]
        if scale == "log":
            values = [value if value is not None and value > 0 else None for value in values]
        fig.add_scatter(x=[float(row["delay_mm"]) for row in points],
                        y=values, yaxis="y" if index == 0 else "y2",
                        name=name, mode="lines+markers", connectgaps=False,
                        line=dict(color=SERIES_COLORS[index], dash="solid" if index == 0 else "dash"),
                        customdata=[[row["folder"], row["counts"], row["selected_frames"]] for row in points],
                        error_y=dict(array=[float(row["poisson_error"]) if row["poisson_error"] else None
                                            for row in points]),
                        hovertemplate="%{customdata[0]}<br>%{x:.6g} mm · %{y:.5g} per frame"
                                      "<br>%{customdata[1]} counts / %{customdata[2]} frames<extra>%{fullData.name}</extra>")
    # An empty trace makes the matched, labelled bottom axis available to Plotly.
    fig.add_scatter(x=[None], y=[None], xaxis="x2", showlegend=False, hoverinfo="skip")
    _style(fig, f"Ion yields · {NAME}")
    positions_mm = [float(row["delay_mm"]) for row in rows]
    low, high = min(positions_mm), max(positions_mm)
    padding = (high - low) * 0.06 or MM_PER_PS
    tickvals, ticktext = delay_ticks(low - padding, high + padding)
    fig.update_layout(height=500, margin=dict(l=96, r=96, t=110, b=66),
                      xaxis=dict(side="top", title_text="Delay-line position (mm)", automargin=True),
                      xaxis2=dict(overlaying="x", matches="x", side="bottom", showgrid=False,
                                  anchor="y", tickmode="array", tickvals=tickvals, ticktext=ticktext,
                                  title_text="Delay (ps)", automargin=True))
    for index, (name, scale) in enumerate(zip(names, scales)):
        color = SERIES_COLORS[index]
        axis = dict(title=dict(text=f"{name} · {y_label(options)}", font=dict(color=color)),
                    tickfont=dict(color=color), automargin=True, type=scale,
                    rangemode="tozero" if scale == "linear" else "normal")
        if index == 0:
            fig.update_layout(yaxis=axis)
        else:
            fig.update_layout(yaxis2=dict(**axis, overlaying="y", side="right", anchor="x", showgrid=False))
    return fig


def ratio_figure(rows, numerator, denominator):
    """Pair peaks by source folder, preserving gaps for undefined ratios."""
    pairs = {}
    for row in rows:
        if row["peak"] in (numerator, denominator):
            pairs.setdefault(row["folder"], {})[row["peak"]] = row
    x, y, folders = [], [], []
    for folder, pair in sorted(pairs.items(), key=lambda item: float(next(iter(item[1].values()))["delay_mm"])):
        first, second = pair.get(numerator), pair.get(denominator)
        value = None
        if first and second and first["status"] == second["status"] == "OK":
            a, b = float(first["hits_per_frame"]), float(second["hits_per_frame"])
            if isfinite(a) and isfinite(b) and b > 0:
                quotient = a / b
                if isfinite(quotient):
                    value = quotient
        x.append(float(next(iter(pair.values()))["delay_mm"]))
        y.append(value)
        folders.append(folder)
    fig = go.Figure(go.Scatter(x=x, y=y, customdata=folders, mode="lines+markers", connectgaps=False,
                              name=f"{numerator} / {denominator}",
                              hovertemplate="%{customdata}<br>%{x:.6g} mm · ratio %{y:.6g}<extra></extra>"))
    _style(fig, f"Yield ratio · {numerator} / {denominator}")
    fig.update_layout(height=460)
    fig.update_xaxes(title_text="Delay-line position (mm)")
    fig.update_yaxes(title_text=f"{numerator} / {denominator}", rangemode="tozero")
    return fig


def scaled_plot(figure):
    """Change the display scale using saved results, without rerunning analysis."""
    def update():
        display = go.Figure(figure)
        if scale.value == "log":
            for trace in display.data:
                trace.y = [value if value is not None and value > 0 else None for value in trace.y]
        display.update_yaxes(type=scale.value, range=None, autorange=True,
                             rangemode="tozero" if scale.value == "linear" else "normal")
        chart.figure = plot_options(display)
        chart.update()

    def replace(next_figure):
        nonlocal figure
        figure = next_figure
        update()

    scale = ui.select(Y_SCALES, value="linear", label="Y-axis scale",
                      on_change=update).props("outlined dense").classes("w-48")
    chart = ui.plotly(plot_options(figure)).classes("w-full")
    return replace


def peak_yield_plot(rows, options, names):
    def update():
        peaks = [left.value, right.value] if right is not None else [left.value]
        scales = [left_scale.value, right_scale.value] if right_scale is not None else [left_scale.value]
        chart.update_figure(plot_options(yield_figure(rows, options, peaks, scales)))

    with field_grid():
        left = ui.select(names, value=names[0], label="Left peak", on_change=update).props("outlined dense")
        right = (ui.select(names, value=names[1], label="Right peak", on_change=update).props("outlined dense")
                 if len(names) > 1 else None)
        left_scale = ui.select(Y_SCALES, value="linear", label="Left Y-axis scale", on_change=update).props("outlined dense")
        right_scale = (ui.select(Y_SCALES, value="linear", label="Right Y-axis scale", on_change=update)
                       .props("outlined dense") if right is not None else None)
    ui.label("0.15 mm = 1 ps. Zero yields are omitted on logarithmic axes.").classes("vmi-caption")
    chart = ui.plotly(plot_options(yield_figure(rows, options, names[:2]))).classes("w-full")

    def update_ticks(event):
        tickvals, ticktext = delay_ticks(*event.args)
        chart.run_plot_method("relayout", {"xaxis2.tickvals": tickvals, "xaxis2.ticktext": ticktext})

    chart.on("plotly_relayout", update_ticks, js_handler=fr"""event => {{
        if (event && Object.keys(event).some(key => /^xaxis2?\.(range|autorange)/.test(key))) {{
            emit(getHtmlElement({chart.id})._fullLayout.xaxis.range);
        }}
    }}""")


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
        names = [peak["name"] for peak in options["peaks"]]
        with ui.expansion("Peak yield comparison", icon="show_chart", value=False).classes("w-full"):
            peak_yield_plot(rows, options, names)
        with theme.panel_card():
            theme.form_section("Peak yield ratio")
            if len(names) < 2:
                ui.label("Analyze at least two peaks to plot their ratio.").classes("vmi-subtitle")
            else:
                def update_ratio():
                    replace_ratio(ratio_figure(rows, numerator.value, denominator.value))

                with field_grid():
                    numerator = ui.select(names, value=names[0], label="Numerator peak",
                                          on_change=update_ratio).props("outlined dense")
                    denominator = ui.select(names, value=names[1], label="Denominator peak",
                                            on_change=update_ratio).props("outlined dense")
                ui.label("Ratio = numerator yield per frame / denominator yield per frame for the same folder. "
                         "Missing measurements and zero denominators appear as gaps. "
                         "Zero ratios are omitted on logarithmic axes.").classes("vmi-subtitle")
                replace_ratio = scaled_plot(ratio_figure(rows, numerator.value, denominator.value))
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
                           window_sigma=width.value, fit_tof=fit_tof.value,
                           keep_dumps=keep.value, workers=int(workers.value))
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
            shift = save(num_input("GMD2 alignment shift d (frames)", DEFAULT_SHIFT, step=1), "gmd_shift")
        for field in (gmd_min, gmd_max, shift):
            field.bind_enabled_from(gate, "value")
        ui.label(SHIFT_HELP + " The pipeline uses this shift for every delay position. "
                 "Use the Ion TOF GMD diagnostic to check the shift for your data. "
                 "Both GMD2 limits are inclusive.").classes("vmi-subtitle")
        theme.form_section("Peak windows")
        ui.label("Fits each Gaussian separately using the selected TOF source for each dataset. "
                 "Counts individual arrivals from selected frames in each window and divides by the selected frame count. "
                 "Overlapping windows can count the same arrival. With GMD2 filtering off, both fit sources use all frames."
                 ).classes("vmi-subtitle")
        with field_grid(3):
            fit_tof = save(ui.select({"ungated": "Ungated TOF (all frames)",
                                      "gated": "GMD2-gated TOF (selected frames)"},
                                     value="gated", label="TOF used to fit peak windows")
                           .props("outlined dense"), "fit_tof")
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
