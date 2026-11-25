"""Ion workbench for measuring peak yields across delay positions."""

import csv
import json
from math import ceil, floor, isfinite, log10
from pathlib import Path
import re
import sys

from nicegui import ui
import plotly.graph_objects as go

from vmi_analysis.ion_pipeline import (
    CSV_NAME, DEFAULT_PEAKS, FIGURE_NAME, RATIO_NAME, positions, validate_settings, yield_ratios,
)
from vmi_analysis.tpx3_analysis import UNWRAP_MODE
from vmi_analysis.ion_pipeline_backend import DEFAULT_SHIFT, NAME, SHIFT_HELP, y_label

from .. import shared, theme
from ..agent_panels import PanelOperation, boolean, number, resolve_path
from ..centroid_plots import SERIES_COLORS, _style, plot_options
from ..jobs import Job
from .common import (JobView, action_bar, adopt_agent_form, field_grid, file_url, num_input,
                     open_folder_button, path_input, persist)
from .ion_pipeline_options import LUNA_DEFAULTS, WSL_FIELDS, options_form


JOB_NAME = "Ion Pipeline"
FALLBACK_JOB_NAME = "Ion Pipeline (fallback)"
MM_PER_PS = 0.15
Y_SCALES = {"linear": "Linear", "log": "Logarithmic"}


RUN_DEFAULTS = {
    "gate_gmd": True, "gmd_min": 2, "gmd_max": 6, "gmd_shift": DEFAULT_SHIFT,
    "fit_tof": "gated", "fit_bins": 60, "window_sigma": 3, "keep_dumps": False, "workers": 1,
}
COUNT_SOURCES = {"pixels": "Raw pixel hits", "clusters": "Clustered events"}
FIT_SOURCES = {"gated": "GMD2-gated TOF", "ungated": "Ungated TOF"}
ALGORITHMS = ("DFSCluster", "Kuttruff2024")


def agent_operations():
    parameters = {
        "input": {"type": "path", "description": "Root directory with one subfolder per delay position"},
        "output": {"type": "path", "description": "Parent directory for a new run folder in your workspace"},
        "peaks": {"type": "list of objects", "default": DEFAULT_PEAKS,
                  "description": "Each peak needs a name, fit low, fit high, and initial center. "
                                 "The keys are name, low, high, center. Times are in ns."},
        "gate_gmd": {"type": "boolean", "description": "Use an inclusive GMD2 gate"},
        "gmd_min": {"type": "number or null", "unit": "nA"},
        "gmd_max": {"type": "number or null", "unit": "nA"},
        "gmd_shift": {"type": "integer", "unit": "frames", "description": SHIFT_HELP},
        "fit_tof": {"type": "string", "choices": list(FIT_SOURCES)},
        "fit_bins": {"type": "integer", "description": "Histogram bins per Gaussian fit"},
        "window_sigma": {"type": "number", "description": "Count-window half-width in fitted standard deviations"},
        "keep_dumps": {"type": "boolean", "description": "Keep the intermediate HDF5 files and metadata"},
        "workers": {"type": "integer", "description": "Number of delay positions to process at once, from 1 to 8"},
        "source": {"type": "string", "choices": list(COUNT_SOURCES)},
        "timewalk": {"type": "boolean", "description": "Calculate and apply timewalk correction per acquisition segment"},
        "clustering_algorithm": {"type": "string", "choices": list(ALGORITHMS)},
        "eps_t": {"type": "string", "description": "Clustering time gap with units, such as 100ns"},
        "eps_s": {"type": "integer", "unit": "pixels", "description": "Clustering space gap"},
        "kuttruff_buffer_size": {"type": "integer", "description": "Buffer size for Kuttruff2024 clustering"},
        "ctot_cut": {"type": "integer", "unit": "ns", "description": "Timewalk cluster ToT cut"},
        "disable_clustering": {"type": "boolean"},
    }
    for key, value in (RUN_DEFAULTS | LUNA_DEFAULTS).items():
        parameters[key]["default"] = value
    description = {
        "panel": JOB_NAME,
        "summary": "Run LUNA on a Timepix3 delay scan, fit each peak separately, and report yields per "
                   "selected frame. The ratio CSV uses the first peak divided by the second.",
        "parameters": parameters,
        "manual": JOB_NAME,
    }
    return (PanelOperation("ion_pipeline", description, _prepare_pipeline, (JOB_NAME,),
                           outputs=("yield_csv", "yield_plot", "ratio_csv")),)


def _integer(value, label):
    value = number(value)
    if value != int(value):
        raise ValueError(f"Enter a whole number for {label}.")
    return int(value)


def _pipeline_settings(values, *, fallback):
    options = {**RUN_DEFAULTS, "fit_tof": "ungated" if fallback else "gated"}
    options.update({key: values[key] for key in options.keys() & values.keys()})
    for key in ("gate_gmd", "keep_dumps"):
        options[key] = boolean(options[key])
    for key in ("gmd_min", "gmd_max"):
        if options[key] is not None:
            options[key] = number(options[key])
    for key in ("fit_bins", "gmd_shift", "workers"):
        options[key] = _integer(options[key], key.replace("_", " "))
    options["window_sigma"] = number(options["window_sigma"])
    peaks = values.get("peaks", DEFAULT_PEAKS)
    if not isinstance(peaks, list):
        raise ValueError("Provide a list of peak settings.")
    options["peaks"] = []
    for peak in peaks:
        if (not isinstance(peak, dict) or peak.keys() != {"name", "low", "high", "center"}
                or not isinstance(peak["name"], str)):
            raise ValueError("Each peak needs a name, fit minimum, fit maximum, and initial center.")
        options["peaks"].append({"name": peak["name"].strip(),
                                 **{key: number(peak[key]) for key in ("low", "high", "center")}})
    if fallback:
        options.update(backend="fallback", source="pixels", unwrap=values.get("unwrap", UNWRAP_MODE))
    else:
        options.update({key: values.get(key, default) for key, default in LUNA_DEFAULTS.items()})
        if options["source"] not in tuple(COUNT_SOURCES) or options["clustering_algorithm"] not in ALGORITHMS:
            raise ValueError("Choose a valid count source and clustering algorithm.")
        for key in ("timewalk", "disable_clustering"):
            options[key] = boolean(options[key])
        if options["disable_clustering"] and (options["timewalk"] or options["source"] == "clusters"):
            raise ValueError("Enable clustering for timewalk correction or clustered counts.")
        gap = options["eps_t"]
        if not isinstance(gap, str) or not gap.strip() or gap.strip().startswith("-") or "\x00" in gap:
            raise ValueError("Enter a valid LUNA clustering time gap.")
        options["eps_t"] = gap.strip()
        for key, minimum in (("eps_s", 0), ("kuttruff_buffer_size", 1), ("ctot_cut", 0)):
            options[key] = _integer(options[key], key.replace("_", " "))
            if options[key] < minimum:
                raise ValueError(f"{key.replace('_', ' ').capitalize()} must be at least {minimum}.")
        if options["timewalk"] and options["ctot_cut"] % 25:
            raise ValueError("Cluster ToT cut must be a multiple of 25 ns.")
        for key in WSL_FIELDS:
            value = "" if shared.SERVER else values.get(key, "")
            if not isinstance(value, str) or "\x00" in value:
                raise ValueError("Enter valid Windows WSL2 settings.")
            options[key] = value.strip()
    validate_settings(options)
    return options


def _capture_progress(job, line):
    if match := re.fullmatch(r"Delay positions written: (\d+)/(\d+)", line):
        completed, total = map(int, match.groups())
        job.progress = completed / total
    return False


def pipeline_job(source, parent, values, *, fallback=False):
    source = shared.checked_path(source)
    parent = shared.checked_path(parent, write=True)
    if not source.is_dir():
        raise ValueError("Choose an existing delay-scan root directory.")
    if parent.exists() and not parent.is_dir():
        raise ValueError("Choose an output directory.")
    inputs = [shared.checked_path(path) for _, path in positions(source)]
    if not inputs:
        raise ValueError("No delay-position folders were found.")
    options = _pipeline_settings(values, fallback=fallback)
    name = FALLBACK_JOB_NAME if fallback else JOB_NAME
    folder = "ion-pipeline-fallback" if fallback else "ion-pipeline"
    job = Job(name, cwd=str(shared.workspace_root()))
    destination = shared.checked_path(parent / f"{folder}-{job.id}", write=True)
    job.cmd = [sys.executable, "-m", "vmi_analysis.ion_pipeline", "--input", str(source),
               "--output-dir", str(destination), "--settings", json.dumps(options, allow_nan=False)]
    job.reads = [str(path) for path in inputs]
    job.extra.update(output=str(destination), settings=options, yield_csv=str(destination / CSV_NAME),
                     yield_plot=str(destination / FIGURE_NAME))
    if not fallback:
        job.consume_output = _capture_progress
        if len(options["peaks"]) > 1:
            job.extra["ratio_csv"] = str(destination / RATIO_NAME)
    return job


def _prepare_pipeline(jobs, params):
    source = resolve_path(params.get("input"))
    parent = resolve_path(params.get("output"), write=True)
    saved = shared.settings_store()
    wsl = {key: saved.get(f"ion_pipeline.{key}", "") for key in WSL_FIELDS}
    job = pipeline_job(source, parent, params | wsl)
    options = job.extra["settings"]
    form = {**options, "input": str(source), "output": str(parent), "gate": options["gate_gmd"],
            "peaks": json.dumps(options["peaks"])}
    del form["gate_gmd"]
    if shared.SERVER:
        for key in WSL_FIELDS:
            del form[key]
    job.extra["panel_form"] = form
    summary = {
        "Delay-scan root": str(source),
        "Delay positions": len(job.reads),
        "Run folder": job.extra["output"],
        "Count source": COUNT_SOURCES[options["source"]],
        "GMD2 gate (nA, inclusive)": [options["gmd_min"], options["gmd_max"]] if options["gate_gmd"] else "Off",
        "GMD2 alignment shift d (frames)": options["gmd_shift"],
        "Pairing": SHIFT_HELP,
        "TOF used to fit peak windows": FIT_SOURCES[options["fit_tof"]],
        "Peaks and fit ranges (ns)": options["peaks"],
        "Histogram bins per peak fit": options["fit_bins"],
        "Count-window half-width (σ)": options["window_sigma"],
        "Parallel delay positions": options["workers"],
        "Keep intermediate HDF5 dumps": options["keep_dumps"],
        "Apply timewalk correction": options["timewalk"],
        "Disable clustering": options["disable_clustering"],
        "Clustering algorithm": options["clustering_algorithm"],
        "Clustering time gap": options["eps_t"],
        "Clustering space gap (pixels)": options["eps_s"],
    }
    if options["clustering_algorithm"] == "Kuttruff2024":
        summary["Kuttruff buffer size"] = options["kuttruff_buffer_size"]
    if options["timewalk"]:
        summary["Cluster ToT cut (ns)"] = options["ctot_cut"]
    for key in WSL_FIELDS:
        if options[key]:
            summary[key.replace("_", " ")] = options[key]
    if "ratio_csv" in job.extra:
        summary["Ratio CSV peak pair"] = [peak["name"] for peak in options["peaks"][:2]]
    return job, summary


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
    decoder = "Standalone decoder" if options.get("backend") == "fallback" else NAME
    _style(fig, f"Ion yields · {decoder}")
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
    points = list(yield_ratios(rows, numerator, denominator))
    folders, x, y = zip(*points) if points else ([], [], [])
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
        files = [(csv_file, "Yield CSV"), (output / FIGURE_NAME, "Yield plot PNG")]
        if "ratio_csv" in job.extra:
            files.append((output / RATIO_NAME, "Default peak ratio CSV"))
        for path, label in files:
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


def build(jobs, *, fallback=False):
    peak_rows = []
    widgets = {}
    job_name = FALLBACK_JOB_NAME if fallback else JOB_NAME
    settings_key = "ion_pipeline_fallback" if fallback else "ion_pipeline"
    output_folder = "ion-pipeline-fallback" if fallback else "ion-pipeline"
    def restore_form(job):
        form = adopt_agent_form(job, settings_key)
        for key in widgets.keys() & form.keys() - {"peaks"}:
            widgets[key].value = form[key]
        if "peaks" in form:
            peak_store.value = form["peaks"]
            load_peaks()

    view = JobView(jobs, job_name=job_name, results_builder=show_results,
                   attach_builder=None if fallback else restore_form)
    theme.panel_header(job_name, "Plot GMD-gated peak yields from a TPX3 delay scan.",
                       view=view)

    def save(field, key):
        widgets[key] = field
        return persist(field, f"{settings_key}.{key}")

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

    def load_peaks():
        try:
            peaks = json.loads(peak_store.value)
            if not isinstance(peaks, list) or any(not isinstance(peak, dict) for peak in peaks):
                raise ValueError("Invalid peak settings.")
        except (TypeError, ValueError):
            peaks = DEFAULT_PEAKS
        peak_rows.clear()
        peak_list.clear()
        for peak in peaks:
            add_peak(peak)

    def run():
        try:
            if not root.value.strip() or not output.value.strip():
                raise ValueError("Choose the delay-scan root and an output directory.")
            options = {key: field.value for key, field in widgets.items()
                       if key not in ("input", "output", "peaks", "gate")}
            options.update(peaks=peak_values(), gate_gmd=gate.value)
            job = pipeline_job(root.value.strip(), output.value.strip(), options, fallback=fallback)
            view.attach(jobs.submit(job))
        except (OSError, ValueError, OverflowError) as exc:
            ui.notify(str(exc), type="warning")

    with theme.panel_card():
        theme.form_section("Delay scan")
        root = save(path_input("Dataset root (one subfolder per delay position)", browse="dir"), "input")
        output = save(path_input("Output directory", str(Path(shared.workspace_root()) / output_folder),
                                 browse="dir"), "output")
        with field_grid():
            save(num_input("Parallel delay positions", RUN_DEFAULTS["workers"], min=1, max=8, step=1), "workers")
            save(ui.checkbox("Keep intermediate HDF5 dumps", value=RUN_DEFAULTS["keep_dumps"]), "keep_dumps")
        options_form(save, fallback=fallback)
        theme.form_section("GMD2 gate")
        gate = save(ui.checkbox("Filter by GMD2", value=RUN_DEFAULTS["gate_gmd"]), "gate")
        with field_grid(3):
            gmd_min = save(num_input("GMD2 minimum (nA)", RUN_DEFAULTS["gmd_min"], step="any"), "gmd_min")
            gmd_max = save(num_input("GMD2 maximum (nA)", RUN_DEFAULTS["gmd_max"], step="any"), "gmd_max")
            shift = save(num_input("GMD2 alignment shift d (frames)", DEFAULT_SHIFT, step=1), "gmd_shift")
        for field in (gmd_min, gmd_max, shift):
            field.bind_enabled_from(gate, "value")
        tof_panel = "Ion TOF (fallback)" if fallback else "Ion TOF"
        ui.label(SHIFT_HELP + " The pipeline uses this shift for every delay position. "
                 f"Use the {tof_panel} GMD diagnostic to check the shift for your data. "
                 "Both GMD2 limits are inclusive.").classes("vmi-subtitle")
        theme.form_section("Peak windows")
        ui.label("Fits each Gaussian separately using the selected TOF source for each dataset. "
                 "Counts individual arrivals from selected frames in each window and divides by the selected frame count. "
                 "Overlapping windows can count the same arrival. With GMD2 filtering off, both fit sources use all frames."
                 ).classes("vmi-subtitle")
        with field_grid(3):
            save(ui.select({"ungated": "Ungated TOF (all frames)", "gated": "GMD2-gated TOF (selected frames)"},
                           value="ungated" if fallback else RUN_DEFAULTS["fit_tof"], label="TOF used to fit peak windows")
                 .props("outlined dense"), "fit_tof")
            save(num_input("Histogram bins per peak fit", RUN_DEFAULTS["fit_bins"], min=4, step=1), "fit_bins")
            save(num_input("Count-window half-width (σ)", RUN_DEFAULTS["window_sigma"], step="any"), "window_sigma")
        peak_store = save(ui.input(value=json.dumps(DEFAULT_PEAKS)), "peaks")
        peak_store.set_visibility(False)
        peak_list = ui.column().classes("w-full")
        load_peaks()
        with action_bar(view, "Run ion pipeline", on_click=run):
            ui.button("Add peak", icon="add", on_click=lambda: (add_peak({}), store_peaks())).props("flat")
    view.place_console()
