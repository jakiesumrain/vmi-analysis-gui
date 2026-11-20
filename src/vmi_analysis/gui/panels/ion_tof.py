"""Unfiltered time-of-flight spectra from LUNA Timepix3 dumps."""

import asyncio
import json
from math import isfinite
import sys

from nicegui import ui
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vmi_analysis.ion_tof import DEFAULT_REFERENCE, GMD_SHIFTS, REFERENCES, gmd_count_pairs, gmd_diagnostics, gmd_limits
from vmi_analysis.ion_acquisition import collection_reads
from vmi_analysis.pes import BASELINE_PARAMETERS, evaluate_fit_for_plot
from vmi_analysis.tof_bins import DEFAULT_BIN_SIZE_NS

from .. import shared, theme
from ..centroid_plots import _style, plot_options
from ..jobs import Job, ResourceBusyError
from .common import JobView, action_bar, field_grid, num_input, path_input, persist


JOB_NAME = "Ion TOF"
FILTER_JOB_NAME = "GMD-filtered TOF"
FIT_JOB_NAME = "TOF peak fit"
UNFILTERED_FIT_JOB_NAME = "Unfiltered TOF peak fit"
SOURCES = {"pixels": "Raw pixel hits", "clusters": "Clustered events", "both": "Both"}
SOURCE_COLORS = {"pixels": "#1e40af", "clusters": "#c2410c"}
REFERENCE_LABELS = {name: name.replace("Rising", " rising").replace("Falling", " falling")
                    for name in REFERENCES}


def _gmd_histogram(label, values):
    counts, edges = np.histogram(values[np.isfinite(values)], bins=80)
    fig = go.Figure(go.Bar(x=(edges[1:] + edges[:-1]) / 2, y=counts, width=np.diff(edges),
                           marker=dict(color="#1e40af", line=dict(color="#f5f7fa", width=0.5))))
    _style(fig, f"{label} histogram")
    fig.update_layout(height=360)
    fig.update_xaxes(title_text=f"{label} (nA)")
    fig.update_yaxes(title_text="Frames / bin", rangemode="tozero")
    return fig


def _correlation_plot(result, sources):
    fig = go.Figure()
    best_shifts = {}
    issue = result["gmd_alignment_issue"]
    if issue:
        fig.add_annotation(text="Alignment unavailable", x=0.5, y=0.5, xref="paper", yref="paper",
                           showarrow=False)
    else:
        shifts = result["gmd_shifts"]
        for source in sources:
            coefficients = result[f"{source}_gmd_correlation"]
            pairs = result[f"{source}_gmd_pairs"]
            for column, (statistic, dash) in enumerate((("Pearson", "solid"), ("Spearman", "dash"))):
                fig.add_scatter(
                    x=shifts, y=coefficients[:, column], customdata=pairs, mode="lines+markers",
                    line=dict(dash=dash, color=SOURCE_COLORS[source]), name=f"{SOURCES[source]} · {statistic}",
                    hovertemplate="d = %{x:+d}<br>Correlation %{y:.3f}<br>%{customdata:,} pairs",
                )
            pearson = coefficients[:, 0]
            maxima = np.isfinite(pearson) & (pearson == np.max(pearson, where=np.isfinite(pearson), initial=-np.inf))
            if maxima.sum() == 1:
                best_shifts[source] = int(shifts[maxima][0])
        fig.update_xaxes(tickmode="array", tickvals=shifts)
        fig.add_vline(x=0, line_dash="dot", line_color="#94a3b8")
    _style(fig, "GMD2 versus counts per frame")
    fig.update_layout(height=360, showlegend=not issue, legend=dict(orientation="h", y=-0.24),
                      margin=dict(b=110))
    fig.update_xaxes(title_text="Shift d (frames)")
    fig.update_yaxes(title_text="Correlation coefficient", range=[-1.05, 1.05])
    return fig, best_shifts


def _count_scatter(result, counts, shifts, boundaries, title):
    fig = go.Figure()
    if not result["gmd_alignment_issue"]:
        for source, shift in shifts.items():
            x, y = gmd_count_pairs(result["gmd2_na"], counts[source], shift, boundaries)
            index = GMD_SHIFTS.index(shift)
            pearson = result[f"{source}_gmd_correlation"][index, 0]
            score = f"r = {pearson:.3f}" if np.isfinite(pearson) else "r unavailable"
            fig.add_scattergl(x=x, y=y, mode="markers",
                              name=f"{SOURCES[source]} · d = {shift:+d} · {score}",
                              marker=dict(size=4, opacity=0.55, color=SOURCE_COLORS[source],
                                          symbol="circle" if source == "pixels" else "diamond"),
                              hovertemplate="%{x:.4g} nA · %{y:,} counts<extra>%{fullData.name}</extra>")
    if not fig.data:
        message = "Alignment unavailable" if result["gmd_alignment_issue"] else "No unique best Pearson shift"
        fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    _style(fig, title)
    fig.update_layout(height=520, showlegend=bool(fig.data), legend=dict(orientation="h", y=-0.24),
                      margin=dict(b=110))
    fig.update_xaxes(title_text="GMD2 (nA)")
    fig.update_yaxes(title_text="Counts per frame", rangemode="tozero")
    return fig


def _gmd_results(result, counts, boundaries):
    first, second = result["gmd1_na"], result["gmd2_na"]
    valid = np.isfinite(first) & np.isfinite(second)
    scatter = go.Figure(go.Scattergl(x=second[valid], y=first[valid], mode="markers",
                                    marker=dict(size=4, opacity=0.55)))
    _style(scatter, "GMD1 versus GMD2")
    scatter.update_layout(height=360)
    scatter.update_xaxes(title_text="GMD2 (nA)")
    scatter.update_yaxes(title_text="GMD1 (nA)")
    correlation, best_shifts = _correlation_plot(result, counts)
    zero = _count_scatter(result, counts, dict.fromkeys(counts, 0), boundaries, "Counts versus GMD2 · No shift")
    best = _count_scatter(result, counts, best_shifts, boundaries, "Counts versus GMD2 · Best Pearson shift")
    with ui.expansion("GMD diagnostic plots", icon="insights", value=False).classes("w-full"):
        with ui.element("div").classes("grid grid-cols-1 md:grid-cols-2 gap-4 w-full"):
            for fig in (_gmd_histogram("GMD1", first), _gmd_histogram("GMD2", second), scatter, correlation,
                        zero, best):
                ui.plotly(plot_options(fig)).classes("w-full min-w-0")
        ui.label("Correlation compares GMD2 row j with camera frame j + d across all assigned hits. "
                 "It uses positive GMD2 and counts, with at least 10 pairs per shift. "
                 "The scatter plots compare d = 0 with the unique best Pearson shift for each count source."
                 ).classes("vmi-subtitle")
    if len(best_shifts) == len(counts) and len(set(best_shifts.values())) == 1:
        return next(iter(best_shifts.values()))
    return None

def _load_gmd_diagnostics(result_path, metadata):
    with np.load(result_path, allow_pickle=False) as result:
        counts = {source: result[f"{source}_shot_counts"] for source in ("pixels", "clusters")
                  if f"{source}_shot_counts" in result}
        boundaries = result["segment_offsets"] if "segment_offsets" in result else None
    if not counts:
        raise ValueError("Plot TOF again to save the counts per trigger for GMD diagnostics.")
    return gmd_diagnostics(metadata, counts, boundaries), counts, boundaries


def _drag_interval(event):
    ranges = event.args or {}
    return next((sorted(limits) for axis, limits in ranges.items()
                 if axis.startswith("x") and len(limits) == 2), None)


def _mark_interval(fig, low, high, selecting):
    has_range = low is not None and high is not None and isfinite(low) and isfinite(high) and low < high
    if has_range:
        fig.add_vrect(x0=low, x1=high, fillcolor="#1e40af", opacity=0.15, line_width=0)
    fig.update_layout(dragmode="select" if selecting else "zoom", selectdirection="h", selections=[],
                      selectionrevision=f"{low}:{high}")


def _tof_figure(edges, spectra, title, *, selecting=False):
    fig = make_subplots(rows=len(spectra), cols=1, shared_xaxes=True,
                        subplot_titles=[SOURCES[source] for source in spectra])
    for row, (source, counts) in enumerate(spectra.items(), start=1):
        # An invisible bin-center trace enables selection without shifting the step histogram.
        fig.add_scatter(x=edges, y=np.r_[counts, 0], mode="lines", line_shape="hv",
                        name=SOURCES[source], hovertemplate="TOF %{x:.4~g} ns<br>Count %{y:,}<extra></extra>",
                        row=row, col=1)
        if selecting:
            fig.add_scatter(x=(edges[:-1] + edges[1:]) / 2, y=counts, mode="markers",
                            marker=dict(opacity=0), showlegend=False, hoverinfo="skip", row=row, col=1)
        fig.update_yaxes(title_text="Counts / bin", rangemode="tozero", row=row, col=1)
    _style(fig, title)
    fig.update_layout(height=400 if len(spectra) == 1 else 650)
    fig.update_xaxes(title_text="Time of flight (ns)", row=len(spectra), col=1)
    return fig


def _results(job):
    if job.status != "done":
        return
    result_path = shared.checked_path(job.extra["histogram"])
    with np.load(result_path, allow_pickle=False) as result:
        edges = result["edges"]
        sources = [name for name in ("pixels", "clusters") if name in result]
        ui.label(f'{job.extra["input_name"]} · {REFERENCE_LABELS[job.extra["reference"]]} '
                 f'· {int(result["triggers"]):,} triggers').classes("vmi-caption break-all")
        for source in sources:
            counts = result[source]
            ui.label(f'{SOURCES[source]} · {int(counts.sum()):,} plotted · '
                     f'{int(result[f"{source}_unmatched"]):,} with no preceding trigger · '
                     f'{int(result[f"{source}_outside"]):,} outside the TOF range').classes("vmi-subtitle")
        fig = _tof_figure(edges, {source: result[source] for source in sources}, "Time-of-flight spectrum")
        ui.plotly(plot_options(fig)).classes("w-full")


def build(jobs):
    view = JobView(jobs, job_name=JOB_NAME, results_builder=_results)
    theme.panel_header(JOB_NAME, "Plot unfiltered time-of-flight spectra from a Timepix3 dump.", view=view)

    def save(widget, key):
        return persist(widget, f"ion_tof.{key}")

    def use_latest():
        candidates = [job for job in jobs.jobs() if job.name == "TPX3 Dump" and job.status == "done"]
        if not candidates:
            ui.notify("Complete a TPX3 dump first or choose a saved collection.", type="warning")
            return
        latest = max(candidates, key=lambda job: job.finished)
        input_file.value = latest.extra.get("collection", latest.extra["output"])
        set_metadata(latest.extra.get("collection", latest.extra.get("metadata", "")))

    def run():
        try:
            if not input_file.value.strip():
                raise ValueError("Choose a LUNA HDF5 file or acquisition collection.")
            source_file = shared.checked_path(input_file.value.strip())
            if not source_file.is_file():
                raise ValueError("The selected HDF5 file does not exist.")
            for path in collection_reads(source_file):
                shared.checked_path(path)
            width = bin_size.value
            if width is None or not isfinite(width) or width <= 0:
                raise ValueError("TOF bin size must be finite and greater than zero.")
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
                       "--bin-size", str(width), "--minimum", str(low)]
            if high is not None:
                job.cmd += ["--maximum", str(high)]
            job.extra.update(histogram=str(result), reference=reference.value, input_name=source_file.name,
                             source_file=str(source_file))
            view.attach(jobs.submit(job))
            reset_gmd()
            if source_file.suffix.lower() == ".json":
                set_metadata(str(source_file))
        except (OSError, ValueError) as exc:
            ui.notify(str(exc), type="warning")

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
        with action_bar(view, "Plot TOF", on_click=run):
            pass
    view.place_console()
    reset_gmd, set_metadata = _gmd_section(jobs, view, save)


def _gmd_section(jobs, view, save):
    loaded_job = None
    loaded_metadata = None
    readings = None

    def reset_gmd():
        nonlocal loaded_job, loaded_metadata, readings
        loaded_job = loaded_metadata = readings = None
        gmd_display.clear()
        filtered_display.clear()
        filter_card.set_visibility(False)

    async def plot_gmd():
        nonlocal loaded_job, loaded_metadata, readings
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
            with jobs.preview_read_access([result_path, *collection_reads(metadata)]):
                result, counts, boundaries = await asyncio.to_thread(_load_gmd_diagnostics, result_path, metadata)
            if view.job is job and metadata_file.value.strip() == metadata_value:
                reset_gmd()
                loaded_job, loaded_metadata = job, metadata
                readings = result["gmd2_na"]
                with gmd_display:
                    ui.label(f'{job.extra["input_name"]} · {REFERENCE_LABELS[job.extra["reference"]]} '
                             f'· {metadata.name}').classes("vmi-caption break-all")
                    suggested = _gmd_results(result, counts, boundaries)
                    if issue := result["gmd_alignment_issue"]:
                        ui.label(f"Cannot filter by GMD2. {issue}").classes("text-negative")
                gmd_low.value = gmd_high.value = None
                shift.value = suggested
                filter_view.action_button.set_visibility(not issue)
                filter_card.set_visibility(True)
                refresh_gmd_plot()
                if (finished := filter_view.job) and finished.status == "done":
                    show_filtered(finished)
        except (OSError, ValueError, ResourceBusyError) as exc:
            ui.notify(str(exc), type="warning")
        finally:
            gmd_button.enable()

    with theme.panel_card():
        theme.form_section("GMD diagnostics")
        ui.label("Uses the count source and trigger reference from the latest TOF plot. Choose the metadata "
                 "exported with that dump. A collection loads each segment's matching metadata. "
                 "GMD pairing stays within each acquisition.").classes("vmi-subtitle")
        metadata_file = save(path_input("Matching acquisition metadata HDF5 or collection",
                                        browse="file", filters=[".h5", ".hdf5", ".json"]), "metadata")
        gmd_button = ui.button("Plot GMD diagnostics", icon="show_chart", on_click=plot_gmd)
        gmd_display = ui.column().classes("w-full gap-4")

    def refresh_gmd_plot():
        if readings is None:
            return
        fig = _gmd_histogram("GMD2", readings)
        try:
            low, high = gmd_limits(readings, gmd_low.value, gmd_high.value)
        except ValueError:
            low = high = None
        _mark_interval(fig, low, high, gmd_mode.value == "Select GMD2 range")
        fig.update_layout(uirevision=loaded_job.id)
        gmd_plot.update_figure(plot_options(fig))

    def select_gmd(event):
        if gmd_mode.value == "Select GMD2 range" and (limits := _drag_interval(event)):
            gmd_low.value, gmd_high.value = limits
            if filter_view.action_button.visible:
                run_filter()

    def run_filter():
        if filter_view.job and filter_view.job.status in ("running", "queued"):
            ui.notify("Wait for the current filter or stop it before applying a new range.", type="warning")
            return
        try:
            if loaded_job is None or loaded_job is not view.job:
                raise ValueError("Load GMD diagnostics for the latest TOF plot.")
            low, high = gmd_limits(readings, gmd_low.value, gmd_high.value)
            if shift.value is None:
                raise ValueError("Select an alignment shift before filtering.")
            if "source_file" not in loaded_job.extra:
                raise ValueError("Plot TOF again before filtering this older result.")
            source_file = shared.checked_path(loaded_job.extra["source_file"])
            original = shared.checked_path(loaded_job.extra["histogram"])
            metadata = shared.checked_path(loaded_metadata)
            job = Job(FILTER_JOB_NAME, cwd=str(shared.workspace_root()))
            result = shared.generated_dir() / f"ion-tof-filtered-{job.id}.npz"
            job.cmd = [sys.executable, "-m", "vmi_analysis.ion_tof", "--input", str(source_file),
                       "--output", str(result), "--tof-result", str(original), "--metadata", str(metadata),
                       "--reference", loaded_job.extra["reference"], f"--gmd-min={low}", f"--gmd-max={high}",
                       "--gmd-shift", str(shift.value)]
            job.extra.update(histogram=str(result), parent_id=loaded_job.id, metadata=str(metadata),
                             gmd_min=low, gmd_max=high, gmd_shift=shift.value,
                             source_file=str(source_file), reference=loaded_job.extra["reference"])
            job = jobs.submit(job)
            filtered_display.clear()
            filter_view.attach(job)
        except (OSError, ValueError, ResourceBusyError) as exc:
            ui.notify(str(exc), type="warning")

    def show_filtered(job):
        if (job.status != "done" or loaded_job is None or loaded_job is not view.job
                or job.extra["parent_id"] != loaded_job.id or job.extra["metadata"] != str(loaded_metadata)):
            return
        with np.load(shared.checked_path(job.extra["histogram"]), allow_pickle=False) as result:
            edges = result["edges"]
            spectra = {source: result[source] for source in ("pixels", "clusters") if source in result}
            selected = int(result["selected_triggers"])
            total = int(result["triggers"])
        original_path = shared.checked_path(loaded_job.extra["histogram"])
        with np.load(original_path, allow_pickle=False) as original:
            unfiltered_spectra = {source: original[source] for source in spectra}
        filtered_display.clear()
        with filtered_display:
            theme.form_section("GMD2-filtered TOF")
            ui.label(f'GMD2 {job.extra["gmd_min"]:g}–{job.extra["gmd_max"]:g} nA · '
                     f'shift {job.extra["gmd_shift"]:+d} · {selected:,} of {total:,} triggers').classes("vmi-caption")
            if not any(counts.any() for counts in spectra.values()):
                ui.label("No counts in the TOF range for these triggers.").classes("vmi-subtitle")
                fig = _tof_figure(edges, spectra, "GMD2-filtered time-of-flight spectra")
                ui.plotly(plot_options(fig)).classes("w-full")
            else:
                _peak_fitting(jobs, job, edges, spectra)
            with ui.expansion("Unfiltered TOF", icon="show_chart", value=False).classes("w-full gap-4").props(
                    'data-tof-fit-source="unfiltered"'):
                ui.label("Fits the unfiltered spectrum to define peak windows. Counting and normalization "
                         f"still use the {selected:,} GMD2-selected triggers above.").classes("vmi-subtitle")
                if selected and any(counts.any() for counts in unfiltered_spectra.values()):
                    _peak_fitting(jobs, job, edges, unfiltered_spectra, unfiltered_result=original_path)
                else:
                    ui.plotly(plot_options(_tof_figure(edges, unfiltered_spectra,
                                                       "Unfiltered time-of-flight spectra"))).classes("w-full")
                    ui.label("Select triggers with TOF data before fitting peak yields.").classes("vmi-caption")

    filter_view = JobView(jobs, job_name=FILTER_JOB_NAME, results_builder=show_filtered)
    with theme.panel_card() as filter_card:
        filter_card.set_visibility(False)
        theme.form_section("GMD2 range")
        gmd_mode = ui.toggle(["Zoom", "Select GMD2 range"], value="Zoom",
                             on_change=lambda _: refresh_gmd_plot())
        gmd_plot = ui.plotly(go.Figure()).classes("w-full")
        gmd_plot.on("plotly_selected", select_gmd, js_handler="event => emit(event?.range ?? null)")
        with field_grid(3):
            gmd_low = num_input("GMD2 min (nA, blank = data min)", None, step="any")
            gmd_high = num_input("GMD2 max (nA, blank = data max)", None, step="any")
            shift = ui.select({d: f"{d:+d}" for d in GMD_SHIFTS}, label="Alignment shift d (frames)") \
                .props("outlined dense")
        for field in (gmd_low, gmd_high):
            field.on_value_change(lambda _: refresh_gmd_plot())
        ui.label("GMD2 row j selects camera frame j + d. Drag to apply a range, or enter limits and "
                 "click Apply GMD2 range. Pairing stays within each acquisition. The limits are inclusive."
                 ).classes("vmi-subtitle")
        filter_view.run_button("Apply GMD2 range", on_click=run_filter)
        filter_view.place_console("Filtered TOF console")
        filtered_display = ui.column().classes("w-full gap-4")
    metadata_file.on_value_change(lambda _: reset_gmd())
    def set_metadata(value):
        metadata_file.value = value

    return reset_gmd, set_metadata


def _peak_fitting(jobs, filtered_job, edges, spectra, *, unfiltered_result=None):
    peak_rows = []
    fit_job_name = UNFILTERED_FIT_JOB_NAME if unfiltered_result is not None else FIT_JOB_NAME
    title = "Unfiltered time-of-flight spectra" if unfiltered_result is not None else "GMD2-filtered time-of-flight spectra"

    def refresh_plot():
        selecting = mode.value == "Select fit range"
        fig = _tof_figure(edges, spectra, title, selecting=selecting)
        _mark_interval(fig, fit_low.value, fit_high.value, selecting)
        fig.update_layout(uirevision=fit_job_name)
        plot.update_figure(plot_options(fig))

    def select_fit_range(event):
        if mode.value == "Select fit range" and (limits := _drag_interval(event)):
            fit_low.value, fit_high.value = limits

    def clear_fit_range():
        fit_low.value = fit_high.value = None
        fit_view.clear_results()

    def add_peak():
        with peak_list, ui.row().classes("w-full items-center gap-3") as row:
            center = num_input("Peak center (ns)", None, step="any")
            gaussian = num_input("Gaussian FWHM (ns)", None, min=0, step="any")
            fields = (center, gaussian)
            peak_rows.append(fields)

            def remove():
                peak_rows.remove(fields)
                row.delete()

            ui.button(icon="remove_circle_outline", on_click=remove).props('flat round aria-label="Remove peak"')

    def fit():
        try:
            region = fit_low.value, fit_high.value
            if None in region or not np.isfinite(region).all() or region[0] >= region[1]:
                raise ValueError("Select a fit range or enter increasing, finite TOF limits.")
            if not targets.value:
                raise ValueError("Choose at least one TOF spectrum.")
            estimates = []
            for center, gaussian in peak_rows:
                if center.value is None and gaussian.value is None:
                    continue
                if center.value is None or gaussian.value is None:
                    raise ValueError("Enter a center and Gaussian FWHM for each peak.")
                if not np.isfinite([center.value, gaussian.value]).all():
                    raise ValueError("Use finite peak centers and widths.")
                estimates.append((center.value, gaussian.value))
            if "source_file" not in filtered_job.extra:
                raise ValueError("Apply the GMD2 range again before fitting this older spectrum.")
            source = shared.checked_path(filtered_job.extra["source_file"])
            filtered = shared.checked_path(filtered_job.extra["histogram"])
            spec = dict(sources=targets.value, region=region, estimates=estimates, baseline=baseline.value)
            job = Job(fit_job_name, cwd=str(shared.workspace_root()))
            output = shared.generated_dir() / f"ion-tof-peaks-{job.id}.npz"
            job.cmd = [sys.executable, "-m", "vmi_analysis.ion_tof", "--input", str(source),
                       "--filtered-result", str(filtered), "--output", str(output),
                       "--reference", filtered_job.extra["reference"], "--fit-spec", json.dumps(spec)]
            if unfiltered_result is not None:
                job.cmd += ["--tof-result", str(shared.checked_path(unfiltered_result))]
            job.extra.update(peak_result=str(output), filtered_id=filtered_job.id, baseline=baseline.value,
                             fit_range=region)
            fit_view.attach(jobs.submit(job))
        except (OSError, ValueError, ResourceBusyError) as exc:
            ui.notify(str(exc), type="warning")

    def show_fits(job):
        if job.status != "done" or job.extra["filtered_id"] != filtered_job.id:
            return
        with np.load(shared.checked_path(job.extra["peak_result"]), allow_pickle=False) as data:
            for source, error in json.loads(data["fit_errors"].item()).items():
                ui.label(f"{SOURCES[source]}: {error}").classes("text-negative")
            results = {source: (data[f"{source}_curve"], json.loads(data[f"{source}_peaks"].item()))
                       for source in spectra if f"{source}_curve" in data}
            if results:
                low, high = job.extra["fit_range"]
                _show_peak_fits(results, job.extra["baseline"], int(data["selected_triggers"]), low, high)

    with ui.row().classes("items-center gap-2"):
        mode = ui.toggle(["Zoom", "Select fit range"], value="Zoom", on_change=lambda _: refresh_plot())
        ui.button("Clear fit range", on_click=clear_fit_range).props("flat")
    plot = ui.plotly(go.Figure()).classes("w-full")
    plot.on("plotly_selected", select_fit_range, js_handler="event => emit(event?.range ?? null)")
    theme.form_section("Gaussian TOF peak fitting")
    ui.label("Drag a fit range on the spectrum above or enter the limits. Leave peak estimates blank "
             "for automatic detection.").classes("vmi-subtitle")
    with field_grid():
        fit_low = num_input("Fit TOF min (ns)", None, step="any")
        fit_high = num_input("Fit TOF max (ns)", None, step="any")
        targets = ui.select({source: SOURCES[source] for source in spectra}, value=list(spectra),
                            label="TOF spectra to fit", multiple=True).props("outlined dense use-chips")
        baseline = ui.select(list(BASELINE_PARAMETERS), value="Constant", label="Baseline").props("outlined dense")
    for field in (fit_low, fit_high):
        field.on_value_change(lambda _: refresh_plot())
    peak_list = ui.column().classes("w-full")
    fit_view = JobView(jobs, job_name=fit_job_name, results_builder=show_fits)
    with ui.row().classes("gap-2"):
        ui.button("Add peak estimate", icon="add", on_click=add_peak).props("flat")
        fit_view.run_button("Fit TOF peaks", on_click=fit)
    fit_view.place_console("TOF peak fit results")
    refresh_plot()


def _show_peak_fits(results, model, selected, fit_low, fit_high):
    ui.label(f"Fit range {fit_low:g}–{fit_high:g} ns · {selected:,} GMD-selected triggers").classes("vmi-caption")
    ui.label("Counts use individual arrivals from μ − 3σ (inclusive) to μ + 3σ (exclusive). "
             "Counts include background and any overlapping peaks. Yield divides counts by all selected triggers, "
             "including those with zero hits.").classes("vmi-subtitle")
    columns = [("spectrum", "Spectrum"), ("peak", "Peak"), ("center", "Center (ns)"),
               ("fwhm", "FWHM (ns)"), ("low", "Window min (ns)"), ("high", "Window max (ns)"),
               ("counts", "Counts in window"), ("yield_per_trigger", "Counts per selected trigger"),
               ("bound", "At bound")]
    rows = [dict(spectrum=SOURCES[source], **peak) for source, (_, peaks) in results.items() for peak in peaks]
    ui.table(columns=[dict(name=key, field=key, label=label) for key, label in columns],
             rows=[dict(id=index, **{key: f"{value:.6g}" if isinstance(value, float) else value
                                     for key, value in row.items()})
                   for index, row in enumerate(rows)], row_key="id").classes("w-full")
    target = ui.select({source: SOURCES[source] for source in results}, value=next(iter(results)),
                       label="Inspect TOF fit").props("outlined dense")
    chart = ui.plotly(go.Figure()).classes("w-full")

    def redraw():
        (x, fitted, background, residual), peaks = results[target.value]
        grid, total, baseline, components = evaluate_fit_for_plot(x, background, peaks)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.74, 0.26], vertical_spacing=0.09)
        fig.add_scatter(x=x, y=fitted + residual, mode="markers", name="Counts", marker=dict(size=8), row=1, col=1)
        fig.add_scatter(x=grid, y=total, mode="lines", name="Total fit", row=1, col=1)
        if model != "None":
            fig.add_scatter(x=grid, y=baseline, mode="lines", name="Baseline", line=dict(dash="dot"), row=1, col=1)
        suffix = " + baseline" if model != "None" else ""
        for index, peak in enumerate(components, 1):
            fig.add_scatter(x=grid, y=baseline + peak, mode="lines", name=f"Peak {index}{suffix}",
                            line=dict(dash="dash"), row=1, col=1)
        fig.add_scatter(x=x, y=residual, name="Residual", showlegend=False, row=2, col=1)
        _style(fig, "TOF peak fit and residuals")
        fig.update_layout(height=570, showlegend=True)
        fig.update_xaxes(title_text="Time of flight (ns)", row=2, col=1)
        fig.update_yaxes(title_text="Counts / bin", row=1, col=1)
        fig.update_yaxes(title_text="Residual", row=2, col=1)
        chart.update_figure(plot_options(fig))

    target.on_value_change(lambda _: redraw())
    redraw()
