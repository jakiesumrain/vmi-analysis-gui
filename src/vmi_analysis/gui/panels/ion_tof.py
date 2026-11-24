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
from ..agent_panels import (PanelOperation, PanelView, RunCapture, boolean, number, require_completed_job,
                            resolve_path)
from ..centroid_plots import _style, plot_options
from ..jobs import Checkpoint, Job, ResourceBusyError
from . import ion_luna
from .common import (JobView, action_bar, adopt_agent_form, checkpoint_view, field_grid, num_input,
                     path_input, persist)


JOB_NAME = "Ion TOF"
FILTER_JOB_NAME = "GMD-filtered TOF"
FIT_JOB_NAME = "TOF peak fit"
UNFILTERED_FIT_JOB_NAME = "Unfiltered TOF peak fit"
GMD_PAIRING = "GMD2 row j selects camera frame j + d within the same acquisition."
SOURCES = {"pixels": "Raw pixel hits", "clusters": "Clustered events", "both": "Both"}
SOURCE_COLORS = {"pixels": "#1e40af", "clusters": "#c2410c"}
REFERENCE_LABELS = {name: name.replace("Rising", " rising").replace("Falling", " falling")
                    for name in REFERENCES}


MEASUREMENT_PARAMETERS = {
    "source": {"type": "string", "choices": ["pixels", "clusters"], "description": "Required if both traces were saved"},
    "region_ns": {"type": "number pair", "unit": "ns", "description": "Increasing limits around the visible peak"},
}


def _measure_tof(jobs, job, parameters):
    if parameters.keys() - MEASUREMENT_PARAMETERS.keys():
        raise ValueError("Use source and region_ns for TOF measurements.")
    region = parameters.get("region_ns")
    if not isinstance(region, list) or len(region) != 2:
        raise ValueError("Enter region_ns as two limits around the visible peak.")
    low, high = map(number, region)
    if low >= high:
        raise ValueError("Use increasing TOF limits.")
    fitted = job.name in (FIT_JOB_NAME, UNFILTERED_FIT_JOB_NAME)
    path = shared.checked_path(job.extra["peak_result" if fitted else "histogram"])
    with jobs.preview_read_access([path]), np.load(path, allow_pickle=False) as data:
        available = [source for source in SOURCE_COLORS if (f"{source}_peaks" if fitted else source) in data]
        source = parameters.get("source") or (available[0] if len(available) == 1 else None)
        if source not in available:
            raise ValueError("Choose an available TOF trace. " + ", ".join(available))
        result = {"source": source, "region_ns": [low, high], "unit": "ns",
                  "spectrum": ("GMD2-gated" if job.name in (FILTER_JOB_NAME, FIT_JOB_NAME) else "ungated")}
        if "selected_triggers" in data:
            result["selected_triggers"] = int(data["selected_triggers"])
        if fitted:
            peaks = json.loads(data[f"{source}_peaks"].item())
            result.update(method="Saved Gaussian fit", peaks=[
                {key: peak[key] for key in ("center", "fwhm", "bound", "counts", "yield_per_trigger")}
                for peak in peaks if low <= peak["center"] <= high
            ])
            return result
        edges = data["edges"]
        counts = data[source]
        centers = edges[:-1] + np.diff(edges) / 2
        indices = np.flatnonzero((centers >= low) & (centers <= high))
        result.update(method="Highest histogram bin with its center inside the requested interval",
                      strongest_bin=None)
        if indices.size and (maximum := int(counts[indices].max())) > 0:
            maxima = indices[counts[indices] == maximum]
            index = maxima[0]
            result["strongest_bin"] = {
                "center_ns": float(centers[index]), "edges_ns": edges[index:index + 2].tolist(),
                "counts": maximum, "bins_tied_for_maximum": len(maxima),
            }
        return result


def _capture_plot(chart, job, ask_agent):
    if ask_agent is None:
        return

    async def capture():
        button.disable()
        try:
            filename = f"ion-tof-{job.id}.png"
            result = await ui.run_javascript(
                f"return await window.vmiAgentCapturePlot(getElement({chart.id}), {json.dumps(filename)})",
                timeout=30,
            )
            if not isinstance(result, dict) or "paths" not in result:
                raise ValueError(result.get("error", "Could not capture this plot.") if isinstance(result, dict)
                                 else "Could not capture this plot.")
            ask_agent(RunCapture(job.id, result["paths"]))
        except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
            ui.notify(str(exc), type="warning")
        finally:
            if not button.is_deleted:
                button.enable()

    button = ui.button("Capture plot for Agent", icon="add_a_photo", on_click=capture).props("flat")


def agent_operations():
    spectrum = {
        "panel": JOB_NAME,
        "summary": "Use the preceding selected TDC trigger in each acquisition to calculate TOF. "
                   "Histogram raw pixel hits, clustered events, or both.",
        "parameters": {
            "input": {"type": "path", "description": "LUNA HDF5 or acquisition collection"},
            "dump_job_id": {"type": "job ID", "description": "Completed TPX3 Dump job ID, instead of input"},
            "source": {"type": "string", "default": "pixels", "choices": list(SOURCES)},
            "reference": {"type": "string", "default": DEFAULT_REFERENCE, "choices": list(REFERENCES)},
            "bin_size_ns": {"type": "number", "unit": "ns", "default": DEFAULT_BIN_SIZE_NS},
            "minimum_ns": {"type": "number", "unit": "ns", "default": 0},
            "maximum_ns": {"type": "number or null", "unit": "ns", "default": None},
        },
        "manual": JOB_NAME,
    }
    gate = {
        "panel": JOB_NAME,
        "summary": "Apply an inclusive GMD2 gate. Metadata row j selects camera frame j + d "
                   "within the same acquisition. Use the alignment diagnostic to choose d.",
        "parameters": {
            "tof_job_id": {"type": "job ID", "description": "Completed Ion TOF job ID"},
            "metadata": {"type": "path", "description": "Matching metadata, optional after a TPX3 Dump handoff"},
            "gmd_min_na": {"type": "number", "unit": "nA"},
            "gmd_max_na": {"type": "number", "unit": "nA"},
            "shift": {"type": "integer", "unit": "frames", "choices": list(GMD_SHIFTS)},
        },
    }
    fit = {
        "panel": JOB_NAME,
        "summary": "Fit Gaussian peaks on gated or unfiltered TOF. Count individual arrivals within "
                   "the fitted windows on the GMD2-gated frames in either case.",
        "parameters": {
            "gate_job_id": {"type": "job ID", "description": "Completed GMD2-gated TOF job ID"},
            "sources": {"type": "list of strings", "default": ["pixels"], "choices": ["pixels", "clusters"]},
            "region_ns": {"type": "number pair", "unit": "ns", "description": "[Minimum, maximum]"},
            "peaks_ns": {"type": "list of number pairs", "unit": "ns", "default": [],
                         "description": "[Center, Gaussian FWHM] for each peak. Empty means automatic."},
            "baseline": {"type": "string", "default": "Constant", "choices": list(BASELINE_PARAMETERS)},
            "unfiltered": {"type": "boolean", "default": False},
        },
    }
    for description in (spectrum, gate, fit):
        description["measurements"] = MEASUREMENT_PARAMETERS
    return (
        PanelOperation("ion_tof", spectrum, _spectrum, (JOB_NAME,), outputs=("histogram",),
                       measure_result=_measure_tof),
        PanelOperation("ion_tof_gate", gate, _gate, (FILTER_JOB_NAME,), outputs=("histogram",),
                       measure_result=_measure_tof),
        PanelOperation("ion_tof_fit", fit, _fit, (FIT_JOB_NAME, UNFILTERED_FIT_JOB_NAME), outputs=("peak_result",),
                       measure_result=_measure_tof),
    )


def _spectrum(jobs, params, *, pending_input=False):
    dump_id, path = params.get("dump_job_id"), params.get("input")
    if bool(dump_id) == bool(path):
        raise ValueError("Supply one TPX3 Dump job ID or one input file.")
    metadata = None
    if dump_id:
        dump = require_completed_job(jobs, dump_id, name=ion_luna.JOB_NAME)
        path, metadata = ion_luna.completed_dump_input(dump)
    source = params.get("source", "pixels")
    reference = params.get("reference", DEFAULT_REFERENCE)
    width = number(params.get("bin_size_ns", DEFAULT_BIN_SIZE_NS))
    low = number(params.get("minimum_ns", 0))
    high = params.get("maximum_ns")
    if high is not None:
        high = number(high)
    job = spectrum_job(resolve_path(path), source=source, reference=reference, width=width, low=low, high=high,
                               pending_input=pending_input)
    if not metadata and job.extra["source_file"].lower().endswith(".json"):
        metadata = job.extra["source_file"]
    if metadata:
        job.extra["metadata"] = str(resolve_path(metadata))
    job.extra["panel_form"] = {"input": job.extra["source_file"], "source": source, "reference": reference,
                               "bin_size_ns": width, "minimum": low, "maximum": high,
                               "metadata": job.extra.get("metadata", "")}
    return job, {"LUNA input": job.extra["source_file"], "Count": SOURCES[source],
                 "Trigger reference": reference, "Bin size (ns)": width, "TOF minimum (ns)": low,
                 "TOF maximum (ns)": high if high is not None else "Automatic"}


def _gate(jobs, params):
    parent = require_completed_job(jobs, params["tof_job_id"], name=JOB_NAME)
    metadata = params.get("metadata") or parent.extra.get("metadata")
    if not metadata and parent.extra["source_file"].lower().endswith(".json"):
        metadata = parent.extra["source_file"]
    low, high = number(params["gmd_min_na"]), number(params["gmd_max_na"])
    shift = params["shift"]
    job = filtered_job(parent, resolve_path(metadata), low, high, shift)
    return job, {"TOF job": parent.id, "Metadata": job.extra["metadata"], "GMD2 minimum (nA)": low,
                 "GMD2 maximum (nA)": high, "Alignment shift d (frames)": shift}


def _fit(jobs, params):
    parent = require_completed_job(jobs, params["gate_job_id"], name=FILTER_JOB_NAME)
    targets = params.get("sources", ["pixels"])
    if not isinstance(targets, list) or not all(isinstance(item, str) for item in targets):
        raise ValueError("Supply a list of spectrum sources.")
    region = params["region_ns"]
    estimates = params.get("peaks_ns", [])
    if not isinstance(region, list) or not isinstance(estimates, list) or len(estimates) > 20:
        raise ValueError("Supply numeric fit bounds and up to 20 pairs of peak estimates.")
    region = [number(value) for value in region]
    parsed = []
    for pair in estimates:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("Supply [center, Gaussian FWHM] for each peak estimate.")
        parsed.append([number(value) for value in pair])
    baseline = params.get("baseline", "Constant")
    unfiltered = boolean(params.get("unfiltered", False))
    original = require_completed_job(jobs, parent.extra["parent_id"], name=JOB_NAME) if unfiltered else None
    job = peak_fit_job(parent, targets, region, parsed, baseline,
                              unfiltered_result=original.extra["histogram"] if original else None)
    return job, {"Gated TOF job": parent.id, "Spectra": targets, "Fit range (ns)": region,
                 "Peak estimates (ns)": parsed or "Automatic", "Baseline": baseline,
                 "Fit spectrum": "Unfiltered" if original else "GMD2-gated",
                 "Counting and normalization": "GMD2-gated"}


def _gmd_histogram(label, values):
    counts, edges = np.histogram(values[np.isfinite(values)], bins=80)
    fig = go.Figure(go.Bar(x=(edges[1:] + edges[:-1]) / 2, y=counts, width=np.diff(edges),
                           marker=dict(color="#1e40af", line=dict(color="#f5f7fa", width=0.5))))
    _style(fig, f"{label} histogram")
    fig.update_layout(height=360)
    fig.update_xaxes(title_text=f"{label} (nA)")
    fig.update_yaxes(title_text="Frames / bin", rangemode="tozero")
    return fig


def _automatic_shift(result, sources):
    """Require agreement and a separated positive maximum for both statistics."""
    shifts = set()
    for source in sources:
        for coefficients in result[f"{source}_gmd_correlation"].T:
            valid = np.flatnonzero(np.isfinite(coefficients))
            order = valid[np.argsort(coefficients[valid])]
            if (len(order) < 2 or coefficients[order[-1]] < 0.5
                    or coefficients[order[-1]] - coefficients[order[-2]] < 0.05):
                return None
            shifts.add(int(result["gmd_shifts"][order[-1]]))
    return shifts.pop() if len(shifts) == 1 else None


def workflow_alignment(tof_job, gate_job):
    """Prepare a gate on the exact histogram and metadata produced by this workflow."""
    server, account = shared.SERVER, shared.current_account()
    histogram = gate_job.cmd[gate_job.cmd.index("--tof-result") + 1]
    metadata = gate_job.extra["metadata"]
    shift_arg = gate_job.cmd.index("--gmd-shift") + 1

    def apply(shift):
        gate_job.cmd[shift_arg] = str(shift)
        gate_job.extra["gmd_shift"] = shift

    def check():
        if account:
            for path in collection_reads(metadata):
                server.path(account, path)
        result, counts, _ = _load_gmd_diagnostics(histogram, metadata)
        if issue := result["gmd_alignment_issue"]:
            raise ValueError(f"Cannot filter by GMD2. {issue}")
        if (shift := _automatic_shift(result, counts)) is not None:
            apply(shift)
            gate_job.extra["gmd_shift_source"] = "automatic"
            return None

        sources = tuple(counts)
        chart_data = {key: value for key, value in result.items()
                      if key in ("gmd_shifts", "gmd_alignment_issue")
                      or key.endswith(("_gmd_correlation", "_gmd_pairs"))}

        def confirm(shift):
            apply(shift)
            gate_job.extra["gmd_shift_source"] = "confirmed"

        def render(jobs, job, checkpoint):
            ui.label(checkpoint.summary["message"]).classes("vmi-subtitle").props('role="status"')
            ui.label(GMD_PAIRING).classes("vmi-caption")
            ui.label(checkpoint.summary["reason"]).classes("vmi-caption")
            ui.plotly(plot_options(_correlation_plot(chart_data, sources)[0])).classes("w-full min-w-0")

            def resume():
                try:
                    jobs.resolve_checkpoint(job.id, checkpoint, field.value)
                except ValueError as exc:
                    ui.notify(str(exc), type="warning")
                else:
                    ui.notify(f"Continuing with alignment shift d = {field.value:+d}.", type="positive")

            with ui.row().classes("items-center gap-3"):
                field = ui.select({d: f"{d:+d}" for d in checkpoint.choices}, label="Shift to confirm (frames)")
                field.props("outlined dense").classes("min-w-52")
                submit = ui.button("Confirm shift and continue", on_click=resume)
                submit.bind_enabled_from(field, "value", lambda value: value is not None)

        coefficients = {
            source: [[float(value) if np.isfinite(value) else None for value in row]
                     for row in result[f"{source}_gmd_correlation"]]
            for source in counts
        }
        reason = ("The diagnostic requires at least 10 pairs with positive GMD2 and counts per shift."
                  if not any(np.isfinite(chart_data[f"{source}_gmd_correlation"]).any() for source in sources)
                  else "The correlation peaks are weak, close together, or disagree on the best shift.")
        return Checkpoint(
            summary={
                "message": "Confirm a GMD2 alignment shift before filtering.",
                "pairing": GMD_PAIRING,
                "reason": reason,
                "tof_job_id": tof_job.id,
                "shifts": list(GMD_SHIFTS),
                "correlations": coefficients,
                "statistics": ["Pearson", "Spearman"],
            },
            choices=GMD_SHIFTS, apply=confirm, render=render,
        )

    return check


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


def _results(job, *, ask_agent=None):
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
        chart = ui.plotly(plot_options(fig)).classes("w-full")
        _capture_plot(chart, job, ask_agent)


def spectrum_job(path, *, source="pixels", reference=DEFAULT_REFERENCE, width=DEFAULT_BIN_SIZE_NS,
                 low=0, high=None, pending_input=False):
    if not path:
        raise ValueError("Choose a LUNA HDF5 file or acquisition collection.")
    source_file = shared.checked_path(path)
    if not pending_input:
        if not source_file.is_file():
            raise ValueError("The selected HDF5 file or collection does not exist.")
        for member in collection_reads(source_file):
            shared.checked_path(member)
    if source not in SOURCES or reference not in REFERENCES:
        raise ValueError("Choose a valid count source and trigger reference.")
    if width is None or not isfinite(width) or width <= 0:
        raise ValueError("TOF bin size must be finite and greater than zero.")
    if low is None or not isfinite(low) or low < 0:
        raise ValueError("TOF minimum must be finite and at least zero.")
    if high is not None and (not isfinite(high) or high <= low):
        raise ValueError("TOF maximum must be greater than the minimum.")
    job = Job(JOB_NAME, cwd=str(shared.workspace_root()))
    result = shared.generated_dir() / f"ion-tof-{job.id}.npz"
    job.cmd = [sys.executable, "-m", "vmi_analysis.ion_tof", "--input", str(source_file),
               "--output", str(result), "--source", source, "--reference", reference,
               "--bin-size", str(width), "--minimum", str(low)]
    if high is not None:
        job.cmd += ["--maximum", str(high)]
    job.extra.update(histogram=str(result), reference=reference, input_name=source_file.name,
                     source_file=str(source_file), source=source)
    return job


def filtered_job(original_job, metadata, low, high, shift, *, pending_metadata=False):
    if "source_file" not in original_job.extra:
        raise ValueError("Plot TOF again before filtering this older result.")
    source = shared.checked_path(original_job.extra["source_file"])
    original = shared.checked_path(original_job.extra["histogram"])
    metadata = shared.checked_path(metadata)
    if not pending_metadata and not metadata.is_file():
        raise ValueError("Choose an existing acquisition metadata file or collection.")
    if (not isfinite(low) or not isfinite(high) or low > high or type(shift) is not int
            or shift not in GMD_SHIFTS):
        raise ValueError("Enter increasing GMD2 limits and a valid alignment shift.")
    job = Job(FILTER_JOB_NAME, cwd=str(shared.workspace_root()))
    result = shared.generated_dir() / f"ion-tof-filtered-{job.id}.npz"
    reference = original_job.extra["reference"]
    job.cmd = [sys.executable, "-m", "vmi_analysis.ion_tof", "--input", str(source),
               "--output", str(result), "--tof-result", str(original), "--metadata", str(metadata),
               "--reference", reference, f"--gmd-min={low}", f"--gmd-max={high}", "--gmd-shift", str(shift)]
    job.extra.update(histogram=str(result), parent_id=original_job.id, metadata=str(metadata),
                     gmd_min=low, gmd_max=high, gmd_shift=shift, source_file=str(source), reference=reference)
    return job


def peak_fit_job(gated_job, targets, region, estimates, baseline, *, unfiltered_result=None):
    if len(region) != 2 or None in region or not np.isfinite(region).all() or region[0] >= region[1]:
        raise ValueError("Select a fit range or enter increasing, finite TOF limits.")
    if not targets or any(target not in ("pixels", "clusters") for target in targets):
        raise ValueError("Choose at least one TOF spectrum.")
    if baseline not in BASELINE_PARAMETERS:
        raise ValueError("Choose a supported baseline.")
    if any(len(peak) != 2 or not np.isfinite(peak).all() or peak[1] <= 0 for peak in estimates):
        raise ValueError("Enter a finite center and positive Gaussian FWHM for each peak.")
    if "source_file" not in gated_job.extra:
        raise ValueError("Apply the GMD2 range again before fitting this older spectrum.")
    source = shared.checked_path(gated_job.extra["source_file"])
    filtered = shared.checked_path(gated_job.extra["histogram"])
    spec = dict(sources=targets, region=region, estimates=estimates, baseline=baseline)
    name = UNFILTERED_FIT_JOB_NAME if unfiltered_result is not None else FIT_JOB_NAME
    job = Job(name, cwd=str(shared.workspace_root()))
    output = shared.generated_dir() / f"ion-tof-peaks-{job.id}.npz"
    job.cmd = [sys.executable, "-m", "vmi_analysis.ion_tof", "--input", str(source),
               "--filtered-result", str(filtered), "--output", str(output),
               "--reference", gated_job.extra["reference"], "--fit-spec", json.dumps(spec)]
    if unfiltered_result is not None:
        job.cmd += ["--tof-result", str(shared.checked_path(unfiltered_result))]
    job.extra.update(peak_result=str(output), filtered_id=gated_job.id, baseline=baseline, fit_range=region)
    return job


def build(jobs, *, ask_agent=None):
    def restore_form(job):
        """Show an Agent run's resolved inputs, as if the user had entered them."""
        form = adopt_agent_form(job, "ion_tof")
        if not form:
            return
        for key, widget in (("input", input_file), ("source", source), ("reference", reference),
                            ("bin_size_ns", bin_size), ("minimum", minimum), ("maximum", maximum)):
            if key in form:
                widget.value = form[key]
        if "metadata" in form:
            set_metadata(form["metadata"])

    def show_job(job):
        if job.status == "done" and job.extra.get("agent_panel"):
            reset_gmd()
        _results(job, ask_agent=ask_agent)
        if job.status == "done":
            restore_agent_gate(job)

    view = JobView(jobs, job_name=JOB_NAME, results_builder=show_job, attach_builder=restore_form)
    theme.panel_header(JOB_NAME, "Plot unfiltered time-of-flight spectra from a Timepix3 dump.", view=view)

    def save(widget, key):
        return persist(widget, f"ion_tof.{key}")

    def use_latest():
        candidates = [job for job in jobs.jobs() if job.name == ion_luna.JOB_NAME and job.status == "done"]
        if not candidates:
            ui.notify("Complete a TPX3 dump first or choose a saved collection.", type="warning")
            return
        latest = max(candidates, key=lambda job: job.finished)
        path, metadata = ion_luna.completed_dump_input(latest)
        input_file.value = path
        set_metadata(metadata or "")

    def run():
        try:
            job = spectrum_job(input_file.value.strip(), source=source.value, reference=reference.value,
                               width=bin_size.value, low=minimum.value, high=maximum.value)
            view.attach(jobs.submit(job))
            reset_gmd()
            if job.extra["source_file"].lower().endswith(".json"):
                set_metadata(job.extra["source_file"])
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
    console = ui.column().classes("w-full")
    reset_gmd, set_metadata, restore_agent_gate = _gmd_section(jobs, view, save, ask_agent=ask_agent)
    with console:
        view.place_console()


def _gmd_section(jobs, view, save, *, ask_agent=None):
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
                    restore_agent_filter(finished)
                    refresh_gmd_plot()
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
            job = jobs.submit(filtered_job(loaded_job, loaded_metadata, low, high, shift.value))
            filtered_display.clear()
            filter_view.attach(job)
        except (OSError, ValueError, ResourceBusyError) as exc:
            ui.notify(str(exc), type="warning")

    def show_filtered(job):
        original_job = view.job
        if job.status != "done" or original_job is None or job.extra["parent_id"] != original_job.id:
            return
        if (not job.extra.get("agent_panel") and
                (loaded_job is not original_job or job.extra["metadata"] != str(loaded_metadata))):
            return
        with np.load(shared.checked_path(job.extra["histogram"]), allow_pickle=False) as result:
            edges = result["edges"]
            spectra = {source: result[source] for source in ("pixels", "clusters") if source in result}
            selected = int(result["selected_triggers"])
            total = int(result["triggers"])
        original_path = shared.checked_path(original_job.extra["histogram"])
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
                chart = ui.plotly(plot_options(fig)).classes("w-full")
                _capture_plot(chart, job, ask_agent)
            else:
                _peak_fitting(jobs, job, edges, spectra, ask_agent=ask_agent)
            with ui.expansion("Unfiltered TOF", icon="show_chart", value=False).classes("w-full gap-4").props(
                    'data-tof-fit-source="unfiltered"'):
                ui.label("Fits the unfiltered spectrum to define peak windows. Counting and normalization "
                         f"still use the {selected:,} GMD2-selected triggers above.").classes("vmi-subtitle")
                if selected and any(counts.any() for counts in unfiltered_spectra.values()):
                    _peak_fitting(jobs, job, edges, unfiltered_spectra, unfiltered_result=original_path,
                                  ask_agent=ask_agent, display_job=original_job)
                else:
                    chart = ui.plotly(plot_options(_tof_figure(edges, unfiltered_spectra,
                                                               "Unfiltered time-of-flight spectra"))).classes("w-full")
                    _capture_plot(chart, original_job, ask_agent)
                    ui.label("Select triggers with TOF data before fitting peak yields.").classes("vmi-caption")

    def restore_agent_filter(job):
        if job.extra.get("agent_panel") and job.extra["parent_id"] == getattr(view.job, "id", None):
            gmd_low.value = job.extra["gmd_min"]
            gmd_high.value = job.extra["gmd_max"]
            shift.value = job.extra["gmd_shift"]

    with theme.panel_card().props('data-ion-tof-checkpoint') as checkpoint_card:
        checkpoint_card.set_visibility(False)
        theme.form_section("GMD2 alignment decision")
        show_checkpoint = checkpoint_view(jobs, stop_label="Stop alignment workflow")

    shown_diagnostics = None

    async def refresh_workflow():
        nonlocal shown_diagnostics
        current_id = getattr(view.job, "id", None)
        pending = next((job for job in jobs.jobs()
                        if job.status == "waiting" and (checkpoint := job.checkpoint) is not None
                        and checkpoint.summary.get("tof_job_id") == current_id), None)
        checkpoint_card.set_visibility(pending is not None)
        show_checkpoint(pending)
        filtered = filter_view.job
        if (filtered is not None and filtered.status == "done" and filtered.id != shown_diagnostics
                and filtered.extra.get("gmd_shift_source")
                and filtered.extra["parent_id"] == current_id
                and filtered.extra["metadata"] == metadata_file.value.strip()):
            shown_diagnostics = filtered.id
            if loaded_job is not view.job:
                await plot_gmd()

    shared.workspace_timer(0.5, refresh_workflow)
    filter_view = JobView(jobs, job_name=FILTER_JOB_NAME, results_builder=show_filtered,
                          attach_builder=restore_agent_filter)
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
        filter_console = ui.column().classes("w-full")
    filtered_display = ui.column().classes("w-full gap-4")
    metadata_file.on_value_change(lambda _: reset_gmd())
    def set_metadata(value):
        metadata_file.value = value

    def restore_agent_gate(parent):
        filtered = filter_view.job
        if (filtered is not None and filtered.extra.get("agent_panel")
                and filtered.extra.get("parent_id") == parent.id):
            restore_agent_filter(filtered)
            show_filtered(filtered)

    with filter_console:
        filter_view.place_console("Filtered TOF console")

    return reset_gmd, set_metadata, restore_agent_gate


def _peak_fitting(jobs, filtered_job, edges, spectra, *, unfiltered_result=None, ask_agent=None, display_job=None):
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

    def show_agent_fit(job, settings, approve, decline):
        if fit_low.is_deleted:
            raise ValueError("The source spectrum changed. Open that result to ask the Agent again.")
        fit_low.value, fit_high.value = job.extra["fit_range"]
        mode.value = "Select fit range"
        refresh_plot()
        with agent_review:
            ui.label(f"Agent fit draft · {settings['Spectra']} · {settings['Baseline']} baseline · "
                     f"Peak estimates {settings['Peak estimates (ns)']}").classes("vmi-caption")
        async def approve_fit():
            await approve({"region_ns": [fit_low.value, fit_high.value]})

        with agent_review, ui.row().classes("items-center gap-2"):
            ui.label("Adjust the fit range on the plot or in the form.")
            ui.button("Approve Agent fit", on_click=approve_fit)
            ui.button("Decline", on_click=decline).props("flat")

    def open_agent():
        if (fit_low.value is None or fit_high.value is None or fit_low.value >= fit_high.value
                or not targets.value):
            ui.notify("Select a spectrum and an increasing fit range first.", type="warning")
            return
        settings = {
            "gate_job_id": filtered_job.id, "sources": list(targets.value or []),
            "region_ns": [fit_low.value, fit_high.value], "baseline": baseline.value,
            "unfiltered": unfiltered_result is not None,
        }

        def bind(parameters):
            if fit_low.is_deleted:
                raise ValueError("The source spectrum changed. Open that result to ask the Agent again.")
            bound = {**settings, **parameters, "gate_job_id": settings["gate_job_id"],
                     "unfiltered": settings["unfiltered"]}
            if not isinstance(bound["sources"], list) or any(
                    source not in settings["sources"] for source in bound["sources"]):
                raise ValueError("Choose a source selected on the Ion TOF plot.")
            return bound

        ask_agent(PanelView(
            panel=JOB_NAME, operation="ion_tof_fit",
            label=f"Ion TOF fit context · {fit_low.value:g}–{fit_high.value:g} ns",
            instruction="Use the selected range unless the user requests a change.",
            settings=settings, bind=bind, render_review=show_agent_fit, review_container=agent_review,
        ))

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
            estimates = []
            for center, gaussian in peak_rows:
                if center.value is None and gaussian.value is None:
                    continue
                if center.value is None or gaussian.value is None:
                    raise ValueError("Enter a center and Gaussian FWHM for each peak.")
                estimates.append((center.value, gaussian.value))
            job = peak_fit_job(filtered_job, targets.value, (fit_low.value, fit_high.value),
                               estimates, baseline.value, unfiltered_result=unfiltered_result)
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
                _show_peak_fits(results, job.extra["baseline"], int(data["selected_triggers"]), low, high,
                                job=job, ask_agent=ask_agent)

    with ui.row().classes("items-center gap-2"):
        mode = ui.toggle(["Zoom", "Select fit range"], value="Zoom", on_change=lambda _: refresh_plot())
        ui.button("Clear fit range", on_click=clear_fit_range).props("flat")
        if ask_agent:
            ui.button("Ask Agent", icon="smart_toy", on_click=open_agent).props("flat")
    agent_review = ui.column().classes("w-full")
    plot = ui.plotly(go.Figure()).classes("w-full")
    plot.on("plotly_selected", select_fit_range, js_handler="event => emit(event?.range ?? null)")
    _capture_plot(plot, display_job or filtered_job, ask_agent)
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


def _show_peak_fits(results, model, selected, fit_low, fit_high, *, job, ask_agent):
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
    _capture_plot(chart, job, ask_agent)

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
