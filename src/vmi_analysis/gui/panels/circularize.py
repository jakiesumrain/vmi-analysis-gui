"""Panel: circularization.py — single-image circularization."""

import sys
from pathlib import Path

import numpy as np
from nicegui import ui
from plotly.subplots import make_subplots
from vmi_analysis.paths import validate_distinct_output

from ..centroid_plots import _heatmap, _image_axes, _style, plot_options
from ..jobs import Job, JobManager
from .. import theme
from ..settings import centroid_settings, output_dir
from .common import (GENERATED, JobView, REPO_ROOT, file_url, num_input,
                     open_folder_button, path_input, persist,
                     zoomable_image)
from .polar_preview import (bind_percentiles, polar_controls, polar_preview,
                            set_polar_source, set_polar_view)

CIRC_DIR = REPO_ROOT / "circularization"


def build(jobs: JobManager):
    theme.section_title(
        "Circularize",
        "Correct radial distortion in one VMI image and compare the raw and corrected polar views.")

    def results(job):
        _results(job)
        if job.status == "done" and job.extra.get("polar_preview"):
            show_polar(job.extra["polar_preview"])

    view = JobView(jobs, job_name='Circularize', results_builder=results)

    with theme.panel_card():
        inp = persist(path_input("Input image (.npy or .txt)",
                                 str(output_dir(centroid_settings()) / "vmi_2d_raw.npy"),
                                 browse="file", filters=[".npy", ".txt"]),
                      "circularize.inp")
        out = persist(path_input("Output image (.txt)",
                                 str(CIRC_DIR / "image_circularized.txt"),
                                 browse="save", filters=[".txt"],
                                 default_name="image_circularized.txt"),
                      "circularize.out")
        with ui.row().classes("gap-4 flex-wrap"):
            shape_y = persist(num_input("Shape NY", 1023, step=1),
                              "circularize.shape_y")
            shape_x = persist(num_input("Shape NX", 1023, step=1),
                              "circularize.shape_x")
            center_x = persist(num_input("Center X", 511.0, step="any"),
                               "circularize.center_x")
            center_y = persist(num_input("Center Y", 511.0, step="any"),
                               "circularize.center_y")
            brightest_center = persist(ui.checkbox("Use brightest pixel as center", value=True),
                                       "circularize.brightest_center")
        extracted = persist(path_input(
            "Extracted trig-fit file",
            str(CIRC_DIR / "extracted-FOC100eV-20260710.txt"),
            browse="file", filters=[".txt"]), "circularize.extracted")
        reassign = persist(path_input(
            "Reassignment file (optional)",
            browse="file", filters=[".txt"]), "circularize.reassign")
        with ui.row().classes("gap-4 flex-wrap"):
            radial_model = persist(ui.select(
                ["interp", "poly"], value="interp",
                label="Radial model").props("outlined dense"),
                "circularize.radial_model")
            poly_order = persist(num_input("Poly order", 2, step=1),
                                 "circularize.poly_order")
            poly_order.set_enabled(radial_model.value == "poly")
            radial_model.on_value_change(lambda e: poly_order.set_enabled(e.value == "poly"))
            interp_order = persist(num_input("Interp order", 3, step=1),
                                   "circularize.interp_order")
            max_trig_n = persist(num_input("Max trig n (0 = auto)", 0, step=1),
                                 "circularize.max_trig_n")
            polar_dr = persist(num_input("Polar dr", 1.0, step=0.5),
                               "circularize.polar_dr")
        with ui.row().classes("gap-4 flex-wrap"):
            no_jacobian = persist(ui.checkbox("No Jacobian", value=True),
                                  "circularize.no_jacobian")
            no_plots = persist(ui.checkbox("No diagnostic plots", value=False),
                               "circularize.no_plots")
            summary_only = persist(ui.checkbox("Summary only", value=False),
                                   "circularize.summary_only")
        plot_dir = persist(path_input(
            "Plot output dir", str(CIRC_DIR / "trig_fit_plots"),
            browse="dir"), "circularize.plot_dir")

        ui.label("NumPy input keeps its original grid. Use the same center as Trig Calibration.") \
            .classes("vmi-subtitle")

        def raw_input():
            return Path(inp.value or "").suffix.lower() == ".npy"

        def update_geometry():
            raw = raw_input()
            for field in (shape_y, shape_x, polar_dr):
                field.set_enabled(not raw)
            for field in (center_x, center_y):
                field.set_enabled(not raw or not brightest_center.value)
            brightest_center.set_visibility(raw)

        inp.on_value_change(lambda _: update_geometry())
        brightest_center.on_value_change(lambda _: update_geometry())
        update_geometry()

        def run():
            if not summary_only.value:
                try:
                    validate_distinct_output(out.value, (
                        ("input image", inp.value), ("extracted calibration file", extracted.value),
                        ("radial reassignment file", reassign.value),
                    ), cwd=CIRC_DIR)
                except (OSError, ValueError) as exc:
                    ui.notify(str(exc), type="warning")
                    return
            cmd = [sys.executable, "-m", "vmi_analysis.circularization.circularization",
                   "--input", inp.value,
                   "--output", out.value,
                   "--extracted", extracted.value,
                   "--radial-model", radial_model.value,
                   "--interp-order", str(int(interp_order.value)),
                   "--plot-dir", plot_dir.value]
            if radial_model.value == "poly":
                cmd += ["--poly-order", str(int(poly_order.value))]
            if not raw_input():
                cmd += ["--shape", str(int(shape_y.value)), str(int(shape_x.value)),
                        "--polar-dr", str(polar_dr.value)]
            if not raw_input() or not brightest_center.value:
                if center_x.value is None or center_y.value is None:
                    ui.notify("Enter both Center X and Center Y.", type="warning")
                    return
                cmd += ["--center", str(center_x.value), str(center_y.value)]
            if max_trig_n.value:
                cmd += ["--max-trig-n", str(int(max_trig_n.value))]
            if reassign.value:
                cmd += ["--reassign-file", reassign.value]
            if no_jacobian.value:
                cmd.append("--no-jacobian")
            if no_plots.value:
                cmd.append("--no-plots")
            if summary_only.value:
                cmd.append("--summary-only")
            job = Job(name="Circularize", cmd=cmd, cwd=str(CIRC_DIR),
                      extra={"output": out.value, "plot_dir": plot_dir.value,
                             "reassignment": bool(reassign.value),
                             "summary_only": summary_only.value})
            if not summary_only.value:
                preview_path = GENERATED / f"circ_{job.id}_polar.npz"
                cmd += ["--polar-preview", str(preview_path)]
                job.extra["polar_preview"] = str(preview_path)
            polar_data.clear()
            polar_card.set_visibility(False)
            view.attach(jobs.submit(job))

        view.run_button(on_click=run)

        polar_data, polar_widgets = {}, {}
        with ui.column().classes("w-full gap-2 border-t border-stone-200 pt-4 polar-guide-group") as polar_card:
            polar_card.set_visibility(False)
            scale, colormap, guides = polar_controls("circularize", radius_only=True)
            for key, label in (("raw", "Raw polar"), ("circularized", "Circularized polar")):
                polar_widgets[key] = polar_preview(
                    "circularize", key, label, guides, lambda: ui.interactive_image(""),
                    radius_only=True)

    def refresh_polar(key):
        if key in polar_data:
            image, limits, status = polar_widgets[key]
            warning = set_polar_source(image, polar_data[key], colormap, scale, limits)
            status.text = f"Showing automatic linear colors. {warning}" if warning else ""
            status.set_visibility(bool(warning))

    def refresh_polar_pair():
        for key in polar_widgets:
            refresh_polar(key)

    def show_polar(path):
        if not Path(path).is_file():
            return  # The job console reports preview generation errors.
        with np.load(path, allow_pickle=False) as data:
            polar_data.update({key: data[key] for key in polar_widgets})
            dr = float(data["dr"])
        n_theta, n_r = polar_data["raw"].shape
        set_polar_view(polar_card, (0, n_r - 1), (0, n_theta - 1), dr, 360 / n_theta)
        refresh_polar_pair()
        polar_card.set_visibility(True)

    for key, (_, limits, _) in polar_widgets.items():
        bind_percentiles(limits, lambda key=key: refresh_polar(key))
    for control in (scale, colormap):
        control.on_value_change(lambda _: refresh_polar_pair())
    view.place_console()

    def use_calibration(extracted_path, image_path=None, manual_center=None, *,
                        radial_model_value=None, poly_order_value=None):
        extracted.value = extracted_path
        if image_path is not None:
            inp.value = image_path
            brightest_center.value = manual_center is None
            if manual_center is not None:
                center_x.value, center_y.value = manual_center
        if radial_model_value is not None:
            radial_model.value = radial_model_value
        if poly_order_value is not None:
            poly_order.value = poly_order_value
        return int(max_trig_n.value or 0)

    return use_calibration


def _results(job: Job):
    if job.status != "done" or job.extra.get("summary_only"):
        return
    cwd = Path(job.cwd or CIRC_DIR)
    out = cwd / job.extra["output"]
    open_folder_button(out)
    if out.exists():
        try:
            figure = _corrected_figure(np.loadtxt(out, ndmin=2))
            ui.plotly(plot_options(figure)).classes("w-full max-w-4xl")
        except (OSError, ValueError) as e:
            ui.label(f"Could not render {out}: {e}").classes("vmi-subtitle")
    if job.extra.get("reassignment") and job.started is not None:
        reassignment_plot = cwd / job.extra["plot_dir"] / "reassignment_map.png"
        if reassignment_plot.is_file() and reassignment_plot.stat().st_mtime >= job.started:
            ui.label("Radial reassignment map").classes("vmi-caption")
            zoomable_image(file_url(reassignment_plot)).classes("w-full max-w-2xl")


def _corrected_figure(image):
    trace = _heatmap(image, logarithmic=True, colorbar={"title": {"text": "Intensity"}})
    positive = image[np.isfinite(image) & (image > 0)]
    if positive.size:
        high = np.log10(np.percentile(positive, 99.999))
        low = high - 4
        ticks = list(range(int(np.ceil(low)), int(np.floor(high)) + 1))
        trace.update(zmin=low, zmax=high, colorbar={
            "tickvals": ticks, "ticktext": [f"10<sup>{tick}</sup>" for tick in ticks]})
    trace.update(colorscale="Viridis",
                 hovertemplate="X: %{x:.0f} px<br>Y: %{y:.0f} px<br>Intensity: %{customdata:.6g}<extra></extra>")
    figure = make_subplots(rows=1, cols=1)
    figure.add_trace(trace, row=1, col=1)
    _image_axes(figure, image, 1, 1, "x")
    _style(figure, "Circularized image · log scale")
    figure.update_layout(height=740)
    return figure
