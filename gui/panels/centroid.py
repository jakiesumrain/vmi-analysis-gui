"""Panel: Picasso centroiding — config.toml form editor + batch run."""

import sys
from pathlib import Path

import tomlkit
from nicegui import ui

from ..jobs import Job, JobManager
from .. import theme
from .common import JobView, REPO_ROOT, file_url

CENTROID_DIR = REPO_ROOT / "centroiding"
CONFIG_PATH = CENTROID_DIR / "config.toml"
SCRIPT = CENTROID_DIR / "process_andor_vmi_PICASSO_v5_h5_batch.py"

# (section path, key, kind, label) — kind: str/int/float/bool/select:opts
FIELDS = [
    ("vmi_picasso", "h5_dir", "str", "HDF5 directory"),
    ("vmi_picasso", "h5_pattern", "str", "HDF5 pattern"),
    ("vmi_picasso", "h5_dataset", "str", "Frame dataset"),
    ("vmi_picasso", "h5_meta", "str", "Metadata dataset"),
    ("vmi_picasso", "gmd_map_path", "str", "GMD map file (empty = raw/meta)"),
    ("vmi_picasso", "dark_data_dir", "str", "Dark HDF5 directory"),
    ("vmi_picasso", "dark_h5_pattern", "str", "Dark pattern"),
    ("vmi_picasso", "dark_file", "str", "Pre-computed dark .npy"),
    ("vmi_picasso", "output_dir", "str", "Output directory"),
    ("vmi_picasso", "checkpoint_file", "str", "Checkpoint filename"),
    ("vmi_picasso", "box_size", "int", "Box size (odd)"),
    ("vmi_picasso", "min_net_gradient", "float", "Min net gradient"),
    ("vmi_picasso", "camera_baseline", "float", "Camera baseline"),
    ("vmi_picasso", "camera_sensitivity", "float", "Camera sensitivity"),
    ("vmi_picasso", "camera_gain", "float", "Camera gain"),
    ("vmi_picasso", "camera_qe", "float", "Camera QE"),
    ("vmi_picasso", "mle_convergence", "float", "MLE convergence"),
    ("vmi_picasso", "mle_max_iterations", "int", "MLE max iterations"),
    ("vmi_picasso", "mle_method", "select:sigma,sigmaxy", "MLE method"),
    ("vmi_picasso", "fitting_method", "select:mle,lq", "Fitting method"),
    ("vmi_picasso", "tuning_frame_index", "int", "Tuning frame index"),
    ("processing", "enable_parallel", "bool", "Parallel processing"),
    ("processing", "num_cores", "int", "CPU cores"),
    ("processing", "enable_dark_subtraction", "bool", "Dark subtraction"),
    ("processing", "dark_method", "select:median,mean", "Dark method"),
    ("processing", "dark_frame_count", "int", "Dark frame count"),
    ("processing", "enable_flat_field", "bool", "Flat field"),
    ("processing", "flat_field_file", "str", "Flat field file"),
    ("processing", "tuning_only", "bool", "Tuning mode only"),
    ("histogram", "resolution_factor", "int", "Histogram resolution factor"),
    ("histogram", "enable_abel", "bool", "Abel inversion"),
    ("gmd", "enable_gmd_weighting", "bool", "GMD weighting"),
    ("gmd", "gmd_weight_min", "float", "GMD weight min"),
    ("gmd", "gmd_weight_max", "float", "GMD weight max"),
]

GROUPS = {
    "vmi_picasso": "Paths, detection and fitting",
    "processing": "Processing",
    "histogram": "Histogram",
    "gmd": "GMD normalization",
}


def _load_doc():
    return tomlkit.parse(CONFIG_PATH.read_text(encoding="utf-8"))


def _section(doc, name):
    sec = doc["vmi_picasso"]
    return sec if name == "vmi_picasso" else sec[name]


def build(jobs: JobManager) -> None:
    theme.section_title(
        "Picasso Centroiding",
        "Edit centroiding/config.toml and run the Picasso localization "
        "(process_andor_vmi_PICASSO_v5_h5_batch.py). "
        "Set 'Tuning mode only' to optimize detection on a single frame first.")

    view = JobView(jobs, results_builder=_results)

    doc = _load_doc()
    widgets = {}

    with theme.panel_card():
        for group, caption in GROUPS.items():
            ui.label(caption).classes("vmi-caption")
            fields = [f for f in FIELDS if f[0] == group]
            with ui.grid(columns=2).classes("w-full gap-x-4"):
                for section, key, kind, label in fields:
                    sec = _section(doc, section)
                    value = sec.get(key, "")
                    wid = _make_widget(kind, label, value)
                    widgets[(section, key, kind)] = wid

        def run():
            _save(widgets)
            cmd = [sys.executable, str(SCRIPT),
                   "--config", str(CONFIG_PATH)]
            job = jobs.submit(Job(name="Centroiding", cmd=cmd,
                                  cwd=str(CENTROID_DIR)))
            view.attach(job)

        with ui.row().classes("gap-2"):
            theme.run_button(on_click=run)
            ui.button("Save config only", on_click=lambda: _save(widgets)) \
                .props("flat color=secondary")


def _make_widget(kind: str, label: str, value):
    if kind == "bool":
        return ui.checkbox(label, value=bool(value))
    if kind.startswith("select:"):
        options = kind.split(":", 1)[1].split(",")
        return ui.select(options, value=str(value) if value != "" else options[0],
                         label=label).props("outlined dense")
    if kind in ("int", "float"):
        return ui.number(label, value=value).props("outlined dense")
    return ui.input(label, value=str(value)).props("outlined dense")


def _save(widgets) -> None:
    doc = _load_doc()
    for (section, key, kind), wid in widgets.items():
        sec = _section(doc, section)
        v = wid.value
        if kind == "bool":
            sec[key] = bool(v)
        elif kind == "int":
            sec[key] = int(v or 0)
        elif kind == "float":
            sec[key] = float(v or 0.0)
        else:
            sec[key] = str(v)
    CONFIG_PATH.write_text(tomlkit.dumps(doc), encoding="utf-8")
    ui.notify(f"Saved {CONFIG_PATH}", type="positive")


def _results(job: Job):
    if job.status != "done":
        return
    doc = _load_doc()
    out_dir = CENTROID_DIR / doc["vmi_picasso"].get("output_dir", "")
    tuning = out_dir / "01_tuning_check_picasso.png"
    summary = out_dir / "04_final_summary.png"
    if tuning.exists():
        ui.label("Tuning check:").classes("vmi-caption")
        ui.image(file_url(tuning)).classes("w-full max-w-4xl")
    if summary.exists():
        ui.label("Final summary:").classes("vmi-caption")
        ui.image(file_url(summary)).classes("w-full max-w-4xl")
