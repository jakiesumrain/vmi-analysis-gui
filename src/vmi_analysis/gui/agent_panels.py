"""The panel operations available to a signed-in user's Agent session."""

from math import isfinite
from pathlib import Path, PurePosixPath

from vmi_analysis.ion_acquisition import manifest_path
from vmi_analysis.ion_tof import DEFAULT_REFERENCE, GMD_SHIFTS
from vmi_analysis.tof_bins import DEFAULT_BIN_SIZE_NS

from . import shared
from .panels import ion_luna, ion_tof


DUMP_OPTIONS = ("layout", "raw-only", "disable-clustering", "tof-tdc-reference",
                "clustering-algorithm", "eps-t", "eps-s", "kuttruff-buffer-size",
                "calculate-timewalk-statistics", "ctot-cut")

OPERATIONS = {
    "tpx3_dump": {
        "panel": ion_luna.JOB_NAME,
        "summary": "Decode TPX3 files with LUNA. Acquisition prefixes supply timestamps and GMD readings. "
                   "Detected clock resets split independent streams. Output is an HDF5 file or a collection.",
        "parameters": {"input": "TPX3 file or directory", "output": "Workspace HDF5 file",
                       "prefixed": True, "timewalk": "Optional existing statistics HDF5 seed",
                       "options": {key: ion_luna.PANEL_DEFAULTS.get(key, ion_luna.OPTIONS[key].default)
                                   for key in DUMP_OPTIONS}},
        "option_choices": {key: list(ion_luna.OPTIONS[key].choices) for key in DUMP_OPTIONS
                           if ion_luna.OPTIONS[key].choices},
        "manual": "TPX3 Dump",
    },
    "ion_tof": {
        "panel": ion_tof.JOB_NAME,
        "summary": "Use the preceding selected TDC trigger in each acquisition to calculate TOF. "
                   "Histogram raw pixel hits, clustered events, or both.",
        "parameters": {"input": "LUNA HDF5 or acquisition collection, or supply dump_job_id",
                       "dump_job_id": "Completed TPX3 Dump job ID, instead of input",
                       "source": "pixels", "reference": DEFAULT_REFERENCE,
                       "bin_size_ns": DEFAULT_BIN_SIZE_NS, "minimum_ns": 0, "maximum_ns": None},
        "choices": {"source": list(ion_tof.SOURCES), "reference": list(ion_tof.REFERENCES)},
        "manual": "Ion TOF",
    },
    "ion_tof_gate": {
        "panel": ion_tof.JOB_NAME,
        "summary": "Apply an inclusive GMD2 gate. Metadata row j selects camera frame j + d "
                   "within the same acquisition. Use the alignment diagnostic to choose d.",
        "parameters": {"tof_job_id": "Completed Ion TOF job ID",
                       "metadata": "Matching acquisition metadata, optional after a TPX3 Dump handoff",
                       "gmd_min_na": "Lower bound", "gmd_max_na": "Upper bound",
                       "shift": "Alignment shift d in frames"},
        "choices": {"shift": list(GMD_SHIFTS)},
    },
    "ion_tof_fit": {
        "panel": ion_tof.JOB_NAME,
        "summary": "Fit Gaussian peaks on gated or unfiltered TOF. Count individual arrivals within "
                   "the fitted windows on the GMD2-gated frames in either case.",
        "parameters": {"gate_job_id": "Completed GMD2-gated TOF job ID", "sources": ["pixels"],
                       "region_ns": ["Fit minimum", "Fit maximum"],
                       "peaks_ns": "Optional list of [center, Gaussian FWHM] pairs. Blank means automatic.",
                       "baseline": "Constant", "unfiltered": False},
        "choices": {"sources": ["pixels", "clusters"], "baseline": list(ion_tof.BASELINE_PARAMETERS)},
    },
}

JOB_NAMES = {ion_luna.JOB_NAME, ion_tof.JOB_NAME, ion_tof.FILTER_JOB_NAME,
             ion_tof.FIT_JOB_NAME, ion_tof.UNFILTERED_FIT_JOB_NAME}


def describe(operation=None):
    return OPERATIONS if operation is None else OPERATIONS[operation]


def _number(value):
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError("Use finite numeric values for limits, widths, and centers.")
    return value


def _bool(value):
    if type(value) is not bool:
        raise ValueError("Use true or false for Boolean parameters.")
    return value


def _path(value, *, write=False):
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError("Enter a file or directory path.")
    cwd = shared.workspace_root()
    if shared.SERVER:
        parts = PurePosixPath(value).parts
        if parts[:2] == ("/", "workspace"):
            value = cwd.joinpath(*parts[2:])
        elif parts[:2] == ("/", "datasets"):
            value = shared.SERVER.datasets.joinpath(*parts[2:])
    return shared.checked_path(value, write=write, cwd=cwd)


def _job(jobs, job_id, *, name=None):
    job = jobs.get(job_id) if isinstance(job_id, str) else None
    if job is None or job.name not in JOB_NAMES or (name is not None and job.name != name):
        raise ValueError("Choose a supported job ID from your workspace.")
    if name is not None and job.status != "done":
        raise ValueError(f"Wait for {name} to finish successfully.")
    return job


def _dump(params):
    source = _path(params["input"])
    if not source.is_dir() and not (source.is_file() and source.suffix.lower() == ".tpx3"):
        raise ValueError("Choose an existing TPX3 file or directory.")
    target = _path(params["output"], write=True)
    if target.suffix.lower() not in ion_luna.HDF5_TYPES:
        raise ValueError("Choose an .h5 or .hdf5 output path.")
    prefixed = _bool(params.get("prefixed", True))
    requested = params.get("options", {})
    if not isinstance(requested, dict) or requested.keys() - set(DUMP_OPTIONS):
        raise ValueError("Use the TPX3 Dump options listed by describe.")
    values = {}
    for key in DUMP_OPTIONS:
        option = ion_luna.OPTIONS[key]
        default = ion_luna.PANEL_DEFAULTS.get(key, option.default)
        value = requested.get(key, default)
        if type(value) is not type(default) or (option.choices and value not in option.choices):
            raise ValueError(f"Choose a valid value for {option.label}.")
        if type(value) is int and value < (1 if key == "kuttruff-buffer-size" else 0):
            raise ValueError(f"{option.label} is below the permitted minimum.")
        if isinstance(value, str) and (not value.strip() or value.startswith("-") or "\x00" in value):
            raise ValueError(f"Choose a valid value for {option.label}.")
        values[key] = value
    if values["raw-only"]:
        values.pop("tof-tdc-reference")
    if values["raw-only"] or values["disable-clustering"]:
        for key in ("clustering-algorithm", "eps-t", "eps-s", "calculate-timewalk-statistics", "ctot-cut"):
            values.pop(key)
    if values.get("clustering-algorithm") != "Kuttruff2024":
        values.pop("kuttruff-buffer-size")
    if not values.get("calculate-timewalk-statistics"):
        values.pop("ctot-cut", None)
    matrix = _path(params["timewalk"]) if params.get("timewalk") else None
    if matrix and (not values.get("calculate-timewalk-statistics") or not matrix.is_file()):
        raise ValueError("A timewalk seed requires statistics calculation and an existing HDF5 file.")
    wsl = ion_luna.wsl_settings()
    job = ion_luna.dump_job(source, target, prefixed=prefixed, options=values, timewalk=matrix,
                           wsl=ion_luna.wsl_arguments(wsl))
    summary = {"TPX3 input": str(source), "Strip acquisition prefix": prefixed,
               "Split detected acquisitions": prefixed, "Timewalk seed": str(matrix) if matrix else None}
    if prefixed:
        summary["Collection output"] = str(manifest_path(target))
        summary["Segment folder"] = str(target.with_name(f"{target.stem}_segments"))
    else:
        summary["HDF5 output"] = str(target)
    summary.update({ion_luna.OPTIONS[key].label: value for key, value in values.items()})
    summary.update({ion_luna.WSL_FIELDS[key]: value.strip() for key, value in wsl.items()
                    if isinstance(value, str) and value.strip()})
    return job, summary


def _spectrum(jobs, params):
    dump_id, path = params.get("dump_job_id"), params.get("input")
    if bool(dump_id) == bool(path):
        raise ValueError("Supply one TPX3 Dump job ID or one input file.")
    metadata = None
    if dump_id:
        extra = _job(jobs, dump_id, name=ion_luna.JOB_NAME).extra
        path = extra.get("collection", extra["output"])
        metadata = extra.get("collection", extra.get("metadata"))
    source = params.get("source", "pixels")
    reference = params.get("reference", DEFAULT_REFERENCE)
    width = _number(params.get("bin_size_ns", DEFAULT_BIN_SIZE_NS))
    low = _number(params.get("minimum_ns", 0))
    high = params.get("maximum_ns")
    if high is not None:
        high = _number(high)
    job = ion_tof.spectrum_job(_path(path), source=source, reference=reference, width=width, low=low, high=high)
    if metadata:
        job.extra["metadata"] = str(_path(metadata))
    return job, {"LUNA input": job.extra["source_file"], "Count": ion_tof.SOURCES[source],
                 "Trigger reference": reference, "Bin size (ns)": width, "TOF minimum (ns)": low,
                 "TOF maximum (ns)": high if high is not None else "Automatic"}


def _gate(jobs, params):
    parent = _job(jobs, params["tof_job_id"], name=ion_tof.JOB_NAME)
    metadata = params.get("metadata") or parent.extra.get("metadata")
    if not metadata and parent.extra["source_file"].lower().endswith(".json"):
        metadata = parent.extra["source_file"]
    low, high = _number(params["gmd_min_na"]), _number(params["gmd_max_na"])
    shift = params["shift"]
    job = ion_tof.filtered_job(parent, _path(metadata), low, high, shift)
    return job, {"TOF job": parent.id, "Metadata": job.extra["metadata"], "GMD2 minimum (nA)": low,
                 "GMD2 maximum (nA)": high, "Alignment shift d (frames)": shift}


def _fit(jobs, params):
    parent = _job(jobs, params["gate_job_id"], name=ion_tof.FILTER_JOB_NAME)
    targets = params.get("sources", ["pixels"])
    if not isinstance(targets, list) or not all(isinstance(item, str) for item in targets):
        raise ValueError("Supply a list of spectrum sources.")
    region = params["region_ns"]
    estimates = params.get("peaks_ns", [])
    if not isinstance(region, list) or not isinstance(estimates, list) or len(estimates) > 20:
        raise ValueError("Supply numeric fit bounds and up to 20 pairs of peak estimates.")
    region = [_number(value) for value in region]
    parsed = []
    for pair in estimates:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("Supply [center, Gaussian FWHM] for each peak estimate.")
        parsed.append([_number(value) for value in pair])
    baseline = params.get("baseline", "Constant")
    unfiltered = _bool(params.get("unfiltered", False))
    original = _job(jobs, parent.extra["parent_id"], name=ion_tof.JOB_NAME) if unfiltered else None
    job = ion_tof.peak_fit_job(parent, targets, region, parsed, baseline,
                              unfiltered_result=original.extra["histogram"] if original else None)
    return job, {"Gated TOF job": parent.id, "Spectra": targets, "Fit range (ns)": region,
                 "Peak estimates (ns)": parsed or "Automatic", "Baseline": baseline,
                 "Fit spectrum": "Unfiltered" if original else "GMD2-gated",
                 "Counting and normalization": "GMD2-gated"}


def prepare(jobs, operation, params):
    if operation not in OPERATIONS or not isinstance(params, dict):
        raise ValueError("Choose a supported panel operation and its named parameters.")
    if params.keys() - OPERATIONS[operation]["parameters"].keys():
        raise ValueError("Use only parameters listed by describe for this operation.")
    if operation == "tpx3_dump":
        job, summary = _dump(params)
    else:
        build = {"ion_tof": _spectrum, "ion_tof_gate": _gate, "ion_tof_fit": _fit}[operation]
        job, summary = build(jobs, params)
    for key in ("metadata", "histogram", "peak_result"):
        if key in job.extra and key != "metadata":
            summary[f"{key.replace('_', ' ').capitalize()} output"] = job.extra[key]
    job.extra["agent_panel"] = True
    return job, summary


def _container_path(value):
    path = Path(value)
    if shared.SERVER:
        for root, mount in ((shared.workspace_root(), "/workspace"), (shared.SERVER.datasets, "/datasets")):
            if path.is_relative_to(root):
                return str(PurePosixPath(mount, *path.relative_to(root).parts))
    return value


def status(jobs, job_id):
    job = _job(jobs, job_id)
    result = {"job_id": job.id, "panel": job.name, "status": job.status,
              "progress": job.progress, "note": job.wait_reason,
              "log_tail": [line[:500] for line in tuple(job.log.copy())[-15:]]}
    if job.status == "done":
        keys = ("histogram", "peak_result")
        if job.name == ion_luna.JOB_NAME:
            keys = ("collection",) if job.extra.get("collection") else ("output", "metadata")
        result["files"] = {key: _container_path(job.extra[key]) for key in keys
                           if key in job.extra and _path(job.extra[key]).is_file()}
    return result
