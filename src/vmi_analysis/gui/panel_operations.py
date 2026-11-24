"""Composition of the panels and workflows available to the embedded Agent."""

from functools import partial

from vmi_analysis.ion_tof import GMD_SHIFTS

from .agent_panels import PanelOperation, PanelRegistry, number
from .jobs import Job, Stage
from .panels import accumulate, binning, calibration, centroid, circularize, dark, gmd_map, ion_live, ion_luna, ion_pipeline, ion_tof, movie, pes


DUMP_TO_TOF = "TPX3 Dump → Ion TOF"


def _dump_to_tof(registry, jobs, params):
    dump, dump_summary = registry.prepare(jobs, "tpx3_dump", params["dump"])
    tof_params = params.get("tof", {})
    allowed = registry.describe("ion_tof")["parameters"].keys() - {"input", "dump_job_id"}
    if not isinstance(tof_params, dict) or tof_params.keys() - allowed:
        raise ValueError("Use Ion TOF parameters without specifying an input for the second stage.")
    data = ion_luna.planned_dump_input(dump)
    tof, tof_summary = registry.prepare_stage(
        jobs, "ion_tof", {**tof_params, "input": str(data)}, pending_input=True
    )
    stages = [Stage(step.name, step.cmd, step.cwd, result=step) for step in (dump, tof)]
    gate_summary = {}
    if "gate" in params:
        gate_params = params["gate"]
        limits = {"gmd_min_na", "gmd_max_na"}
        if (not isinstance(gate_params, dict) or not limits <= gate_params.keys()
                or gate_params.keys() - (limits | {"shift"})):
            raise ValueError("Provide gmd_min_na and gmd_max_na for the gate, with an optional shift.")
        if not (metadata := tof.extra.get("metadata")):
            raise ValueError("Enable the acquisition prefix in TPX3 Dump to use the GMD2 gate.")
        low, high = (number(gate_params[key]) for key in ("gmd_min_na", "gmd_max_na"))
        shift = gate_params.get("shift")
        if shift is not None and (type(shift) is not int or shift not in GMD_SHIFTS):
            raise ValueError("Use a listed integer GMD2 shift or omit it for the alignment diagnostic.")
        gate = ion_tof.filtered_job(tof, metadata, low, high, shift if shift is not None else 0,
                                    pending_metadata=True)
        gate.extra["agent_panel"] = True
        stage = Stage(gate.name, gate.cmd, gate.cwd, result=gate)
        if shift is None:
            stage.checkpoint = ion_tof.workflow_alignment(tof, gate)
        else:
            gate.extra["gmd_shift_source"] = "specified"
        stages.append(stage)
        gate_summary = {"3. GMD2 gate (nA, inclusive)": [low, high],
                        "3. Alignment shift d": shift if shift is not None else "Diagnose and confirm if ambiguous",
                        "3. Pairing": ion_tof.GMD_PAIRING}
    job = Job(DUMP_TO_TOF, cwd=dump.cwd, stages=stages)
    return job, {
        **{f"1. {label}": value for label, value in dump_summary.items()},
        **{f"2. {label}": value for label, value in tof_summary.items()},
        **gate_summary,
    }


def build_registry():
    registry = PanelRegistry()
    for panel in (dark, gmd_map, accumulate, centroid, binning, circularize, calibration, pes, movie, ion_luna, ion_tof, ion_pipeline, ion_live):
        for operation in panel.agent_operations():
            registry.register(operation)
    workflow = {
        "panel": DUMP_TO_TOF,
        "summary": "Unpack TPX3 with LUNA and plot TOF from that exact dump. An optional GMD2 gate "
                   "uses the matching acquisition metadata and pauses for user confirmation if alignment is ambiguous.",
        "parameters": {
            "dump": {"type": "object", "description": "Parameters from tpx3_dump"},
            "tof": {"type": "object", "description": "Optional ion_tof parameters without input or dump_job_id"},
            "gate": {"type": "object", "description": "Optional GMD2 gate. Provide gmd_min_na and "
                     "gmd_max_na. Omit shift to diagnose alignment or supply an approved integer shift.",
                     "shift_choices": list(GMD_SHIFTS)},
        },
    }
    registry.register(PanelOperation(
        "ion_dump_to_tof", workflow, partial(_dump_to_tof, registry),
        (DUMP_TO_TOF,), workflow_panels=(ion_luna.JOB_NAME, ion_tof.JOB_NAME),
    ))
    return registry
