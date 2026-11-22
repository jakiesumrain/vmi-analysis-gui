"""Composition of the panels and workflows available to the embedded Agent."""

from functools import partial

from .agent_panels import PanelOperation, PanelRegistry
from .jobs import Job, Stage
from .panels import ion_luna, ion_tof


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
    job = Job(DUMP_TO_TOF, cwd=dump.cwd, stages=stages)
    return job, {
        **{f"1. {label}": value for label, value in dump_summary.items()},
        **{f"2. {label}": value for label, value in tof_summary.items()},
    }


def build_registry():
    registry = PanelRegistry()
    for panel in (ion_luna, ion_tof):
        for operation in panel.agent_operations():
            registry.register(operation)
    workflow = {
        "panel": DUMP_TO_TOF,
        "summary": "Unpack TPX3 with LUNA, then plot unfiltered TOF from that exact dump. "
                   "One approval covers both stages.",
        "parameters": {
            "dump": {"type": "object", "description": "Parameters from tpx3_dump"},
            "tof": {"type": "object", "description": "Optional ion_tof parameters without input or dump_job_id"},
        },
    }
    registry.register(PanelOperation(
        "ion_dump_to_tof", workflow, partial(_dump_to_tof, registry),
        (DUMP_TO_TOF,), workflow_panels=(ion_luna.JOB_NAME, ion_tof.JOB_NAME),
    ))
    return registry
