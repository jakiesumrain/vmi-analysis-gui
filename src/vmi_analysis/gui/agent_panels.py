"""Registration and dispatch for embedded Agent panel operations."""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path, PurePosixPath

from . import shared
from .jobs import Job, TERMINAL


@dataclass(frozen=True)
class PanelOperation:
    id: str
    description: dict
    prepare: Callable[..., tuple[Job, dict]] | None
    job_names: tuple[str, ...]
    artifacts: Callable[[Job], tuple[str, ...]] | None = None
    outputs: tuple[str, ...] = ()
    workflow_panels: tuple[str, ...] = ()
    measure_result: Callable[[object, Job, dict], dict] | None = None
    render_review: Callable[[Job], None] | None = None


@dataclass(frozen=True)
class PanelView:
    """A proposal and review surface belonging to one browser tab."""

    panel: str
    operation: str
    label: str
    instruction: str
    settings: dict
    bind: Callable[[dict], dict]
    render_review: Callable
    review_container: object
    run_interactive: Callable | None = None
    context: dict | None = None


@dataclass(frozen=True)
class PanelInspection:
    """Read-only access to a single browser tab's panel state."""

    panel: str
    operation: str
    label: str
    read_snapshot: Callable[[], dict]


@dataclass(frozen=True)
class RunCapture:
    job_id: str
    paths: list[str]


class PanelRegistry:
    def __init__(self):
        self._operations = {}
        self._job_operations = {}

    def register(self, operation: PanelOperation):
        if operation.id in self._operations or any(
                name in self._job_operations for name in operation.job_names):
            raise ValueError(f"Duplicate panel operation or job name for {operation.id}.")
        self._operations[operation.id] = operation
        self._job_operations.update(dict.fromkeys(operation.job_names, operation))

    def describe(self, operation=None):
        if operation is None:
            return {key: self.describe(key) for key in self._operations}
        spec = self._operations[operation]
        return {"outputs": list(spec.outputs), **spec.description}

    def can_measure(self, job):
        spec = self._job_operations.get(job.name)
        return job.status == "done" and spec is not None and spec.measure_result is not None

    def evidence_run(self, jobs, job_id):
        job = jobs.get(job_id) if isinstance(job_id, str) else None
        if job is None or not self.can_measure(job):
            raise ValueError("Choose a completed run with measurements in your workspace.")
        return job

    def measurement_context(self, job):
        spec = self._job_operations[job.name]
        return {"job_id": job.id, "panel": job.name, "operation": spec.id,
                "measurement_parameters": spec.description["measurements"]}

    def measure(self, jobs, job_id, parameters):
        job = self.evidence_run(jobs, job_id)
        if not isinstance(parameters, dict):
            raise ValueError("Use named measurement parameters.")
        spec = self._job_operations[job.name]
        return {"job_id": job.id, "panel": job.name, **spec.measure_result(jobs, job, parameters)}

    def render_review(self, job):
        renderer = self._job_operations[job.name].render_review
        if renderer is not None:
            renderer(job)

    def workflows(self):
        return {name: operation.workflow_panels for name, operation in self._job_operations.items()
                if operation.workflow_panels}

    def prepare(self, jobs, operation, params):
        return self.prepare_stage(jobs, operation, params)

    def prepare_stage(self, jobs, operation, params, **options):
        """Trusted workflows can pass builder options for inputs produced by earlier stages."""
        spec = self._operations.get(operation) if isinstance(operation, str) else None
        if spec is None or not isinstance(params, dict):
            raise ValueError("Choose a supported panel operation and its named parameters.")
        if params.keys() - spec.description["parameters"].keys():
            raise ValueError("Use only parameters listed by describe for this operation.")
        if spec.prepare is None:
            raise ValueError("Open the panel's Ask Agent control to propose this operation.")
        job, summary = spec.prepare(jobs, params, **options)
        if job.name not in spec.job_names:
            raise ValueError(f"Job {job.name} is not registered for {operation}.")
        for key in spec.outputs:
            if key in job.extra:
                summary[f"{key.replace('_', ' ').capitalize()} output"] = job.extra[key]
        job.extra["agent_panel"] = True
        return job, summary

    def status(self, jobs, job_id):
        job = jobs.get(job_id) if isinstance(job_id, str) else None
        spec = self._job_operations.get(job.name) if job is not None else None
        if spec is None:
            raise ValueError("Choose a supported job ID from your workspace.")
        result = {"job_id": job.id, "panel": job.name, "status": job.status,
                  "progress": job.progress, "note": job.wait_reason,
                  "log_tail": [line[:500] for line in tuple(job.log.copy())[-15:]]}
        if spec.workflow_panels:
            if (checkpoint := job.checkpoint) is not None and job.status == "waiting":
                result["checkpoint"] = checkpoint.summary
            result["steps"] = []
            for stage in job.stages:
                step = {"panel": stage.name, "status": stage.status}
                if (stage.status in TERMINAL and stage.result is not None
                        and stage.result.name in self._job_operations and jobs.get(stage.result.id)):
                    step = self.status(jobs, stage.result.id)
                    step.pop("log_tail")
                result["steps"].append(step)
        elif job.status == "done":
            if "gmd_shift_source" in job.extra:
                result["alignment"] = {"shift": job.extra["gmd_shift"],
                                       "source": job.extra["gmd_shift_source"]}
            keys = spec.artifacts(job) if spec.artifacts else spec.outputs
            result["files"] = {key: _container_path(job.extra[key]) for key in keys
                               if key in job.extra and resolve_path(job.extra[key]).is_file()}
        return result


def require_completed_job(jobs, job_id, *, name):
    job = jobs.get(job_id) if isinstance(job_id, str) else None
    if job is None or job.name != name:
        raise ValueError("Choose a supported job ID from your workspace.")
    if job.status != "done":
        raise ValueError(f"Wait for {name} to finish successfully.")
    return job


def number(value):
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError("Use finite numeric values for limits, widths, and centers.")
    return value


def boolean(value):
    if type(value) is not bool:
        raise ValueError("Use true or false for Boolean parameters.")
    return value


def resolve_path(value, *, write=False):
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


def _container_path(value):
    path = Path(value)
    if shared.SERVER:
        for root, mount in ((shared.workspace_root(), "/workspace"), (shared.SERVER.datasets, "/datasets")):
            if path.is_relative_to(root):
                return str(PurePosixPath(mount, *path.relative_to(root).parts))
    return value
