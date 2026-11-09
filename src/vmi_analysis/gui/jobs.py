"""Subprocess job manager for the VMI GUI.

Jobs run concurrently within a configurable limit and file/resource reservations.
Each job retains its own subprocess tree, log, progress and cancellation state.
"""

from __future__ import annotations

import os
import math
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager

from .resources import Access, canonical_path, job_access

TERMINAL = {"done", "failed", "cancelled"}
_PROGRESS_RE = re.compile(r"(\d{1,3})%\s*\|")


@dataclass
class Stage:
    """One step of a pipeline job."""
    name: str
    cmd: list[str]
    cwd: str | None = None
    outputs: list[str] = field(default_factory=list)  # skip if all exist
    status: str = "queued"


@dataclass
class Job:
    name: str
    cmd: list[str] | None = None
    cwd: str | None = None
    stages: list[Stage] | None = None
    skip_existing: bool = False
    env: dict | None = None  # extra env overrides ("" clears a variable)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: str = "queued"
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    returncode: int | None = None
    progress: float | None = None  # 0..1, or None if unknown
    log: deque = field(default_factory=lambda: deque(maxlen=5000))
    extra: dict = field(default_factory=dict)
    reads: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)
    exclusive: list[str] = field(default_factory=list)
    wait_reason: str = ""
    _access: Access = field(default_factory=Access, repr=False)
    _proc: subprocess.Popen | None = field(default=None, repr=False)
    _cancel_requested: bool = field(default=False, repr=False)
    _termination_thread: threading.Thread | None = field(default=None, repr=False)

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested


class ResourceBusyError(RuntimeError):
    pass


class JobManager:
    """Bounded concurrent jobs with conservative file and shared-resource locks."""

    def __init__(self, max_concurrent=2):
        self._jobs: dict[str, Job] = {}
        self._queue: deque[str] = deque()
        self._lock = threading.Lock()
        self._stopping = False
        self._workers: dict[str, threading.Thread] = {}
        self._preview_reads: dict[object, Access] = {}
        self._stop_previews = threading.Event()
        self._max_concurrent = self._validate_limit(max_concurrent)

    # ------------------------------------------------------------------ API

    def submit(self, job: Job) -> Job:
        access = job_access(job)
        with self._lock:
            if self._stopping:
                raise RuntimeError("The job manager is shutting down")
            if job.id in self._jobs:
                raise ValueError("This job has already been submitted")
            job._access = access
            self._jobs[job.id] = job
            self._queue.append(job.id)
            self._schedule_locked()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def jobs(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created)

    def running(self) -> list[Job]:
        with self._lock:
            return [self._jobs[job_id] for job_id in self._workers]

    @staticmethod
    def _validate_limit(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 1 or value != int(value):
            raise ValueError("Maximum concurrent jobs must be a positive whole number")
        return int(value)

    @property
    def max_concurrent(self):
        with self._lock:
            return self._max_concurrent

    def set_max_concurrent(self, value):
        limit = self._validate_limit(value)
        with self._lock:
            self._max_concurrent = limit
            self._schedule_locked()

    @contextmanager
    def write_access(self, paths):
        """Keep GUI output deletion atomic with job submission and scheduling."""
        access = Access(writes=frozenset(canonical_path(p) for p in paths))
        with self._lock:
            self._check_active_access_locked(access)
            yield

    @contextmanager
    def preview_read_access(self, paths):
        """Protect preview inputs without occupying a subprocess job slot."""
        access = Access(reads=frozenset(canonical_path(p) for p in paths))
        token = object()
        with self._lock:
            if self._stopping:
                raise ResourceBusyError("The job manager is shutting down")
            self._check_active_access_locked(access)
            self._preview_reads[token] = access
        try:
            yield self._stop_previews.is_set
        finally:
            with self._lock:
                del self._preview_reads[token]
                self._schedule_locked()

    def cancel(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in TERMINAL:
                return
            self._cancel_locked(job)
            self._schedule_locked()

    def begin_shutdown(self) -> None:
        """Stop queue advancement and request cancellation without blocking the UI."""
        with self._lock:
            self._stopping = True
            self._stop_previews.set()
            for job in self._jobs.values():
                if job.status not in TERMINAL:
                    self._cancel_locked(job)

    def shutdown(self) -> None:
        """Cancel all work and wait for process cleanup; call outside the UI loop."""
        self.begin_shutdown()
        with self._lock:
            workers = list(self._workers.values())
        deadline = time.monotonic() + 15.0
        for worker in workers:
            worker.join(timeout=max(0, deadline - time.monotonic()))
            if worker.is_alive():
                raise RuntimeError("Job process cleanup did not finish within 15 seconds")

    def _cancel_locked(self, job: Job) -> None:
        job._cancel_requested = True
        if job.status == "queued":
            job.status = "cancelled"
            job.finished = time.time()
            self._queue.remove(job.id)
            if job.stages:
                for stage in job.stages:
                    stage.status = "cancelled"
        elif job._proc is not None:
            self._terminate_locked(job)

    def _terminate_locked(self, job: Job) -> None:
        if job._termination_thread is not None:
            return
        job.log.append("[gui] cancel requested — terminating process tree...")
        job._termination_thread = threading.Thread(
            target=self._terminate_tree, args=(job, job._proc), daemon=True)
        job._termination_thread.start()

    def _terminate_tree(self, job: Job, proc: subprocess.Popen) -> None:
        try:
            if sys.platform == "win32":
                # CREATE_NEW_PROCESS_GROUP alone does not make terminate() kill
                # descendants (notably the MATLAB process launched by a script).
                result = subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    capture_output=True, timeout=5.0,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                if result.returncode and proc.poll() is None:
                    raise RuntimeError(f"taskkill failed: {result.stderr!r}")
            else:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        pass
                finally:
                    # Also kill descendants which ignored SIGTERM even if the
                    # direct child has already exited.
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            proc.wait(timeout=5.0)
        except Exception as exc:
            job.log.append(f"[gui] process tree cleanup error: {exc}")
            try:
                proc.kill()
                proc.wait(timeout=5.0)
            except Exception as fallback_exc:
                job.log.append(f"[gui] process cleanup error: {fallback_exc}")

    # -------------------------------------------------------------- internals

    def _check_active_access_locked(self, access):
        for job in self._jobs.values():
            if job.status in ("running", "queued"):
                conflict = access.conflict(job._access)
                if conflict:
                    raise ResourceBusyError(f"Files are reserved by {job.name}: {conflict}")
        for preview in self._preview_reads.values():
            if conflict := access.conflict(preview):
                raise ResourceBusyError(f"Files are reserved by a histogram preview: {conflict}")

    def _schedule_locked(self) -> None:
        if self._stopping:
            return
        waiting = []
        for job_id in list(self._queue):
            job = self._jobs[job_id]
            blockers = [self._jobs[key] for key in self._workers] + waiting
            blocker = next(((other, reason) for other in blockers
                            if (reason := job._access.conflict(other._access))), None)
            if blocker:
                other, reason = blocker
                job.wait_reason = f"Waiting for {other.name}: {reason}"
                waiting.append(job)
                continue
            preview_conflict = next((reason for preview in self._preview_reads.values()
                                     if (reason := job._access.conflict(preview))), None)
            if preview_conflict:
                job.wait_reason = f"Waiting for histogram preview: {preview_conflict}"
                waiting.append(job)
                continue
            if len(self._workers) >= self._max_concurrent:
                job.wait_reason = "Waiting for a concurrent job slot"
                waiting.append(job)
                continue
            self._queue.remove(job_id)
            job.wait_reason = ""
            job.status = "running"
            job.started = time.time()
            worker = threading.Thread(target=self._run, args=(job,), daemon=True)
            self._workers[job_id] = worker
            worker.start()

    def _run(self, job: Job) -> None:
        try:
            if job.stages is not None:
                self._run_pipeline(job)
            else:
                rc = self._run_cmd(job, job.cmd, job.cwd)
                job.returncode = rc
        except Exception as e:  # defensive: a GUI job must never die silently
            job.log.append(f"[gui] job runner error: {e}")
            job.returncode = -1
        with self._lock:
            job.finished = time.time()
            if job._cancel_requested:
                job.status = "cancelled"
            elif job.returncode == 0:
                job.status = "done"
                job.progress = 1.0
            else:
                job.status = "failed"
            self._workers.pop(job.id)
            self._schedule_locked()

    def _run_pipeline(self, job: Job) -> None:
        n = len(job.stages)
        for i, stage in enumerate(job.stages):
            if job._cancel_requested:
                for rest in job.stages[i:]:
                    rest.status = "cancelled"
                job.returncode = -1
                return
            if job.skip_existing and stage.outputs and all(
                    Path(p).exists() for p in stage.outputs):
                stage.status = "skipped"
                job.log.append(f"\n[stage {i+1}/{n}] {stage.name} — SKIPPED "
                               f"(outputs exist)")
                continue
            stage.status = "running"
            job.log.append(f"\n[stage {i+1}/{n}] {stage.name}")
            job.extra["stage_index"] = i
            job.progress = None
            rc = self._run_cmd(job, stage.cmd, stage.cwd,
                               prefix=f"[{stage.name}] ")
            stage.status = ("cancelled" if job._cancel_requested
                            else "done" if rc == 0 else "failed")
            if rc != 0 or job._cancel_requested:
                outcome = "CANCELLED" if job._cancel_requested else "FAILED"
                job.log.append(f"[stage {i+1}/{n}] {stage.name} {outcome} (rc={rc});"
                               f" stopping pipeline")
                job.returncode = rc
                for rest in job.stages[i + 1:]:
                    rest.status = "cancelled"
                return
        job.returncode = 0

    def _run_cmd(self, job: Job, cmd: list[str], cwd: str | None,
                 prefix: str = "") -> int:
        with self._lock:
            if job._cancel_requested:
                return -1
        env = dict(os.environ)
        env.setdefault("MPLBACKEND", "Agg")
        if job.env:
            env.update(job.env)
        env["PYTHONUNBUFFERED"] = "1"
        # The pipe is decoded as UTF-8 below; without this the child (Windows
        # locale) writes GBK and tqdm bars / em-dashes arrive as U+FFFD.
        env["PYTHONIOENCODING"] = "utf-8"
        job.log.append(f"[gui] $ {' '.join(str(c) for c in cmd)}")
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **kwargs,
        )
        with self._lock:
            job._proc = proc
            job._termination_thread = None
            # cancel() may have run while Popen was creating the process.
            if job._cancel_requested:
                self._terminate_locked(job)
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                # tqdm updates one line via \r — split and keep the last segment
                for seg in line.replace("\r", "\n").split("\n"):
                    seg = seg.rstrip()
                    if not seg:
                        continue
                    m = _PROGRESS_RE.search(seg)
                    if m:
                        job.progress = min(1.0, int(m.group(1)) / 100.0)
                    job.log.append(prefix + seg)
            proc.wait()
        finally:
            with self._lock:
                if proc.poll() is None:
                    self._terminate_locked(job)
                # Unregister atomically so a late cancel cannot create another
                # termination thread after we captured the thread to join.
                termination = job._termination_thread
                job._proc = None
            if termination is not None:
                termination.join()
            proc.stdout.close()
        return proc.returncode


MANAGER = JobManager()
