"""Subprocess job manager for the VMI GUI.

One job at a time (the MATLAB CPBASEX step uses shared global state, and the
network drive does not benefit from parallel readers). Jobs are subprocesses:
isolation, real cancellation, and streamed logs for free.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

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
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: str = "queued"
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    returncode: int | None = None
    progress: float | None = None  # 0..1, or None if unknown
    log: deque = field(default_factory=lambda: deque(maxlen=5000))
    extra: dict = field(default_factory=dict)
    _proc: subprocess.Popen | None = field(default=None, repr=False)
    _cancel_requested: bool = field(default=False, repr=False)


class JobManager:
    """Serial queue of subprocess jobs with log streaming and cancellation."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._queue: deque[str] = deque()
        self._current_id: str | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ API

    def submit(self, job: Job) -> Job:
        with self._lock:
            self._jobs[job.id] = job
            self._queue.append(job.id)
            if self._current_id is None:
                self._start_next_locked()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def jobs(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created)

    def current(self) -> Job | None:
        return self._jobs.get(self._current_id) if self._current_id else None

    def cancel(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in TERMINAL:
                return
            job._cancel_requested = True
            if job.status == "queued":
                job.status = "cancelled"
                job.finished = time.time()
                try:
                    self._queue.remove(job_id)
                except ValueError:
                    pass
                return
            proc = job._proc
        # running: terminate outside the lock
        if proc is not None:
            job.log.append("[gui] cancel requested — terminating process...")
            try:
                proc.terminate()
            except Exception:
                pass
            threading.Thread(target=self._kill_after, args=(proc, 5.0),
                             daemon=True).start()

    def _kill_after(self, proc: subprocess.Popen, timeout: float) -> None:
        try:
            proc.wait(timeout=timeout)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    # -------------------------------------------------------------- internals

    def _start_next_locked(self) -> None:
        while self._queue:
            job_id = self._queue.popleft()
            job = self._jobs[job_id]
            if job.status == "cancelled":
                continue
            self._current_id = job_id
            job.status = "running"
            job.started = time.time()
            threading.Thread(target=self._run, args=(job,), daemon=True).start()
            return
        self._current_id = None

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
        job.finished = time.time()
        if job._cancel_requested:
            job.status = "cancelled"
        elif job.returncode == 0:
            job.status = "done"
            job.progress = 1.0
        else:
            job.status = "failed"
        with self._lock:
            self._start_next_locked()

    def _run_pipeline(self, job: Job) -> None:
        n = len(job.stages)
        for i, stage in enumerate(job.stages):
            if job._cancel_requested:
                stage.status = "cancelled"
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
            stage.status = "done" if rc == 0 else "failed"
            if rc != 0:
                job.log.append(f"[stage {i+1}/{n}] {stage.name} FAILED (rc={rc});"
                               f" stopping pipeline")
                job.returncode = rc
                for rest in job.stages[i + 1:]:
                    rest.status = "cancelled"
                return
        job.returncode = 0

    def _run_cmd(self, job: Job, cmd: list[str], cwd: str | None,
                 prefix: str = "") -> int:
        env = dict(os.environ)
        env.setdefault("MPLBACKEND", "Agg")
        env["PYTHONUNBUFFERED"] = "1"
        job.log.append(f"[gui] $ {' '.join(str(c) for c in cmd)}")
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
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
        job._proc = proc
        assert proc.stdout is not None
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
        job._proc = None
        return proc.returncode


MANAGER = JobManager()
