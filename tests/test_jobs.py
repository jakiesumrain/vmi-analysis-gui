"""Regression checks for cancellation and shutdown, using harmless child processes."""

import ctypes
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from gui.jobs import Job, JobManager, Stage


def wait_for(predicate, timeout=8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for job state")


class JobShutdownTests(unittest.TestCase):
    def setUp(self):
        self.manager = JobManager()
        self.addCleanup(self.manager.shutdown)

    def sleeping_job(self, name="sleep"):
        return Job(name=name, cmd=[sys.executable, "-c", "import time; time.sleep(60)"])

    def test_shutdown_cancels_queue_and_rejects_new_jobs(self):
        running = self.manager.submit(self.sleeping_job())
        queued = self.manager.submit(self.sleeping_job("queued"))
        wait_for(lambda: running._proc is not None)
        proc = running._proc
        self.manager.shutdown()
        self.assertEqual(running.status, "cancelled")
        self.assertEqual(queued.status, "cancelled")
        self.assertIsNone(queued.started)
        self.assertIsNotNone(proc.poll())
        self.assertIsNone(self.manager.current())
        self.manager.shutdown()  # repeated shutdown is safe
        with self.assertRaises(RuntimeError):
            self.manager.submit(self.sleeping_job())

    def test_cancel_during_process_creation(self):
        entered, release = threading.Event(), threading.Event()
        original = subprocess.Popen

        def delayed_spawn(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("Spawn test was not released")
            return original(*args, **kwargs)

        with patch("gui.jobs.subprocess.Popen", delayed_spawn):
            job = self.manager.submit(self.sleeping_job())
            try:
                self.assertTrue(entered.wait(5))
                self.manager.begin_shutdown()
            finally:
                release.set()
            self.manager.shutdown()
        self.assertEqual(job.status, "cancelled")
        self.assertIsNone(job._proc)
        self.assertFalse(any("cleanup error" in line for line in job.log), list(job.log))

    def test_cancel_does_not_stop_normal_queue_advancement(self):
        first = self.manager.submit(self.sleeping_job())
        second = self.manager.submit(Job(name="next", cmd=[sys.executable, "-c", "print('next')"]))
        wait_for(lambda: first._proc is not None)
        self.manager.cancel(first.id)
        wait_for(lambda: second.status == "done")
        self.assertEqual(first.status, "cancelled")

    def test_shutdown_cancels_remaining_pipeline_stages(self):
        stages = [Stage(name=str(i), cmd=self.sleeping_job().cmd) for i in range(3)]
        job = self.manager.submit(Job(name="pipeline", stages=stages))
        wait_for(lambda: job._proc is not None)
        self.manager.shutdown()
        self.assertEqual([s.status for s in stages], ["cancelled"] * 3)

    def test_shutdown_kills_descendant_process(self):
        with tempfile.TemporaryDirectory() as folder:
            pid_path = Path(folder) / "child.pid"
            code = (
                "import subprocess,sys,time; from pathlib import Path; "
                "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                "Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)"
            )
            job = self.manager.submit(Job(name="tree", cmd=[sys.executable, "-c", code, str(pid_path)]))
            wait_for(lambda: pid_path.exists() and pid_path.stat().st_size > 0)
            child_pid = int(pid_path.read_text())
            # Keep a process handle so a re-used PID cannot confuse the Windows check.
            handle = None
            if sys.platform == "win32":
                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel.OpenProcess.restype = ctypes.c_void_p
                kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
                kernel.CloseHandle.argtypes = [ctypes.c_void_p]
                handle = kernel.OpenProcess(0x00100000, False, child_pid)
                self.assertTrue(handle)
            try:
                self.manager.shutdown()
                if handle:
                    self.assertEqual(kernel.WaitForSingleObject(handle, 1000), 0)
                else:
                    # On Linux a killed orphan may remain as a zombie briefly.
                    def dead():
                        stat = Path(f"/proc/{child_pid}/stat")
                        if stat.exists():
                            return stat.read_text().split()[2] == "Z"
                        try:
                            os.kill(child_pid, 0)
                        except ProcessLookupError:
                            return True
                        return False
                    wait_for(dead)
                self.assertFalse(any("cleanup error" in line for line in job.log), list(job.log))
            finally:
                if handle:
                    kernel.CloseHandle(handle)


if __name__ == "__main__":
    unittest.main()
