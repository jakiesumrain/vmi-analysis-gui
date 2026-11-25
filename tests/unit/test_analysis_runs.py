import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from vmi_analysis.gui import analysis_runs, shared
from vmi_analysis.gui.jobs import JobManager, TERMINAL
from vmi_analysis.gui.panel_operations import build_registry


class AnalysisRunTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.enterContext(patch.object(shared, "workspace_root", return_value=self.root))
        self.enterContext(patch.object(shared, "SERVER", None))
        self.jobs = JobManager(max_concurrent=1)
        self.addCleanup(self.jobs.shutdown)
        self.registry = build_registry()
        self.source = self.root / "analysis.py"

    def wait_for(self, predicate):
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.02)
        self.fail("Analysis run did not reach the expected state")

    def prepare(self, script_text, **kwargs):
        self.source.write_text(script_text, encoding="utf-8")
        job, _ = self.registry.prepare(self.jobs, "analysis_python", {"script": self.source.name, **kwargs})
        folder = self.root / ".agent" / "analysis" / "runs" / job.id
        return job, folder

    def test_approval_runs_frozen_script_and_records_input_parameters_and_generated_file(self):
        self.root.joinpath("counts.txt").write_text("15", encoding="utf-8")
        source = """\
import json
import sys
from pathlib import Path
run = json.loads(Path(sys.argv[1]).read_text())
counts = int(Path(run["inputs"]["data"]["path"]).read_text())
Path("result.txt").write_text(str(counts * run["parameters"]["scale"]))
print("wrote result")
"""
        job, folder = self.prepare(source, inputs={"data": "counts.txt"}, parameters={"scale": 2})
        self.assertFalse(folder.exists())
        self.source.write_text("raise RuntimeError('edited after review')", encoding="utf-8")
        self.jobs.submit(job)
        self.wait_for(lambda: job.status in TERMINAL)
        self.assertEqual(job.status, "done", "\n".join(job.log))
        self.assertEqual((folder / "script.py").read_text(), source)
        self.assertEqual((folder / "outputs/result.txt").read_text(), "30")
        record = json.loads((folder / "run.json").read_text())
        self.assertEqual(record["parameters"], {"scale": 2})
        self.assertEqual(record["status"], "done")
        self.assertEqual(record["artifacts"], ["result.txt"])
        self.assertIn("wrote result", (folder / "run.log").read_text())
        self.assertIn("output/result.txt", self.registry.status(self.jobs, job.id)["files"])
        self.assertEqual(analysis_runs._history()[0][0]["id"], job.id)

    def test_cancellation_and_failure_leave_run_records(self):
        running, folder = self.prepare("import time\nprint('running')\ntime.sleep(30)\n")
        self.jobs.submit(running)
        self.wait_for(lambda: "running" in running.log)
        queued, queued_folder = self.prepare("raise AssertionError('must not run')\n")
        self.jobs.submit(queued)
        self.assertEqual(queued.status, "queued")
        self.jobs.cancel(queued.id)
        self.jobs.cancel(running.id)
        self.wait_for(lambda: running.status in TERMINAL)
        for recorded_folder in (folder, queued_folder):
            self.assertEqual(json.loads((recorded_folder / "run.json").read_text())["status"], "cancelled")

        failing, failed_folder = self.prepare("raise ValueError('invalid calibration')\n")
        self.jobs.submit(failing)
        self.wait_for(lambda: failing.status in TERMINAL)
        self.assertEqual(failing.status, "failed")
        self.assertEqual(json.loads((failed_folder / "run.json").read_text())["status"], "failed")
        self.assertIn("invalid calibration", (failed_folder / "run.log").read_text())

    def test_saved_panel_reuses_snapshot_and_validates_new_inputs(self):
        declaration = {
            "schema": "vmi.analysis-panel/v1", "title": "Peak drift",
            "fields": {"factor": {"type": "number", "label": "Scale"}},
            "results": {"drift.txt": {"label": "Drift measurement", "view": "download"}},
        }
        job, folder = self.prepare(
            "import sys,json\nfrom pathlib import Path\n"
            "factor=json.loads(Path(sys.argv[1]).read_text())['parameters']['factor']\n"
            "Path('drift.txt').write_text(str(factor * 5))\n",
            panel_definition=declaration, parameters={"factor": 2},
        )
        self.jobs.submit(job)
        self.wait_for(lambda: job.status in TERMINAL)
        record = json.loads((folder / "run.json").read_text())
        analysis_runs._save_panel(record, folder)
        params = {"saved_panel": job.id, "parameters": {"factor": 3}}
        with self.assertRaisesRegex(ValueError, "requires a number"):
            self.registry.prepare(self.jobs, "analysis_python", {**params, "parameters": {"factor": True}})
        replay, _ = self.registry.prepare(self.jobs, "analysis_python", params)
        self.jobs.submit(replay)
        self.wait_for(lambda: replay.status in TERMINAL)
        replay_folder = folder.with_name(replay.id)
        self.assertEqual((replay_folder / "outputs/drift.txt").read_text(), "15")
        self.assertEqual((replay_folder / "script.py").read_bytes(), (folder / "script.py").read_bytes())

    def test_paths_outside_local_workspace_are_rejected(self):
        with TemporaryDirectory() as other:
            script = Path(other) / "external.py"
            script.touch()
            with self.assertRaisesRegex(ValueError, "inside this workspace"):
                self.registry.prepare(self.jobs, "analysis_python", {"script": str(script)})
