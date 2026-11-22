"""Agent proposals use the panel builders and the signed-in account's job scope."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from vmi_analysis.agent_backend import AgentRequest, AgentSession, ContainerLauncher, PANEL_REQUEST_TITLE
from vmi_analysis.gui import agent_panels, shared
from vmi_analysis.gui.jobs import Job, JobManager
from vmi_analysis.gui.resources import job_access
from vmi_analysis.gui.panels import ion_luna, ion_tof
from vmi_analysis.gui.panels.common import adopt_agent_form


class AgentPanelTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / "files"
        self.workspace.mkdir()
        self.datasets = self.root / "data"
        self.datasets.mkdir()
        self.account = SimpleNamespace(name="analyst", root=self.workspace)
        server = shared.SharedServer.__new__(shared.SharedServer)
        server.datasets = self.datasets
        self.enterContext(patch.object(shared, "SERVER", server))
        self.enterContext(patch.object(shared, "current_account", return_value=self.account))
        self.jobs = JobManager()

    def test_dump_proposal_uses_normal_defaults_and_does_not_submit(self):
        (self.datasets / "sample.tpx3").touch()
        job, summary = agent_panels.prepare(self.jobs, "tpx3_dump", {
            "input": "/datasets/sample.tpx3", "output": "/workspace/ions.h5",
        })
        self.assertEqual(job.name, ion_luna.JOB_NAME)
        self.assertEqual(job.consume_output, ion_luna.capture_command)
        self.assertIn("--split-acquisitions", job.cmd)
        self.assertIn("--calculate-timewalk-statistics", job.cmd)
        self.assertFalse(any(arg.startswith("--wsl-") for arg in job.cmd))
        self.assertEqual(summary["Collection output"], str(self.workspace / "ions.segments.json"))
        self.assertEqual(self.jobs.jobs(), [])
        with self.assertRaises(shared.WorkspaceDenied):
            agent_panels.prepare(self.jobs, "tpx3_dump", {
                "input": "/datasets/sample.tpx3", "output": "/datasets/ions.h5",
            })

    def test_local_windows_dump_uses_the_panel_wsl_settings(self):
        source = self.datasets / "sample.tpx3"
        source.touch()
        executable = "/opt/my luna/bin/tpx3dump"
        settings = {"ion_luna.distribution": " Ubuntu-20.04 ", "ion_luna.executable": executable}
        with (patch.object(shared, "SERVER", None),
              patch.object(shared, "current_account", return_value=None),
              patch.object(shared, "workspace_root", return_value=self.workspace),
              patch.object(shared, "settings_store", return_value=settings),
              patch.object(ion_luna.sys, "platform", "win32")):
            job, summary = agent_panels.prepare(self.jobs, "tpx3_dump", {
                "input": str(source), "output": str(self.workspace / "ions.h5"),
            })
        self.assertIn("--wsl-distribution=Ubuntu-20.04", job.cmd)
        self.assertIn(f"--wsl-executable={executable}", job.cmd)
        self.assertEqual(summary["Linux LUNA executable"], executable)

    def test_dump_proposal_records_the_panel_form(self):
        (self.datasets / "sample.tpx3").touch()
        job, _ = agent_panels.prepare(self.jobs, "tpx3_dump", {
            "input": "/datasets/sample.tpx3", "output": "/workspace/ions.h5", "prefixed": False,
            "options": {"layout": "quad", "raw-only": True, "ctot-cut": 250},
        })
        form = job.extra["panel_form"]
        self.assertEqual(form["input_mode"], "File")
        self.assertEqual(form["input"], str(self.datasets / "sample.tpx3"))
        self.assertEqual(form["output"], str(self.workspace / "ions.h5"))
        self.assertIs(form["acquisition_prefix"], False)
        self.assertEqual(form["layout"], "quad")
        self.assertIs(form["raw-only"], True)
        self.assertEqual(form["timewalk_file"], "")
        self.assertNotIn("tof-tdc-reference", form)  # absent from a raw-only run

    def test_folder_dump_proposal_marks_the_folder_input_mode(self):
        self.datasets.joinpath("frames").mkdir()
        job, _ = agent_panels.prepare(self.jobs, "tpx3_dump", {
            "input": "/datasets/frames", "output": "/workspace/ions.h5",
        })
        self.assertEqual(job.extra["panel_form"]["input_mode"], "Folder")

    def test_spectrum_proposal_records_the_panel_form(self):
        output = self.workspace / "ions.h5"
        output.touch()
        dump = Job(ion_luna.JOB_NAME, status="done", owner=self.account.name, extra={"output": str(output)})
        self.jobs._jobs[dump.id] = dump
        job, _ = agent_panels.prepare(self.jobs, "ion_tof", {
            "dump_job_id": dump.id, "source": "both", "reference": "TDC2Falling",
            "bin_size_ns": 10, "minimum_ns": 5, "maximum_ns": 20000,
        })
        self.assertEqual(job.extra["panel_form"], {
            "input": str(output), "source": "both", "reference": "TDC2Falling",
            "bin_size_ns": 10.0, "minimum": 5.0, "maximum": 20000, "metadata": "",
        })

    def test_adoption_replaces_old_inputs_once_and_respects_later_edits(self):
        saved = {"ion_tof.metadata": "old-metadata.h5", "ion_tof.bin_size_ns": 5}
        job = Job(ion_tof.JOB_NAME, extra={"agent_panel": True,
                                          "panel_form": {"metadata": "", "bin_size_ns": 10}})
        with patch.object(shared, "settings_store", return_value=saved):
            self.assertEqual(adopt_agent_form(job, "ion_tof"), {"metadata": "", "bin_size_ns": 10})
            self.assertEqual(saved["ion_tof.metadata"], "")
            saved["ion_tof.metadata"] = "new-metadata.h5"
            saved["ion_tof.bin_size_ns"] = 20
            self.assertEqual(adopt_agent_form(job, "ion_tof"),
                             {"metadata": "new-metadata.h5", "bin_size_ns": 20})
            self.assertEqual(job.extra["panel_form"], {"metadata": "", "bin_size_ns": 10})

    def test_handoff_keeps_acquisition_metadata_and_owner_scope(self):
        output, metadata = self.workspace / "ions.h5", self.workspace / "meta.h5"
        for path in (output, metadata):
            path.touch()
        dump = Job(ion_luna.JOB_NAME, status="done", owner=self.account.name,
                   extra={"output": str(output), "metadata": str(metadata)})
        self.jobs._jobs[dump.id] = dump
        spectrum, options = agent_panels.prepare(self.jobs, "ion_tof", {"dump_job_id": dump.id})
        self.assertEqual(options["LUNA input"], str(output))
        self.assertEqual(spectrum.extra["metadata"], str(metadata))
        self.assertEqual(agent_panels.status(self.jobs, dump.id)["files"]["output"], "/workspace/ions.h5")
        spectrum.status = "done"
        spectrum.owner = self.account.name
        self.jobs._jobs[spectrum.id] = spectrum
        gate, options = agent_panels.prepare(self.jobs, "ion_tof_gate", {
            "tof_job_id": spectrum.id, "gmd_min_na": 2, "gmd_max_na": 8, "shift": -1,
        })
        self.assertEqual(options["Alignment shift d (frames)"], -1)
        self.assertEqual(gate.extra["metadata"], str(metadata))
        gate.status = "done"
        gate.owner = self.account.name
        self.jobs._jobs[gate.id] = gate
        fit, options = agent_panels.prepare(self.jobs, "ion_tof_fit", {
            "gate_job_id": gate.id, "region_ns": [9000, 9800], "peaks_ns": [[9400, 30]],
            "unfiltered": True,
        })
        self.assertEqual(fit.name, ion_tof.UNFILTERED_FIT_JOB_NAME)
        self.assertEqual(fit.extra["filtered_id"], gate.id)
        self.assertEqual(options["Counting and normalization"], "GMD2-gated")
        dump.owner = "another_user"
        with self.assertRaisesRegex(ValueError, "your workspace"):
            agent_panels.status(self.jobs, dump.id)

    def test_workflow_reserves_the_future_collection_and_stops_after_dump_failure(self):
        (self.datasets / "sample.tpx3").touch()
        workflow, summary = agent_panels.prepare(self.jobs, "ion_dump_to_tof", {
            "dump": {"input": "/datasets/sample.tpx3", "output": "/workspace/ions.h5"},
            "tof": {"bin_size_ns": 10},
        })
        collection = self.workspace / "ions.segments.json"
        self.assertFalse(collection.exists())
        self.assertEqual(summary["2. LUNA input"], str(collection))
        self.assertIn(collection.resolve(), job_access(workflow).writes)
        self.assertEqual(self.jobs.jobs(), [])
        self.addCleanup(self.jobs.shutdown)
        with patch.object(self.jobs, "_run_cmd", return_value=1) as run:
            self.jobs.submit(workflow)
            deadline = time.monotonic() + 5
            while workflow.status in ("queued", "running") and time.monotonic() < deadline:
                time.sleep(.01)
        run.assert_called_once()
        outcome = agent_panels.status(self.jobs, workflow.id)
        self.assertEqual(outcome["status"], "failed")
        self.assertEqual([step["status"] for step in outcome["steps"]], ["failed", "cancelled"])
        self.assertEqual(self.jobs.get(outcome["steps"][0]["job_id"]).owner, self.account.name)
        with patch.object(shared, "current_account", return_value=SimpleNamespace(name="other")):
            with self.assertRaisesRegex(ValueError, "your workspace"):
                agent_panels.status(self.jobs, workflow.id)

    def test_stopping_workflow_cancels_the_waiting_tof_stage(self):
        (self.datasets / "sample.tpx3").touch()
        workflow, _ = agent_panels.prepare(self.jobs, "ion_dump_to_tof", {
            "dump": {"input": "/datasets/sample.tpx3", "output": "/workspace/ions.h5"},
        })
        running, release = Event(), Event()

        def run_step(*args, **kwargs):
            running.set()
            release.wait(5)
            return 0

        self.addCleanup(self.jobs.shutdown)
        with patch.object(self.jobs, "_run_cmd", side_effect=run_step) as run:
            self.jobs.submit(workflow)
            try:
                self.assertTrue(running.wait(5))
                self.jobs.cancel(workflow.id)
            finally:
                release.set()
            deadline = time.monotonic() + 5
            while workflow.status in ("queued", "running") and time.monotonic() < deadline:
                time.sleep(.01)
        run.assert_called_once()
        self.assertEqual(workflow.status, "cancelled")
        self.assertEqual([stage.status for stage in workflow.stages], ["cancelled", "cancelled"])

    def test_stale_and_duplicate_approvals_cannot_be_claimed(self):
        clock = [0.0]
        request = AgentRequest(self.account.name, self.workspace, self.root / ".agent",
                               "https://provider.example", "test", "test-key")
        session = AgentSession(request, None, clock=lambda: clock[0])
        session.status = "running"
        record = {"type": "extension_ui_request", "method": "input", "title": PANEL_REQUEST_TITLE,
                  "id": "test-id", "placeholder": json.dumps({"action": "describe"})}
        session.handle_record(record)
        self.assertEqual(session.panel_request(), ("test-id", {"action": "describe"}))
        self.assertTrue(session.take_panel_request("test-id"))
        self.assertFalse(session.take_panel_request("test-id"))
        session.handle_record(record)
        clock[0] = 301
        self.assertFalse(session.take_panel_request("test-id"))

    def test_approved_tof_gate_and_fit_use_real_scheduler_and_artifacts(self):
        data = self.datasets / "luna.h5"
        metadata = self.datasets / "metadata.h5"
        ticks = np.array([100_000, 300_000], dtype=np.uint64)
        hits = np.random.default_rng(5).normal(50_000, 5_000, (2, 300)).astype(np.uint64)
        with h5py.File(data, "w") as handle:
            handle["TDCEvents"] = np.array([(1, tick) for tick in ticks],
                                           dtype=[("tdc_type", "u1"), ("timestamp", "u8")])
            handle["PixelHits"] = np.array([(tick,) for tick in (hits + ticks[:, None]).flat],
                                            dtype=[("toa", "u8")])
        with h5py.File(metadata, "w") as handle:
            handle["frames"] = np.array([(1, 3e-9, 3e-9), (2, 9e-9, 9e-9)],
                                        dtype=[("timestamp_ns", "i8"), ("gmd1", "f8"), ("gmd2", "f8")])

        def approve(operation, params):
            job, _ = agent_panels.prepare(self.jobs, operation, params)
            self.jobs.submit(job)
            deadline = time.monotonic() + 30
            while job.status in ("queued", "running") and time.monotonic() < deadline:
                time.sleep(.05)
            self.assertEqual(job.status, "done", "\n".join(job.log))
            return job

        self.addCleanup(self.jobs.shutdown)
        original = approve("ion_tof", {"input": "/datasets/luna.h5", "bin_size_ns": .2, "maximum_ns": 10})
        gated = approve("ion_tof_gate", {"tof_job_id": original.id, "metadata": "/datasets/metadata.h5",
                                       "gmd_min_na": 2, "gmd_max_na": 4, "shift": 0})
        fitted = approve("ion_tof_fit", {"gate_job_id": gated.id, "region_ns": [3, 7], "peaks_ns": [[5, 1]]})
        with np.load(fitted.extra["peak_result"]) as result:
            self.assertEqual(int(result["selected_triggers"]), 1)
            self.assertIn("pixels_peaks", result)
        self.assertIn("peak_result", agent_panels.status(self.jobs, fitted.id)["files"])

    def test_rootless_launch_arguments_keep_the_private_mounts_accessible(self):
        request = AgentRequest(self.account.name, self.workspace, self.root / ".agent",
                               "https://provider.example", "test", "test-key")
        config = self.root / ".agent" / "test"
        self.assertIn("--userns=keep-id", ContainerLauncher("podman").command(request, config, "test", rootless=True))
        self.assertNotIn("--user", ContainerLauncher("docker").command(request, config, "test", rootless=True))
