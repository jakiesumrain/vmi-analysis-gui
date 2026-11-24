"""Agent proposals use the panel builders and the signed-in account's job scope."""

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import h5py
import numpy as np
from nicegui import Client, ui

from vmi_analysis.agent_backend import AgentRequest, AgentSession, ContainerLauncher, PANEL_REQUEST_TITLE
from vmi_analysis.gui import shared
from vmi_analysis.gui.agent_completion import path_completions
from vmi_analysis.gui.panel_operations import build_registry
from vmi_analysis.gui.jobs import Job, JobManager, ResourceBusyError
from vmi_analysis.ion_acquisition import SCHEMA
from vmi_analysis.gui.resources import job_access
from vmi_analysis.gui.panels import agent, dark, ion_luna, ion_pipeline, ion_tof
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
        self.account = SimpleNamespace(name="analyst", root=self.workspace, settings={})
        server = shared.SharedServer.__new__(shared.SharedServer)
        server.datasets = self.datasets
        self.enterContext(patch.object(shared, "SERVER", server))
        self.enterContext(patch.object(shared, "current_account", return_value=self.account))
        self.jobs = JobManager()
        self.registry = build_registry()

    def test_dark_frame_registration_validates_and_reserves_paths(self):
        raw = self.datasets / "dark"
        raw.mkdir()
        (raw / "RAW-dark.h5").touch()
        description = self.registry.describe()["dark_frame"]
        self.assertEqual(description["panel"], "Dark Frame")
        self.assertEqual(description["parameters"]["method"]["choices"], ["median", "mean"])
        job, summary = self.registry.prepare(self.jobs, "dark_frame", {
            "dark_dir": "/datasets/dark", "frames": 0,
        })
        output = self.workspace / "centroiding"
        self.assertEqual(summary["Dark frame count"], "All frames")
        access = job_access(job)
        self.assertIn(raw.resolve(), access.reads)
        self.assertEqual(access.writes, {(output / name).resolve()
                                        for name in ("dark_median.npy", "dark_median_info.txt")})
        self.assertEqual(self.jobs.jobs(), [])
        for invalid in ({"out_dir": "/datasets/dark"}, {"pattern": "../RAW-*.h5"}):
            with self.assertRaises(shared.WorkspaceDenied):
                self.registry.prepare(self.jobs, "dark_frame", {"dark_dir": "/datasets/dark", **invalid})
        for invalid in ({"frames": 1.5}, {"method": "unknown"}, {"pattern": "missing-*.h5"}):
            with self.assertRaises(ValueError):
                self.registry.prepare(self.jobs, "dark_frame", {"dark_dir": "/datasets/dark", **invalid})

    def test_manual_and_agent_dark_frame_outputs_and_panel_handoff(self):
        raw = self.datasets / "dark"
        raw.mkdir()
        data = np.arange(6 * 8 * 8, dtype=np.uint16).reshape(6, 8, 8)
        for index, frames in enumerate(np.split(data, 2)):
            with h5py.File(raw / f"RAW-{index}.h5", "w") as handle:
                handle["raw/data"] = frames
        target = self.workspace / "centroiding"
        self.addCleanup(self.jobs.shutdown)

        def run_and_read(job):
            self.jobs.submit(job)
            deadline = time.monotonic() + 20
            while job.status in ("queued", "running") and time.monotonic() < deadline:
                time.sleep(.04)
            self.assertEqual(job.status, "done", "\n".join(job.log))
            self.assertTrue(any("% |" in line for line in job.log))
            stats = Path(job.extra["info"]).read_text().splitlines()
            return np.load(job.extra["dark_npy"]), [line for line in stats if not line.startswith("computed_seconds=")]

        for method, calculate in (("mean", np.mean), ("median", np.median)):
            with self.subTest(method=method):
                manual = dark.dark_frame_job(raw, method=method, frames=4, out_dir=target)
                prepared, _ = self.registry.prepare(self.jobs, "dark_frame", {
                    "dark_dir": "/datasets/dark", "method": method, "frames": 4,
                    "out_dir": "/workspace/centroiding",
                })
                self.assertEqual(manual.cmd, prepared.cmd)
                manual_frame, manual_info = run_and_read(manual)
                agent_frame, agent_info = run_and_read(prepared)
                np.testing.assert_array_equal(agent_frame, manual_frame)
                np.testing.assert_array_equal(agent_frame, calculate(data[:4], axis=0))
                self.assertEqual(agent_info, manual_info)
                outcome = self.registry.status(self.jobs, prepared.id)
                self.assertEqual(outcome["progress"], 1.0)
                self.assertEqual(outcome["files"], {
                    "dark_npy": f"/workspace/centroiding/dark_{method}.npy",
                    "info": f"/workspace/centroiding/dark_{method}_info.txt",
                })

        views = []
        real_view = dark.JobView

        def track_view(*args, **kwargs):
            view = real_view(*args, **kwargs)
            views.append(view)
            return view

        client = Client(ui.page("/_test_dark_frame_agent_result"))
        self.addCleanup(client.delete)
        with (client, patch.object(dark, "JobView", side_effect=track_view),
              patch.object(shared, "workspace_timer")):
            dark.build(self.jobs)
            views[-1].refresh()
            self.assertIs(views[-1].job, prepared)
            count = next(element for element in client.elements.values()
                         if element._props.get("label") == "Dark frame count (0 = all)")
            self.assertEqual(count.value, 4)
            self.assertTrue(any(str(getattr(element, "text", "")).startswith("Dark frame: ")
                                for element in client.elements.values()))

    def test_agent_dark_frame_can_be_cancelled_while_queued_in_its_panel(self):
        (self.datasets / "RAW-dark.h5").touch()
        manual = dark.dark_frame_job(self.datasets)
        proposal, _ = self.registry.prepare(self.jobs, "dark_frame", {"dark_dir": "/datasets"})
        running, release = Event(), Event()
        self.addCleanup(self.jobs.shutdown)

        def hold_job(*_args, **_kwargs):
            running.set()
            release.wait(5)
            return 0

        with patch.object(self.jobs, "_run_cmd", side_effect=hold_job) as run:
            self.jobs.submit(manual)
            try:
                self.assertTrue(running.wait(5))
                self.jobs.submit(proposal)
                self.assertEqual(self.registry.status(self.jobs, proposal.id)["status"], "queued")
                self.jobs.cancel(proposal.id)
                self.assertEqual(self.registry.status(self.jobs, proposal.id)["status"], "cancelled")
            finally:
                release.set()
        run.assert_called_once()

    def test_dump_proposal_uses_normal_defaults_and_does_not_submit(self):
        (self.datasets / "sample.tpx3").touch()
        job, summary = self.registry.prepare(self.jobs, "tpx3_dump", {
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
            self.registry.prepare(self.jobs, "tpx3_dump", {
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
            job, summary = self.registry.prepare(self.jobs, "tpx3_dump", {
                "input": str(source), "output": str(self.workspace / "ions.h5"),
            })
        self.assertIn("--wsl-distribution=Ubuntu-20.04", job.cmd)
        self.assertIn(f"--wsl-executable={executable}", job.cmd)
        self.assertEqual(summary["Linux LUNA executable"], executable)

    def test_dump_proposal_records_the_panel_form(self):
        (self.datasets / "sample.tpx3").touch()
        job, _ = self.registry.prepare(self.jobs, "tpx3_dump", {
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

    def test_discovery_exposes_typed_inputs_and_rejects_internal_builder_options(self):
        spectrum = self.registry.describe("ion_tof")
        self.assertEqual(spectrum["parameters"]["bin_size_ns"]["unit"], "ns")
        self.assertEqual(spectrum["outputs"], ["histogram"])
        with self.assertRaisesRegex(ValueError, "listed by describe"):
            self.registry.prepare(self.jobs, "ion_tof", {"input": "missing.h5", "pending_input": True})
        with self.assertRaisesRegex(ValueError, "does not exist"):
            self.registry.prepare(self.jobs, "ion_tof", {"input": "missing.h5"})

    def test_ion_pipeline_proposal_uses_manual_builder_and_workspace_paths(self):
        scan = self.datasets / "scan"
        point = scan / "Tr200p8_ion"
        point.mkdir(parents=True)
        params = {
            "input": "/datasets/scan", "output": "/workspace/ion-results",
            "source": "clusters", "fit_tof": "ungated", "gmd_shift": -2,
            "peaks": [{"name": " A ", "low": 10150, "high": 11050, "center": 10545},
                      {"name": "B", "low": 8800, "high": 9600, "center": 9155}],
        }
        description = self.registry.describe("ion_pipeline")
        self.assertEqual(description["parameters"]["gmd_shift"]["default"], -1)
        self.assertNotIn("wsl_executable", description["parameters"])
        self.assertNotIn("ion_pipeline_fallback", self.registry.describe())
        self.account.settings["ion_pipeline.wsl_executable"] = "tpx3dump-custom"
        proposed, summary = self.registry.prepare(self.jobs, "ion_pipeline", params)
        manual = ion_pipeline.pipeline_job(scan, self.workspace / "ion-results", params)
        self.assertEqual(proposed.extra["settings"], manual.extra["settings"])
        self.assertEqual(proposed.extra["settings"]["wsl_executable"], "")
        self.assertEqual(proposed.extra["panel_form"]["gate"], True)
        self.assertEqual(summary["Ratio CSV peak pair"], ["A", "B"])
        self.assertEqual(proposed.reads, [str(point)])
        self.assertEqual(self.jobs.jobs(), [])
        access = job_access(proposed)
        self.assertIn(point, access.reads)
        self.assertIn(Path(proposed.extra["output"]), access.writes)
        for change, message in (
            ({"output": "/datasets/output"}, "read-only"),
            ({"fit_bins": True}, "finite numeric"),
            ({"disable_clustering": True}, "Enable clustering"),
            ({"wsl_executable": "custom"}, "listed by describe"),
            ({"peaks": [params["peaks"][0]] * 2}, "distinct name"),
        ):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                self.registry.prepare(self.jobs, "ion_pipeline", {**params, **change})

    def test_folder_dump_proposal_marks_the_folder_input_mode(self):
        self.datasets.joinpath("frames").mkdir()
        job, _ = self.registry.prepare(self.jobs, "tpx3_dump", {
            "input": "/datasets/frames", "output": "/workspace/ions.h5",
        })
        self.assertEqual(job.extra["panel_form"]["input_mode"], "Folder")

    def test_spectrum_proposal_records_the_panel_form(self):
        output = self.workspace / "ions.h5"
        output.touch()
        dump = Job(ion_luna.JOB_NAME, status="done", owner=self.account.name, extra={"output": str(output)})
        self.jobs._jobs[dump.id] = dump
        job, _ = self.registry.prepare(self.jobs, "ion_tof", {
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
        spectrum, options = self.registry.prepare(self.jobs, "ion_tof", {"dump_job_id": dump.id})
        self.assertEqual(options["LUNA input"], str(output))
        self.assertEqual(spectrum.extra["metadata"], str(metadata))
        self.assertEqual(self.registry.status(self.jobs, dump.id)["files"]["output"], "/workspace/ions.h5")
        spectrum.status = "done"
        spectrum.owner = self.account.name
        self.jobs._jobs[spectrum.id] = spectrum
        gate, options = self.registry.prepare(self.jobs, "ion_tof_gate", {
            "tof_job_id": spectrum.id, "gmd_min_na": 2, "gmd_max_na": 8, "shift": -1,
        })
        self.assertEqual(options["Alignment shift d (frames)"], -1)
        self.assertEqual(gate.extra["metadata"], str(metadata))
        gate.status = "done"
        gate.owner = self.account.name
        self.jobs._jobs[gate.id] = gate
        fit, options = self.registry.prepare(self.jobs, "ion_tof_fit", {
            "gate_job_id": gate.id, "region_ns": [9000, 9800], "peaks_ns": [[9400, 30]],
            "unfiltered": True,
        })
        self.assertEqual(fit.name, ion_tof.UNFILTERED_FIT_JOB_NAME)
        self.assertEqual(fit.extra["filtered_id"], gate.id)
        self.assertEqual(options["Counting and normalization"], "GMD2-gated")
        dump.owner = "another_user"
        with self.assertRaisesRegex(ValueError, "your workspace"):
            self.registry.status(self.jobs, dump.id)

    def test_workflow_reserves_the_future_collection_and_stops_after_dump_failure(self):
        (self.datasets / "sample.tpx3").touch()
        workflow, summary = self.registry.prepare(self.jobs, "ion_dump_to_tof", {
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
        outcome = self.registry.status(self.jobs, workflow.id)
        self.assertEqual(outcome["status"], "failed")
        self.assertEqual([step["status"] for step in outcome["steps"]], ["failed", "cancelled"])
        self.assertEqual(self.jobs.get(outcome["steps"][0]["job_id"]).owner, self.account.name)
        with patch.object(shared, "current_account", return_value=SimpleNamespace(name="other")):
            with self.assertRaisesRegex(ValueError, "your workspace"):
                self.registry.status(self.jobs, workflow.id)

    def test_stopping_workflow_cancels_the_waiting_tof_stage(self):
        (self.datasets / "sample.tpx3").touch()
        workflow, _ = self.registry.prepare(self.jobs, "ion_dump_to_tof", {
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

    def test_gmd_checkpoint_releases_worker_and_preserves_exact_inputs(self):
        (self.datasets / "sample.tpx3").touch()
        self.jobs.set_max_concurrent(1)
        self.addCleanup(self.jobs.shutdown)
        workflow, _ = self.registry.prepare(self.jobs, "ion_dump_to_tof", {
            "dump": {"input": "/datasets/sample.tpx3", "output": "/workspace/ions.h5"},
            "tof": {"bin_size_ns": 1, "maximum_ns": 10},
            "gate": {"gmd_min_na": 2, "gmd_max_na": 4},
        })
        dump, spectrum, gate = workflow.stages
        collection = Path(spectrum.result.extra["source_file"])
        data = self.workspace / "ions_segments" / "hits.h5"
        metadata = data.with_name("metadata.h5")
        run_command = self.jobs._run_cmd
        other, _ = self.registry.prepare(self.jobs, "tpx3_dump", {
            "input": "/datasets/sample.tpx3", "output": "/workspace/other.h5",
        })

        def simulate_luna(job, command, cwd, **kwargs):
            if job is other:
                return 0
            if command != dump.cmd:
                return run_command(job, command, cwd, **kwargs)
            data.parent.mkdir()
            with h5py.File(data, "w") as handle:
                handle["TDCEvents"] = np.array([(1, 100_000), (1, 300_000)],
                                               dtype=[("tdc_type", "u1"), ("timestamp", "u8")])
                handle["PixelHits"] = np.array([(150_000,), (350_000,)], dtype=[("toa", "u8")])
            with h5py.File(metadata, "w") as handle:
                handle["frames"] = np.array([(1, 9e-9, 9e-9), (2, 3e-9, 3e-9)],
                                           dtype=[("timestamp_ns", "i8"), ("gmd1", "f8"), ("gmd2", "f8")])
            collection.write_text(json.dumps({
                "schema": SCHEMA, "segments": [{"data": "ions_segments/hits.h5",
                                                 "metadata": "ions_segments/metadata.h5"}],
            }), encoding="utf-8")
            return 0

        def wait(job, status):
            deadline = time.monotonic() + 15
            while job.status not in (status, "failed", "cancelled") and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertEqual(job.status, status, "\n".join(job.log))

        with patch.object(self.jobs, "_run_cmd", side_effect=simulate_luna) as run:
            self.jobs.submit(workflow)
            wait(workflow, "waiting")
            checkpoint = workflow.checkpoint
            self.assertEqual(self.registry.status(self.jobs, workflow.id)["checkpoint"]["tof_job_id"],
                             spectrum.result.id)
            self.assertEqual([step.status for step in workflow.stages], ["done", "done", "waiting"])
            with self.assertRaises(ResourceBusyError), self.jobs.write_access([metadata]):
                pass
            with patch.object(shared, "current_account", return_value=SimpleNamespace(name="other")):
                with self.assertRaisesRegex(ValueError, "no longer pending"):
                    self.jobs.resolve_checkpoint(workflow.id, checkpoint, -1)
            with self.assertRaisesRegex(ValueError, "available option"):
                self.jobs.resolve_checkpoint(workflow.id, checkpoint, True)
            self.jobs.submit(other)
            wait(other, "done")
            self.jobs.resolve_checkpoint(workflow.id, checkpoint, -1)
            with self.assertRaisesRegex(ValueError, "no longer pending"):
                self.jobs.resolve_checkpoint(workflow.id, checkpoint, 0)
            wait(workflow, "done")
            self.assertEqual(sum(call.args[1] == dump.cmd for call in run.call_args_list), 1)
            self.assertEqual(sum(call.args[1] == spectrum.cmd for call in run.call_args_list), 1)
        filtered = self.jobs.get(gate.result.id)
        self.assertEqual(filtered.extra["parent_id"], spectrum.result.id)
        self.assertEqual(filtered.extra["metadata"], str(collection))
        self.assertEqual(self.registry.status(self.jobs, filtered.id)["alignment"],
                         {"shift": -1, "source": "confirmed"})
        with np.load(filtered.extra["histogram"], allow_pickle=False) as result:
            self.assertEqual(int(result["selected_triggers"]), 1)
            self.assertEqual(int(result["pixels"].sum()), 1)

    def test_gmd_automatic_shift_requires_agreement_and_clear_peaks(self):
        coefficients = np.tile([.1, .1], (len(ion_tof.GMD_SHIFTS), 1))
        result = {"gmd_shifts": np.array(ion_tof.GMD_SHIFTS),
                  "pixels_gmd_correlation": coefficients,
                  "clusters_gmd_correlation": coefficients.copy()}
        index = ion_tof.GMD_SHIFTS.index(-1)
        coefficients[index] = [.8, .75]
        result["clusters_gmd_correlation"][index] = [.7, .65]
        self.assertEqual(ion_tof._automatic_shift(result, ("pixels", "clusters")), -1)
        coefficients[index + 1, 0] = .77
        self.assertIsNone(ion_tof._automatic_shift(result, ("pixels", "clusters")))
        coefficients[index + 1, 0] = .92
        self.assertIsNone(ion_tof._automatic_shift(result, ("pixels", "clusters")))

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

    def test_at_completion_matches_nested_paths_and_stays_within_the_workspace(self):
        folder = self.workspace / "Results"
        folder.mkdir()
        expected = folder / "ion spectrum.h5"
        expected.touch()
        hidden = self.workspace / ".ion-secret"
        hidden.touch()
        outside = self.root / "ion-external.h5"
        outside.touch()
        roots = (("workspace", self.workspace), ("datasets", self.datasets))
        self.assertEqual(path_completions("iospc", roots)[0]["value"], "/workspace/Results/ion spectrum.h5")
        self.assertEqual(path_completions("iospc", roots[:1])[0]["value"], "Results/ion spectrum.h5")
        self.assertEqual(path_completions("/workspace/Results/ion sp", roots)[0]["label"],
                         "/workspace/Results/ion spectrum.h5")
        self.assertEqual(path_completions("../ion", roots), [])

    def test_agent_attachment_paths_respect_workspace_and_dataset_mounts(self):
        path = self.workspace / "data file.h5"
        path.touch()
        self.assertEqual(agent._attachment_reference(path, container=True), "/workspace/data file.h5")
        self.assertEqual(agent._attachment_reference(self.datasets, container=True), "/datasets")
        outside = self.root / "outside.h5"
        outside.touch()
        with self.assertRaises(shared.WorkspaceDenied):
            agent._attachment_reference(outside, container=True)
        with self.assertRaisesRegex(ValueError, "no longer available"):
            agent._attachment_reference(self.workspace / "deleted.h5", container=True)

    def test_contextual_fit_stays_in_its_tab_and_submits_the_edited_bounds(self):
        data = self.datasets / "sample.h5"
        data.touch()
        gated = Job(ion_tof.FILTER_JOB_NAME, status="done", owner=self.account.name,
                    extra={"source_file": str(data), "histogram": str(self.workspace / "gate.npz"),
                           "reference": ion_tof.DEFAULT_REFERENCE})
        jobs = Mock()
        jobs.get.side_effect = lambda job_id: gated if job_id == gated.id else None
        jobs.jobs.return_value = []
        jobs.submit.side_effect = lambda job: job

        session = Mock(id="session", status="ready")
        session.snapshot.return_value = {
            "id": session.id, "status": "ready", "activity": "", "entries": [], "diagnostics": [],
        }
        page = ui.page("/_test_agent_view")
        origin, other = Client(page), Client(page)
        self.addCleanup(origin.delete)
        self.addCleanup(other.delete)

        def build_tab(client):
            timers, buttons = {}, {}
            navigate = Mock()

            def button(text="", **kwargs):
                element = real_button(text, **kwargs)
                if callback := kwargs.get("on_click"):
                    buttons[text] = callback
                if text == "Send" or kwargs.get("icon") == "arrow_upward":
                    on_click = element.on_click

                    def connect(callback):
                        buttons["Send"] = callback
                        return on_click(callback)

                    element.on_click = connect
                return element

            def timer(_interval, callback):
                timers.setdefault(callback.__name__, callback)

            with client, patch.object(shared, "workspace_timer", timer), patch.object(ui, "button", button):
                ask_agent = agent.build(jobs, registry=self.registry, navigate=navigate)
                ion_tof._peak_fitting(jobs, gated, np.arange(11), {"pixels": np.ones(10)}, ask_agent=ask_agent)
            return timers, buttons, navigate

        def field(label):
            return next(element for element in origin.elements.values() if element._props.get("label") == label)

        real_button = ui.button
        with patch.object(agent.AGENTS, "current", return_value=session):
            timers, buttons, navigate = build_tab(origin)
            other_timers, _, other_navigate = build_tab(other)
            with origin:
                field("Fit TOF min (ns)").value = 2
                field("Fit TOF max (ns)").value = 8
                buttons["Ask Agent"]()
                field("Instruction").value = "Fit this peak"
                asyncio.run(buttons["Send"]())
            context = session.prompt.call_args.kwargs["model_context"]
            view_id = json.loads(context.splitlines()[-1])["view_id"]
            session.status = "running"
            session.snapshot.return_value["status"] = "running"
            session.panel_request.return_value = ("draft", {
                "action": "propose", "operation": "ion_tof_fit", "view_id": view_id,
                "parameters": {"gate_job_id": "different-run", "region_ns": [3, 7]},
            })

            with other:
                asyncio.run(other_timers["refresh_panel"]())
            other_navigate.assert_not_called()
            session.take_panel_request.assert_not_called()

            with origin, patch.object(ui, "button", wraps=real_button) as added:
                asyncio.run(timers["refresh_panel"]())
                navigate.assert_called_with("Ion TOF")
                self.assertEqual(field("Fit TOF min (ns)").value, 3)
                self.assertEqual(field("Fit TOF max (ns)").value, 7)
                field("Fit TOF min (ns)").value = 4
                approve = next(call.kwargs["on_click"] for call in added.call_args_list
                               if call.args == ("Approve Agent fit",))
                asyncio.run(approve())
        jobs.submit.assert_called_once()
        submitted = jobs.submit.call_args.args[0]
        self.assertEqual(submitted.extra["filtered_id"], gated.id)
        self.assertEqual(submitted.extra["fit_range"], [4, 7])
        session.take_panel_request.assert_called_once_with("draft")

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

        def run(job):
            self.jobs.submit(job)
            deadline = time.monotonic() + 30
            while job.status in ("queued", "running") and time.monotonic() < deadline:
                time.sleep(.05)
            self.assertEqual(job.status, "done", "\n".join(job.log))
            return job

        def approve(operation, params):
            job, _ = self.registry.prepare(self.jobs, operation, params)
            return run(job)

        self.addCleanup(self.jobs.shutdown)
        (self.datasets / "sample.tpx3").touch()
        workflow, _ = self.registry.prepare(self.jobs, "ion_dump_to_tof", {
            "dump": {"input": "/datasets/sample.tpx3", "output": "/workspace/dump.h5", "prefixed": False},
            "tof": {"bin_size_ns": .2, "maximum_ns": 10},
        })
        dump, spectrum = workflow.stages
        run_command = self.jobs._run_cmd

        def simulate_luna(job, command, cwd, **kwargs):
            if command == dump.cmd:
                Path(dump.result.extra["output"]).write_bytes(data.read_bytes())
                return 0
            return run_command(job, command, cwd, **kwargs)

        with patch.object(self.jobs, "_run_cmd", side_effect=simulate_luna):
            run(workflow)
        original = self.jobs.get(spectrum.result.id)
        outcome = self.registry.status(self.jobs, workflow.id)
        self.assertEqual([step["status"] for step in outcome["steps"]], ["done", "done"])
        self.assertEqual(original.extra["source_file"], dump.result.extra["output"])
        self.assertTrue(original.extra["agent_panel"])
        self.assertTrue(dump.result.extra["agent_panel"])
        gated = approve("ion_tof_gate", {"tof_job_id": original.id, "metadata": "/datasets/metadata.h5",
                                       "gmd_min_na": 2, "gmd_max_na": 4, "shift": 0})
        fitted = approve("ion_tof_fit", {"gate_job_id": gated.id, "region_ns": [3, 7], "peaks_ns": [[5, 1]]})
        with np.load(fitted.extra["peak_result"]) as result:
            self.assertEqual(int(result["selected_triggers"]), 1)
            self.assertIn("pixels_peaks", result)
        self.assertIn("peak_result", self.registry.status(self.jobs, fitted.id)["files"])
        manual = run(ion_tof.peak_fit_job(gated, ["pixels"], [3, 7], [[5, 1]], "Constant"))
        with np.load(fitted.extra["peak_result"]) as agent_result, np.load(manual.extra["peak_result"]) as manual_result:
            for key in agent_result.files:
                np.testing.assert_array_equal(agent_result[key], manual_result[key])

    def test_rootless_launch_arguments_keep_the_private_mounts_accessible(self):
        request = AgentRequest(self.account.name, self.workspace, self.root / ".agent",
                               "https://provider.example", "test", "test-key")
        config = self.root / ".agent" / "test"
        self.assertIn("--userns=keep-id", ContainerLauncher("podman").command(request, config, "test", rootless=True))
        self.assertNotIn("--user", ContainerLauncher("docker").command(request, config, "test", rootless=True))
