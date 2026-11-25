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
from vmi_analysis.centroiding.plot_data import save_plot_data
from vmi_analysis.gui import shared
from vmi_analysis.gui.agent_completion import path_completions
from vmi_analysis.gui.panel_operations import build_registry
from vmi_analysis.gui.jobs import Job, JobManager, ResourceBusyError
from vmi_analysis.ion_acquisition import SCHEMA
from vmi_analysis.gui.resources import job_access
from vmi_analysis.gui.panels import accumulate, agent, binning, calibration, centroid, circularize, dark, gmd_map, ion_live, ion_luna, ion_momenta, ion_pipeline, ion_tof, movie, pes
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
        # These tests exercise panel workflows on Windows and Linux. The real
        # Linux filesystem boundary is covered by test_shared_job_sandbox.
        self.enterContext(patch("vmi_analysis.gui.jobs.panel_command",
                                side_effect=lambda command, *_: command))
        self.jobs = JobManager()
        self.registry = build_registry()

    def test_movie_proposal_streams_without_writes_and_reserves_both_export_formats(self):
        raw = self.datasets / "camera"
        raw.mkdir()
        params = {"h5_dir": "/datasets/camera", "n_frames": 3}
        preview, _ = self.registry.prepare(self.jobs, "movie", {**params, "save_movie": False})
        self.assertIn("--preview-stream", preview.cmd)
        self.assertIn(raw.resolve(), job_access(preview).reads)
        self.assertFalse(job_access(preview).writes)
        self.assertTrue(movie._consume_preview_frame(preview, "[preview-png] 3 3 12 8 data"))
        self.assertEqual(preview.extra["preview_frame"]["index"], 3)

        exported, _ = self.registry.prepare(self.jobs, "movie", {**params, "movie_dir": "review"})
        folder = self.workspace / "review"
        self.assertTrue({folder / "camera.mp4", folder / "camera.gif"} <= job_access(exported).writes)
        self.assertIn(Path(exported.extra["frames_dir"]), job_access(exported).writes)
        movie._consume_preview_frame(exported, f"Movie written: {raw / 'camera.gif'}")
        self.assertNotIn("movie_file", exported.extra)
        movie._consume_preview_frame(exported, f"Movie written: {folder / 'camera.gif'}")
        self.assertEqual(Path(exported.extra["movie_file"]), folder / "camera.gif")
        with self.assertRaises(shared.WorkspaceDenied):
            self.registry.prepare(self.jobs, "movie", {**params, "movie_dir": "/datasets/camera"})

    def test_live_preview_is_inspection_only_and_summarizes_recent_finite_readings(self):
        from vmi_analysis.ion_live import LiveFrame, LivePreview

        self.assertEqual(self.registry.describe("live_tpx3")["action"], "inspect")
        with self.assertRaises(ValueError):
            self.registry.prepare(self.jobs, "live_tpx3", {})
        receiver = LivePreview("tcp://127.0.0.1:5556", np.array([0, 10, 20]))
        self.assertEqual(ion_live._preview_summary(receiver)["status"], "waiting for frames")
        receiver.add(LiveFrame(0, 999, 999, np.array([258, 258]), np.array([12]), 1))
        now = (ion_live.GMD_TIMELINE_SECONDS + 1) * 10**9
        receiver.add(LiveFrame(now, np.nan, 5, np.array([512]), np.array([]), 0))
        snapshot = ion_live._preview_summary(receiver)
        self.assertEqual(snapshot["detector"]["busiest_pixel"], {"x_px": 2, "y_px": 1, "hits": 2})
        self.assertEqual(snapshot["frames_without_exactly_one_trigger"], 1)
        self.assertEqual(snapshot["tof"]["busiest_bins"], [{"minimum_ns": 10, "maximum_ns": 20, "hits": 1}])
        self.assertIsNone(snapshot["gmd"]["GMD1"]["latest_na"])
        self.assertEqual(snapshot["gmd"]["GMD2"]["mean_na"], 5)

    def test_run_measurement_reads_owned_histogram_and_saved_gaussian_fit(self):
        histogram = self.workspace / "histogram.npz"
        np.savez(histogram, edges=np.array([10490, 10495, 10500, 10505, 10510]),
                 pixels=np.array([1, 3, 8, 2]), clusters=np.array([0, 2, 3, 0]), selected_triggers=12)
        fit_data = self.workspace / "fit.npz"
        row = {"center": 10501.25, "fwhm": 12.3, "bound": False, "counts": 28, "yield_per_trigger": 28 / 12}
        np.savez(fit_data, pixels_peaks=json.dumps([row]))
        spectrum = Job(ion_tof.FILTER_JOB_NAME, status="done", owner=self.account.name,
                       extra={"histogram": str(histogram)})
        fit = Job(ion_tof.FIT_JOB_NAME, status="done", owner=self.account.name, extra={"peak_result": str(fit_data)})
        params = {"region_ns": [10498, 10506]}
        with self.jobs._lock:
            self.jobs._jobs.update({spectrum.id: spectrum, fit.id: fit})
        with self.assertRaisesRegex(ValueError, "available TOF trace"):
            self.registry.measure(self.jobs, spectrum.id, params)
        result = self.registry.measure(self.jobs, spectrum.id, params | {"source": "pixels"})
        self.assertEqual(result["strongest_bin"], {
            "center_ns": 10502.5, "edges_ns": [10500, 10505], "counts": 8, "bins_tied_for_maximum": 1,
        })
        self.assertEqual(result["selected_triggers"], 12)
        self.assertEqual(self.registry.measure(self.jobs, fit.id, params)["peaks"], [row])
        with patch.object(shared, "current_account", return_value=SimpleNamespace(name="other")):
            with self.assertRaisesRegex(ValueError, "in your workspace"):
                self.registry.measure(self.jobs, spectrum.id, params)

    def test_pes_measurement_reads_owned_uniform_bins_and_reports_coverage(self):
        path = self.workspace / "pes.npz"
        x = np.array([1., 4., 9.])
        save_plot_data(path, {"alpha": 0.5, "beta_order": 4,
                              "images": [{"path": "first.npy"}, {"path": "second.npy"}]},
                       x_0=x, y_0=[2., 4., 9.], beta_x_0=x,
                       moments_0=[[1., 2., 7.], [0., 0., 10.], [1., 1., np.nan]],
                       x_1=x[1:], y_1=[8., 8.], beta_x_1=x[1:],
                       moments_1=[[0., 0.], [1., 1.], [1., 1.]])
        job = Job(pes.JOB_NAME, status="done", owner=self.account.name, extra={"pes_spectra": str(path)})
        with self.jobs._lock:
            self.jobs._jobs[job.id] = job
        self.assertEqual(self.registry.measurement_context(job)["operation"], "pes")
        self.assertIn("bin_width", self.registry.describe("pes")["measurements"])
        params = {"coordinate": 4, "bin_width": 3, "unit": "eV"}
        with self.assertRaisesRegex(ValueError, "2. second.npy"):
            self.registry.measure(self.jobs, job.id, params)
        params["spectrum"] = 1
        measured = self.registry.measure(self.jobs, job.id, params)
        self.assertEqual(measured["spectrum"], "1. first.npy")
        self.assertEqual(measured["unit"], "eV")
        self.assertEqual(measured["bin_edges"], [4., 7.])
        self.assertEqual(measured["bin_center"], 5.5)
        self.assertEqual(measured["integrated_yield"], 16.5)
        self.assertEqual(measured["bin_average_intensity"], 5.5)
        self.assertAlmostEqual(measured["beta"]["beta_2"], 6 / 7)
        self.assertIsNone(measured["beta"]["beta_4"])

        partial = self.registry.measure(self.jobs, job.id, params | {"coordinate": 10})
        self.assertEqual(partial["bin_edges"], [7., 10.])
        self.assertEqual(partial["measured_interval"], [7., 9.])
        self.assertAlmostEqual(partial["bin_average_intensity"], 16 / 3)
        uncovered = self.registry.measure(self.jobs, job.id, params | {"coordinate": 2, "spectrum": 2})
        self.assertIsNone(uncovered["measured_interval"])
        self.assertIsNone(uncovered["bin_average_intensity"])
        second = self.registry.measure(self.jobs, job.id, params | {"spectrum": 2})
        self.assertIsNone(second["beta"]["beta_2"])
        for invalid in ({"coordinate": 11}, {"bin_width": 0}, {"bin_width": 0.0001}, {"unit": "pixel²"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.registry.measure(self.jobs, job.id, params | invalid)
        with patch.object(shared, "current_account", return_value=SimpleNamespace(name="other")):
            with self.assertRaisesRegex(ValueError, "in your workspace"):
                self.registry.measure(self.jobs, job.id, params)

    def test_pes_measurement_defaults_to_the_panel_grid_and_uses_saved_units(self):
        path = self.workspace / "uncalibrated.npz"
        save_plot_data(path, {"alpha": None, "images": [{"path": "only.npy"}]}, x_0=[1., 9.], y_0=[2., 2.])
        job = Job(pes.JOB_NAME, status="done", owner=self.account.name, extra={"pes_spectra": str(path)})
        with self.jobs._lock:
            self.jobs._jobs[job.id] = job
        measured = self.registry.measure(self.jobs, job.id, {"coordinate": 1})
        self.assertEqual(measured["unit"], "pixel²")
        self.assertEqual(measured["bin_width"], 0.04)
        self.assertEqual(measured["bin_edges"], [1., 1.04])
        self.assertAlmostEqual(measured["bin_average_intensity"], 2.)
        self.assertEqual(measured["beta"], {})

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

    def test_gmd_map_proposal_uses_the_manual_builder_and_preserves_reservations(self):
        camera = self.datasets / "camera"
        camera.mkdir()
        gmd = self.datasets / "current-time-both.txt"
        gmd.touch()
        settings = {"h5_dir": "/datasets/camera", "gmd_path": "/datasets/current-time-both.txt",
                    "h5_pattern": "*RAW-*.h5", "n_files": 2}
        proposed, summary = self.registry.prepare(self.jobs, "gmd_map", settings)
        manual = gmd_map.gmd_map_job(camera, gmd, h5_pattern="*RAW-*.h5", n_files=2)
        output = self.workspace / "gmd_map_all.txt"
        self.assertEqual(proposed.cmd, manual.cmd)
        self.assertEqual(summary["Output path output"], str(output))
        access = job_access(proposed)
        self.assertEqual(access.reads, {camera, gmd})
        self.assertEqual(access.writes, {output})
        self.assertEqual(access.gmd_map_output, output)
        self.assertEqual(self.jobs.jobs(), [])
        for invalid, message in (
            ({"h5_pattern": "../*.h5"}, "filename pattern"),
            ({"n_files": 2.5}, "whole file count"),
            ({"output_path": "/datasets/map.txt"}, "read-only"),
        ):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, message):
                self.registry.prepare(self.jobs, "gmd_map", settings | invalid)

    def test_accumulate_proposal_uses_the_manual_builder_and_reserves_artifacts(self):
        camera = self.datasets / "camera"
        camera.mkdir()
        dark_frame = self.workspace / "dark.npy"
        dark_frame.touch()
        gmd = self.workspace / "gmd.txt"
        gmd.touch()
        settings = {
            "h5_dir": "/datasets/camera", "output_path": "/workspace/sum.txt",
            "dark_path": "/workspace/dark.npy", "enable_gmd": True, "gmd_map_path": "/workspace/gmd.txt",
            "frame_limit": 3, "simple_sum": True, "max_workers": 2,
        }
        proposed, _ = self.registry.prepare(self.jobs, "accumulate", settings)
        output = self.workspace / "sum.txt"
        manual = accumulate.accumulate_job(camera, output, dark_path=dark_frame, enable_gmd=True,
                                           gmd_map_path=gmd, frame_limit=3, simple_sum=True, max_workers=2)
        self.assertEqual(proposed.extra["panel_form"], manual.extra["panel_form"])
        self.assertEqual(proposed.extra["plot_png"], str(output.with_suffix(".png")))
        self.assertIn("--require-new-output", proposed.cmd)
        self.assertIn("--simple-sum", proposed.cmd)
        self.assertNotIn("--im-threshold", proposed.cmd)
        access = job_access(proposed)
        self.assertEqual(access.reads, {camera, dark_frame, gmd})
        self.assertEqual(access.writes, {output, output.with_suffix(".png"), Path(proposed.extra["live_path"])})
        self.assertEqual(self.jobs.jobs(), [])
        for change, message in (
            ({"enable_gmd": False, "output_path": "/workspace/gmd.png"}, "text file"),
            ({"frame_limit": 1.5}, "whole number"),
            ({"dark_path": "/workspace/missing.npy"}, "existing dark frame"),
            ({"gmd_w_min": 8, "gmd_w_max": 2}, "in increasing order"),
            ({"output_path": "/datasets/other.txt"}, "read-only"),
        ):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                self.registry.prepare(self.jobs, "accumulate", settings | change)
        output.with_suffix(".png").touch()
        with self.assertRaisesRegex(ValueError, "PNG already exists"):
            self.registry.prepare(self.jobs, "accumulate", settings)
        reused, _ = self.registry.prepare(self.jobs, "accumulate", settings | {"reuse_existing": True})
        self.assertNotIn("--require-new-output", reused.cmd)

    def test_centroiding_approval_uses_manual_snapshot_and_mode_specific_artifacts(self):
        camera = self.datasets / "camera"
        camera.mkdir()
        output = self.workspace / "run"
        output.mkdir()
        params = {"h5_dir": "/datasets/camera", "output_dir": "/workspace/run",
                  "frame_limit": 3, "enable_parallel": False, "enable_abel": False,
                  "min_net_gradient": 30, "fitting_method": "lq"}
        proposed, summary = self.registry.prepare(self.jobs, "centroiding", params)
        self.assertEqual(summary["Mode"], "Batch")
        self.assertEqual(proposed.extra["panel_form"]["processing.frame_limit"], 3)
        self.assertIn("--require-new-checkpoint", proposed.cmd)
        self.assertEqual(job_access(proposed).centroid_output_dir, output)
        self.assertEqual(self.jobs.jobs(), [])
        for invalid, message in (
            ({"checkpoint_file": "../escape.h5"}, "inside the output"),
            ({"enable_dark_subtraction": True}, "precomputed dark"),
            ({"enable_gmd_weighting": True}, "GMD map"),
            ({"frame_limit": 2.5}, "valid numeric value"),
            ({"output_dir": "/datasets/results"}, "read-only"),
        ):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, message):
                self.registry.prepare(self.jobs, "centroiding", params | invalid)

        for name in ("checkpoint", "raw_image", "summary_plot", "summary_data", "abel_image", "spectrum_csv"):
            Path(proposed.extra[name]).touch()
        proposed.owner = self.account.name
        proposed.status = "done"
        self.jobs._jobs[proposed.id] = proposed
        self.assertEqual(tuple(self.registry.status(self.jobs, proposed.id)["files"]),
                         ("checkpoint", "raw_image", "summary_plot", "summary_data"))
        with self.assertRaisesRegex(ValueError, "checkpoint already exists"):
            self.registry.prepare(self.jobs, "centroiding", params)
        reused, _ = self.registry.prepare(self.jobs, "centroiding", params | {"reuse_checkpoint": True})
        self.assertEqual(reused.cmd, proposed.cmd[:-1])

        tuning, summary = self.registry.prepare(self.jobs, "centroiding", params | {"tuning_only": True})
        for name in ("tuning_plot", "tuning_data"):
            Path(tuning.extra[name]).touch()
        tuning.owner = self.account.name
        tuning.status = "done"
        self.jobs._jobs[tuning.id] = tuning
        self.assertEqual(summary["Mode"], "Single-frame tuning")
        self.assertEqual(tuple(self.registry.status(self.jobs, tuning.id)["files"]), ("tuning_plot", "tuning_data"))
        self.assertNotIn("--require-new-checkpoint", tuning.cmd)

    def test_binning_resolves_same_owner_handoffs_and_lists_only_current_bin_files(self):
        checkpoint = self.workspace / "detections.h5"
        gmd = self.datasets / "gmd.txt"
        for path in (checkpoint, gmd):
            path.touch()
        parent = Job("Centroiding", status="done", owner=self.account.name,
                     extra={"checkpoint": str(checkpoint)})
        mapping = Job("GMD map", status="done", owner=self.account.name, extra={"output_path": str(gmd)})
        self.jobs._jobs.update({parent.id: parent, mapping.id: mapping})
        params = {"centroid_job_id": parent.id, "gmd_map_job_id": mapping.id,
                  "output_dir": "/workspace/bins", "thresholds": [1e-8, 2e-8, 3e-8, 4e-8],
                  "weight_mode": "gmd", "gmd_col": 5}
        proposed, _ = self.registry.prepare(self.jobs, "binning", params)
        target = self.workspace / "bins"
        self.assertFalse(target.exists())
        manual = binning.binning_job(checkpoint, gmd, target, thresholds="1e-08,2e-08,3e-08,4e-08",
                                     weight_mode="gmd", gmd_col=5)
        self.assertEqual(proposed.cmd, manual.cmd)
        self.assertEqual(job_access(proposed).reads, {checkpoint, gmd})
        self.assertIn(target, job_access(proposed).writes)
        for overrides in ({"thresholds": [2, 1]}, {"thresholds": [1, 1]}, {"thresholds": "1,nan"}):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, "strictly increasing"):
                self.registry.prepare(self.jobs, "binning", params | overrides)

        target.mkdir()
        for artifact in proposed.extra["binning_artifacts"]:
            Path(proposed.extra[artifact]).touch()
        (target / "vmi_2d_gmd_bin0_minusinf_9.99e-09_w-gmd.npy").touch()
        proposed.owner = self.account.name
        proposed.status = "done"
        self.jobs._jobs[proposed.id] = proposed
        files = self.registry.status(self.jobs, proposed.id)["files"]
        self.assertEqual(len(files), 7)
        self.assertEqual(files["bin_0"], "/workspace/bins/vmi_2d_gmd_bin0_minusinf_1.00e-08_w-gmd.npy")
        mapping.owner = "another-user"
        with self.assertRaisesRegex(ValueError, "your workspace"):
            self.registry.prepare(self.jobs, "binning", params)

    def test_trig_fit_requires_a_panel_view_and_finite_band_limits(self):
        operation = calibration.FIT_OPERATION
        params = {"band": [4, 10, 0, 360], "n_terms": 3, "use_beta": True, "iterative": False}
        self.assertEqual(self.registry.describe(operation)["panel"], "Trig Calibration")
        with self.assertRaisesRegex(ValueError, "Ask Agent"):
            self.registry.prepare(self.jobs, operation, params)
        self.assertEqual(calibration._fit_parameters(params, max_radius=15), params)
        with self.assertRaisesRegex(ValueError, "inside the loaded"):
            calibration._fit_parameters(params | {"band": [4, 16, 0, 360]}, max_radius=15)
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            calibration._fit_parameters(params | {"band": [4, float("nan"), 0, 360]}, max_radius=15)

    def test_circularize_resolves_handoff_and_reserves_outputs_by_mode(self):
        image = self.workspace / "raw.npy"
        calibration = self.workspace / "extracted.txt"
        reassign = self.datasets / "radial.txt"
        for path in (image, calibration, reassign):
            path.touch()
        source = Job("GMD binning", status="done", owner=self.account.name,
                     extra={"bin_2": str(image)})
        self.jobs._jobs[source.id] = source
        params = {"source_job_id": source.id, "bin_index": 2, "output": "/workspace/corrected.txt",
                  "no_plots": True, "reassign_file": "/datasets/radial.txt"}
        proposed, summary = self.registry.prepare(self.jobs, "circularize", params)
        manual = circularize.circularize_job(calibration, inp=image, out=self.workspace / "corrected.txt",
                                             reassign=reassign, no_plots=True)
        self.assertEqual(proposed.extra["panel_form"], manual.extra["panel_form"])
        self.assertEqual(summary["Center"], "Brightest pixel")
        access = job_access(proposed)
        self.assertEqual(access.reads, {image, calibration, reassign})
        self.assertIn(self.workspace / "circularization/trig_fit_plots/reassignment_map.png", access.writes)
        self.assertNotIn(self.workspace / "circularization/trig_fit_plots", access.writes)
        self.assertNotIn("--center", proposed.cmd)
        self.assertNotIn("--shape", proposed.cmd)
        with self.assertRaisesRegex(ValueError, "points to the input image"):
            self.registry.prepare(self.jobs, "circularize", params | {"output": "/workspace/raw.npy"})
        with self.assertRaisesRegex(ValueError, "read-only"):
            self.registry.prepare(self.jobs, "circularize", params | {"output": "/datasets/corrected.txt"})

        preview, _ = self.registry.prepare(self.jobs, "circularize", {
            "extracted": "/workspace/extracted.txt", "summary_only": True,
        })
        self.assertEqual(job_access(preview).reads, {calibration})
        self.assertFalse(job_access(preview).writes)
        self.assertNotIn("output", preview.extra)
        source.owner = "other"
        with self.assertRaisesRegex(ValueError, "your workspace"):
            self.registry.prepare(self.jobs, "circularize", params)

    def test_pes_agent_generation_and_contextual_fit_use_resolved_spectra(self):
        first = self.workspace / "vmi.npy"
        second = self.datasets / "other.txt"
        first.touch()
        second.touch()
        source = Job("Circularize", status="done", owner=self.account.name, extra={"output": str(second)})
        self.jobs._jobs[source.id] = source
        params = {"images": ["/workspace/vmi.npy"], "source_job_id": source.id,
                  "geometry": {"center": [4, 5], "angle": 90},
                  "image_geometry": {"/datasets/other.txt": {"center": None, "resolution": 2}},
                  "alpha": .002, "beta_order": 4}
        proposed, _ = self.registry.prepare(self.jobs, "pes", params)
        override = {"center": None, "angle": 90, "resolution": 2}
        key = str(pes.canonical_path(second))
        manual = pes.spectra_job([first, second], pes.Geometry((4, 5), 90), {key: override},
                                 alpha=.002, beta_order=4)
        self.assertEqual(proposed.cmd[:-1], manual.cmd[:-1])
        self.assertEqual(job_access(proposed).reads, {first, second})
        self.assertEqual(json.loads(proposed.extra["panel_form"]["image_geometry"]), {key: override})
        with self.assertRaisesRegex(ValueError, "symmetry axis"):
            self.registry.prepare(self.jobs, "pes", params | {"quadrants": ["ur"]})
        source.owner = "other"
        with self.assertRaisesRegex(ValueError, "your workspace"):
            self.registry.prepare(self.jobs, "pes", params)

        settings = {"targets": ["first"], "region": [1, 3], "baseline": "None", "peaks": [],
                    "auto_profile": "Gaussian"}
        with self.assertRaisesRegex(ValueError, "Ask Agent"):
            self.registry.prepare(self.jobs, pes.FIT_OPERATION, settings)
        self.assertEqual(pes._fit_settings(settings, {"first": None}), settings)
        with self.assertRaisesRegex(ValueError, "displayed spectra"):
            pes._fit_settings(settings | {"targets": ["missing"]}, {"first": None})

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

    def test_ion_momenta_reuses_dump_input_and_manual_builder(self):
        collection = self.workspace / "dump.segments.json"
        source = self.workspace / "hits.h5"
        metadata = self.workspace / "metadata.h5"
        for path in (source, metadata):
            path.touch()
        collection.write_text(json.dumps({"schema": SCHEMA, "segments": [
            {"data": source.name, "metadata": metadata.name},
        ]}), encoding="utf-8")
        dump = Job(ion_luna.JOB_NAME, status="done", owner=self.account.name,
                   extra={"collection": str(collection), "output": str(self.workspace / "dump.h5")})
        self.jobs._jobs[dump.id] = dump
        params = {"dump_job_id": dump.id, "source": "both", "bin_size_ns": 5,
                  "minimum_ns": 8900, "maximum_ns": 11000}
        agent_job, summary = self.registry.prepare(self.jobs, "ion_momenta", params)
        manual = ion_momenta.positions_job(collection, source="both", width=5, low=8900, high=11000)
        self.assertEqual(agent_job.cmd[:6] + agent_job.cmd[7:], manual.cmd[:6] + manual.cmd[7:])
        self.assertEqual(agent_job.extra["panel_form"], manual.extra["panel_form"])
        self.assertEqual(summary["Count"], "Both")
        self.assertIn(collection, job_access(agent_job).reads)
        self.assertIn(source, job_access(agent_job).reads)
        self.assertIn(Path(agent_job.extra["histogram"]), job_access(agent_job).writes)
        for invalid in ({"input": "/workspace/dump.segments.json"}, {"bin_size_ns": 0},
                        {"maximum_ns": 8900}, {"source": "unknown"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.registry.prepare(self.jobs, "ion_momenta", params | invalid)
        with patch.object(shared, "current_account", return_value=SimpleNamespace(name="other")):
            with self.assertRaisesRegex(ValueError, "from your workspace"):
                self.registry.prepare(self.jobs, "ion_momenta", params)

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

    def test_approval_details_render_paths_and_nested_settings_as_plain_text(self):
        client = Client(ui.page("/_test_agent_approval_details"))
        self.addCleanup(client.delete)
        image = r"E:\data\scientist's <sample>.txt"
        with client:
            agent._render_review_details({"VMI images": [image, "second.txt"],
                                          "Shared geometry": {"center": (476, 473), "rotation (degrees)": 90},
                                          "Quadrants": ["ur", "ul", "ll", "lr"]})
        labels = [element.text for element in client.elements.values() if isinstance(element, ui.label)]
        self.assertEqual(labels, ["VMI images", image, "second.txt", "Shared geometry", "center",
                                  "476, 473", "rotation (degrees)", "90", "Quadrants", "ur, ul, ll, lr"])

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
