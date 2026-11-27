import json
from functools import partial
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd

from vmi_analysis.circularization import checkpoint, circularization
from vmi_analysis.centroiding.frame_roster import FAILED, PROCESSED, SKIPPED, write_frame_roster
from vmi_analysis.circularization.polar_fit import save_polar_comparison
from vmi_analysis.gui.panels import circularize
from vmi_analysis.gui.resources import command_access


class CircularizeCheckpointTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.source = self.root / "raw.h5"
        self.corrected = self.root / "corrected.h5"
        self.counts = self.root / "counts.npy"
        self.calibration = self.root / "extracted.txt"
        self.reassign = self.root / "reassign.txt"
        self.reassign.write_text("0 0\n3 2\n", encoding="utf-8")
        self.events = pd.DataFrame({
            "x": [2.75, 4.25, 2.75, 10, np.nan, 6.25, 2.75, 2.75],
            "y": [4.625, 2.75, 2.75, 10, 2, 2.75, 7, 7.75],
            "frame": [0, 0, 2, 2, 2, 2, 2, 2],
            "mass": [4, 7, 13, 9, 15, 8, 12, 7],
        }, index=[0, 1, 0, 1, 2, 3, 4, 5])
        self.events.to_hdf(self.source, key="localizations", format="table")
        with h5py.File(self.source, "r+") as store:
            write_frame_roster(store, [("camera.h5", index) for index in range(5)],
                               [PROCESSED, FAILED, PROCESSED, PROCESSED, SKIPPED])
        self.calibration.write_text(
            '# detector geometry: {"center": [5, 5], "shape": [12, 12]}\n'
            '"Parameter" "Value" "+-" "Uncertainty (1 std dev)"\n'
            '"Gaussian Centre" 3 +- 0.1\n'
            '"Trig Term sin(1*theta)" 0.75 +- 0.1\n', encoding="utf-8",
        )
        self.args = [
            "circularization", "--checkpoint", str(self.source), "--extracted", str(self.calibration),
            "--corrected-checkpoint", str(self.corrected), "--counts-output", str(self.counts),
            "--reassign-file", str(self.reassign), "--shape", "12", "12", "--center", "5", "5",
            "--polar-dr", "2", "--resolution-factor", "2",
        ]

    def run_cli(self):
        with patch.object(sys, "argv", self.args):
            circularization.main()

    def verify_histogram(self, store):
        counts = np.load(self.counts)
        rebuilt = np.histogram2d(store["events/y_corrected"][:], store["events/x_corrected"][:],
                                 bins=(store["geometry/y_edges"][:], store["geometry/x_edges"][:]))[0]
        self.assertEqual(counts.dtype, np.int64)
        np.testing.assert_array_equal(rebuilt, counts)

    def test_inverts_combined_map_past_last_ring_and_reassignment(self):
        preview = self.root / "polar.npz"
        self.args += ["--polar-preview", str(preview)]
        with patch("builtins.print"):
            self.run_cli()
        pd.testing.assert_frame_equal(pd.read_hdf(self.source, "localizations"), self.events)
        expected = np.zeros((12, 12), dtype=np.int64)
        expected[7, 5] = expected[5, 7] = expected[5, 5] = 1
        expected[5, 10] = expected[10, 5] = expected[11, 5] = 1
        histogram = np.load(self.counts)
        self.assertEqual(histogram.dtype, np.int64)
        np.testing.assert_array_equal(histogram, expected)
        raw = np.zeros_like(expected)
        raw[9, 5] = raw[5, 8] = raw[5, 5] = 1
        reference = self.root / "reference.npz"
        save_polar_comparison(reference, raw, expected, (5, 5))
        with np.load(preview) as actual_preview, np.load(reference) as expected_preview:
            for key in ("raw", "circularized", "dr"):
                np.testing.assert_array_equal(actual_preview[key], expected_preview[key])
        with h5py.File(self.corrected, "r") as store:
            self.assertEqual(set(store), {"events", "shots", "geometry"})
            self.assertEqual(store.attrs["schema_version"], 1)
            self.assertEqual(store.attrs["coordinate_correction"], "circularize")
            self.assertRegex(store.attrs["calibration_id"], r"^sha256:[a-f0-9]{64}$")
            events = store["events"]
            self.assertEqual(events["shot_id"].dtype, np.dtype("int64"))
            self.assertEqual(events["x_corrected"].dtype, np.dtype("float64"))
            self.assertEqual(events["y_corrected"].dtype, np.dtype("float64"))
            self.assertTrue(all(column.chunks for column in events.values()))
            np.testing.assert_array_equal(store["shots/id"][:], [0, 2, 3])
            np.testing.assert_array_equal(events["shot_id"][:], [0, 0, 2, 2, 2, 2])
            np.testing.assert_allclose(events["x_corrected"][:], [2.75, 3.75, 2.75, 5.25, 2.75, 2.75])
            np.testing.assert_allclose(events["y_corrected"][:], [3.75, 2.75, 2.75, 2.75, 5.25, 6])
            np.testing.assert_array_equal(store["geometry"].attrs["center_xy"], [2.75, 2.75])
            self.verify_histogram(store)
            metadata = json.loads(store.attrs["circularization"])
        self.assertEqual(metadata["event_totals"],
                         {"counted": 6, "invalid_input": 1, "outside_calibration": 0, "outside_histogram": 1})
        self.assertAlmostEqual(metadata["max_corrected_radius_grid_pixels"], np.hypot(6.5, 6.5))

    def test_rejects_polynomial_that_folds_before_detector_edge(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing radial map"):
            checkpoint.EventWarp({"A1": {"kind": "poly", "coeffs": [0.1, 0]}},
                                 1, {"kind": "identity"}, 1.0, (16, 16), (7, 7))

    def test_measured_export_uses_exact_edges_and_one_preview(self):
        edge = 1 / 3
        just_before = np.nextafter(edge, -np.inf)
        events = pd.DataFrame({"x": [0, just_before, edge, 2, np.nextafter(2., np.inf), np.inf],
                               "y": [0, edge, edge, 2, 2, edge], "frame": [2, 0, 2, 0, 2, 0]})
        events.to_hdf(self.source, key="localizations", format="table")
        preview = self.root / "measured.npz"
        self.args += ["--coordinate-correction", "none", "--shape", "6", "6", "--center", "2", "2",
                      "--resolution-factor", "3", "--polar-preview", str(preview)]
        self.calibration.unlink()
        self.reassign.unlink()
        with (patch("builtins.print"),
              patch.object(checkpoint, "correct_checkpoint", partial(checkpoint.correct_checkpoint, chunksize=2))):
            self.run_cli()
        with h5py.File(self.corrected, "r") as store:
            self.assertEqual(store.attrs["coordinate_correction"], "none")
            self.assertNotIn("calibration_id", store.attrs)
            np.testing.assert_array_equal(store["events/x_corrected"][:], events.x[:4])
            np.testing.assert_array_equal(store["events/y_corrected"][:], events.y[:4])
            np.testing.assert_array_equal(store["events/shot_id"][:], [2, 0, 2, 0])
            np.testing.assert_array_equal(store["shots/id"][:], [0, 2, 3])
            self.verify_histogram(store)
            self.assertEqual(json.loads(store.attrs["circularization"])["event_totals"],
                             {"counted": 4, "invalid_input": 1, "outside_calibration": 0, "outside_histogram": 1})
        with np.load(preview) as polar:
            self.assertEqual(set(polar.files), {"raw", "dr"})

    def test_rejects_missing_roster_and_event_from_failed_shot(self):
        with h5py.File(self.source, "r+") as store:
            del store["frame_roster"]
        with patch("builtins.print"), self.assertRaisesRegex(SystemExit, "valid shot roster"):
            self.run_cli()
        with h5py.File(self.source, "r+") as store:
            write_frame_roster(store, [("camera.h5", index) for index in range(3)],
                               [PROCESSED, PROCESSED, FAILED])
        with patch("builtins.print"), self.assertRaisesRegex(SystemExit, "unsuccessful shots"):
            self.run_cli()
        self.assertFalse(self.corrected.exists())
        self.assertFalse(self.counts.exists())

    def test_checkpoint_publication_failure_restores_existing_histogram(self):
        np.save(self.counts, np.array([[42]]))
        replace = Path.replace

        def fail_checkpoint(path, destination):
            if destination == self.corrected:
                raise PermissionError("Checkpoint is locked")
            return replace(path, destination)

        with patch.object(Path, "replace", fail_checkpoint), patch("builtins.print"):
            with self.assertRaisesRegex(SystemExit, "Checkpoint is locked"):
                self.run_cli()
        np.testing.assert_array_equal(np.load(self.counts), [[42]])
        self.assertFalse(self.corrected.exists())

    def test_panel_builder_and_agent_use_same_checkpoint_reservations(self):
        settings = dict(shape_y=12, shape_x=12, center_x=5, center_y=5, polar_dr=2, resolution_factor=2)
        job = circularize.circularize_job(str(self.calibration), inp=str(self.source),
                                         corrected_checkpoint=str(self.corrected), counts_output=str(self.counts),
                                         reassign=str(self.reassign), **settings)
        agent_job, summary = circularize._prepare_agent(None, dict(
            input=str(self.source), extracted=str(self.calibration), reassign_file=str(self.reassign),
            corrected_checkpoint=str(self.corrected), counts_output=str(self.counts), **settings))
        self.assertEqual(job.cmd[:-2], agent_job.cmd[:-2])
        self.assertEqual(agent_job.cmd[-2:], ["--polar-preview", agent_job.extra["polar_preview"]])
        self.assertEqual(summary["Integer histogram"], str(self.counts))
        access = command_access(job.cmd, job.cwd)
        self.assertEqual(access.reads, {self.calibration, self.reassign, self.source})
        self.assertEqual(access.writes, {self.corrected, self.counts, Path(job.extra["polar_preview"])})
