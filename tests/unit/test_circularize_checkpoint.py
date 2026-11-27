import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd

from vmi_analysis.circularization import circularization
from vmi_analysis.circularization.checkpoint import STATUS
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
            "x": [2.75, 4.25, 2.75, 10, np.nan],
            "y": [4.625, 2.75, 2.75, 10, 2],
            "frame": [0, 0, 2, 2, 2],
            "mass": [4, 7, 13, 9, 15],
        }, index=[0, 1, 0, 1, 2])
        self.frames = pd.DataFrame({"frame": [0, 1, 2, 3]})
        self.events.to_hdf(self.source, key="localizations", format="table")
        self.frames.to_hdf(self.source, key="frames", format="table")
        with h5py.File(self.source, "r+") as store:
            store.attrs["frame_limit"] = 4
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

    def test_inverts_combined_map_and_preserves_events_and_frame_table(self):
        with patch("builtins.print"):
            self.run_cli()
        actual = pd.read_hdf(self.corrected, "localizations")
        pd.testing.assert_frame_equal(actual[list(self.events)], self.events)
        pd.testing.assert_frame_equal(pd.read_hdf(self.corrected, "frames"), self.frames)
        np.testing.assert_allclose(actual.x_corrected, [2.75, 3.75, 2.75, np.nan, np.nan])
        np.testing.assert_allclose(actual.y_corrected, [3.75, 2.75, 2.75, np.nan, np.nan])
        np.testing.assert_array_equal(actual.correction_status, [
            STATUS["counted"], STATUS["counted"], STATUS["counted"],
            STATUS["outside_calibration"], STATUS["invalid_input"],
        ])
        expected = np.zeros((12, 12), dtype=np.int64)
        expected[7, 5] = expected[5, 7] = expected[5, 5] = 1
        histogram = np.load(self.counts)
        self.assertEqual(histogram.dtype, np.int64)
        np.testing.assert_array_equal(histogram, expected)
        with h5py.File(self.corrected, "r") as store:
            self.assertEqual(store.attrs["frame_limit"], 4)
            metadata = json.loads(store.attrs["circularization"])
        self.assertEqual(metadata["event_totals"]["counted"], 3)
        self.assertEqual(metadata["max_corrected_radius"], 4.0)

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
        self.assertEqual(job.cmd, agent_job.cmd)
        self.assertEqual(summary["Integer histogram"], str(self.counts))
        access = command_access(job.cmd, job.cwd)
        self.assertEqual(access.reads, {self.calibration, self.reassign, self.source})
        self.assertEqual(access.writes, {self.corrected, self.counts})
