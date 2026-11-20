import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from vmi_analysis import tpx3_analysis as reference
from vmi_analysis.ion_verification import analyze, read_dump, unpack
from vmi_analysis.ion_pipeline import DEFAULT_PEAKS, run
from vmi_analysis.tof_bins import DEFAULT_BIN_SIZE_NS


class IonVerificationTests(unittest.TestCase):
    def test_dump_replays_standalone_yield_on_timestamp_ordered_frames(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "150_ion"
            source.mkdir()
            generator = np.random.default_rng(27)
            for index in range(12):
                hits = np.r_[generator.normal(10545, 85, 140 + index),
                             generator.normal(9155, 70, 100 + 2 * index)]
                tdc = (0x6F << 56) | (1000 << 12) | (1 << 5)
                pixels = [(0xB << 60) | (round((25_000 + tof) / 25) << 30) | (15 << 16) for tof in hits]
                # Reversed times distinguish timestamp order from filename order.
                (source / f"frame_{index:02}.tpx3").write_bytes(
                    struct.pack("<qqdd", 100, 1000 * (12 - index), 3e-9, (1 + index % 8) * 1e-9)
                    + struct.pack(f"<{len(pixels) + 1}Q", tdc, *pixels)
                )
            position, = reference.scan_dataset(root)
            path = root / "dump.h5"
            unpack(source, path)
            replay, decoded, dump_id = read_dump(path)
            self.assertEqual([r.name for r in replay.records], [r.name for r in position.records])
            for expected, actual in zip(reference.ensure_tof(position), decoded, strict=True):
                np.testing.assert_array_equal(expected, actual)
            for tracked in (True, False):
                for window in ((2, 6), None):
                    expected = reference.compute_yield(position, window, -1, tracked)
                    self.assertTrue(expected["valid"])
                    data = analyze(path, gmd_window=window, shift=-1, tracked=tracked, expected_id=dump_id)
                    actual = json.loads(data["yield_summary"])
                    self.assertEqual(int(data["selected_frames"]), expected["n_window"])
                    for column in ("xe1", "xe2", "y1", "y2", "ratio", "ratio_err", "n_window"):
                        self.assertEqual(actual[column], expected[column])
                    for column in ("win1", "win2"):
                        self.assertEqual(actual[column], list(expected[column]))
            with self.assertRaisesRegex(ValueError, "input was replaced"):
                analyze(path, expected_id="old-dump")
            default = analyze(path)
            same_width = analyze(path, bin_size_ns=DEFAULT_BIN_SIZE_NS)
            np.testing.assert_array_equal(same_width["edges"], default["edges"])
            custom = analyze(path, bin_size_ns=17.5)
            np.testing.assert_allclose(np.diff(custom["edges"])[:-1], 17.5)
            self.assertEqual(custom["edges"][-1], reference.TOF_PLOT_RANGE[1])
            self.assertLess(np.diff(custom["edges"])[-1], 17.5)
            self.assertEqual(json.loads(custom["yield_summary"]), json.loads(default["yield_summary"]))
            pipeline_rows = run(root, root / "pipeline-output", {
                "peaks": DEFAULT_PEAKS, "fit_bins": 60, "window_sigma": 3,
                "gate_gmd": True, "gmd_min": 2, "gmd_max": 6, "gmd_shift": -1,
                "backend": "fallback", "unwrap": "signed", "keep_dumps": False, "fit_tof": "ungated",
            })
            expected = reference.compute_yield(position, (2, 6), shift=1)
            self.assertEqual([row["status"] for row in pipeline_rows], ["OK", "OK"])
            self.assertEqual([row["counts"] for row in pipeline_rows], [expected["xe1"], expected["xe2"]])
            self.assertEqual([row["hits_per_frame"] for row in pipeline_rows], [expected["y1"], expected["y2"]])
            self.assertEqual([row["selected_frames"] for row in pipeline_rows], [expected["n_window"]] * 2)
