"""GMD map CLI: configurable input pattern and clear failure on no input."""

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from vmi_analysis.gmd import build_gmd_map


class GmdMapPatternTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.h5_dir = self.root / "h5"
        self.h5_dir.mkdir()
        for name in ("ODL-150.000-RAW-R0144-150Xe-S00000.h5",
                     "ODL-150.000-RAW-R0144-150Xe-S00001.h5"):
            with h5py.File(self.h5_dir / name, "w") as handle:
                meta = np.zeros((3, 5), dtype=np.uint64)
                meta[:, 3] = np.array([1000, 1001, 1002], dtype=np.uint64)
                handle.create_dataset("raw/meta", data=meta)
        self.gmd_path = self.root / "current-time-both.txt"
        np.savetxt(self.gmd_path, np.array([
            [0, 1000, 1.0, 2.0],
            [1, 1001, 3.0, 4.0],
            [2, 1002, 5.0, 6.0],
        ]), fmt="%.6f")
        self.output = self.root / "map.txt"

    def test_default_pattern_rejects_prefixed_names(self):
        with self.assertRaises(SystemExit) as raised:
            build_gmd_map.main(str(self.h5_dir), str(self.gmd_path),
                               str(self.output), 0)
        self.assertIn("No HDF5 files matching", str(raised.exception))
        self.assertFalse(self.output.exists())

    def test_custom_pattern_matches_prefixed_names(self):
        build_gmd_map.main(str(self.h5_dir), str(self.gmd_path),
                           str(self.output), 0, "*RAW-*.h5")
        lines = self.output.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2 + 6)
        self.assertIn("ODL-150.000-RAW-R0144-150Xe-S00000.h5", lines[2])

    def test_matched_files_without_frames_fail_before_writing(self):
        with h5py.File(self.h5_dir / "RAW-empty.h5", "w") as handle:
            handle.create_dataset("raw/meta", data=np.zeros((0, 5), dtype=np.uint64))
        with self.assertRaises(SystemExit) as raised:
            build_gmd_map.main(str(self.h5_dir), str(self.gmd_path),
                               str(self.output), 0)
        self.assertIn("No frames found", str(raised.exception))
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
