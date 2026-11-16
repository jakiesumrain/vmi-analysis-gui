"""Binning preview helpers: raw GMD values and unclamped histogram weights."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from vmi_analysis.gui.gmd_histograms import (gmd_histogram_samples, gmd_weights,
                                             load_gmd_map)


class GmdHistogramTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.map_path = Path(self.tmp.name) / "gmd_map_all.txt"
        np.savetxt(self.map_path, np.array([
            [0, 0, 0, 0, 1.0e-8, 5.0e-9, 0.0],
            [1, 0, 0, 1, 2.0e-8, 1.0e-8, 0.0],
            [2, 0, 0, 2, 4.0e-8, 2.0e-8, 0.0],
            [3, 0, 0, 3, -1.0e-8, -5.0e-9, 0.0],
        ]), delimiter=" ", header="frame h5 frame_in_file gmd_idx gmd_val1 gmd_val2 dt")

    def test_values_are_raw_and_column_is_selectable(self):
        _, first = load_gmd_map(self.map_path, 4)
        _, second = load_gmd_map(self.map_path, 5)
        np.testing.assert_allclose(first, [1e-8, 2e-8, 4e-8, -1e-8])
        np.testing.assert_allclose(second, [5e-9, 1e-8, 2e-8, -5e-9])

    def test_weight_modes_have_no_floor_or_limits(self):
        values = np.array([1e-8, 2e-8, 3e-8])
        np.testing.assert_allclose(gmd_weights(values, "none"), [1, 1, 1])
        np.testing.assert_allclose(gmd_weights(values, "gmd"), values)
        np.testing.assert_allclose(gmd_weights(values, "gmd_norm", 1.0),
                                   [2.0, 1.0, 2.0 / 3.0])

    def test_samples_return_mean_and_raw_weights(self):
        values, weights, mean = gmd_histogram_samples(self.map_path, 4, "gmd", 1.0)
        self.assertEqual(len(values), 4)
        self.assertAlmostEqual(mean, 1.5e-8)
        np.testing.assert_allclose(weights, values)

    def test_invalid_column_and_mode_are_rejected(self):
        with self.assertRaises(ValueError):
            load_gmd_map(self.map_path, 6)
        with self.assertRaises(ValueError):
            gmd_weights([1.0], "unknown")


if __name__ == "__main__":
    unittest.main()
