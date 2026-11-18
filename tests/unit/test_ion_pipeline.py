import unittest

import numpy as np

from vmi_analysis.ion_pipeline import PositionData, measure
from vmi_analysis.ion_pipeline_backend import TDC_PERIOD_TICKS, acquisition_splits, select_frames


class IonPipelineTests(unittest.TestCase):
    def test_fits_independently_on_all_hits_and_counts_exact_gated_hits(self):
        data = PositionData(145.9, np.array([7., 10., 12., 13., 20., 21.]),
                            np.array([0, 0, 1, 0, 0, 1]), np.array([3., 8., 3.]), [3])
        options = {"fit_bins": 60, "window_sigma": 3, "gmd_min": 2, "gmd_max": 6, "gmd_shift": 0,
                   "peaks": [{"name": "A", "low": 8, "high": 14, "center": 10},
                             {"name": "B", "low": 16, "high": 24, "center": 20}]}
        fits = []

        def fit(values, low, high, center, bins):
            np.testing.assert_array_equal(values, data.tof_ns)
            fits.append((low, high, bins))
            return {"mu": center, "sigma": 1}

        first, second = measure(data, options, select_frames(data, options), fit)
        self.assertEqual(fits, [(8, 14, 60), (16, 24, 60)])
        self.assertEqual((first["counts"], second["counts"]), (2, 1))
        self.assertEqual((first["hits_per_frame"], second["hits_per_frame"]), (1., 0.5))

    def test_luna_splits_reset_and_keeps_expected_rollover(self):
        wrap = TDC_PERIOD_TICKS
        wall = np.array([0, 100_000_000, 8_100_000_000])
        ticks = np.array([wrap - 500_000_000_000, 500_000_000_000, 1_000_000_000])
        np.testing.assert_array_equal(acquisition_splits(wall, ticks), [2])
        data = PositionData(145.9, np.empty(0), np.empty(0, dtype=int), np.full(4, 3.), [2, 2])
        selected = select_frames(data, {"gmd_min": 2, "gmd_max": 6, "gmd_shift": 1})
        np.testing.assert_array_equal(selected, [False, True, False, True])
