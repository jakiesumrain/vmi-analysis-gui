from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import h5py
import numpy as np

from vmi_analysis.ion_tof import histogram


class IonTofTests(unittest.TestCase):
    def test_reference_edge_order_tick_precision_and_missing_trigger(self):
        base = 2**60
        with TemporaryDirectory() as directory:
            path = Path(directory) / "luna.h5"
            with h5py.File(path, "w") as file:
                file["TDCEvents"] = np.array(
                    [(1, base + 100), (2, base + 12), (1, base)],
                    dtype=[("tdc_type", "u1"), ("timestamp", "u8")],
                )
                file["PixelHits"] = np.array(
                    [(base + shift, -1) for shift in (-1, 0, 14, 149, 180)],
                    dtype=[("toa", "u8"), ("tof", "i8")],
                )
                file["Clusters"] = np.array([(base + 102,)], dtype=[("ctoa", "u8")])
            result = histogram(path, ("pixels", "clusters"), bins=2, maximum=0.006)
        np.testing.assert_array_equal(result["pixels"], [2, 1])
        np.testing.assert_array_equal(result["clusters"], [1, 0])
        self.assertEqual(result["triggers"], 2)
        self.assertEqual(result["pixels_unmatched"], 1)
        self.assertEqual(result["pixels_outside"], 1)

    def test_auto_range_uses_latest_arrival_from_either_source(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "luna.h5"
            with h5py.File(path, "w") as file:
                file["TDCEvents"] = np.array([(1, 5000)], dtype=[("tdc_type", "u1"), ("timestamp", "u8")])
                file["PixelHits"] = np.array([(6000,)], dtype=[("toa", "u8")])
                file["Clusters"] = np.array([(8000,)], dtype=[("ctoa", "u8")])
            result = histogram(path, ("pixels", "clusters"), bins=2)
        self.assertAlmostEqual(result["edges"][-1], 0.3)
        np.testing.assert_array_equal(result["pixels"], [1, 0])
        np.testing.assert_array_equal(result["clusters"], [0, 1])
