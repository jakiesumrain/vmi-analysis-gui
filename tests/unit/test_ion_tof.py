from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import h5py
import numpy as np

from vmi_analysis.ion_tof import GMD_SHIFTS, histogram


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

    def test_gmd_offset_uses_selected_count_source_and_all_tof_values(self):
        gmd = np.array([3, 1, 7, 9, 2, 8, 3, 5, 2, 9, 4, 7, 3, 1, 6, 2])
        triggers = np.arange(1, len(gmd) + 1, dtype=np.uint64) * 100_000
        with TemporaryDirectory() as directory:
            path = Path(directory) / "luna.h5"
            metadata = Path(directory) / "metadata.h5"
            with h5py.File(path, "w") as file:
                file["TDCEvents"] = np.array([(1, t) for t in triggers],
                                            dtype=[("tdc_type", "u1"), ("timestamp", "u8")])
                for table, field, counts in (("PixelHits", "toa", np.r_[gmd[1:], 4]),
                                              ("Clusters", "ctoa", gmd)):
                    file[table] = np.array([(t,) for t in np.repeat(triggers + 5000, counts)],
                                           dtype=[(field, "u8")])
            with h5py.File(metadata, "w") as file:
                file["frames"] = np.array([(index, 2 * value * 1e-9, value * 1e-9)
                                           for index, value in enumerate(gmd)],
                                          dtype=[("timestamp_ns", "i8"), ("gmd1", "f8"), ("gmd2", "f8")])
            result = histogram(path, ("pixels", "clusters"), maximum=0.1, metadata=metadata)
            with h5py.File(metadata, "r+") as file:
                file["frames"][-1] = file["frames"][-2]
            misordered = histogram(path, ("pixels",), maximum=0.1, metadata=metadata)
        self.assertEqual(result["pixels"].sum(), 0)
        self.assertEqual(GMD_SHIFTS[np.nanargmax(result["pixels_gmd_correlation"][:, 0])], -1)
        self.assertEqual(GMD_SHIFTS[np.nanargmax(result["clusters_gmd_correlation"][:, 0])], 0)
        np.testing.assert_allclose(result["gmd2_na"], gmd)
        self.assertIn("not strictly increasing", misordered["gmd_alignment_issue"])
        self.assertNotIn("gmd_shifts", misordered)
