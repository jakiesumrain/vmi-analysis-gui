from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import json

import h5py
import numpy as np

from vmi_analysis.ion_tof import GMD_SHIFTS, filtered_histogram, fit_filtered_peaks, gmd_diagnostics, histogram


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
            result = histogram(path, ("pixels", "clusters"), maximum=0.1)
            diagnostics = gmd_diagnostics(metadata, {
                source: result[f"{source}_shot_counts"] for source in ("pixels", "clusters")
            })
            with h5py.File(metadata, "r+") as file:
                file["frames"][-1] = file["frames"][-2]
            misordered = histogram(path, ("pixels",), maximum=0.1, metadata=metadata)
        self.assertEqual(result["pixels"].sum(), 0)
        self.assertNotIn("gmd2_na", result)
        self.assertEqual(GMD_SHIFTS[np.nanargmax(diagnostics["pixels_gmd_correlation"][:, 0])], -1)
        self.assertEqual(GMD_SHIFTS[np.nanargmax(diagnostics["clusters_gmd_correlation"][:, 0])], 0)
        np.testing.assert_allclose(diagnostics["gmd2_na"], gmd)
        self.assertIn("not strictly increasing", misordered["gmd_alignment_issue"])
        self.assertNotIn("gmd_shifts", misordered)

    def test_filter_uses_inclusive_gmd_limits_and_signed_shift_on_original_bins(self):
        with TemporaryDirectory() as directory:
            path, metadata, original = [Path(directory) / name for name in ("luna.h5", "meta.h5", "tof.npz")]
            with h5py.File(path, "w") as file:
                file["TDCEvents"] = np.array([(1, t) for t in (100_000, 200_000, 300_000, 400_000)],
                                             dtype=[("tdc_type", "u1"), ("timestamp", "u8")])
                file["PixelHits"] = np.array([(110_000,), (225_000,), (330_000,), (401_000,)],
                                             dtype=[("toa", "u8")])
                file["Clusters"] = np.array([(102_000,), (210_000,), (330_000,), (431_000,)],
                                            dtype=[("ctoa", "u8")])
            with h5py.File(metadata, "w") as file:
                file["frames"] = np.array([(j, v, v) for j, v in enumerate([2e-9, 3e-9, 9e-9, 9e-9])],
                                          dtype=[("timestamp_ns", "i8"), ("gmd1", "f8"), ("gmd2", "f8")])
            before = histogram(path, ("pixels", "clusters"), bins=4, maximum=4)
            np.savez(original, **before)
            earlier = filtered_histogram(path, original, metadata, "TDC1Rising", 2, 3, -1)
            later = filtered_histogram(path, original, metadata, "TDC1Rising", 2, 3, 1)
            with h5py.File(path, "r+") as file:
                file["TDCEvents"][-1] = (1, 410_000)
            with self.assertRaisesRegex(ValueError, "triggers changed"):
                filtered_histogram(path, original, metadata, "TDC1Rising", 2, 3, -1)
        np.testing.assert_array_equal(earlier["edges"], before["edges"])
        self.assertEqual(earlier["selected_triggers"], 1)
        np.testing.assert_array_equal(earlier["pixels"], [0, 1, 0, 0])
        np.testing.assert_array_equal(earlier["clusters"], [1, 0, 0, 0])
        self.assertEqual(later["selected_triggers"], 2)
        np.testing.assert_array_equal(later["pixels"], [0, 0, 1, 1])
        np.testing.assert_array_equal(later["clusters"], [0, 1, 0, 1])

    def test_peak_yield_counts_exact_arrivals_on_selected_triggers(self):
        with TemporaryDirectory() as directory:
            path, filtered = [Path(directory) / name for name in ("luna.h5", "filtered.npz")]
            triggers = np.array([100_000, 200_000, 300_000], dtype="u8")
            with h5py.File(path, "w") as file:
                file["TDCEvents"] = np.array([(1, t) for t in triggers],
                                             dtype=[("tdc_type", "u1"), ("timestamp", "u8")])
                file["PixelHits"] = np.array([(99_999,), (109_999,), (110_000,), (140_000,),
                                              (169_999,), (170_000,), (240_000,)], dtype=[("toa", "u8")])
            # The newest selected trigger has zero hits. The count window extends beyond the TOF plot.
            np.savez(filtered, edges=[2.0, 3.0, 4.0, 5.0, 6.0], pixels=[0, 0, 1, 0],
                     trigger_times=triggers, selected_mask=[True, False, True], selected_triggers=2)
            x = np.array([2.5, 3.5, 4.5, 5.5])
            peak = {"center": 4.0, "gaussian_fwhm": 2 * np.sqrt(2 * np.log(2))}
            with patch("vmi_analysis.pes.fit_peaks", return_value=(x, x, x, None, x, [peak])):
                result = fit_filtered_peaks(path, filtered, "TDC1Rising", ["pixels"], (2, 6), [], "Constant")
        counted = json.loads(result["pixels_peaks"])[0]
        self.assertEqual((counted["low"], counted["high"]), (1.0, 7.0))
        self.assertEqual(counted["counts"], 3)
        self.assertEqual(counted["yield_per_trigger"], 1.5)
