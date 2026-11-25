import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from vmi_analysis.ion_pipeline import CSV_NAME, GMD2_REFERENCE, PositionData, measure, run, validate_settings
from vmi_analysis.ion_pipeline_backend import TDC_PERIOD_TICKS, acquisition_splits, select_frames


class IonPipelineTests(unittest.TestCase):
    def test_weighted_scan_uses_one_reference_for_both_fit_sources_and_all_selected_frames(self):
        options = {"source": "pixels", "gate_gmd": True, "gmd_min": 0, "gmd_max": 5, "gmd_shift": 1,
                   "weight_by_gmd2": True, "workers": 2, "fit_bins": 60, "window_sigma": 3,
                   "peaks": [{"name": "A", "low": 8, "high": 14, "center": 10}]}
        positions = {
            "1_ion": PositionData(1, np.array([10., 10., 10.]), np.array([0, 1, 1]), np.array([2., 2., 0., 9.]), [4]),
            "2_ion": PositionData(2, np.full(7, 10.), np.array([0, 1, 1, 2, 2, 3, 3]), np.array([4., 4., 4., 9.]), [4]),
        }
        with self.assertRaisesRegex(ValueError, "enabled GMD2 gate"):
            validate_settings(options | {"gate_gmd": False})
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in positions:
                (root / name).mkdir()
            for fit_source in ("gated", "ungated"):
                output = root / fit_source
                with self.subTest(fit_source=fit_source), \
                        patch("vmi_analysis.ion_pipeline_backend.decode", side_effect=lambda path, *_: positions[path.name]), \
                        patch("vmi_analysis.ion_pipeline_backend.fit_peak", return_value={"mu": 10., "sigma": 1.}) as fit, \
                        patch("vmi_analysis.ion_pipeline.save_plot"):
                    rows = run(root, output, options | {"fit_tof": fit_source})
                self.assertEqual([row[GMD2_REFERENCE] for row in rows], [3.2, 3.2])
                self.assertEqual([row["selected_frames"] for row in rows], [2, 3])
                np.testing.assert_allclose([row["counts"] for row in rows], [3.2, 4.8])
                np.testing.assert_allclose([row["hits_per_frame"] for row in rows], [1.6, 1.6])
                np.testing.assert_allclose([row["poisson_error"] for row in rows], [1.6 / np.sqrt(2), 1.6 / np.sqrt(6)])
                calls = sorted(fit.call_args_list, key=lambda call: len(call.args[0]))
                if fit_source == "gated":
                    np.testing.assert_array_equal(calls[0].kwargs["weights"], [.5, .5])
                    np.testing.assert_array_equal(calls[1].kwargs["weights"], [.25] * 6)
                else:
                    self.assertEqual([len(call.args[0]) for call in calls], [3, 7])
                    self.assertFalse(any(call.kwargs for call in calls))
                with (output / CSV_NAME).open(encoding="utf-8", newline="") as file:
                    self.assertEqual([float(row[GMD2_REFERENCE]) for row in csv.DictReader(file)], [3.2, 3.2])
                self.assertEqual(json.loads((output / "settings.json").read_text(encoding="utf-8"))[GMD2_REFERENCE], 3.2)

    def test_fits_independently_on_all_hits_and_counts_exact_gated_hits(self):
        data = PositionData(145.9, np.array([7., 10., 12., 13., 20., 21.]),
                            np.array([0, 0, 1, 0, 0, 1]), np.array([3., 8., 3.]), [3])
        options = {"fit_bins": 60, "window_sigma": 3, "fit_tof": "ungated", "gmd_min": 2, "gmd_max": 6, "gmd_shift": 0,
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

    def test_gated_fit_uses_selected_frames_and_keeps_counting_normalization(self):
        data = PositionData(145.9, np.array([7., 10., 12., 13., 20., 21.]),
                            np.array([0, 0, 1, 0, 0, 1]), np.array([3., 8., 3.]), [3])
        selected = np.array([True, False, True])
        options = {"fit_bins": 60, "window_sigma": 3, "fit_tof": "gated",
                   "peaks": [{"name": "A", "low": 8, "high": 14, "center": 10}]}

        def fit(values, low, high, center, bins):
            np.testing.assert_array_equal(values, [7., 10., 13., 20.])
            return {"mu": 10., "sigma": 1.}

        row, = measure(data, options, selected, fit)
        self.assertEqual((row["window_min_ns"], row["window_max_ns"]), (7., 13.))
        self.assertEqual((row["counts"], row["selected_frames"], row["hits_per_frame"]), (2, 2, 1.))
        del options["fit_tof"]
        default_row, = measure(data, options, selected, fit)
        self.assertEqual(default_row, row)

        # A failed gated fit must not silently retry with ungated arrivals.
        row, = measure(data, options, selected, lambda *args: None)
        self.assertNotEqual(row["status"], "OK")

        def no_fit(*args):
            self.fail("An empty selection must not start a fit")

        row, = measure(data, options, np.zeros(3, dtype=bool), no_fit)
        self.assertNotEqual(row["status"], "OK")

        # Disabling the gate selects all frames, making the fit inputs identical.
        def fit_all(values, *args):
            np.testing.assert_array_equal(values, data.tof_ns)
            return {"mu": 10., "sigma": 1.}

        row, = measure(data, options, np.ones(3, dtype=bool), fit_all)
        self.assertEqual(row["selected_frames"], 3)

    def test_luna_splits_reset_and_keeps_expected_rollover(self):
        wrap = TDC_PERIOD_TICKS
        wall = np.array([0, 100_000_000, 8_100_000_000])
        ticks = np.array([wrap - 500_000_000_000, 500_000_000_000, 1_000_000_000])
        np.testing.assert_array_equal(acquisition_splits(wall, ticks), [2])
        data = PositionData(145.9, np.empty(0), np.empty(0, dtype=int), np.full(4, 3.), [2, 2])
        selected = select_frames(data, {"gmd_min": 2, "gmd_max": 6, "gmd_shift": 1})
        np.testing.assert_array_equal(selected, [False, True, False, True])
