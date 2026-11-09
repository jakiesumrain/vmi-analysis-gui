"""Calibration checks with known peak centers; no experimental dataset needed."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from vmi_analysis.centroiding.spectrum_calibration import (
    CalibrationSettings, XE_4D_3_2_IP_EV, XE_4D_5_2_IP_EV, fit_xe_4d, scale_spectrum,
)

ROOT = Path(__file__).resolve().parents[2]


def synthetic_spectrum():
    # Uniform radius -> nonuniform pixel² bins, as in the real pipeline.
    q = np.arange(1., 512.)**2
    alpha, hv = 0.0004, 82.0
    q1 = (hv - XE_4D_3_2_IP_EV) / alpha
    q2 = (hv - XE_4D_5_2_IP_EV) / alpha
    y = (1 + 0.00001 * q
         + 15 * np.exp(-0.5 * ((q - q1) / 500)**2)
         + 20 * np.exp(-0.5 * ((q - q2) / 600)**2))
    return q, y, alpha, hv, q1, q2


class SpectrumCalibrationTests(unittest.TestCase):
    def test_peak_fit_recovers_sub_bin_centers_and_photoelectric_parameters(self):
        q, y, alpha, hv, q1, q2 = synthetic_spectrum()
        y += np.random.default_rng(12).normal(0, 0.1, y.size)
        result = fit_xe_4d(q, y)
        self.assertAlmostEqual(result.q_3_2, q1, delta=30)
        self.assertAlmostEqual(result.q_5_2, q2, delta=30)
        self.assertAlmostEqual(result.alpha, alpha, delta=2e-6)
        self.assertAlmostEqual(result.photon_energy_ev, hv, delta=0.1)
        self.assertAlmostEqual(result.alpha * result.q_3_2,
                               result.photon_energy_ev - XE_4D_3_2_IP_EV)
        self.assertAlmostEqual(result.alpha * result.q_5_2,
                               result.photon_energy_ev - XE_4D_5_2_IP_EV)
        self.assertGreater(result.alpha_std, 0)
        # A fitted center must not just be the nearest sampled peak location.
        self.assertGreater(np.min(abs(q - result.q_3_2)), 1)

    def test_window_excludes_stronger_unrelated_peaks(self):
        q, y, alpha, hv, _, _ = synthetic_spectrum()
        y += 100 * np.exp(-0.5 * ((q - 90000) / 2000)**2)
        y += 80 * np.exp(-0.5 * ((q - 120000) / 2000)**2)
        result = fit_xe_4d(q, y, 20000, 45000)
        self.assertAlmostEqual(result.alpha, alpha, places=8)
        self.assertAlmostEqual(result.photon_energy_ev, hv, places=5)

    def test_single_peak_and_flat_spectrum_do_not_produce_calibration(self):
        q = np.arange(1., 512.)**2
        for y in (np.ones_like(q), np.exp(-0.5 * ((q - 35000) / 1500)**2)):
            with self.subTest(y=y.max()), self.assertRaises(ValueError):
                fit_xe_4d(q, y)

    def test_invalid_input_and_fit_ranges(self):
        q, y, *_ = synthetic_spectrum()
        for x, z, lo, hi in ((q, y, 10, 5), (q, y, -1, 10),
                              (q[:8], y[:8], None, None),
                              (q[::-1], y, None, None),
                              (q, np.full(q.size, np.nan), None, None)):
            with self.subTest(lo=lo, hi=hi), self.assertRaises(ValueError):
                fit_xe_4d(x, z, lo, hi)

    def test_calibration_changes_only_x_and_preserves_intensity(self):
        q, y, alpha, *_ = synthetic_spectrum()
        original_q, original_y = q.copy(), y.copy()
        energy, density = scale_spectrum(q, y, alpha)
        np.testing.assert_allclose(energy, q * alpha)
        np.testing.assert_array_equal(density, y)
        np.testing.assert_array_equal(q, original_q)
        np.testing.assert_array_equal(y, original_y)
        x, z = scale_spectrum(q, y)
        np.testing.assert_array_equal(x, q)
        np.testing.assert_array_equal(z, y)

    def test_optional_settings_and_invalid_alpha(self):
        self.assertEqual(CalibrationSettings.from_config({}), CalibrationSettings())
        for value in (0, -1, np.nan, np.inf):
            with self.subTest(alpha=value), self.assertRaises(ValueError):
                CalibrationSettings.from_config({"alpha": value})


class CentroidPanelCalibrationTests(unittest.TestCase):
    def test_save_roundtrip_optional_fields_and_validation(self):
        from vmi_analysis.gui.panels import centroid
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "config.toml"
            config.write_text('[vmi_picasso]\nh5_dir = "unchanged"\n', encoding="utf-8")
            widgets = {("energy_calibration", key, kind): SimpleNamespace(value=value)
                       for key, kind, value in (("alpha", "float", .0004),
                                               ("fit_xe_4d", "bool", True),
                                               ("fit_min_px2", "float", 20000),
                                               ("fit_max_px2", "float", 45000))}
            from vmi_analysis.gui.settings import centroid_settings, storage_key
            store = {}
            def snapshot():
                return centroid_settings(store=store, legacy_path=config)
            with patch.object(centroid, "centroid_settings", snapshot), \
                    patch.object(centroid.app.storage, "_general", store), \
                    patch.object(centroid.ui, "notify"):
                original = config.read_bytes()
                self.assertTrue(centroid._save(widgets))
                settings = CalibrationSettings.from_config(snapshot()["vmi_picasso"]["energy_calibration"])
                self.assertEqual(settings, CalibrationSettings(.0004, True, 20000, 45000))
                saved = dict(store)
                widgets[("energy_calibration", "alpha", "float")].value = -1
                self.assertFalse(centroid._save(widgets))
                self.assertEqual(store, saved)
                widgets[("energy_calibration", "alpha", "float")].value = None
                self.assertTrue(centroid._save(widgets))
                self.assertIsNone(store[storage_key("energy_calibration", "alpha")])
                self.assertEqual(config.read_bytes(), original)
                self.assertEqual(snapshot()["vmi_picasso"]["h5_dir"], "unchanged")



class CentroidScriptCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        module = "vmi_analysis.centroiding.process_andor_vmi_PICASSO_v5_h5_batch"
        with patch.object(sys, "argv", [module, "--h5-dir", "unused-test-data"]):
            cls.namespace = runpy.run_module(module, run_name="calibration_test")
        cls.env = cls.namespace["perform_abel_transform"].__globals__

    def test_abel_preprocessing_centers_then_applies_requested_rotation(self):
        raw = np.zeros((9, 9))
        raw[2, 3] = 10.0
        centered = np.zeros_like(raw)
        centered[4, 4] = 10.0
        centered[4, 6] = 4.0
        with patch.object(self.env["abel"].tools.center, "center_image",
                          return_value=centered) as center:
            for angle, turns in ((0, 0), (90, 1), (-90, 3), (180, 2)):
                with self.subTest(angle=angle), patch.dict(self.env, ABEL_ROTATION_DEG=angle):
                    image = self.env["center_and_rotate"](raw, verbose=False)
                    np.testing.assert_array_equal(image, np.rot90(centered, turns))
            with patch.dict(self.env, ABEL_ROTATION_DEG=45.0):
                image = self.env["center_and_rotate"](raw, verbose=False)
            self.assertEqual(image.shape, centered.shape)
            self.assertAlmostEqual(image[4, 4], 10.0)
            self.assertGreater(image[3, 5], 1.0)
            self.assertEqual(image[5, 5], 0.0)
        self.assertEqual(center.call_args.args[1], (2, 3))

    def test_script_csv_and_console_use_input_alpha_independently_of_fit(self):
        q, y, _, _, _, _ = synthetic_spectrum()
        radius = np.sqrt(q)
        fake_abel = SimpleNamespace(
            Transform=lambda *a, **kw: SimpleNamespace(transform=np.ones((1024, 1024))),
            tools=SimpleNamespace(vmi=SimpleNamespace(
                angular_integration_3D=lambda image: (radius, y.copy()),
                toPES=lambda r, speed, alpha: (r**2, speed),
            )))
        for manual_alpha in (None, .0007):
            with self.subTest(alpha=manual_alpha), tempfile.TemporaryDirectory() as folder:
                output = io.StringIO()
                settings = CalibrationSettings(manual_alpha, True, 20000, 45000)
                with patch.dict(self.env, abel=fake_abel, abel_available=True,
                                center_and_rotate=lambda image: image,
                                ENERGY_CALIBRATION=settings, OUTPUT_DIR=folder), redirect_stdout(output):
                    _, energy, density = self.env["perform_abel_transform"](np.ones((32, 32)))
                factor = manual_alpha or 1
                np.testing.assert_allclose(energy, q * factor)
                np.testing.assert_array_equal(density, y)
                saved = np.loadtxt(Path(folder) / "energy_spectrum.csv", delimiter=",", skiprows=1)
                np.testing.assert_allclose(saved[:, 1], q)
                np.testing.assert_allclose(saved[:, 2], energy)
                np.testing.assert_allclose(saved[:, 4], density)
                self.assertEqual((Path(folder) / "energy_spectrum.csv").read_text().splitlines()[0],
                                 "r_px,r2_px2,energy,dN_per_dr,intensity")
                self.assertIn("hv = 82.000000", output.getvalue())
                self.assertIn("alpha = 0.0004", output.getvalue())

    def test_plot_labels_and_png_follow_manual_calibration(self):
        import matplotlib.pyplot as plt
        q, y, *_ = synthetic_spectrum()
        for alpha, label in ((None, "Squared radius (pixel²)"), (.0004, "Kinetic energy (eV)")):
            with self.subTest(alpha=alpha), tempfile.TemporaryDirectory() as folder:
                with patch.dict(self.env, ENERGY_CALIBRATION=CalibrationSettings(alpha),
                                OUTPUT_DIR=folder, abel_available=False), \
                        patch.object(plt, "show"), patch.object(plt, "close"):
                    energy, density = scale_spectrum(q, y, alpha)
                    self.env["plot_final_summary"](np.ones((32, 32)), None, 100, 1, energy, density)
                    axes = [a for a in plt.gcf().axes if a.get_title() == "Photoelectron Spectrum"]
                    self.assertEqual(axes[0].get_xlabel(), label)
                    self.assertEqual(axes[0].get_ylabel(), "Intensity (arb. units)")
                    np.testing.assert_array_equal(axes[0].lines[0].get_ydata(), y)
                    self.assertTrue((Path(folder) / "04_final_summary.png").exists())
                    from vmi_analysis.centroiding.plot_data import load_plot_data, SUMMARY_DATA
                    metadata, arrays = load_plot_data(Path(folder) / SUMMARY_DATA)
                    self.assertEqual(metadata["alpha"], alpha)
                    np.testing.assert_array_equal(arrays["energy"], energy)
                    np.testing.assert_array_equal(arrays["intensity"], density)
                plt.close("all")

    def test_tuning_script_exports_corrected_frame_and_localizations(self):
        import h5py
        import pandas as pd
        from vmi_analysis.centroiding.plot_data import load_plot_data, TUNING_DATA
        locs = pd.DataFrame({"x": [2., 3.], "y": [4., 5.], "photons": [100., 200.],
                             "net_gradient": [500., 600.], "lpx": [.1, .2], "lpy": [.2, .3]})
        with tempfile.TemporaryDirectory() as folder:
            raw = np.arange(64, dtype=np.uint16).reshape(8, 8)
            path = Path(folder) / "frame.h5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("raw/data", data=np.stack([raw, raw + 100]))
            output = io.StringIO()
            with patch.dict(self.env, OUTPUT_DIR=folder, TUNING_FRAME_INDEX=None,
                            _run_picasso_batch=lambda *args: locs,
                            sigma_clipped_stats=lambda *args, **kw: (2., 2., 0.)), \
                    patch.dict(self.env, randrange=lambda count: count - 1), \
                    patch.object(self.env["plt"], "show"), redirect_stdout(output):
                self.env["run_tuning_check"]([(str(path), 0), (str(path), 1)], None, None)
            self.assertIn("Global frame index: 1", output.getvalue())
            metadata, arrays = load_plot_data(Path(folder) / TUNING_DATA)
            self.assertEqual(metadata["n_events"], 2)
            np.testing.assert_array_equal(arrays["frame"], raw.astype(float) + 98)
            np.testing.assert_array_equal(arrays["x"], locs.x)
            np.testing.assert_array_equal(arrays["photons"], locs.photons)


if __name__ == "__main__":
    unittest.main()
