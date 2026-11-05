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
import tomlkit

from centroiding.spectrum_calibration import (
    CalibrationSettings, XE_4D_3_2_IP_EV, XE_4D_5_2_IP_EV, fit_xe_4d, scale_spectrum,
)

ROOT = Path(__file__).resolve().parents[1]


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

    def test_scaling_changes_density_and_preserves_integral(self):
        q, y, alpha, *_ = synthetic_spectrum()
        original_q, original_y = q.copy(), y.copy()
        energy, density = scale_spectrum(q, y, alpha)
        np.testing.assert_allclose(energy, q * alpha)
        np.testing.assert_allclose(density, y / alpha)
        self.assertAlmostEqual(np.trapezoid(density, energy), np.trapezoid(y, q), places=6)
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
        from gui.panels import centroid
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "config.toml"
            config.write_text('[vmi_picasso]\nh5_dir = "unchanged"\n', encoding="utf-8")
            widgets = {("energy_calibration", key, kind): SimpleNamespace(value=value)
                       for key, kind, value in (("alpha", "float", .0004),
                                               ("fit_xe_4d", "bool", True),
                                               ("fit_min_px2", "float", 20000),
                                               ("fit_max_px2", "float", 45000))}
            with patch.object(centroid, "CONFIG_PATH", config), patch.object(centroid.ui, "notify"):
                self.assertTrue(centroid._save(widgets))
                doc = tomlkit.parse(config.read_text(encoding="utf-8"))
                settings = CalibrationSettings.from_config(doc["vmi_picasso"]["energy_calibration"])
                self.assertEqual(settings, CalibrationSettings(.0004, True, 20000, 45000))
                saved = config.read_bytes()
                widgets[("energy_calibration", "alpha", "float")].value = -1
                self.assertFalse(centroid._save(widgets))
                self.assertEqual(config.read_bytes(), saved)
                widgets[("energy_calibration", "alpha", "float")].value = None
                self.assertTrue(centroid._save(widgets))
                doc = tomlkit.parse(config.read_text(encoding="utf-8"))
                self.assertNotIn("alpha", doc["vmi_picasso"]["energy_calibration"])
                self.assertEqual(doc["vmi_picasso"]["h5_dir"], "unchanged")


class CentroidScriptCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script = ROOT / "centroiding" / "process_andor_vmi_PICASSO_v5_h5_batch.py"
        with patch.object(sys, "argv", [str(script), "--config", str(ROOT / "centroiding/config.toml")]), \
                patch.object(sys, "path", [str(script.parent), *sys.path]):
            cls.namespace = runpy.run_path(str(script), run_name="calibration_test")
        cls.env = cls.namespace["perform_abel_transform"].__globals__

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
                np.testing.assert_allclose(density, y / factor)
                saved = np.loadtxt(Path(folder) / "energy_spectrum.csv", delimiter=",", skiprows=1)
                np.testing.assert_allclose(saved[:, 1], q)
                np.testing.assert_allclose(saved[:, 2], energy)
                np.testing.assert_allclose(saved[:, 4], density)
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
                    self.assertTrue((Path(folder) / "04_final_summary.png").exists())
                plt.close("all")


if __name__ == "__main__":
    unittest.main()
