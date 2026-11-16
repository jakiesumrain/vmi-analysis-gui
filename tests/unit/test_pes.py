import tempfile
import unittest
from pathlib import Path

import numpy as np
import abel
from scipy.special import eval_legendre
from scipy.special import voigt_profile

from vmi_analysis.centroiding.plot_data import load_plot_data
from vmi_analysis.pes import (ABEL_METHODS, Geometry, PeakGuess, anisotropy_values, average_anisotropy,
                              evaluate_fit_for_plot, extract_spectrum, fit_peaks, main)


class SpectrumTests(unittest.TestCase):
    def test_supported_inverse_methods(self):
        y, x = np.indices((45, 45)) - 22
        image = np.exp(-(x * x + y * y) / 80)
        for method in ABEL_METHODS:
            with self.subTest(method=method):
                _, energy, intensity, _ = extract_spectrum(image, method=method)
                self.assertEqual(energy.shape, intensity.shape)
                self.assertTrue(np.isfinite(intensity).all())

    def test_projected_shell_recovers_anisotropy_and_keeps_the_pes(self):
        y, x = np.indices((129, 129)) - 64
        r = np.hypot(x, y)
        vertical = np.divide(y, r, out=np.zeros_like(r), where=r != 0)
        shell = np.exp(-0.5 * ((r - 40) / 4) ** 2)
        slice_image = shell * (1 + 0.55 * eval_legendre(2, vertical) - 0.22 * eval_legendre(4, vertical))
        projected = abel.Transform(slice_image, direction="forward", method="hansenlaw").transform
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "shell.npy"
            output = root / "spectra.npz"
            np.save(source, projected)
            main(["--image", str(source), "64", "64", "0", "1", "--output", str(output), "--alpha", "0.02",
                  "--abel-method", "three_point", "--beta-order", "4"])
            meta, arrays = load_plot_data(output)
            self.assertEqual(meta["abel_method"], "three_point")
            self.assertEqual(meta["beta_order"], 4)
            _, _, intensity, _ = extract_spectrum(projected, Geometry((64, 64)), 0.02, method="three_point")
            np.testing.assert_array_equal(arrays["y_0"], intensity)
            beta = average_anisotropy(arrays["beta_x_0"], arrays["moments_0"], 32, 10)
            np.testing.assert_allclose(beta, [0.55, -0.22], atol=0.04)

    def test_window_average_uses_angular_moments_and_energy_spacing(self):
        x = np.array([1., 2., 5., 7.])
        moments = np.vstack((1 + x, 2 * x - 1, x - 3))
        np.testing.assert_allclose(average_anisotropy(x, moments, 4., 5.), [7 / 5, 1 / 5])
        self.assertEqual(average_anisotropy(x, moments, 4., 0.2), [None, None])
        self.assertEqual(average_anisotropy(x, moments, 6., 3.), [None, None])
        moments[0, 1] = -1
        self.assertTrue(np.isnan(anisotropy_values(moments)[:, 1]).all())
        moments[:, 2] = np.nan
        self.assertEqual(average_anisotropy(x, moments, 4., 5.), [None, None])

    def test_center_rotation_and_energy_scaling(self):
        y, x = np.indices((65, 65)) - 32
        image = np.exp(-(x * x + 2 * y * y) / 200)
        q, energy, intensity, center = extract_spectrum(image)
        self.assertEqual(center, (32, 32))
        np.testing.assert_array_equal(energy, q)
        q2, e2, i2, _ = extract_spectrum(np.rot90(image), Geometry((32, 32), -90, 2), alpha=0.5)
        np.testing.assert_allclose(q2, q / 4)
        np.testing.assert_allclose(e2, q / 8)
        np.testing.assert_allclose(i2, intensity)
        with self.assertRaisesRegex(ValueError, "coverage"):
            extract_spectrum(image, Geometry((0, 32)))

    def test_mixed_batch_skips_invalid_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            y, x = np.indices((66, 78))
            image = np.exp(-((x - 30) ** 2 + (y - 32) ** 2) / 80)
            files = [root / name for name in ("raw.npy", "missing.npy", "circularized.txt")]
            np.save(files[0], image)
            np.savetxt(files[2], image)
            output = root / "spectra.npz"
            args = ["--output", str(output), "--alpha", "0.001"]
            for path in files:
                args += ["--image", str(path), "30", "32", "0", "1"]
            main(args)
            meta, arrays = load_plot_data(output)
            self.assertEqual([row["path"] for row in meta["images"]], [str(files[0]), str(files[2])])
            np.testing.assert_array_equal(arrays["r2_0"], arrays["r2_1"])
            np.testing.assert_allclose(arrays["y_0"], arrays["y_1"])


class PeakTests(unittest.TestCase):
    def test_no_baseline_keeps_background_fixed_at_zero(self):
        x = np.linspace(0, 10, 161)
        y = 5 * voigt_profile(x - 4.2, 0.4, 0)
        for guesses in ((), (PeakGuess("Gaussian", 4, 1),)):
            with self.subTest(automatic=not guesses):
                _, total, background, components, residual, peaks = fit_peaks(x, y, (1, 9), guesses, "None")
                np.testing.assert_array_equal(background, 0)
                np.testing.assert_allclose(total, components.sum(axis=0))
                np.testing.assert_allclose(residual, 0, atol=1e-7)
                self.assertAlmostEqual(peaks[0]["center"], 4.2)

    def test_automatic_fit_detects_multiple_peaks_on_a_nonuniform_grid(self):
        x = np.linspace(0, np.sqrt(10), 180) ** 2
        y = 5 + 1.5 * x + 8 * voigt_profile(x - 3.0, 0.21, 0) + 4 * voigt_profile(x - 6.8, 0.33, 0)
        *_, residual, peaks = fit_peaks(x, y, (1, 9), baseline="Linear")
        np.testing.assert_allclose([peak["center"] for peak in peaks], [3, 6.8])
        np.testing.assert_allclose([peak["area"] for peak in peaks], [8, 4])
        np.testing.assert_allclose(residual, 0, atol=1e-7)

    def test_automatic_fit_uses_selected_profile(self):
        x = np.linspace(0, 10, 200)
        y = 5 + 4 * voigt_profile(x - 4, 0.33, 0.19)
        *_, peaks = fit_peaks(x, y, (1, 8), auto_profile="Voigt")
        self.assertEqual(len(peaks), 1)
        self.assertEqual(peaks[0]["profile"], "Voigt")
        self.assertAlmostEqual(peaks[0]["center"], 4)
        self.assertAlmostEqual(peaks[0]["lorentzian_fwhm"], 0.38)

    def test_automatic_fit_reports_a_region_without_peaks(self):
        x = np.linspace(0, 10, 301)
        rng = np.random.default_rng(12)
        for y in (np.ones_like(x), 0.1 + 2 * x, 0.1 + 2 * x + rng.normal(0, 0.05, x.shape)):
            with self.assertRaisesRegex(ValueError, "No prominent peaks found"):
                fit_peaks(x, y, (1, 9), baseline="Linear")

    def test_overlapping_gaussian_and_voigt_with_linear_baseline(self):
        x = np.linspace(0, 10, 301)
        y = 1 + 0.13 * x + 15 * voigt_profile(x - 3.7, 0.35, 0) + 9 * voigt_profile(x - 4.9, 0.4, 0.24)
        *_, residual, peaks = fit_peaks(x, y, (1, 8), [
            PeakGuess("Gaussian", 3.5, 0.95), PeakGuess("Voigt", 5.1, 1.1, 0.4),
        ], "Linear")
        np.testing.assert_allclose(residual, 0, atol=1e-7)
        np.testing.assert_allclose([p["center"] for p in peaks], [3.7, 4.9])
        np.testing.assert_allclose([p["area"] for p in peaks], [15, 9])
        self.assertAlmostEqual(peaks[1]["lorentzian_fwhm"], 0.48)
        self.assertGreater(peaks[1]["fwhm"], peaks[1]["gaussian_fwhm"])

    def test_reports_underdetermined_fit(self):
        with self.assertRaisesRegex(ValueError, "distinct spectrum samples"):
            fit_peaks(np.arange(8), np.ones(8), (1, 4), [PeakGuess("Gaussian", 2, 1)])

    def test_display_grid_evaluates_the_same_fit_between_sparse_samples(self):
        x = np.arange(0, 12, 0.5)
        center = 4.2
        y = 1 + 0.2 * x + 5 * voigt_profile(x - center, 0.45, 0)
        guess = PeakGuess("Gaussian", 4.1, 1.2)
        x, fitted, background, peaks, _, rows = fit_peaks(x, y, (1, 8), [guess], "Linear")
        grid, total, baseline_curve, components = evaluate_fit_for_plot(x, background, rows)
        self.assertGreater(grid.size, x.size)
        native = np.searchsorted(grid, x)
        np.testing.assert_allclose(total[native], fitted)
        np.testing.assert_allclose(baseline_curve[native], background)
        np.testing.assert_allclose(components[:, native], peaks, atol=1e-10)
        self.assertAlmostEqual(grid[np.argmax(components[0])], center)
        self.assertGreater(components.max(), peaks.max())
