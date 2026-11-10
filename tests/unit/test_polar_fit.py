import tempfile
import unittest
from pathlib import Path

import numpy as np

from vmi_analysis.circularization import polar_fit as pf
from vmi_analysis.circularization.circularization import (parse_extracted_text,
                                                          parse_extracted_txt)


def make_ring_image(center=(100, 100), size=201, radius=50.0, sigma=1.6):
    """Cartesian image with one bright circular ridge."""
    ys, xs = np.mgrid[0:size, 0:size]
    rr = np.hypot(xs - center[0], ys - center[1])
    return np.exp(-0.5 * ((rr - radius) / sigma) ** 2)


def make_polar_ridge(n_theta=180, n_r=101, r0=50.0, sigma=2.0, intensity=100.0,
                     offset=2.0, beta=0.3, trig=None):
    """Polar image generated from the ridge model with known parameters."""
    trig = trig or {}
    theta = np.linspace(0.0, pf.TWO_PI, n_theta, endpoint=False)
    r = np.linspace(0.0, n_r - 1, n_r)
    theta_grid, r_grid = np.meshgrid(theta, r, indexing="ij")
    rg = np.full_like(theta_grid, r0)
    for key, value in trig.items():
        n = int(key[1:])
        rg += value * (np.sin(n * theta_grid) if key.startswith("A")
                       else np.cos(n * theta_grid))
    angle = intensity * (1.0 + beta * (3.0 * np.cos(theta_grid) ** 2 - 1.0) / 2.0)
    polar = angle * np.exp(-0.5 * ((r_grid - rg) / sigma) ** 2) + offset
    return polar, theta, r


class PolarRemapTests(unittest.TestCase):
    def test_radial_bin_size_sets_sample_spacing_and_interpolates(self):
        image = np.tile(np.arange(20, dtype=float), (20, 1))
        polar, _, radii, dr, _ = pf.remap_polar(
            image, (8, 8), radial_bin_size=0.5, r_max=4.2, n_theta=8)
        self.assertEqual(dr, 0.5)
        np.testing.assert_array_equal(radii, np.arange(9) * 0.5)
        np.testing.assert_allclose(polar[0], 8 + radii)
        _, radii, _ = pf.polar_grid(image.shape, (8, 8), radial_bin_size=0.1, r_max=0.9)
        self.assertEqual(len(radii), 10)
        self.assertEqual(radii[-1], 0.9)

    def test_remap_places_ring_at_expected_radius(self):
        image = make_ring_image()
        polar, theta, r, dr, valid = pf.remap_polar(
            image, center=(100, 100), n_r=120, n_theta=180, r_max=90)
        self.assertGreater(dr, 0)
        self.assertTrue(valid.any())
        profile = np.nanmean(polar, axis=0)
        peak = r[int(np.nanargmax(profile))]
        self.assertAlmostEqual(peak, 50.0, delta=2.0)

    def test_remap_outside_image_is_nan(self):
        image = make_ring_image(size=101, center=(50, 50))
        polar, theta, r, dr, valid = pf.remap_polar(
            image, center=(50, 50), n_r=80, n_theta=90, r_max=70)
        self.assertFalse(valid.all())
        self.assertTrue(np.isnan(polar[~valid]).all())

class AbelTests(unittest.TestCase):
    def test_hansenlaw_inverts_uniform_ball_projection(self):
        radius = 96
        r = np.arange(radius, dtype=float)
        projection = 2.0 * np.sqrt(np.clip(radius ** 2 - r ** 2, 0.0, None))
        polar = np.tile(projection, (6, 1))
        inverted = pf.inverse_abel(polar, dr=1.0, method="hansenlaw")
        self.assertEqual(inverted.shape, polar.shape)
        inside = inverted[:, 10:80]
        mean = float(np.mean(inside))
        spread = float(np.std(inside) / max(abs(mean), 1e-12))
        self.assertGreater(mean, 0.8)
        self.assertLess(spread, 0.05)

    def test_nan_rows_stay_nan(self):
        polar = np.ones((3, 32))
        polar[1, :] = np.nan
        inverted = pf.inverse_abel(polar, dr=1.0)
        self.assertTrue(np.isnan(inverted[1]).all())


class EnhancementTests(unittest.TestCase):
    def test_bandpass_accepts_zero_lower_width_and_rejects_invalid_widths(self):
        polar, _, _ = make_polar_ridge(n_theta=24, n_r=60)
        result = pf.enhance_bandpass(polar, low_r=0, high_r=5)
        self.assertTrue(np.isfinite(result).all())
        self.assertGreater(result.max(), 0)
        for low, high in ((-1, 5), (0, 0), (5, 2), (0, np.inf)):
            with self.subTest(low=low, high=high), self.assertRaisesRegex(ValueError, "Bandpass widths"):
                pf.enhance_bandpass(polar, low_r=low, high_r=high)

    def test_bandpass_preserves_ridge_position(self):
        n_theta, n_r = 40, 80
        theta = np.linspace(0, pf.TWO_PI, n_theta, endpoint=False)
        r = np.arange(n_r, dtype=float)
        ridge = np.exp(-0.5 * ((r - 42.0) / 1.5) ** 2)
        polar = np.tile(ridge, (n_theta, 1)) + 0.02 * r
        enhanced = pf.enhance_polar(polar, "bandpass", {})
        profile = np.nanmean(enhanced, axis=0)
        peak = int(np.argmax(profile))
        self.assertAlmostEqual(float(r[peak]), 42.0, delta=1.0)

    def test_display_filters_do_not_change_fit_image(self):
        polar, _, _ = make_polar_ridge(n_theta=24, n_r=60)
        for abel in (False, True):
            expected = pf.inverse_abel(polar) if abel else polar
            for placement in ("raw", "pre_abel", "post_abel"):
                with self.subTest(abel=abel, placement=placement):
                    fit_image, display, options = pf.process_polar(
                        polar, dr=1.0, abel=abel,
                        enhancement={"placement": placement, "mode": "bandpass"})
                    np.testing.assert_array_equal(fit_image, expected)
                    self.assertFalse(np.allclose(display, expected))
                    self.assertEqual(options["enhancement"]["placement"], placement)

    def test_fit_can_use_processed_polar_data_with_and_without_abel(self):
        polar, _, _ = make_polar_ridge(n_theta=24, n_r=60)
        for abel, placement in ((False, "raw"), (True, "post_abel")):
            base = pf.inverse_abel(polar) if abel else polar
            for selected in (placement, "off"):
                with self.subTest(abel=abel, placement=selected):
                    data, display, options = pf.process_polar(
                        polar, 1.0, abel=abel, fit_filtered=True,
                        enhancement={"placement": selected, "mode": "bandpass"})
                    expected = base if selected == "off" else pf.enhance_polar(base, "bandpass")
                    self.assertIs(data, display)
                    np.testing.assert_array_equal(data, expected)
                    self.assertIn("fit_filter=", pf.describe_processing(options))


class SelectionTests(unittest.TestCase):
    def test_full_angle_band_and_mask(self):
        theta = np.linspace(0, pf.TWO_PI, 72, endpoint=False)
        r = np.arange(50, dtype=float)
        band = pf.Rect(10, 20)
        mask = pf.selection_mask(theta, r, band)
        self.assertEqual(int(mask.sum()), 72 * 11)
        masked = pf.selection_mask(theta, r, band, [pf.Rect(12, 15, 0, np.pi)])
        self.assertLess(int(masked.sum()), int(mask.sum()))
        self.assertFalse(masked[:36, 12:16].any())

    def test_angular_wrap(self):
        theta = np.array([0.1, 1.0, 3.0, 5.9])
        mask = pf._angular_mask(theta, 5.5, 0.5)
        self.assertTrue(mask[0] and mask[3])
        self.assertFalse(mask[1] or mask[2])


class FitTests(unittest.TestCase):
    def test_masked_inner_ring_remnants_do_not_seed_the_fit(self):
        theta = np.linspace(0, 2 * np.pi, 180, endpoint=False)
        radii = np.arange(205, 225.1, 0.5)
        inner = 3 * np.exp(-0.5 * ((radii - 210) / 0.6) ** 2)
        ridge = 218 + 1.3 * np.sin(theta)
        outer = np.exp(-0.5 * ((radii - ridge[:, None]) / 0.9) ** 2)
        mask = pf.Rect(207.5, 213, np.deg2rad(10), 2 * np.pi - 0.01)
        fit = pf.fit_ring(outer + inner, theta, radii, pf.Rect(206, 224),
                          masks=[mask], n_terms=1, use_beta=False)
        self.assertAlmostEqual(fit.params["r0"], 218, delta=0.1)
        self.assertAlmostEqual(fit.params["A1"], 1.3, delta=0.1)
        self.assertEqual(fit.masked_ridge_fraction(theta), 0)

    def test_one_angle_shows_unconstrained_terms_and_blocks_export(self):
        polar, theta, r = make_polar_ridge()
        fit = pf.fit_ring(polar, theta, r, pf.Rect(34, 46, 0, 0.001),
                          n_terms=2, use_beta=False)
        self.assertIn("A1", fit.unconstrained)
        self.assertIn("A2", fit.unconstrained)
        self.assertTrue(np.isnan(fit.sigmas["A1"]))
        with self.assertRaisesRegex(ValueError, "cannot determine"):
            pf.ring_block_text(fit)

    def test_optimizer_warning_does_not_block_identifiable_fit(self):
        polar, theta, r = make_polar_ridge()
        fit = pf.fit_ring(polar, theta, r, pf.Rect(34, 46),
                          n_terms=2, use_beta=False, iterative=False, max_nfev=1)
        self.assertFalse(fit.converged)
        self.assertIsNone(fit.export_issue())
        self.assertIn(pf.EXTRACTED_HEADER, pf.ring_block_text(fit))

    def test_fit_recovers_known_trig_terms(self):
        trig = {"A1": 1.0, "B1": -2.0, "A2": 0.5, "B2": 1.2}
        polar, theta, r = make_polar_ridge(n_theta=180, n_r=101, r0=50.0,
                                           sigma=2.0, intensity=100.0,
                                           offset=2.0, beta=0.3, trig=trig)
        rng = np.random.default_rng(4)
        polar = polar + rng.normal(0.0, 0.3, polar.shape)
        fit = pf.fit_ring(polar, theta, r, pf.Rect(43.0, 57.0), n_terms=2)
        self.assertTrue(fit.converged)
        self.assertAlmostEqual(fit.params["r0"], 50.0, delta=0.3)
        self.assertAlmostEqual(fit.params["sigma"], 2.0, delta=0.3)
        for key, expected in trig.items():
            self.assertAlmostEqual(fit.params[key], expected, delta=0.15,
                                   msg=key)
        self.assertGreater(fit.params["intensity"], 80.0)

    def test_fit_respects_masks(self):
        trig = {"A1": 1.0, "B1": -2.0}
        polar, theta, r = make_polar_ridge(n_theta=120, n_r=80, r0=40.0,
                                           sigma=1.5, trig=trig)
        # A huge corrupt wedge must not change the fit when masked.
        corrupt = polar.copy()
        corrupt[:30, :] = 1e4
        fit = pf.fit_ring(corrupt, theta, r, pf.Rect(34.0, 46.0),
                          masks=[pf.Rect(0.0, 70.0, 0.0, np.deg2rad(90.0))],
                          n_terms=1)
        self.assertAlmostEqual(fit.params["A1"], 1.0, delta=0.2)
        self.assertAlmostEqual(fit.params["B1"], -2.0, delta=0.2)
        self.assertAlmostEqual(fit.masked_ridge_fraction(theta), 0.25, delta=0.02)

    def test_fit_uses_beta_by_default(self):
        polar, theta, r = make_polar_ridge(n_theta=120, n_r=80, r0=40.0,
                                           sigma=1.5, beta=0.8)
        fit = pf.fit_ring(polar, theta, r, pf.Rect(35.0, 45.0), n_terms=0)
        self.assertIn("beta", fit.params)
        self.assertAlmostEqual(fit.params["beta"], 0.8, delta=0.05)


class OutputTests(unittest.TestCase):
    def setUp(self):
        trig = {"A1": 1.0, "B1": -2.0, "A2": 0.5, "B2": 0.2}
        polar, theta, r = make_polar_ridge(n_theta=120, n_r=80, r0=40.0,
                                           trig=trig)
        self.fit = pf.fit_ring(polar, theta, r, pf.Rect(34.0, 46.0), n_terms=2)

    def test_block_text_parses_with_comments(self):
        text = pf.ring_block_text(self.fit, provenance=True)
        blocks = parse_extracted_text(text)
        self.assertEqual(len(blocks), 1)
        self.assertAlmostEqual(blocks[0]["radius"], self.fit.params["r0"])
        self.assertAlmostEqual(blocks[0]["A1"], self.fit.params["A1"])
        self.assertAlmostEqual(blocks[0]["B1_sigma"], self.fit.sigmas["B1"])

    def test_refresh_preserves_saved_ring_data_and_source_for_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "extracted.txt"
            first = pf.ring_block_text(self.fit)
            second = (f"{pf.EXTRACTED_HEADER}\n"
                      '"Gaussian Centre" 90 +- 0.1\n'
                      '"Trig Term cos(3*theta)" 2 +- 0.4\n')
            path.write_text(first + "\n" + second, encoding="utf-8")
            expected = parse_extracted_txt(path)
            rings = pf.load_accepted_rings(path)
            self.assertAlmostEqual(rings[0].rms, self.fit.rms, places=3)
            self.assertTrue(np.isnan(rings[1].rms))
            self.assertEqual(rings[1].n_terms, 3)
            restored = [pf.RingFit.from_dict(ring.to_dict()) for ring in rings]
            pf.write_ring_blocks(path, [ring.source_block for ring in restored])
            self.assertEqual(parse_extracted_txt(path), expected)
            self.assertIn("# ring", path.read_text(encoding="utf-8"))

    def test_append_and_parse_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "extracted.txt"
            first = pf.ring_block_text(self.fit, provenance=False)
            second = first.replace("Gaussian Centre", "Gaussian Centre")
            pf.append_ring_block(path, first)
            pf.append_ring_block(path, second)
            blocks = parse_extracted_txt(path)
            self.assertEqual(len(blocks), 2)
            self.assertEqual(pf.existing_radii(path), [self.fit.params["r0"]] * 2)

    def test_validate_rejects_malformed_block(self):
        with self.assertRaises(ValueError):
            pf.validate_block_text("not a block")

    def test_existing_radii_ignores_empty_and_comment_only_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "extracted.txt"
            path.write_text("", encoding="utf-8")
            self.assertEqual(pf.existing_radii(path), [])
            path.write_text("# just a comment\n", encoding="utf-8")
            self.assertEqual(pf.existing_radii(path), [])

    def test_session_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.json"
            session = {"image": {"path": "img.npy"},
                       "rings": [self.fit.to_dict()],
                       "output": "extracted.txt"}
            pf.save_session(path, session)
            loaded = pf.load_session(path)
            self.assertEqual(len(loaded["rings"]), 1)
            ring = pf.session_rings(loaded)[0]
            self.assertAlmostEqual(ring.params["r0"], self.fit.params["r0"])

    def test_legacy_session_requires_refitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.json"
            path.write_text('{"schema": "vmi-trig-calibration/1"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Refit the rings"):
                pf.load_session(path)


def make_cartesian_rings(size=257, center=128.0, rings=(60.0, 100.0),
                         slope_a=None, slope_b=None, sigma=1.8,
                         noise=0.3, seed=3):
    """Cartesian image with rings distorted by known a_n, b_n slopes."""
    slope_a = slope_a or {1: 0.02, 2: 0.01}
    slope_b = slope_b or {1: -0.03, 2: 0.015}
    ys, xs = np.mgrid[0:size, 0:size]
    rr = np.hypot(xs - center, ys - center)
    theta = np.arctan2(ys - center, xs - center)
    image = np.full((size, size), 0.5)
    for radius in rings:
        distortion = sum(slope_a[n] * np.sin(n * theta)
                         + slope_b[n] * np.cos(n * theta)
                         for n in slope_a)
        ridge_radius = radius * (1.0 + distortion)
        image += 100.0 * np.exp(-0.5 * ((rr - ridge_radius) / sigma) ** 2)
    if noise:
        image += np.random.default_rng(seed).normal(0.0, noise, image.shape)
    return image


class EndToEndTests(unittest.TestCase):
    def test_clamped_interpolant_has_zero_derivative_outside_fitted_range(self):
        from vmi_analysis.circularization import circularization as circ

        models, _ = circ.build_all_trig_interpolators(
            {"radii": np.array([10., 20.]), "B1": np.array([2., 4.])})
        model = models["B1"]
        r = np.array([-5., 5., 15., 30.])
        values, derivatives = circ.eval_radial_model(model, r)
        step = 1e-3
        left, _ = circ.eval_radial_model(model, r - step)
        right, _ = circ.eval_radial_model(model, r + step)
        np.testing.assert_allclose(values, [0., 1., 3., 4.])
        np.testing.assert_allclose(derivatives, (right - left) / (2 * step), atol=1e-10)
        np.testing.assert_allclose(derivatives, [0., 0.2, 0.2, 0.])
        _, edge_derivatives = circ.eval_radial_model(model, [0., 20.])
        np.testing.assert_allclose(edge_derivatives, [0.2, 0.2])

    def test_circularize_uses_clamped_derivative_in_jacobian(self):
        from vmi_analysis.circularization import circularization as circ

        models, _ = circ.build_all_trig_interpolators(
            {"radii": np.array([10., 20.]), "B1": np.array([2., 4.])})
        image = np.ones((151, 151))
        corrected = circ.circularize_vmi(image, (75, 75), models, interp_order=1)
        self.assertAlmostEqual(corrected[75, 90], 1.44)  # r_source/r = 1.2, dr_source/dr = 1.2
        self.assertAlmostEqual(corrected[75, 105], 34 / 30)  # dr_source/dr = 1 beyond r=20

    def test_circularize_rejects_outputs_aliasing_its_inputs(self):
        import sys
        from unittest.mock import patch

        from vmi_analysis.circularization import circularization

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "raw.npy"
            extracted = root / "extracted.txt"
            reassign = root / "reassign.txt"
            for source in (image, extracted, reassign):
                source.write_bytes(b"original")
            hardlink = root / "alias.txt"
            hardlink.hardlink_to(image)
            cases = [(image, "input image"), (extracted, "extracted calibration file"),
                     (reassign, "radial reassignment file"), (hardlink, "input image")]
            symlink = root / "symlink.txt"
            try:
                symlink.symlink_to(extracted)
            except OSError:
                pass  # Windows may require developer mode for symbolic links.
            else:
                cases.append((symlink, "extracted calibration file"))
            args = ["circularization", "--input", str(image), "--extracted", str(extracted),
                    "--reassign-file", str(reassign)]
            for output, label in cases:
                with self.subTest(output=output), patch.object(sys, "argv", args + ["--output", str(output)]):
                    with self.assertRaisesRegex(SystemExit, label):
                        circularization.main()
                    self.assertEqual(output.read_bytes(), b"original")

    def test_atomic_image_write_preserves_output_on_failure(self):
        from unittest.mock import patch

        from vmi_analysis.circularization import circularization

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "corrected.txt"
            path.write_bytes(b"previous complete result")
            image = np.arange(6).reshape(2, 3)

            def fail(handle, *_args, **_kwargs):
                handle.write("partial result")
                raise OSError("write interrupted")

            with patch.object(circularization.np, "savetxt", side_effect=fail):
                with self.assertRaisesRegex(OSError, "write interrupted"):
                    circularization.save_image_txt(path, image)
            self.assertEqual(path.read_bytes(), b"previous complete result")
            self.assertEqual(list(Path(folder).iterdir()), [path])
            circularization.save_image_txt(path, image)
            np.testing.assert_array_equal(np.loadtxt(path), image)

    def test_signed_harmonics_round_trip_through_circularize(self):
        from vmi_analysis.circularization import circularization as circ

        center = (77, 57)
        y, x = np.indices((128, 144), dtype=float)
        dx, dy = x - center[0], y - center[1]
        theta = np.arctan2(dy, dx)
        distortion = (.035 * np.sin(theta) - .028 * np.cos(theta)
                      + .026 * np.sin(2 * theta) + .014 * np.cos(2 * theta))
        r_true = np.hypot(dx, dy) / (1 + distortion)
        radii = (27.0, 40.0, 51.0)
        raw = sum(np.exp(-0.5 * ((r_true - radius) / 1.7) ** 2) for radius in radii)

        def fit_rings(image):
            polar, angles, r, _, _ = pf.remap_polar(image, center, n_r=229, n_theta=360)
            return [pf.fit_ring(polar, angles, r, pf.Rect(radius - 5.8, radius + 5.8),
                                n_terms=2, use_beta=False) for radius in radii]

        before = fit_rings(raw)
        blocks = circ.parse_extracted_text("".join(pf.ring_block_text(fit) for fit in before))
        fit_params, _ = circ.build_all_trig_interpolators(circ.build_fit_dataset(blocks))
        corrected = circ.circularize_vmi(raw, center, fit_params, apply_jacobian=False)
        self.assertEqual(corrected.shape, raw.shape)
        for fit, expected_radius in zip(fit_rings(corrected)[:2], radii):
            self.assertAlmostEqual(fit.params["r0"], expected_radius, delta=.03)
            for key in ("A1", "B1", "A2", "B2"):
                self.assertAlmostEqual(fit.params[key], 0.0, delta=.03)

    def test_circularize_cli_keeps_the_original_numpy_detector_grid(self):
        from contextlib import redirect_stdout
        import io
        import sys
        from unittest.mock import patch

        from vmi_analysis.circularization import circularization

        image = np.arange(12 * 16, dtype=float).reshape(12, 16)
        image[4, 10] = 300
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "vmi_2d_raw.npy"
            calibration = root / "extracted.txt"
            output = root / "corrected.txt"
            np.save(source, image)
            calibration.write_text(
                f'{pf.EXTRACTED_HEADER}\n"Gaussian Centre" 5 +- 0.1\n'
                '"Trig Term cos(1*theta)" 0 +- 0.1\n', encoding="utf-8")
            args = ["circularization", "--input", str(source), "--output", str(output),
                    "--extracted", str(calibration), "--no-plots", "--interp-order", "1"]
            log = io.StringIO()
            with patch.object(sys, "argv", args), redirect_stdout(log):
                circularization.main()
            self.assertIn("NY=12, NX=16. Center X=10, Center Y=4.", log.getvalue())
            np.testing.assert_allclose(np.loadtxt(output), image)
            np.testing.assert_array_equal(np.load(source), image)
            with self.assertRaisesRegex(ValueError, "cannot be reshaped"):
                circularization.load_image_and_center(source, shape=(16, 12))
            legacy = root / "legacy.txt"
            np.savetxt(legacy, image.ravel())
            text_image, text_center = circularization.load_image_and_center(
                legacy, shape=image.shape, center=(7.5, 5.5))
            np.testing.assert_array_equal(text_image, image)
            self.assertEqual(text_center, (7.5, 5.5))

    def test_cartesian_rings_recover_slopes_through_full_chain(self):
        image = make_cartesian_rings()
        polar, theta, r, dr, _valid = pf.remap_polar(
            image, center=(128.0, 128.0), n_r=129, n_theta=360)
        fit_image, _, _ = pf.process_polar(polar, dr)
        fits = [pf.fit_ring(fit_image, theta, r, pf.Rect(radius - 6, radius + 6),
                            n_terms=2)
                for radius in (60.0, 100.0)]
        reduced = pf.reduce_rings(fits, model="poly", order=1)
        for key, expected in (("A1", 0.02), ("B1", -0.03),
                              ("A2", 0.01), ("B2", 0.015)):
            self.assertAlmostEqual(reduced["a"][key], expected, delta=0.002,
                                   msg=key)

    def test_projected_velocity_shell_recovers_absolute_radius(self):
        radius, width = 25.0, 4.0
        impact = np.linspace(0.0, 120.0, 1201)
        los = np.linspace(0.0, 85.0, 681)
        speed = np.hypot(impact[:, None], los)
        density = np.exp(-0.5 * ((speed - radius) / width) ** 2)
        projected = 2.0 * np.trapezoid(density, los, axis=1)
        y, x = np.mgrid[-80:81, -80:81]
        image = np.interp(np.hypot(x, y), impact, projected)
        polar, theta, r, dr, _ = pf.remap_polar(
            image, center=(80, 80), n_r=161, n_theta=72)

        fit_image, _, _ = pf.process_polar(
            polar, dr, enhancement={"placement": "post_abel", "mode": "bandpass"})
        fit = pf.fit_ring(fit_image, theta, r, pf.Rect(17.0, 33.0),
                          n_terms=2, use_beta=False)
        self.assertTrue(fit.converged)
        self.assertAlmostEqual(fit.params["r0"], radius, delta=0.15)
        self.assertAlmostEqual(fit.params["sigma"], width, delta=0.15)
        for name in ("A1", "B1", "A2", "B2"):
            self.assertAlmostEqual(fit.params[name], 0.0, delta=0.02)

    def test_cli_writes_blocks_and_session(self):
        from contextlib import redirect_stdout
        import io

        from vmi_analysis.circularization import trig_fit

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            image = make_cartesian_rings(noise=0.0)
            image_path = tmp / "rings.npy"
            np.save(image_path, image)
            output = tmp / "extracted.txt"
            session = tmp / "session.json"
            argv = ["--image", str(image_path), "--center", "128", "128",
                    "--n-r", "129", "--n-theta", "360",
                    "--band", "54", "66", "--band", "94", "106",
                    "--n-terms", "2", "--output", str(output),
                    "--session", str(session)]
            with redirect_stdout(io.StringIO()):
                code = trig_fit.main(argv)
            self.assertEqual(code, 0)
            blocks = parse_extracted_txt(output)
            self.assertEqual(len(blocks), 2)
            self.assertEqual(len(pf.session_rings(pf.load_session(session))), 2)


if __name__ == "__main__":
    unittest.main()
