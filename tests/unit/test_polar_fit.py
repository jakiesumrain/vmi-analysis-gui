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
