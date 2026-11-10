from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

from vmi_analysis.circularization import polar_fit as pf
from vmi_analysis.gui.panels import calibration, circularize
from vmi_analysis.imaging.centering import brightest_pixel


def make_state():
    return {
        "view": {"r0": 0.0, "r1": 100.0, "theta0": 0.0, "theta1": pf.TWO_PI},
        "img_size": (100, 360),
        "band": pf.Rect(10.0, 20.0),
        "masks": [pf.Rect(30.0, 40.0, 0.0, np.pi)],
        "drag_rect": None,
        "last_fit": None,
    }


class PanelHelperTests(unittest.TestCase):
    def test_radial_plots_follow_both_models_with_unsorted_rings(self):
        fits = [pf.RingFit(
            params={"r0": radius, "A1": radius / 100, "B1": -radius / 200},
            sigmas={"A1": 0.1, "B1": 0.2}, n_terms=1, use_beta=False,
            band=pf.Rect(radius - 5, radius + 5))
            for radius in (100, 50)]
        for mode in ("interp", "poly"):
            with self.subTest(mode=mode):
                reduced = pf.reduce_rings(fits, model=mode, order=1)
                a, b = calibration._radial_model_figures(reduced)
                for figure, slope, sigma in ((a, 0.01, 0.1), (b, -0.005, 0.2)):
                    curve, points = figure.data
                    np.testing.assert_allclose(curve.y, np.asarray(curve.x) * slope)
                    np.testing.assert_array_equal(points.x, [100, 50])
                    np.testing.assert_array_equal(points.error_y.array, [sigma, sigma])

    def test_fit_comparison_uses_shared_scale_and_marks_excluded_pixels(self):
        theta = np.linspace(0, pf.TWO_PI, 36, endpoint=False)
        r = np.arange(24, dtype=float)
        values = np.tile(np.exp(-0.5 * ((r - 12) / 2) ** 2), (theta.size, 1))
        fit = pf.fit_ring(values, theta, r, pf.Rect(7, 17),
                          masks=[pf.Rect(9, 11, 0, 0.5)], n_terms=0, use_beta=False)
        figure = calibration._fit_comparison_figure(values, theta, r, fit)
        data, model, ridge = figure.data
        self.assertEqual(data.coloraxis, model.coloraxis)
        np.testing.assert_array_equal(data.x, model.x)
        np.testing.assert_array_equal(data.y, model.y)
        np.testing.assert_array_equal(np.isnan(data.z), np.isnan(model.z))
        self.assertTrue(np.isnan(data.z[0, 2]))
        self.assertEqual(ridge.type, "scatter")
        self.assertEqual(figure.layout.xaxis2.matches, "x")

    def test_brightest_pixel_uses_row_column_and_first_finite_maximum(self):
        image = np.array([[np.nan, 1, 8], [8, np.inf, 1]])
        self.assertEqual(brightest_pixel(image), (0, 2))
        with self.assertRaisesRegex(ValueError, "no finite pixels"):
            brightest_pixel(np.full((2, 3), np.nan))

    def test_pixel_data_round_trip(self):
        state = make_state()
        r, theta = calibration._px_to_data(state, 50, 90)
        self.assertAlmostEqual(r, 50.0)
        self.assertAlmostEqual(theta, np.pi / 2)
        x, y = calibration._data_to_px(state, r, theta)
        self.assertAlmostEqual(x, 50.0)
        self.assertAlmostEqual(y, 90.0)

    def test_drag_coordinates_clamp_to_view_edges(self):
        state = make_state()
        self.assertEqual(calibration._px_to_data(state, -5, 375), (0.0, pf.TWO_PI))
        self.assertEqual(calibration._px_to_data(state, 105, -20), (100.0, 0.0))

    def test_pointer_reads_displayed_samples_after_crop(self):
        raw = np.arange(20).reshape(4, 5)
        state = {"raw": raw, "processed": raw * 2, "r": np.arange(5) / 2,
                 "theta": np.deg2rad([0, 90, 180, 270]),
                 "view_indices": (np.array([1, 2]), np.array([2, 3, 4]))}
        self.assertEqual(calibration._polar_sample(state, "raw", 1.9, 0.9), (1.5, 90, 8))
        self.assertEqual(calibration._polar_sample(state, "processed", 1.9, 0.9), (1.5, 90, 16))
        self.assertIsNone(calibration._polar_sample(state, "raw", 3, 0.9))

    def test_overlay_draws_band_mask_and_ridge(self):
        state = make_state()
        band_only = calibration._overlay_svg(state)
        self.assertIn("<rect", band_only)
        state["last_fit"] = pf.RingFit(
            params={"r0": 50.0, "A1": 1.0, "B1": 0.0},
            sigmas={}, n_terms=1, use_beta=False, band=pf.Rect(45, 55))
        svg = calibration._overlay_svg(state)
        self.assertEqual(svg.count("<rect"), 2)
        self.assertIn("<path", svg)

    def test_render_image_has_one_pixel_per_bin(self):
        polar = np.random.default_rng(0).random((360, 100))
        self.assertEqual(calibration._render_polar_image(polar).size, (100, 360))

    def test_percentile_limits_apply_linear_or_log_mapping(self):
        data = np.array([[-1., 1., 10., 100., np.nan]])
        linear = np.array(calibration._render_polar_image(data, "Gray", vmin_pct=0, vmax_pct=100))[0, :, 0]
        log = np.array(calibration._render_polar_image(data, "Gray", log_scale=True,
                                                      vmin_pct=0, vmax_pct=100))[0, :, 0]
        self.assertLess(linear[2], 30)
        self.assertGreater(log[2], 120)
        self.assertEqual(log[[0, 1, 3, 4]].tolist(), [0, 0, 255, 0])
        for lower, upper in ((-1, 100), (0, 101), (90, 10), (np.nan, 100)):
            with self.subTest(lower=lower, upper=upper), \
                    self.assertRaisesRegex(ValueError, "Enter percentiles"):
                calibration._render_polar_image(data, vmin_pct=lower, vmax_pct=upper)

    def test_percentile_limits_follow_each_images_intensity_distribution(self):
        data = np.array([[0, 25, 50, 75, 100]])
        for values in (data, 3 * data - 100):
            image = calibration._render_polar_image(values, "Gray", vmin_pct=25, vmax_pct=75)
            self.assertEqual(np.array(image)[0, :, 0].tolist(), [0, 0, 128, 255, 255])


class PanelBuildTests(unittest.TestCase):
    def test_panel_builds(self):
        from nicegui import Client, ui

        client = Client(ui.page("/_test_calibration_panel"))
        try:
            with client:
                calibration.build(None)
            names = {element.__class__.__name__
                     for element in client.elements.values()}
            self.assertIn("InteractiveImage", names)
            self.assertIn("Toggle", names)
            self.assertIn("Table", names)
        finally:
            client.delete()


class PanelLoadTests(unittest.IsolatedAsyncioTestCase):
    async def test_load_uses_brightest_pixel_and_logs_circularize_coordinates(self):
        from nicegui import Client, app, ui

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "vmi_2d_raw.npy"
            image = np.zeros((12, 16))
            image[4, 10] = 30
            np.save(path, image)
            store = {"calibration.image": str(path), "calibration.abel": False,
                     "calibration.rotation": 90, "calibration.brightest_center": True,
                     "calibration.output": str(Path(tmp) / "extracted.txt"),
                     "calibration.radial_bin_size": 0.5, "calibration.n_theta": 24,
                     "calibration.radial_model": "poly", "calibration.slope_order": 2,
                     "calibration.interp_order": 5, "circularize.interp_order": 1}
            callbacks = {}
            manager = Mock()
            manager.jobs.return_value = []

            def capture(button, callback):
                callbacks[button.text] = callback

            client = Client(ui.page("/_test_calibration_brightest"))
            try:
                with client, patch.object(app.storage, "_general", store):
                    with patch.object(ui.button, "on_click", capture):
                        use_calibration = circularize.build(manager)
                        calibration.build(use_calibration=use_calibration)
                    with patch.object(pf, "remap_polar", wraps=pf.remap_polar) as remap:
                        await callbacks["Load image & remap"]()
                    sources = [element.source for element in client.elements.values()
                               if isinstance(element, ui.interactive_image)]
                    self.assertEqual(len(sources), 2)
                    self.assertTrue(all(source.startswith("data:image/png;base64,") for source in sources))
                    np.testing.assert_array_equal(remap.call_args.args[0], image)
                    self.assertEqual(remap.call_args.args[1], (10, 4))
                    self.assertEqual(remap.call_args.kwargs["radial_bin_size"], 0.5)
                    lines = [line.text for element in client.elements.values()
                             if isinstance(element, ui.log) for line in element.default_slot.children]
                    self.assertTrue(any("row=4, col=10" in line for line in lines), lines)
                    self.assertTrue(any("Center X=10, Center Y=4" in line for line in lines), lines)
                    self.assertTrue(any("9 radial samples from 0 to 4 px at 0.5 px spacing" in line
                                        for line in lines), lines)
                    callbacks["Export to Circularize"]()
                    self.assertEqual(store["circularize.inp"], str(path))
                    with patch.object(circularize.JobView, "attach"):
                        callbacks["Run"]()
                    cmd = manager.submit.call_args.args[0].cmd
                    self.assertEqual(cmd[cmd.index("--input") + 1], str(path))
                    self.assertEqual(cmd[cmd.index("--extracted") + 1], store["calibration.output"])
                    for flag, value in (("--radial-model", "poly"), ("--poly-order", "2"), ("--interp-order", "1")):
                        self.assertEqual(cmd[cmd.index(flag) + 1], value)
                    self.assertTrue({"--shape", "--center", "--polar-dr"}.isdisjoint(cmd))
                    model = next(element for element in client.elements.values()
                                 if isinstance(element, ui.select) and element.label == "Radial model")
                    poly = next(element for element in client.elements.values()
                                if isinstance(element, ui.number) and element._props.get("label") == "Poly order")
                    self.assertTrue(poly.enabled)
                    model.value = "interp"
                    self.assertFalse(poly.enabled)
                    with patch.object(circularize.JobView, "attach"):
                        callbacks["Run"]()
                    cmd = manager.submit.call_args.args[0].cmd
                    self.assertNotIn("--poly-order", cmd)
                    self.assertEqual(cmd[cmd.index("--interp-order") + 1], "1")
                    model.value = "poly"
                    self.assertTrue(poly.enabled)
                    self.assertEqual(poly.value, 2)

                    auto = [element for element in client.elements.values()
                            if isinstance(element, ui.checkbox)
                            and element.text == "Use brightest pixel as center"][-1]
                    x, y = [element for element in client.elements.values()
                            if isinstance(element, ui.number)
                            and element._props.get("label") in ("Centre X", "Centre Y")]
                    self.assertTrue(auto.enabled)
                    self.assertFalse(x.enabled or y.enabled)
                    auto.value = False
                    self.assertTrue(x.enabled and y.enabled)
                    x.value, y.value = 8.5, 5.5
                    with patch.object(pf, "remap_polar", wraps=pf.remap_polar) as remap:
                        await callbacks["Load image & remap"]()
                    self.assertEqual(remap.call_args.args[1], (8.5, 5.5))
                    callbacks["Export to Circularize"]()
                    with patch.object(circularize.JobView, "attach"):
                        callbacks["Run"]()
                    cmd = manager.submit.call_args.args[0].cmd
                    self.assertEqual(cmd[cmd.index("--center") + 1:cmd.index("--center") + 3], ["8.5", "5.5"])
                    self.assertTrue({"--shape", "--polar-dr"}.isdisjoint(cmd))
            finally:
                client.delete()


if __name__ == "__main__":
    unittest.main()
