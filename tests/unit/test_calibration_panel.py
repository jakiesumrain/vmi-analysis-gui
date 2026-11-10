import asyncio
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
    def test_handoff_compares_ring_values_without_order_or_export_rounding(self):
        a = pf.RingFit(
            params={"r0": 10.1234567894, "A3": 0.125}, sigmas={"A3": 0.05},
            n_terms=3, use_beta=False, band=pf.Rect(9, 11))
        saved_a = pf.RingFit.from_dict(a.to_dict())
        saved_a.params["r0"] = 10.12345679
        b = pf.RingFit(params={"r0": 20., "B1": 1.}, sigmas={"B1": 0.1},
                       n_terms=1, use_beta=False, band=pf.Rect(19, 21))
        self.assertEqual(calibration._handoff_warnings([a, b], [b, saved_a], 0), [])
        self.assertIn("excludes A3", calibration._handoff_warnings([a, b], [b, saved_a], 1)[0])
        saved_a.params["A3"] = 0.25
        self.assertIn("measurements differ", calibration._handoff_warnings([a, b], [b, saved_a], 0)[0])

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
    async def test_session_restores_pending_fit_only_for_matching_image(self):
        from nicegui import Client, app, ui

        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "raw.npy"
            session_path = Path(tmp) / "session.json"
            calibration_path = Path(tmp) / "extracted.txt"
            pixels = np.zeros((31, 31))
            pixels[15, 15] = 10
            np.save(image_path, pixels)
            options = pf.process_options(abel=False, enhancement={"placement": "off"})
            ring = pf.RingFit(params={"r0": 10, "sigma": 1, "offset": 0, "intensity": 5},
                              sigmas={}, n_terms=0, use_beta=False, band=pf.Rect(8, 12),
                              processing=options, unconstrained=[])
            pf.save_session(session_path, {
                "image": {"path": str(image_path), "center": [15, 15], "brightest_center": True,
                          "radial_bin_size": 1, "n_theta": 24},
                "processing": options, "band": [8, 12, 0, pf.TWO_PI],
                "masks": [[8, 9, 0, 1]], "rings": [ring.to_dict()],
                "output": str(calibration_path),
            })
            callbacks, buttons = {}, {}

            def capture(button, callback):
                callbacks[button.text] = callback
                buttons[button.text] = button

            def fit_ring(*_args, processing, **_kwargs):
                fit = pf.RingFit.from_dict(ring.to_dict())
                fit.processing = processing
                return fit

            client = Client(ui.page("/_test_calibration_session_image"))
            try:
                with client, patch.object(app.storage, "_general", {"calibration.session": str(session_path)}), \
                        patch.object(ui.button, "on_click", capture), patch.object(ui, "notify") as notify, \
                        patch.object(pf, "fit_ring", side_effect=fit_ring), \
                        patch.object(calibration, "_fit_comparison_figure", return_value=None):
                    calibration.build()
                    await callbacks["Load session"]()
                    await callbacks["Fit selection"]()
                    callbacks["Save session"]()
                    saved = pf.load_session(session_path)
                    self.assertTrue(saved["image"]["fingerprint"])
                    accept = buttons["Accept ring (append to file)"]

                    await callbacks["Load session"]()
                    self.assertTrue(accept.enabled)

                    legacy = dict(saved, image={k: v for k, v in saved["image"].items()
                                               if k != "fingerprint"})
                    pf.save_session(session_path, legacy)
                    notify.reset_mock()
                    await callbacks["Load session"]()
                    self.assertFalse(accept.enabled)
                    self.assertTrue(any("no image fingerprint" in call.args[0]
                                        for call in notify.call_args_list))

                    pf.save_session(session_path, saved)
                    pixels[15, 24] = 2  # Same path, shape, and brightest pixel.
                    np.save(image_path, pixels)
                    notify.reset_mock()
                    await callbacks["Load session"]()
                    self.assertFalse(accept.enabled)
                    self.assertTrue(any("detector image changed" in call.args[0]
                                        for call in notify.call_args_list))
                    self.assertTrue(any(isinstance(el, ui.label) and el.text == "Masks: 1"
                                        for el in client.elements.values()))
                    self.assertTrue(any(isinstance(el, ui.table) and el.rows
                                        and el.rows[0].get("radius") == "10.0000"
                                        for el in client.elements.values()))
                    pixels[15, 16] = 20  # The replacement also moved the brightest pixel.
                    np.save(image_path, pixels)
                    await callbacks["Load session"]()
                    self.assertFalse(accept.enabled)
                    await callbacks["Fit selection"]()
                    self.assertTrue(accept.enabled)
                    callbacks["Save session"]()
                    self.assertEqual(pf.load_session(session_path)["image"]["center"], [16, 15])

                    original = f'{pf.EXTRACTED_HEADER}\n"Gaussian Centre" 5 +- 0.1\n'
                    calibration_path.write_text(original, encoding="utf-8")
                    session_input = next(el for el in client.elements.values()
                                         if isinstance(el, ui.input)
                                         and el._props.get("label") == "Session file (.json)")
                    session_input.value = str(calibration_path)
                    callbacks["Save session"]()
                    self.assertIn("ending in .json", notify.call_args.args[0])
                    self.assertEqual(calibration_path.read_text(encoding="utf-8"), original)
                    notify.reset_mock()
                    await callbacks["Accept ring (append to file)"]()
                    self.assertTrue(any("ending in .json" in call.args[0] for call in notify.call_args_list))
                    self.assertEqual(pf.existing_radii(calibration_path), [5, 10])
            finally:
                client.delete()

    async def test_rebuild_confirms_removed_and_replaced_rings_and_rechecks_file(self):
        from nicegui import Client, app, ui

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "extracted.txt"
            ring = f'{pf.EXTRACTED_HEADER}\n"Gaussian Centre" 20 +- 0.1\n'
            old = (f'{pf.EXTRACTED_HEADER}\n"Gaussian Centre" 10 +- 0.1\n\n'
                   + ring.replace("0.1", "0.2"))
            path.write_text(ring, encoding="utf-8")
            callbacks = {}

            def capture(button, callback):
                callbacks[button.text] = callback

            client = Client(ui.page("/_test_calibration_rebuild"))
            try:
                with client, patch.object(app.storage, "_general", {"calibration.output": str(path)}), \
                        patch.object(ui.button, "on_click", capture), patch.object(ui, "notify") as notify:
                    calibration.build()
                    await callbacks["Refresh rings from file"]()
                    path.write_text(old, encoding="utf-8")
                    callbacks["Rebuild file from rings"]()
                    labels = [el.text for el in client.elements.values() if isinstance(el, ui.label)]
                    self.assertIn("Will remove rings at 10 px.", labels)
                    self.assertIn("Will replace rings at 20 px.", labels)
                    self.assertEqual(path.read_text(encoding="utf-8"), old)
                    callbacks["Cancel"]()
                    self.assertEqual(path.read_text(encoding="utf-8"), old)

                    callbacks["Rebuild file from rings"]()
                    changed = old + "\n# external edit\n"
                    path.write_text(changed, encoding="utf-8")
                    callbacks["Rebuild file"]()
                    self.assertEqual(path.read_text(encoding="utf-8"), changed)
                    self.assertIn("changed", notify.call_args.args[0])

                    callbacks["Rebuild file from rings"]()
                    callbacks["Rebuild file"]()
                    self.assertEqual(path.read_text(encoding="utf-8"), ring)
            finally:
                client.delete()

    async def test_latest_remap_and_processing_request_wins(self):
        from nicegui import Client, app, ui

        with TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f"raw-{i}.npy" for i in range(3)]
            for index, (path, size) in enumerate(zip(paths, (21, 31, 31)), start=1):
                image = np.zeros((size, size))
                image[size // 2, size // 2] = index
                np.save(path, image)
            store = {"calibration.image": str(paths[0]), "calibration.abel": False,
                     "calibration.placement": "off", "calibration.n_theta": 24}
            callbacks, buttons = {}, {}

            def capture(button, callback):
                callbacks[button.text] = callback
                buttons[button.text] = button

            client = Client(ui.page("/_test_calibration_requests"))
            try:
                with client, patch.object(app.storage, "_general", store), patch.object(ui, "notify") as notify:
                    with patch.object(ui.button, "on_click", capture):
                        calibration.build()
                    image_input = next(el for el in client.elements.values()
                                       if isinstance(el, ui.input) and el._props.get("label") == "VMI image (.npy)")
                    mode_input = next(el for el in client.elements.values()
                                      if isinstance(el, ui.select) and el.label == "Enhancement")
                    fit = buttons["Fit selection"]
                    apply = buttons["Apply processing"]

                    async def run_overlapping(button, change, *, stale_failure=False):
                        started, release = asyncio.Event(), asyncio.Event()
                        requests = []

                        async def dispatch(function, *args, **kwargs):
                            if function is pf.process_polar:
                                requests.append(kwargs["enhancement"]["mode"])
                                if len(requests) == 1:
                                    started.set()
                                    await release.wait()
                                    if stale_failure:
                                        raise ValueError("stale processing failure")
                            return function(*args, **kwargs)

                        with patch.object(asyncio, "to_thread", side_effect=dispatch), \
                                patch.object(calibration, "_render_polar_image", wraps=calibration._render_polar_image) as render:
                            first = asyncio.create_task(callbacks[button]())
                            try:
                                await asyncio.wait_for(started.wait(), timeout=5)
                                self.assertFalse(fit.enabled)
                                self.assertEqual(apply.enabled, button == "Apply processing")
                                change()
                                await callbacks[button]()
                                self.assertTrue(fit.enabled and apply.enabled)
                                self.assertEqual(render.call_count, 2)
                                latest = render.call_args.args[0].copy()
                            finally:
                                release.set()
                                await first
                            self.assertEqual(render.call_count, 2)
                        return latest, requests

                    for first, second in ((paths[0], paths[1]), (paths[1], paths[2])):
                        image_input.value = str(first)
                        latest, _ = await run_overlapping(
                            "Load image & remap", lambda: setattr(image_input, "value", str(second)))
                        self.assertEqual(latest.shape, (24, 16))
                        self.assertEqual(latest[0, 0], 2 if second == paths[1] else 3)

                    placement = next(el for el in client.elements.values()
                                     if isinstance(el, ui.select) and el.label == "Filter placement")
                    placement.value = "raw"
                    raw, *_ = pf.remap_polar(np.load(paths[2]), (15, 15), n_theta=24, radial_bin_size=1)
                    expected = pf.enhance_polar(raw, "bandpass")
                    for stale_failure in (False, True):
                        mode_input.value = "ridge"
                        latest, requests = await run_overlapping(
                            "Apply processing", lambda: setattr(mode_input, "value", "bandpass"),
                            stale_failure=stale_failure)
                        self.assertEqual(requests, ["ridge", "bandpass"])
                        np.testing.assert_array_equal(latest, expected)
                    notify.assert_not_called()
            finally:
                client.delete()

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
                     "calibration.interp_order": 5, "circularize.interp_order": 1,
                     "circularize.max_trig_n": 1}
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
                               if isinstance(element, ui.interactive_image) and element.source]
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
                    Path(store["calibration.output"]).write_text(
                        f'{pf.EXTRACTED_HEADER}\n"Gaussian Centre" 12 +- 0.1\n'
                        '"Trig Term sin(3*theta)" 0.2 +- 0.1\n', encoding="utf-8")
                    with patch.object(ui, "notify") as notify:
                        callbacks["Send to Circularize"]()
                    message = notify.call_args.args[0]
                    self.assertIn("preview has 0 ring(s) and the saved file has 1", message)
                    self.assertIn("Max trig n = 1 excludes A3", message)
                    self.assertEqual(store["circularize.inp"], str(path))
                    output = next(element for element in client.elements.values()
                                  if isinstance(element, ui.input)
                                  and element._props.get("label") == "Output image (.txt)")
                    original_output = output.value
                    output.value = str(path)
                    with patch.object(ui, "notify") as notify:
                        callbacks["Run"]()
                    manager.submit.assert_not_called()
                    self.assertIn("input image", notify.call_args.args[0])
                    output.value = original_output
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
                    callbacks["Send to Circularize"]()
                    with patch.object(circularize.JobView, "attach"):
                        callbacks["Run"]()
                    cmd = manager.submit.call_args.args[0].cmd
                    self.assertEqual(cmd[cmd.index("--center") + 1:cmd.index("--center") + 3], ["8.5", "5.5"])
                    self.assertTrue({"--shape", "--polar-dr"}.isdisjoint(cmd))
            finally:
                client.delete()


if __name__ == "__main__":
    unittest.main()
