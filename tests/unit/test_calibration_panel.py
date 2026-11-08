import unittest

import numpy as np

from vmi_analysis.circularization import polar_fit as pf
from vmi_analysis.gui.panels import calibration


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
    def test_pixel_data_round_trip(self):
        state = make_state()
        r, theta = calibration._px_to_data(state, 50, 90)
        self.assertAlmostEqual(r, 50.0)
        self.assertAlmostEqual(theta, np.pi / 2)
        x, y = calibration._data_to_px(state, r, theta)
        self.assertAlmostEqual(x, 50.0)
        self.assertAlmostEqual(y, 90.0)

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

    def test_render_png_has_one_pixel_per_bin(self):
        import tempfile
        from pathlib import Path

        polar = np.random.default_rng(0).random((360, 100))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "polar.png"
            calibration._render_polar_png(polar, path)
            from PIL import Image
            with Image.open(path) as image:
                self.assertEqual(image.size, (100, 360))


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


if __name__ == "__main__":
    unittest.main()
