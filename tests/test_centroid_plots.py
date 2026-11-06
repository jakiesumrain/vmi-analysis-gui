"""Interactive figures use exported numerical values, not PNG pixels."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from centroiding.plot_data import save_plot_data, load_plot_data, SUMMARY_DATA, TUNING_DATA
from gui.centroid_plots import summary_figure, tuning_figure, plot_options, _compact_values


def summary_data(path, *, alpha=.001, abel=True, spectrum=True):
    meta = {"n_events": 100, "n_frames": 10, "box_size": 5, "min_net_gradient": 400.,
            "alpha": alpha, "x_min": -1, "x_max": 150, "y_min": -1, "y_max": 20}
    raw = np.array([[0., 1., 10.], [100., -2., np.nan]])
    inverse = np.array([[.1, .01, -1.], [.001, 0., 10.]]) if abel else np.empty((0, 0))
    energy = np.array([1., 2., 3.]) if spectrum else np.array([])
    intensity = np.array([5., 10., 8.]) if spectrum else np.array([])
    save_plot_data(path, meta, raw=raw, abel=inverse, energy=energy, intensity=intensity)
    return meta, raw, inverse, energy, intensity


def tuning_data(path, *, empty=False):
    size = 0 if empty else 3
    values = np.arange(size, dtype=float)
    frame = np.arange(24, dtype=float).reshape(4, 6) - 2
    save_plot_data(path, {"n_events": size, "frame_index": 50, "fitting_method": "mle",
                          "min_net_gradient": 400, "bins_photons": 4,
                          "bins_net_gradient": 3, "bins_precision": 2},
                   frame=frame, x=values + .25, y=values + .5,
                   photons=values + 10, net_gradient=values + 400,
                   lpx=values * .01 + .1, lpy=values * .01 + .2)
    return frame, values


class CentroidPlotTests(unittest.TestCase):
    def test_colormaps_target_only_the_selected_image_trace(self):
        from nicegui import Client, ui
        from plotly.colors import get_colorscale
        from gui.centroid_plots import COLORMAPS
        from gui.panels.centroid import _interactive_plot
        with tempfile.TemporaryDirectory() as folder:
            for tuning, filename, builder, writer in (
                (True, TUNING_DATA, tuning_figure, tuning_data),
                (False, SUMMARY_DATA, summary_figure, summary_data),
            ):
                path = Path(folder) / filename
                writer(path)
                figure = builder(path)
                client = Client(ui.page("/_test_centroid_colors"))
                try:
                    with client, patch("gui.panels.centroid.persist", side_effect=lambda w, key: w):
                        plot = _interactive_plot(figure, tuning=tuning)
                    selects = [e for e in client.elements.values() if e.__class__.__name__ == "Select"]
                    self.assertEqual(len(selects), 1 if tuning else 2)
                    for index, select in enumerate(selects):
                        with patch.object(plot, "run_plot_method") as restyle:
                            for name in COLORMAPS:
                                if select.value == name:
                                    continue
                                select.value = name
                                expected = get_colorscale(name)
                                restyle.assert_called_with("restyle", {"colorscale": [expected]}, [index])
                                self.assertEqual(plot.figure["data"][index]["colorscale"], expected)
                    # Localization markers, histogram colors and spectrum stay intact.
                    self.assertEqual(figure.data[1].marker.color if tuning else figure.data[-1].line.color,
                                     "red" if tuning else "#c64545")
                finally:
                    client.delete()

    def test_compact_encoding_is_lossless_and_does_not_downsample(self):
        for data in (np.arange(12).reshape(3, 4).astype(float),
                     np.array([[.1, .2], [np.nan, 1e100]]),
                     np.array([[.5, np.nan], [.25, 1.]])):
            with self.subTest(shape=data.shape):
                compact = _compact_values(data)
                self.assertEqual(compact.shape, data.shape)
                np.testing.assert_array_equal(compact, data)
        self.assertEqual(_compact_values(np.arange(10.)).dtype, np.int16)
        self.assertEqual(_compact_values(np.array([.1])).dtype, np.float64)

    def test_summary_matches_counts_log_colors_pixel_orientation_and_spectrum(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / SUMMARY_DATA
            meta, raw, inverse, energy, intensity = summary_data(path)
            fig = summary_figure(path)
            np.testing.assert_array_equal(fig.data[0].customdata, raw)
            np.testing.assert_array_equal(fig.data[1].customdata, inverse)
            self.assertTrue(np.isnan(fig.data[0].z[0, 0]))
            self.assertTrue(np.isnan(fig.data[0].z[1, 1]))
            self.assertAlmostEqual(fig.data[0].z[1, 0], 2)
            self.assertAlmostEqual(fig.data[0].zmin, -2)
            self.assertAlmostEqual(fig.data[0].zmax, 2)
            self.assertEqual(fig.layout.yaxis.scaleanchor, "x")
            self.assertEqual(fig.layout.yaxis2.scaleanchor, "x2")
            self.assertEqual(tuple(fig.layout.yaxis.range), (-.5, 1.5))
            np.testing.assert_array_equal(fig.data[2].x, energy)
            np.testing.assert_array_equal(fig.data[2].y, intensity)
            self.assertEqual(fig.layout.xaxis3.title.text, "Kinetic energy (eV)")
            self.assertEqual(fig.layout.yaxis3.title.text, "Intensity (arb. units)")
            self.assertEqual(tuple(fig.layout.yaxis3.range), (0, 20))
            info = fig.layout.annotations[-1]
            self.assertEqual(info.xref, "paper")
            self.assertGreater(info.x, .5)  # stays in the right-hand summary panel
            self.assertLess(info.y, .5)
            self.assertTrue(plot_options(fig)["config"]["scrollZoom"])
            self.assertIn("Counts: %{customdata", fig.data[0].hovertemplate)

    def test_absent_abel_spectrum_and_uncalibrated_units(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / SUMMARY_DATA
            summary_data(path, alpha=None, abel=False, spectrum=False)
            fig = summary_figure(path)
            self.assertEqual(len(fig.data), 1)
            self.assertEqual(fig.layout.xaxis3.title.text, "Squared radius (pixel²)")
            self.assertEqual(len([a for a in fig.layout.annotations if "not available" in a.text]), 2)
            fig.to_json()  # empty arrays and masked values are serializable

    def test_tuning_frame_markers_and_histograms_match_exported_values(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / TUNING_DATA
            frame, values = tuning_data(path)
            fig = tuning_figure(path)
            np.testing.assert_array_equal(fig.data[0].z, frame)
            np.testing.assert_array_equal(fig.data[1].x, values + .25)
            np.testing.assert_array_equal(fig.data[1].y, values + .5)
            np.testing.assert_array_equal(fig.data[1].customdata[:, 0], values + 10)
            np.testing.assert_array_equal(fig.data[1].customdata[:, 1], values + 400)
            np.testing.assert_array_equal(fig.data[2].y, np.histogram(values + 10, bins=4)[0])
            self.assertEqual(sum(fig.data[3].y), 3)
            self.assertEqual(sum(fig.data[4].y), 3)
            self.assertEqual(fig.layout.yaxis.scaleanchor, "x")
            self.assertIn(400, [shape.x0 for shape in fig.layout.shapes if shape.type == "line"])

    def test_empty_tuning_and_pickle_free_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / TUNING_DATA
            frame, _ = tuning_data(path, empty=True)
            meta, arrays = load_plot_data(path)
            self.assertEqual(meta["n_events"], 0)
            np.testing.assert_array_equal(arrays["frame"], frame)
            with np.load(path, allow_pickle=False) as data:
                self.assertFalse(any(data[key].dtype.hasobject for key in data.files))
            fig = tuning_figure(path)
            self.assertEqual(sum(fig.data[2].y), 0)
            fig.to_json()

    def test_panel_shows_both_interactive_outputs_and_png_fallback(self):
        from nicegui import Client, ui
        from gui.panels.centroid import _result_files
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            summary_data(output / SUMMARY_DATA)
            tuning_data(output / TUNING_DATA)
            client = Client(ui.page("/_test_centroid_plots"))
            try:
                with client:
                    _result_files(output)
                plots = [element for element in client.elements.values() if element.__class__.__name__ == "Plotly"]
                self.assertEqual(len(plots), 2)
            finally:
                client.delete()
            (output / SUMMARY_DATA).unlink()
            (output / TUNING_DATA).unlink()
            (output / "04_final_summary.png").write_bytes(b"old PNG placeholder")
            client = Client(ui.page("/_test_legacy_plots"))
            try:
                with client:
                    _result_files(output)
                self.assertTrue(any("Saved PNG only" in str(getattr(e, "text", "")) for e in client.elements.values()))
                self.assertFalse(any(e.__class__.__name__ == "Plotly" for e in client.elements.values()))
            finally:
                client.delete()


if __name__ == "__main__":
    unittest.main()
