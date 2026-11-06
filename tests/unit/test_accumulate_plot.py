import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from vmi_analysis.gui.accumulate_plot import accumulation_figure


class AccumulatePlotTests(unittest.TestCase):
    def test_dropdown_sends_explicit_color_stops_for_every_map(self):
        from nicegui import Client, ui
        from plotly.colors import get_colorscale
        from vmi_analysis.gui.accumulate_plot import COLORMAPS
        from vmi_analysis.gui.panels.accumulate import _interactive_plot
        client = Client(ui.page("/_test_accumulate_colors"))
        try:
            with client, patch("vmi_analysis.gui.panels.accumulate.persist", side_effect=lambda widget, key: widget):
                plot, colors = _interactive_plot(accumulation_figure([[1, 10], [100, 1000]]))
            with patch.object(plot, "run_plot_method") as restyle:
                for name in (*COLORMAPS[1:], COLORMAPS[0]):
                    colors.value = name
                    expected = get_colorscale(name)
                    restyle.assert_called_with("restyle", {"colorscale": [expected]}, [0])
                    self.assertEqual(plot.figure["data"][0]["colorscale"], expected)
        finally:
            client.delete()

    def test_original_counts_orientation_and_script_log_range(self):
        values = np.array([[0., 1., 10.], [-1., 100., np.nan]])
        fig = accumulation_figure(values, revision="job1")
        trace = fig.data[0]
        np.testing.assert_array_equal(trace.customdata, values)
        np.testing.assert_allclose(trace.z[0, 1:], [0, 1])
        self.assertTrue(np.isnan(trace.z[0, 0]))
        self.assertTrue(np.isnan(trace.z[1, 0]))
        high = np.log10(np.percentile(values[np.isfinite(values)], 99.999999))
        self.assertAlmostEqual(trace.zmax, high)
        self.assertAlmostEqual(trace.zmin, high - 2)
        self.assertEqual(fig.layout.yaxis.scaleanchor, "x")
        self.assertEqual(list(fig.layout.yaxis.range), [-.5, 1.5])
        self.assertEqual(fig.layout.uirevision, "job1")
        fig.to_json()

    def test_empty_signal_is_serializable_and_bad_shape_rejected(self):
        accumulation_figure(np.zeros((3, 3))).to_json()
        accumulation_figure(np.full((3, 3), np.nan)).to_json()
        with self.assertRaises(ValueError):
            accumulation_figure(np.array([1, 2]))

    def test_saved_text_result_builds_interactive_plot_and_png_fallback(self):
        from nicegui import Client, ui
        from vmi_analysis.gui.panels.accumulate import _result_file
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "result.txt"
            np.savetxt(path, [[0, 1], [10, 100]])
            path.with_suffix(".png").write_bytes(b"placeholder")
            client = Client(ui.page("/_test_accumulate"))
            try:
                with client:
                    _result_file(str(path))
                plots = [e for e in client.elements.values() if e.__class__.__name__ == "Plotly"]
                self.assertEqual(len(plots), 1)
                self.assertTrue(any(e.__class__.__name__ == "Expansion" for e in client.elements.values()))
            finally:
                client.delete()
            path.unlink()
            client = Client(ui.page("/_test_accumulate_png"))
            try:
                with client:
                    _result_file(str(path))
                self.assertFalse(any(e.__class__.__name__ == "Plotly" for e in client.elements.values()))
                self.assertTrue(any(e.__class__.__name__ == "Image" for e in client.elements.values()))
            finally:
                client.delete()


if __name__ == "__main__":
    unittest.main()
