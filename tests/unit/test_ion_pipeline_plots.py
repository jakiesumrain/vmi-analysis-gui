import unittest

from vmi_analysis.gui.panels.ion_pipeline import ratio_figure, yield_figure


def measurement(folder, peak, delay, value, status="OK"):
    return dict(folder=folder, peak=peak, delay_mm=str(delay), hits_per_frame=str(value),
                status=status, counts="10", selected_frames="5", poisson_error="0.6")


class IonPipelinePlotTests(unittest.TestCase):
    def test_ratio_pairs_by_folder_and_keeps_undefined_gaps(self):
        rows = [measurement("run-b", "A", 10, 8), measurement("run-a", "B", 10, 2),
                measurement("run-a", "A", 10, 6), measurement("run-b", "B", 10, 4),
                measurement("zero", "A", 20, 1), measurement("zero", "B", 20, 0),
                measurement("missing", "A", 30, 1),
                measurement("failed", "A", 40, 1), measurement("failed", "B", 40, 2, "Fit failed"),
                measurement("zero-numerator", "A", 50, 0), measurement("zero-numerator", "B", 50, 2)]
        trace = ratio_figure(rows, "A", "B").data[0]
        self.assertEqual(dict(zip(trace.customdata, trace.y)),
                         {"run-a": 3, "run-b": 2, "zero": None, "missing": None,
                          "failed": None, "zero-numerator": 0})
        self.assertFalse(trace.connectgaps)
        reverse = ratio_figure(rows[:4], "B", "A").data[0]
        self.assertEqual(dict(zip(reverse.customdata, reverse.y)), {"run-a": 1/3, "run-b": 0.5})

    def test_comparison_has_independent_scales_and_converted_delay_ticks(self):
        rows = [measurement("a", "A", 10, 2), measurement("a", "B", 10, 0),
                measurement("b", "B", 20, "", "Fit failed")]
        options = {"peaks": [{"name": "A"}, {"name": "B"}], "source": "pixels"}
        combined = yield_figure(rows, options, ["A", "B"])
        single = yield_figure(rows, options, ["B"])
        self.assertEqual(tuple(single.data[0].y), (0, None))
        self.assertEqual(tuple(single.data[0].error_y.array), (0.6, 0.6))
        self.assertEqual((combined.data[0].yaxis, combined.data[1].yaxis), ("y", "y2"))
        self.assertEqual(combined.layout.yaxis2.side, "right")
        self.assertEqual(combined.layout.xaxis.side, "top")
        self.assertEqual(combined.layout.xaxis2.side, "bottom")
        for mm, ps in zip(combined.layout.xaxis2.tickvals, combined.layout.xaxis2.ticktext, strict=True):
            self.assertAlmostEqual(mm, float(ps) * 0.15)
        logarithmic = yield_figure(rows, options, ["A", "B"], ["linear", "log"])
        self.assertEqual((logarithmic.layout.yaxis.type, logarithmic.layout.yaxis2.type), ("linear", "log"))
        self.assertEqual(tuple(logarithmic.data[1].y), (None, None))


if __name__ == "__main__":
    unittest.main()
