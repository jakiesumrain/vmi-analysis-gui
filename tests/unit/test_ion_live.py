import struct
import unittest

import numpy as np

from vmi_analysis.ion_live import LivePreview, decode_message, gmd_histogram


class LiveTpx3Tests(unittest.TestCase):
    def test_gmd_bins_have_exact_limits_and_ignore_nonfinite_and_outside_readings(self):
        readings = np.array([0.5, 1, 1.5, 2.5, 3.4, 3.5, np.nan])
        counts, edges = gmd_histogram(readings, 1, 3.4, 1, "GMD1")
        np.testing.assert_allclose(edges, [1, 2, 3, 3.4])
        np.testing.assert_array_equal(counts, [2, 1, 1])
        counts, edges = gmd_histogram(readings, 5, None, 0.5, "GMD2")
        np.testing.assert_allclose(edges, [5, 5.5])
        np.testing.assert_array_equal(counts, [0])
        with self.assertRaisesRegex(ValueError, "GMD2 limits"):
            gmd_histogram(readings, 4, 2, 0.5, "GMD2")

    def test_decode_acquisition_prefix_pixel_position_and_trigger_relative_time(self):
        pixel = (0xB << 60) | (8 << 52) | (8 << 45) | (7 << 44) | (440 << 30) | (15 << 16)
        trigger = (0x6F << 56) | (40 << 12) | (1 << 5)
        payload = struct.pack("<qqddQQ", 1700000000, 125, 3e-9, 6e-9, trigger, pixel)
        frame = decode_message(payload)
        self.assertEqual(frame.timestamp_ns, 1700000000000000125)
        np.testing.assert_allclose([frame.gmd1_na, frame.gmd2_na], [3, 6])
        np.testing.assert_array_equal(frame.pixels, [11 * 256 + 9])
        np.testing.assert_allclose(frame.tof_ns, [10_000])
        with self.assertRaisesRegex(ValueError, "message size"):
            decode_message(payload[:-1])

    def test_preview_retains_gmd_for_frames_without_a_trigger_and_resets(self):
        preview = LivePreview("tcp://localhost:5556", np.array([0, 15_000]))
        for ns, value in ((0, 3), (200_000_000, 4)):
            preview.add(decode_message(struct.pack("<qqdd", 1700000000, ns, value * 1e-9, 7e-9)))
        image, tof, gmd, frames, missing, invalid, _ = preview.snapshot()
        self.assertEqual((image.sum(), tof.sum(), frames, missing, invalid), (0, 0, 2, 2, 0))
        np.testing.assert_allclose(gmd, [[0, 3, 7], [0.2, 4, 7]])
        preview.reset()
        self.assertEqual(preview.snapshot()[2].shape, (0, 3))
