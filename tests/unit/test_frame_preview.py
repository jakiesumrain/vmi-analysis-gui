import random
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from vmi_analysis.gui import frame_preview
from vmi_analysis.gui.accumulate_plot import frame_histogram_figure


def write_h5(path, frames):
    with h5py.File(path, "w") as handle:
        handle.create_dataset("raw/data", data=np.stack(frames))


class FramePreviewTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmp.name)
        first = np.arange(2 * 8 * 8, dtype=np.uint16).reshape(2, 8, 8)
        second = first + 1000
        # Write out of order: enumeration must sort by file name.
        write_h5(self.folder / "RAW-0002.h5", second)
        write_h5(self.folder / "RAW-0001.h5", first)

    def tearDown(self):
        self._tmp.cleanup()

    def test_frames_are_enumerated_in_sorted_file_order(self):
        refs = frame_preview.list_raw_frames(self.folder)
        self.assertEqual([r.global_index for r in refs], [0, 1, 2, 3])
        self.assertEqual([r.file_name for r in refs],
                         ["RAW-0001.h5"] * 2 + ["RAW-0002.h5"] * 2)
        self.assertEqual([r.frame_index for r in refs], [0, 1, 0, 1])
        expected = np.arange(2 * 8 * 8, dtype=np.uint16).reshape(2, 8, 8)[0] + 1000
        np.testing.assert_array_equal(frame_preview.load_raw_frame(refs[2]), expected)

    def test_pick_frame_index_random_and_range(self):
        refs = frame_preview.list_raw_frames(self.folder)
        self.assertEqual(frame_preview.pick_frame(refs, 3).global_index, 3)
        picks = {frame_preview.pick_frame(refs, rng=random.Random(seed)).global_index
                 for seed in range(30)}
        self.assertTrue(picks <= {0, 1, 2, 3})
        self.assertGreater(len(picks), 1)
        for bad in (4, -1):
            with self.assertRaises(IndexError):
                frame_preview.pick_frame(refs, bad)

    def test_cache_rebuilds_when_a_file_is_added(self):
        self.assertEqual(len(frame_preview.list_raw_frames(self.folder)), 4)
        write_h5(self.folder / "RAW-0003.h5", np.zeros((1, 8, 8), dtype=np.uint16))
        self.assertEqual(len(frame_preview.list_raw_frames(self.folder)), 5)

    def test_missing_directory_and_dataset_are_reported(self):
        with self.assertRaises(FileNotFoundError):
            frame_preview.list_raw_frames(self.folder / "missing")
        with h5py.File(self.folder / "broken.h5", "w") as handle:
            handle.create_dataset("other/data", data=np.zeros((1, 2, 2), dtype=np.uint16))
        with self.assertRaises(ValueError):
            frame_preview.list_raw_frames(self.folder)

    def test_statistics_count_threshold_and_denoise(self):
        frame = np.zeros((8, 8), dtype=np.uint16)
        frame[3, 3] = 40        # isolated pixel: removed by the 2×2 opening
        frame[5:7, 5:7] = 40    # solid 2×2 block: survives
        stats = frame_preview.frame_statistics(frame, 10, denoise=True)
        self.assertEqual(stats["n_pixels"], 64)
        self.assertEqual(stats["above"], 5)
        self.assertEqual(stats["after_denoise"], 4)
        self.assertEqual(stats["maximum"], 40)
        linear = frame_preview.frame_statistics(frame, 10, denoise=False)
        self.assertIsNone(linear["after_denoise"])
        self.assertEqual(linear["above"], 5)

    def test_histogram_figure_marks_threshold_and_serialises(self):
        values = np.array([[0, 1, 2], [3, 100, 1000]], dtype=np.uint16)
        figure = frame_histogram_figure(values, 50)
        self.assertEqual(figure.data[0].type, "bar")
        self.assertEqual(int(np.sum(figure.data[0].y)), values.size)
        self.assertEqual(figure.layout.yaxis.type, "log")
        self.assertTrue(figure.layout.shapes)
        figure.to_json()
        with self.assertRaises(ValueError):
            frame_histogram_figure(np.array([]), 5)


if __name__ == "__main__":
    unittest.main()
