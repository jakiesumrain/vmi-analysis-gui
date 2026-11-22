import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from vmi_analysis.gui.resources import command_access
from vmi_analysis.ion_acquisition import SCHEMA, collection_reads
from vmi_analysis.ion_momenta import (DEFAULT_MM_PER_PIXEL, LONGITUDINAL_CONVERSION, SpeciesMomentum,
                                     gated_preview, main, momentum_components, momentum_projections,
                                     position_histograms)
from vmi_analysis.ion_tof import histogram


class IonMomentaTests(unittest.TestCase):
    def create_dump(self, path, base=2**60):
        with h5py.File(path, "w") as file:
            file["TDCEvents"] = np.array([(1, base + 100), (1, base + 200)],
                                         dtype=[("tdc_type", "u1"), ("timestamp", "u8")])
            # Deliberately unsorted arrival rows exercise position/TOF alignment.
            file["PixelHits"] = np.array([
                (base + 240, 18., 22.), (base + 99, 1., 1.), (base + 110, 12., 8.),
                (base + 150, 30., 30.), (base + 120, np.nan, 2.),
            ], dtype=[("toa", "u8"), ("x", "f8"), ("y", "f8")])
            file["Clusters"] = np.array([
                (base + 110, 12.25, 8.75), (base + 150, 10., 10.),
                (base + 240, 300.75, 400.25),
            ], dtype=[("ctoa", "u8"), ("cx", "f8"), ("cy", "f8")])

    def test_positions_match_tof_window_and_preserve_clock_precision(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dump.h5"
            self.create_dump(path)
            with patch("vmi_analysis.ion_tof.BLOCK_SIZE", 2):
                result = position_histograms(path, ("pixels", "clusters"), minimum=.001,
                                             maximum=.004, bin_size_ns=.001)
            original = histogram(path, ("pixels", "clusters"), minimum=.001, maximum=.004, bin_size_ns=.001)
            for source in ("pixels", "clusters"):
                np.testing.assert_array_equal(result[source], original[source])
            self.assertEqual(result["pixels"].sum(), 3)
            self.assertEqual(result["pixels_unmatched"], 1)
            self.assertEqual(result["pixels_outside"], 1)
            self.assertEqual(result["pixels_invalid_positions"], 1)
            image = result["pixels_image"]
            self.assertEqual(image.sum(), 2)
            self.assertEqual((image[8, 12], image[22, 18]), (1, 1))
            image = result["clusters_image"]
            self.assertEqual(image.shape, (401, 302))
            self.assertEqual(image.sum(), 2)
            self.assertEqual((image[9, 12], image[400, 301]), (1, 1))

    def test_momenta_use_pixel_center_tof_reference_and_longitudinal_scale(self):
        ion = SpeciesMomentum("clusters", (.0039, .0041), 131.293, 2, .005, 1.5)
        momenta = momentum_components(np.array([14.]), np.array([6.]), np.array([1000.]),
                                      SpeciesMomentum("clusters", (0, 2000), 131.293, 2, 1010, 1.5),
                                      (12., 8.), DEFAULT_MM_PER_PIXEL)
        expected = 349.7122801734099 * 131.293 * 2 / 1000
        np.testing.assert_allclose(momenta, [[expected], [-expected], [LONGITUDINAL_CONVERSION * 30]])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dump.h5"
            self.create_dump(path)
            result = position_histograms(path, ("clusters",), minimum=0, maximum=.003)
            edges, maps, counts = momentum_projections(
                path, "TDC1Rising", [ion], (300.75, 400.25), DEFAULT_MM_PER_PIXEL,
                result["trigger_times"], result["segment_offsets"], stopped=lambda: False,
            )
            np.testing.assert_array_equal(counts, [1])
            np.testing.assert_array_equal(maps.sum(axis=(2, 3)), [[1, 1]])
            zero_bin = np.searchsorted(edges[0], 0, side="right") - 1
            self.assertEqual(maps[0, 0, zero_bin, zero_bin], 1)

    def test_independent_gates_use_unrounded_positions_and_unbinned_tof(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dump.h5"
            self.create_dump(path)
            result = position_histograms(path, ("clusters",), minimum=0, maximum=.006, bin_size_ns=.003)
            args = (path, "TDC1Rising", "clusters", result["edges"], result["trigger_times"],
                    result["segment_offsets"], result["clusters_image"].shape)
            counts = gated_preview(*args, position_gate=(12.2, 12.3, 8.7, 8.8))
            np.testing.assert_array_equal(counts, [1, 0])
            image = gated_preview(*args, tof_gate=(.0039, .0041))
            self.assertEqual(image.sum(), 1)
            self.assertEqual(image[400, 301], 1)

    def test_segment_clock_resets_and_file_reservations(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            segments = []
            for index in range(2):
                path = root / f"dump-{index}.h5"
                self.create_dump(path, base=0)
                metadata = root / f"metadata-{index}.h5"
                with h5py.File(metadata, "w"):
                    pass
                segments.append({"data": path.name, "metadata": metadata.name})
            collection = root / "dump.segments.json"
            collection.write_text(json.dumps({"schema": SCHEMA, "segments": segments}))
            output = root / "result.npz"
            main(["--input", str(collection), "--output", str(output), "--source", "both",
                  "--minimum", ".001", "--maximum", ".004", "--bin-size", ".001"])
            with np.load(output, allow_pickle=False) as result:
                self.assertEqual(result["triggers"], 4)
                self.assertEqual(result["pixels_image"].sum(), 4)
                self.assertEqual(result["clusters_image"].sum(), 4)
            access = command_access(["python", "-m", "vmi_analysis.ion_momenta",
                                     "--input", str(collection), "--output", str(output)])
            self.assertEqual(access.reads, frozenset(collection_reads(collection)))
            self.assertEqual(access.writes, frozenset([output]))

    def test_failed_input_preserves_previous_output(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dump.h5"
            self.create_dump(path)
            with h5py.File(path, "a") as file:
                del file["Clusters"]
            output = Path(folder) / "result.npz"
            output.write_bytes(b"previous result")
            with self.assertRaises(SystemExit):
                main(["--input", str(path), "--output", str(output), "--source", "both",
                      "--maximum", "1", "--bin-size", ".1"])
            self.assertEqual(output.read_bytes(), b"previous result")
