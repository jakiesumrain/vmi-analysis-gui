import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from vmi_analysis.ion_acquisition import SCHEMA, collection_reads, segment_files
from vmi_analysis.ion_luna import dump_acquisitions
from vmi_analysis.ion_tof import filtered_histogram, fit_filtered_peaks, gmd_correlations, histogram
from vmi_analysis.gui.panels.ion_tof import _load_gmd_diagnostics


class IonSegmentTests(unittest.TestCase):
    def test_collection_counts_and_fits_do_not_cross_clock_resets(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            segments = []
            for i in range(2):
                data, metadata = root / f"data-{i}.h5", root / f"metadata-{i}.h5"
                with h5py.File(data, "w") as file:
                    file["TDCEvents"] = np.array([(1, 100_000), (1, 200_000)],
                                                 dtype=[("tdc_type", "u1"), ("timestamp", "u8")])
                    file["PixelHits"] = np.array([(110_000,), (140_000,), (170_000,), (210_000,)],
                                                 dtype=[("toa", "u8")])
                    file["Clusters"] = np.array([(110_000,), (240_000,)], dtype=[("ctoa", "u8")])
                with h5py.File(metadata, "w") as file:
                    file["frames"] = np.array([(i * 10, 2e-9, 2e-9), (i * 10 + 1, 3e-9, 3e-9)],
                                              dtype=[("timestamp_ns", "i8"), ("gmd1", "f8"), ("gmd2", "f8")])
                segments.append(dict(data=data.name, metadata=metadata.name))
            collection = root / "dump.segments.json"
            collection.write_text(json.dumps(dict(schema=SCHEMA, segments=segments)))
            original = root / "original.npz"
            result = histogram(collection, ("pixels", "clusters"), minimum=0, maximum=8, bins=8)
            np.testing.assert_array_equal(result["segment_offsets"], [0, 2, 4])
            self.assertEqual(result["pixels"].sum(), 8)
            np.savez(original, **result)
            diagnostics, sources = _load_gmd_diagnostics(original, collection)
            np.testing.assert_allclose(diagnostics["gmd2_na"], [2, 3, 2, 3])
            self.assertEqual(sources, ["pixels", "clusters"])
            filtered = filtered_histogram(collection, original, collection, "TDC1Rising", None, None, 1)
            np.testing.assert_array_equal(filtered["selected_mask"], [False, True, False, True])
            self.assertEqual(filtered["selected_triggers"], 2)
            self.assertEqual(filtered["pixels"].sum(), 2)
            self.assertAlmostEqual(filtered["gmd_min"], 2)
            self.assertAlmostEqual(filtered["gmd_max"], 3)
            # Both bounds at the single data value are a valid inclusive selection.
            constant = filtered_histogram(collection, original, collection, "TDC1Rising", 2, 2, 0)
            self.assertEqual(constant["selected_triggers"], 2)
            filtered_path = root / "filtered.npz"
            np.savez(filtered_path, **constant)
            x = np.arange(8, dtype=float) + 0.5
            peak = dict(center=4., gaussian_fwhm=2*np.sqrt(2*np.log(2)))
            with patch("vmi_analysis.pes.fit_peaks", return_value=(x, x, x, None, x, [peak])):
                fit = fit_filtered_peaks(collection, filtered_path, "TDC1Rising", ["pixels"], (0, 8), [], "None")
            self.assertEqual(json.loads(fit["pixels_peaks"])[0]["counts"], 4)
            self.assertEqual(json.loads(fit["pixels_peaks"])[0]["yield_per_trigger"], 2)
            self.assertEqual(len(collection_reads(collection)), 5)
            collection.write_text(json.dumps(dict(schema=SCHEMA, segments=[dict(data="../outside.h5",
                                                                               metadata="metadata-0.h5")])))
            with self.assertRaisesRegex(ValueError, "within the collection"):
                segment_files(collection)

    def test_correlation_excludes_pairs_across_acquisitions(self):
        values = np.arange(1, 13, dtype=float)
        _, pairs = gmd_correlations(values, values, [0, 6, 12])
        self.assertEqual(pairs[6], 10)  # d = +1, one excluded pair at each acquisition edge
        self.assertEqual(pairs[4], 10)  # d = -1

    def test_dump_publishes_separate_files_with_reset_explanation(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = []
            for i, (sec, coarse) in enumerate(((100, 800_000_000), (108, 4_000_000))):
                path = root / f"frame_{i}.tpx3"
                word = (0x6F << 56) | (coarse << 12) | (1 << 5)
                path.write_bytes(struct.pack("<qqddQ", sec, 0, 2e-9, 3e-9, word))
                sources.append(path)

            def publish(command, pending, output, **kwargs):
                with h5py.File(output, "w") as file:
                    file["TDCEvents"] = [1]

            options = SimpleNamespace(wsl_distribution="", wsl_executable="")
            with patch("vmi_analysis.ion_luna.publish_result", side_effect=publish):
                dump_acquisitions(sources, root / "ions.h5", None, {}, options, windows=False)
            manifest = root / "ions.segments.json"
            self.assertEqual(len(segment_files(manifest)), 2)
            doc = json.loads(manifest.read_text())
            self.assertIn("more than 1 s", doc["segments"][1]["reason"])
            for data, meta in segment_files(manifest):
                self.assertTrue(data.is_file())
                with h5py.File(meta, "r") as file:
                    self.assertEqual(len(file["frames"]), 1)
