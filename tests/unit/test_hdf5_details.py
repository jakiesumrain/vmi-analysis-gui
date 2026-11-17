from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from vmi_analysis.hdf5_details import inspect_hdf5


class Hdf5DetailsTests(unittest.TestCase):
    def test_summary_and_structure(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "sample.h5"
            with h5py.File(path, "w") as file:
                file.attrs["label"] = "example"
                file.create_dataset("nested/data", data=np.zeros((2, 3), dtype="u2"))
                file.create_dataset("Clusters", data=np.array([(10, 100), (20, 300)],
                                                              dtype=[("cx", "u2"), ("ctoa", "u8")]))
                file.create_dataset("frames", data=np.array([(1_794_488_306_444_736_228, 2.0),
                                                             (1_794_488_306_444_736_229, 4.0)],
                                                            dtype=[("timestamp_ns", "i8"), ("gmd1", "f8")]))
                matrix = file.create_dataset("TimewalkMatrix", data=[[10000, 110], [20000, 0]])
                matrix.attrs["col_names"] = np.array([(b"AverageDToA",), (b"Count",)], dtype=[("inner", "S12")])

            with patch("vmi_analysis.hdf5_details.ROWS_PER_BLOCK", 1):
                rows, structure = inspect_hdf5(path, lambda: False)
            values = {(row["dataset"], row["metric"]): row["value"] for row in rows}
            self.assertEqual(values["/Clusters", "Centroid X"], "Mean 15 px · range 10 to 20 px")
            self.assertEqual(values["/frames", "Time span"], "1e-09 s")
            self.assertEqual(values["/TimewalkMatrix", "Average ΔToA"], "Mean 1 ns · range 1 to 1 ns")
            self.assertIn("/nested/data  dataset  shape=(2, 3)", structure)
            self.assertIn("@label = 'example'", structure)
            self.assertIn("timestamp_ns: int64", structure)
