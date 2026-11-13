"""Exercise real entry points and writable paths after relocating source files."""

import base64
import os
from pathlib import Path
import subprocess
import socket
import sys
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd

from vmi_analysis.centroiding.plot_data import SUMMARY_DATA, TUNING_DATA, load_plot_data

ROOT = Path(__file__).resolve().parents[2]
COMMANDS = (
    "gmd.build_gmd_map", "imaging.h5_accumulate_fast_gmd", "imaging.h5_play_frames_movie",
    "centroiding.compute_dark", "centroiding.vmi_histogram_gmd_bins",
    "centroiding.process_andor_vmi_PICASSO_v5_h5_batch",
    "circularization.circularization", "circularization.process_gmd_bins",
)


class PackageLayoutTests(unittest.TestCase):
    def run_command(self, command, cwd, *, workspace=None, returncode=0):
        env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
        if workspace is not None:
            env["VMI_WORKSPACE_DIR"] = str(workspace)
        result = subprocess.run([sys.executable, *command], cwd=cwd, env=env,
                                capture_output=True, encoding="utf-8", timeout=90)
        self.assertEqual(result.returncode, returncode, result.stdout + result.stderr)
        return result

    def test_all_analysis_modules_from_unrelated_cwd(self):
        with tempfile.TemporaryDirectory() as folder:
            for module in COMMANDS:
                with self.subTest(module=module):
                    result = self.run_command(["-m", "vmi_analysis." + module, "--help"], folder)
                    self.assertIn("usage:", result.stdout.lower())
            self.assertFalse((Path(folder) / "src").exists())

    def test_movie_preview_stream_writes_no_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / "RAW-test.h5"
            with h5py.File(source, "w") as handle:
                handle.create_dataset("raw/data", data=np.arange(192, dtype=np.uint16).reshape(2, 8, 12))
            result = self.run_command(
                ["-m", "vmi_analysis.imaging.h5_play_frames_movie",
                 "--h5-dir", str(folder), "--n-frames", "200", "--interval-ms", "40",
                 "--movie-dir", str(folder / "movies"), "--headless",
                 "--no-save-movie", "--preview-stream"],
                folder,
            )
            frames = [line.split(" ", 5)[1:] for line in result.stdout.splitlines()
                      if line.startswith("[preview-png] ")]
            self.assertEqual([frame[:4] for frame in frames], [["1", "2", "12", "8"], ["2", "2", "12", "8"]])
            for *_, png in frames:
                image = base64.b64decode(png, validate=True)
                self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))
                self.assertEqual(image[16:24], b"\x00\x00\x00\x0c\x00\x00\x00\x08")
            self.assertEqual(list(folder.iterdir()), [source])

    def test_accumulation_module_with_worker_processes(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            data = np.arange(3 * 16 * 16, dtype=np.uint16).reshape(3, 16, 16) % 30 + 1
            with h5py.File(folder / "RAW-test.h5", "w") as handle:
                handle.create_dataset("raw/data", data=data)
            output = folder / "sum.txt"
            self.run_command(["-m", "vmi_analysis.imaging.h5_accumulate_fast_gmd",
                              "--h5-dir", str(folder), "--output-path", str(output),
                              "--simple-sum", "--max-workers", "2", "--no-show"], folder)
            self.assertTrue(output.with_suffix(".png").is_file())
            np.testing.assert_array_equal(np.loadtxt(output), data.sum(axis=0))

    def test_accumulation_limit_spans_files_and_preserves_gmd_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            frames = np.arange(1, 7, dtype=np.uint16)[:, None, None] * np.ones((8, 8), dtype=np.uint16)
            for name, data in (("RAW-c.h5", frames[5:]), ("RAW-b.h5", frames[2:5]), ("RAW-a.h5", frames[:2])):
                with h5py.File(folder / name, "w") as handle:
                    handle.create_dataset("raw/data", data=data)
            gmd = np.array([1, 2, 3, 4, 5, 6], dtype=float)
            gmd_path = folder / "gmd.txt"
            gmd_path.write_text("".join(f"{i} file 0 0 {value}\n" for i, value in enumerate(gmd)), encoding="utf-8")
            output = folder / "sum.txt"
            command = ["-m", "vmi_analysis.imaging.h5_accumulate_fast_gmd",
                       "--h5-dir", str(folder), "--output-path", str(output),
                       "--simple-sum", "--max-workers", "2", "--no-show",
                       "--enable-gmd", "--gmd-map-path", str(gmd_path),
                       "--gmd-w-min", "0", "--gmd-w-max", "0"]
            self.run_command([*command, "--frame-limit", "3"], folder)
            expected = (frames[:3] * (gmd.mean() / gmd[:3])[:, None, None]).sum(axis=0)
            np.testing.assert_allclose(np.loadtxt(output), expected)
            result = self.run_command(command, folder, returncode=1)
            self.assertIn("was created with frame limit 3", result.stderr)

    def test_centroid_limit_preserves_frame_ids_and_tuning_can_use_later_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            y, x = np.indices((32, 32))
            frames = np.stack([100 + 500 * np.exp(-((x - (12.4 + i))**2 + (y - 15.2)**2) / 3)
                               for i in range(5)]).astype(np.uint16)
            for name, data in (("RAW-b.h5", frames[2:]), ("RAW-a.h5", frames[:2])):
                with h5py.File(folder / name, "w") as handle:
                    handle.create_dataset("raw/data", data=data)
            output = folder / "results"
            command = ["-m", "vmi_analysis.centroiding.process_andor_vmi_PICASSO_v5_h5_batch",
                       "--h5-dir", str(folder), "--output-dir", str(output),
                       "--min-net-gradient", "20", "--fitting-method", "lq",
                       "--no-parallel", "--no-abel"]
            self.run_command([*command, "--frame-limit", "3"], folder)
            checkpoint = output / "vmi_results_raw.h5"
            localizations = pd.read_hdf(checkpoint, "localizations")
            np.testing.assert_array_equal(localizations["frame"], [0, 1, 2])
            metadata, _ = load_plot_data(output / SUMMARY_DATA)
            self.assertEqual(metadata["n_frames"], 3)
            result = self.run_command(command, folder, returncode=1)
            self.assertIn("does not contain a complete checkpoint for frame limit 0", result.stderr)
            result = self.run_command([*command, "--frame-limit", "3", "--no-median-subtraction"],
                                      folder, returncode=1)
            self.assertIn("does not match background subtraction mode 'none'", result.stderr)
            self.run_command([*command, "--frame-limit", "3", "--tuning-only",
                              "--tuning-frame-index", "4"], folder)
            metadata, _ = load_plot_data(output / TUNING_DATA)
            self.assertEqual(metadata["frame_index"], 4)

    def test_centroid_dark_subtraction_requires_precomputed_numpy_file(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            with h5py.File(folder / "RAW-dark.h5", "w") as handle:
                handle.create_dataset("raw/data", data=np.full((1, 8, 8), 10, dtype=np.uint16))
            config = folder / "config.toml"
            config.write_text('[vmi_picasso]\ndark_data_dir = "."\n', encoding="utf-8")
            command = ["-m", "vmi_analysis.centroiding.process_andor_vmi_PICASSO_v5_h5_batch",
                       "--config", str(config), "--h5-dir", str(folder), "--dark-subtraction"]
            result = self.run_command(command, folder, returncode=1)
            self.assertIn("Select a precomputed dark .npy file", result.stderr)
            dark = folder / "dark.npy"
            result = self.run_command([*command, "--dark-file", str(dark)], folder, returncode=1)
            self.assertIn("Cannot load precomputed dark frame", result.stderr)
            np.save(dark, np.zeros((1, 8)))
            result = self.run_command([*command, "--dark-file", str(dark)], folder, returncode=1)
            self.assertIn("does not match camera frame shape", result.stderr)
            np.save(dark, np.full((8, 8), 3))
            output = folder / "results"
            self.run_command([*command, "--dark-file", str(dark), "--output-dir", str(output),
                              "--tuning-only", "--no-median-subtraction"], folder)
            _, arrays = load_plot_data(output / TUNING_DATA)
            np.testing.assert_array_equal(arrays["frame"], np.full((8, 8), 7))

    def test_dark_frame_default_writes_to_workspace_instead_of_source(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            data = np.arange(3 * 8 * 8, dtype=np.uint16).reshape(3, 8, 8)
            with h5py.File(folder / "RAW-dark.h5", "w") as handle:
                handle.create_dataset("raw/data", data=data)
            self.run_command(["-m", "vmi_analysis.centroiding.compute_dark", "--dark-dir", str(folder),
                              "--frames", "0", "--method", "mean"], folder, workspace=folder)
            np.testing.assert_array_equal(np.load(folder / "centroiding/dark_mean.npy"), data.mean(axis=0))

    def test_assets_and_legacy_data_locations(self):
        from importlib.resources import files
        from vmi_analysis.paths import CENTROID_DIR, CIRCULARIZATION_DIR, CPBASEX_DIR, GENERATED_DIR
        self.assertEqual(CENTROID_DIR, ROOT / "centroiding")
        self.assertEqual(CIRCULARIZATION_DIR, ROOT / "circularization")
        self.assertEqual(GENERATED_DIR, ROOT / ".cache/vmi-analysis")
        self.assertTrue((CPBASEX_DIR / "examples/new_10.m").is_file())
        assets = files("vmi_analysis.gui.assets")
        self.assertIn("vmiZoomOpen", (assets / "image_zoom.js").read_text(encoding="utf-8"))
        self.assertIn("--", (assets / "theme.css").read_text(encoding="utf-8"))

    def test_busy_gui_port_exits_with_relaunch_commands_before_ready_message(self):
        with socket.socket() as listener, tempfile.TemporaryDirectory() as folder:
            listener.bind(('127.0.0.1', 0))
            listener.listen()
            port = listener.getsockname()[1]
            result = subprocess.run([sys.executable, '-m', 'vmi_analysis.gui.app'], cwd=folder,
                env=dict(os.environ, VMI_GUI_PORT=str(port), VMI_GUI_NO_SHOW='1',
                         VMI_WORKSPACE_DIR=folder, NICEGUI_STORAGE_PATH=str(Path(folder) / 'storage'),
                         PYTHONIOENCODING='utf-8'), capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn(f'port {port} is unavailable', result.stderr)
            self.assertIn('$env:VMI_GUI_PORT = "', result.stderr)
            self.assertIn('uv run vmi-analysis', result.stderr)
            self.assertNotIn('NiceGUI ready', result.stdout + result.stderr)
            self.assertNotIn('Traceback', result.stdout + result.stderr)

    def test_workspace_survives_a_subprocess_working_directory_change(self):
        code = """
import subprocess, sys
from vmi_analysis.paths import CENTROID_DIR, prepare_workspace
prepare_workspace()
result = subprocess.run([sys.executable, '-c',
    'from vmi_analysis.paths import CENTROID_DIR; print(CENTROID_DIR)'],
    cwd=CENTROID_DIR, capture_output=True, text=True, check=True)
assert result.stdout.strip() == str(CENTROID_DIR), result.stdout
"""
        with tempfile.TemporaryDirectory() as folder:
            # A relative workspace is normalized once and inherited by jobs.
            self.run_command(["-c", code], folder, workspace="experiment")


if __name__ == "__main__":
    unittest.main()
