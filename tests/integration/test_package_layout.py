"""Exercise real entry points and writable paths after relocating source files."""

import os
from pathlib import Path
import subprocess
import socket
import sys
import tempfile
import unittest

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COMMANDS = (
    "gmd.build_gmd_map", "imaging.h5_accumulate_fast_gmd", "imaging.h5_play_frames_movie",
    "centroiding.compute_dark", "centroiding.vmi_histogram_gmd_bins",
    "centroiding.process_andor_vmi_PICASSO_v5_h5_batch",
    "circularization.circularization", "circularization.process_gmd_bins",
)


class PackageLayoutTests(unittest.TestCase):
    def run_command(self, command, cwd, *, workspace=None):
        env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
        if workspace is not None:
            env["VMI_WORKSPACE_DIR"] = str(workspace)
        result = subprocess.run([sys.executable, *command], cwd=cwd, env=env,
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_all_analysis_modules_from_unrelated_cwd(self):
        with tempfile.TemporaryDirectory() as folder:
            for module in COMMANDS:
                with self.subTest(module=module):
                    result = self.run_command(["-m", "vmi_analysis." + module, "--help"], folder)
                    self.assertIn("usage:", result.stdout.lower())
            self.assertFalse((Path(folder) / "src").exists())

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
