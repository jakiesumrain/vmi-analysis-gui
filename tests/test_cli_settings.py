"""CLI precedence and GUI persistence without a required TOML file."""

import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest.mock import Mock, patch

from centroiding.options import centroid_cli_args, load_settings, parse_centroid_args
from gui.settings import centroid_settings, storage_key, binning_cli_args, binning_inputs, MIGRATED_KEY

ROOT = Path(__file__).resolve().parents[1]


class CentroidOptionsTests(unittest.TestCase):
    def test_cli_without_toml_and_explicit_false_values(self):
        result = parse_centroid_args(["--h5-dir=data", "--alpha=.0007", "--no-abel",
                                      "--no-parallel", "--no-gmd-weighting",
                                      "--spectrum-x-max=150"])
        vmi = result.settings["vmi_picasso"]
        self.assertIsNone(result.config)
        self.assertEqual(vmi["h5_dir"], "data")
        self.assertEqual(vmi["output_dir"], "vmi_output")
        self.assertEqual(vmi["energy_calibration"]["alpha"], .0007)
        self.assertFalse(vmi["processing"]["enable_parallel"])
        self.assertFalse(vmi["histogram"]["enable_abel"])
        self.assertEqual(vmi["spectrum_plot"]["x_max"], 150)

    def test_round_trip_every_option_including_spaces_and_negative_limits(self):
        config = load_settings()
        config["vmi_picasso"].update(h5_dir="folder with spaces", output_dir="out dir", gmd_map_path="")
        config["vmi_picasso"]["processing"].update(tuning_only=True, mask_center_x=0.0)
        args = centroid_cli_args(config)
        self.assertNotIn("--config", args)
        self.assertEqual(parse_centroid_args(args).settings, config)
        self.assertIn("--no-gmd-weighting", args)

    def test_explicit_cli_overrides_optional_legacy_config(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.toml"
            path.write_text('[vmi_picasso]\nh5_dir="legacy"\nbox_size=7\n'
                            '[vmi_picasso.histogram]\nenable_abel=true\n', encoding="utf-8")
            args = parse_centroid_args(["--config", str(path), "--h5-dir=chosen", "--no-abel"])
            self.assertEqual(args.settings["vmi_picasso"]["h5_dir"], "chosen")
            self.assertEqual(args.settings["vmi_picasso"]["box_size"], 7)
            self.assertFalse(args.settings["vmi_picasso"]["histogram"]["enable_abel"])

    def test_missing_required_input_and_invalid_calibration(self):
        for args in ([], ["--h5-dir=data", "--alpha=-1"], ["--h5-dir=data", "--box-size=4"]):
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_centroid_args(args)


class SavedSettingsTests(unittest.TestCase):
    def test_custom_binning_config_migrates_once_without_overwriting_saved_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bins.toml"
            path.write_text('[vmi_picasso]\noutput_dir="custom bins"\ngmd_map_path="custom map"\n', encoding="utf-8")
            store = {MIGRATED_KEY: True, "binning.config": str(path), "binning.checkpoint": "chosen.h5"}
            result = binning_inputs(store)
            self.assertEqual(result["checkpoint"], "chosen.h5")
            self.assertEqual(result["gmd_map"], "custom map")
            self.assertEqual(Path(result["output_dir"]), ROOT / "centroiding/custom bins")
            path.unlink()
            self.assertEqual(binning_inputs(store), result)

    def test_read_only_one_time_migration_and_restart_without_file(self):
        with tempfile.TemporaryDirectory() as folder:
            legacy = Path(folder) / "old.toml"
            legacy.write_text('[vmi_picasso]\nh5_dir="old data"\noutput_dir="old out"\n'
                              '[vmi_picasso.energy_calibration]\nalpha=.0007\n'.replace('=.0007', '=0.0007'),
                              encoding="utf-8")
            original = legacy.read_bytes()
            store = {storage_key("vmi_picasso", "h5_dir"): "already saved"}
            config = centroid_settings(store, legacy)
            self.assertEqual(config["vmi_picasso"]["h5_dir"], "already saved")
            self.assertEqual(config["vmi_picasso"]["output_dir"], "old out")
            self.assertEqual(config["vmi_picasso"]["energy_calibration"]["alpha"], .0007)
            self.assertEqual(legacy.read_bytes(), original)
            store[storage_key("vmi_picasso", "output_dir")] = "new out"
            legacy.unlink()
            restarted = json.loads(json.dumps(store))
            self.assertEqual(centroid_settings(restarted, legacy)["vmi_picasso"]["output_dir"], "new out")
            self.assertTrue(restarted[MIGRATED_KEY])

    def test_no_legacy_file_uses_defaults_and_persist_listener(self):
        from gui.panels.common import persist
        from nicegui import app
        class Widget:
            def __init__(self):
                self.value = "initial"
            def on_value_change(self, callback):
                self.callback = callback
        with tempfile.TemporaryDirectory() as folder:
            store = {}
            centroid_settings(store, Path(folder) / "absent.toml")
            key = storage_key("vmi_picasso", "h5_dir")
            with patch.object(app.storage, "_general", store):
                widget = persist(Widget(), key)
                widget.value = "edited without Run"
                widget.callback(type("Event", (), {"value": widget.value})())
                restored = persist(Widget(), key)
            self.assertEqual(restored.value, "edited without Run")
            self.assertEqual(centroid_settings(store)["vmi_picasso"]["h5_dir"], "edited without Run")

    def test_binning_arguments_capture_snapshot_paths_and_filters(self):
        config = load_settings()
        vmi = config["vmi_picasso"]
        vmi.update(output_dir="out", checkpoint_file="events.h5", gmd_map_path="map.txt")
        vmi["histogram"]["resolution_factor"] = 2
        vmi["gmd"].update(gmd_weight_min=.5, gmd_weight_max=4)
        args = binning_cli_args(config)
        resolved = dict(zip(args[::2], args[1::2]))
        self.assertEqual(Path(resolved["--checkpoint-path"]), ROOT / "centroiding/out/events.h5")
        self.assertEqual(resolved["--resolution-factor"], "2")
        self.assertEqual(resolved["--gmd-weight-min"], "0.5")


class StandaloneInputsTests(unittest.TestCase):
    def binning_namespace(self, args):
        script = ROOT / "centroiding/vmi_histogram_gmd_bins.py"
        with patch.object(sys, "argv", [str(script), *args]), patch.dict(os.environ, MPLBACKEND="Agg"):
            return runpy.run_path(str(script), run_name="binning_test")

    def test_binning_explicit_inputs_and_legacy_override(self):
        args = ["--checkpoint-path", "chosen.h5", "--gmd-map-path", "chosen.txt",
                "--output-dir", "chosen out", "--resolution-factor", "2", "--gmd-weight-max", "3"]
        for legacy in (False, True):
            with self.subTest(legacy=legacy), tempfile.TemporaryDirectory() as folder:
                if legacy:
                    path = Path(folder) / "old.toml"
                    path.write_text('[vmi_picasso]\noutput_dir="old"\ngmd_map_path="old.txt"\n', encoding="utf-8")
                    argv = ["--config", str(path), *args]
                else:
                    argv = args
                ns = self.binning_namespace(argv)
                self.assertEqual(ns["CHECKPOINT_PATH"], "chosen.h5")
                self.assertEqual(ns["OUTPUT_DIR"], "chosen out")
                self.assertEqual(ns["GMD_MAP_PATH"], "chosen.txt")
                self.assertEqual(ns["RES_FACTOR"], 2)
                self.assertEqual(ns["GMD_WEIGHT_MAX"], 3)

    def test_standalone_scripts_help_from_an_unrelated_directory(self):
        scripts = ["process_andor_vmi_PICASSO_v5_h5_batch.py", "vmi_histogram_gmd_bins.py", "compute_dark.py"]
        with tempfile.TemporaryDirectory() as folder:
            for script in scripts:
                with self.subTest(script=script):
                    result = subprocess.run([sys.executable, str(ROOT / "centroiding" / script), "--help"],
                                            cwd=folder, capture_output=True, text=True, timeout=30,
                                            env=dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8"))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("--config", result.stdout)


class GuiSubmissionTests(unittest.TestCase):
    def submit_form(self, panel, store, label="Run"):
        from nicegui import Client, app, ui
        captured = {}
        def button(text="Run", **kwargs):
            captured[text] = kwargs["on_click"]
            return ui.button(text, **kwargs)
        manager = Mock()
        manager.jobs.return_value = []
        manager.submit.side_effect = lambda job: job
        client = Client(ui.page("/_test_settings"))
        try:
            with client, patch.object(app.storage, "_general", store), \
                    patch.object(panel.theme, "run_button", button):
                panel.build(manager)
                captured[label]()
            return manager.submit.call_args.args[0]
        finally:
            client.delete()

    def test_centroid_form_submits_saved_fields_without_config(self):
        from gui.panels import centroid
        store = {MIGRATED_KEY: True, storage_key("vmi_picasso", "h5_dir"): "chosen data",
                 storage_key("energy_calibration", "alpha"): .0007,
                 storage_key("spectrum_plot", "x_max"): None}
        job = self.submit_form(centroid, store)
        self.assertNotIn("--config", job.cmd)
        cfg = parse_centroid_args(job.cmd[2:]).settings
        self.assertEqual(cfg["vmi_picasso"]["h5_dir"], "chosen data")
        self.assertEqual(cfg["vmi_picasso"]["energy_calibration"]["alpha"], .0007)
        self.assertEqual(cfg["vmi_picasso"]["spectrum_plot"]["x_max"], -1)
        self.assertEqual(Path(job.extra["output_dir"]), ROOT / "centroiding/vmi_output")

    def test_pipeline_overrides_stale_dataset_map_and_tuning_mode(self):
        from gui.panels import pipeline
        store = {MIGRATED_KEY: True,
                 storage_key("vmi_picasso", "h5_dir"): "old data",
                 storage_key("vmi_picasso", "gmd_map_path"): "old map",
                 storage_key("processing", "tuning_only"): True,
                 storage_key("processing", "num_cores"): 3.0,
                 "pipeline.h5_dir": "pipeline data", "pipeline.gmd_path": "gmd.txt",
                 "pipeline.map_out": "new map.txt", "binning.thresholds": "1,2"}
        job = self.submit_form(pipeline, store, "Run pipeline")
        self.assertEqual([s.name for s in job.stages], ["GMD map", "Centroiding", "Binning", "PES"])
        cfg = parse_centroid_args(job.stages[1].cmd[2:]).settings["vmi_picasso"]
        self.assertEqual(cfg["h5_dir"], str(ROOT / "pipeline data"))
        self.assertEqual(cfg["gmd_map_path"], str(ROOT / "new map.txt"))
        self.assertFalse(cfg["processing"]["tuning_only"])
        self.assertEqual(cfg["processing"]["num_cores"], 3)
        self.assertTrue(store[storage_key("processing", "tuning_only")])
        args = job.stages[2].cmd
        self.assertEqual(args[args.index("--gmd-map-path") + 1], cfg["gmd_map_path"])
        self.assertEqual(args[args.index("--thresholds") + 1], "1,2")
        for stage in job.stages:
            self.assertNotIn("--config", stage.cmd)

    def test_binning_uses_explicit_paths_and_dark_preserves_zero_frames(self):
        from gui.panels import binning, dark
        store = {MIGRATED_KEY: True, "binning.checkpoint": "chosen.h5",
                 "binning.output_dir": "chosen out", "binning.gmd_map": "chosen.txt",
                 "dark.frames": 0}
        job = self.submit_form(binning, store)
        self.assertNotIn("--config", job.cmd)
        self.assertEqual(job.cmd[job.cmd.index("--checkpoint-path") + 1], "chosen.h5")
        self.assertEqual(Path(job.extra["output_dir"]), ROOT / "centroiding/chosen out")
        job = self.submit_form(dark, store)
        self.assertNotIn("--config", job.cmd)
        self.assertEqual(job.cmd[job.cmd.index("--frames") + 1], "0")


if __name__ == "__main__":
    unittest.main()
