"""CLI precedence and GUI persistence without a required TOML file."""

import asyncio
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

from vmi_analysis.centroiding.options import centroid_cli_args, load_settings, parse_centroid_args
from vmi_analysis.gui.settings import centroid_settings, storage_key, binning_cli_args, binning_inputs, MIGRATED_KEY

ROOT = Path(__file__).resolve().parents[2]


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
        self.assertEqual(vmi["histogram"]["abel_rotation_deg"], 0.0)
        self.assertIsNone(vmi["tuning_frame_index"])
        self.assertEqual(vmi["spectrum_plot"]["x_max"], 150)

    def test_round_trip_every_option_including_spaces_and_negative_limits(self):
        config = load_settings()
        config["vmi_picasso"].update(h5_dir="folder with spaces", output_dir="out dir", gmd_map_path="")
        config["vmi_picasso"]["tuning_frame_index"] = 0
        config["vmi_picasso"]["processing"].update(tuning_only=True, mask_center_x=0.0)
        config["vmi_picasso"]["histogram"]["abel_rotation_deg"] = -22.5
        args = centroid_cli_args(config)
        self.assertNotIn("--config", args)
        self.assertEqual(parse_centroid_args(args).settings, config)
        self.assertIn("--no-gmd-weighting", args)
        self.assertIn("--tuning-frame-index=0", args)

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
        from vmi_analysis.gui.panels.common import persist
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
        module = "vmi_analysis.centroiding.vmi_histogram_gmd_bins"
        with patch.object(sys, "argv", [module, *args]), patch.dict(os.environ, MPLBACKEND="Agg"):
            return runpy.run_module(module, run_name="binning_test")

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
                    result = subprocess.run([sys.executable, "-m", "vmi_analysis.centroiding." + script.removesuffix(".py"), "--help"],
                                            cwd=folder, capture_output=True, text=True, timeout=30,
                                            env=dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8"))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("--config", result.stdout)


class GuiSubmissionTests(unittest.TestCase):
    def test_clear_actions_delete_panel_outputs_and_preserve_other_files(self):
        from nicegui import Client, app, ui
        from vmi_analysis.circularization.polar_fit import SESSION_SCHEMA
        from vmi_analysis.gui.jobs import Job
        from vmi_analysis.gui.panels import accumulate, binning, calibration, centroid, circularize, dark, gmd_map, movie, pes

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            preserved = [root / name for name in ("detector.npy", "notes.txt", "trig_order_notes.png",
                                                   "vmi_2d_gmd_bin0_minusinf_1.00e-08_w-gmd.npy")]
            for path in preserved:
                path.touch()
            data = root / "dataset"
            data.mkdir()
            frames = root / "movie_frames" / "saved"
            frames.mkdir(parents=True)
            (frames / "movie-source.json").write_text(json.dumps({"dataset": str(data), "output_dir": folder}))
            (frames / "frame_0000.png").touch()
            (frames / "movie-path.txt").touch()
            preview = root / "circ_polar.npz"
            partial = root / "accum_live.npy"
            for path in (preview, partial):
                path.touch()
            store = {MIGRATED_KEY: True, "gmd_map.output_path": str(root / "map.txt"),
                     "movie.h5_dir": str(data), "movie.movie_dir": folder,
                     "dark.out_dir": folder, "accumulate.output_path": str(root / "sum.txt"),
                     storage_key("vmi_picasso", "output_dir"): folder,
                     "binning.output_dir": folder, "pes.output_dir": folder,
                     "circularize.inp": str(preserved[0]), "circularize.out": str(root / "corrected.txt"),
                     "circularize.plot_dir": folder, "calibration.output": str(root / "rings.txt"),
                     "calibration.session": str(root / "session.json")}
            cases = (
                (gmd_map, "Clear output", {"map.txt": "map"}),
                (movie, "Clear outputs", {"dataset.mp4": "movie", "dataset.gif": "movie"}),
                (dark, "Clear output", {"dark_median.npy": "dark", "dark_median_info.txt": "info"}),
                (accumulate, "Clear output", {"sum.txt": "image", "sum.png": "plot"}),
                (centroid, "Clear outputs", {**dict.fromkeys(centroid.OUTPUT_NAMES, "output"), "vmi_results_raw.h5": "checkpoint"}),
                (binning, "Clear outputs", dict.fromkeys(("vmi_2d_gmd_all_w-gmd_norm.npy", "vmi_gmd_bins_w-gmd_norm.png",
                                                         "vmi_2d_gmd_bin0_minusinf_1.00e-08_w-gmd_norm.npy"), "bin")),
                (calibration, "Clear calibration file", {"rings.txt": "# empty calibration"}),
                (calibration, "Clear session file", {"session.json": json.dumps({"schema": SESSION_SCHEMA})}),
                (circularize, "Clear outputs", dict.fromkeys(("corrected.txt", "trig_order_1.png", "reassignment_map.png"), "plot")),
                (pes, "Clear outputs", {"pes-spectra-test.npz": "PES"}),
            )
            manager = Mock()
            for panel, label, files in cases:
                with self.subTest(panel=panel.__name__, label=label):
                    for name, content in files.items():
                        (root / name).write_text(content, encoding="utf-8")
                    manager.jobs.return_value = [
                        Job("Accumulate", extra={"output_path": str(root / "sum.txt"), "live_path": str(partial)}, status="done"),
                        Job("PES", extra={"pes_spectra": str(root / "pes-spectra-test.npz")}, status="done"),
                        Job("Circularize", extra={"output": str(root / "corrected.txt"), "polar_preview": str(preview)}, status="done"),
                    ]
                    callbacks = {}
                    button = ui.button

                    def capture(text="", *args, **kwargs):
                        if kwargs.get("on_click"):
                            callbacks[text] = kwargs["on_click"]
                        return button(text, *args, **kwargs)

                    client = Client(ui.page("/_test_clear_outputs"))
                    try:
                        with client, patch.object(app.storage, "_general", store), patch.object(ui, "button", capture), \
                                patch.object(movie, "GENERATED", root):
                            panel.build(manager)
                            if label == "Clear calibration file":
                                element = next(el for el in client.elements.values()
                                               if isinstance(el, button) and el.text == label)
                                listener, = element._event_listeners.values()
                                listener.handler(None)
                            else:
                                callbacks[label]()
                            callbacks["Delete"]()
                        self.assertTrue(all(not (root / name).exists() for name in files))
                        self.assertTrue(all(path.exists() for path in preserved))
                    finally:
                        client.delete()
            self.assertFalse(frames.exists())
            self.assertFalse(preview.exists())
            self.assertFalse(partial.exists())

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
        from vmi_analysis.gui.panels import centroid
        store = {MIGRATED_KEY: True, storage_key("vmi_picasso", "h5_dir"): "chosen data",
                 storage_key("energy_calibration", "alpha"): .0007,
                 storage_key("histogram", "abel_rotation_deg"): 17.5,
                 storage_key("vmi_picasso", "tuning_frame_index"): None,
                 storage_key("spectrum_plot", "x_max"): None}
        job = self.submit_form(centroid, store)
        self.assertNotIn("--config", job.cmd)
        cfg = parse_centroid_args(job.cmd[3:]).settings
        self.assertEqual(cfg["vmi_picasso"]["h5_dir"], "chosen data")
        self.assertEqual(cfg["vmi_picasso"]["energy_calibration"]["alpha"], .0007)
        self.assertEqual(cfg["vmi_picasso"]["histogram"]["abel_rotation_deg"], 17.5)
        self.assertIsNone(cfg["vmi_picasso"]["tuning_frame_index"])
        self.assertEqual(cfg["vmi_picasso"]["spectrum_plot"]["x_max"], -1)
        self.assertEqual(Path(job.extra["output_dir"]), ROOT / "centroiding/vmi_output")

    def test_ion_momenta_submits_its_own_inputs_and_reservations(self):
        from vmi_analysis.gui.panels import ion_momenta
        from vmi_analysis.gui.resources import canonical_path, job_access

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "ions.h5"
            source.touch()
            store = {MIGRATED_KEY: True, "ion_momenta.input": str(source),
                     "ion_momenta.source": "both", "ion_momenta.reference": "TDC2Falling",
                     "ion_momenta.bin_size_ns": 10, "ion_momenta.minimum": 8000,
                     "ion_momenta.maximum": 12000,
                     "ion_tof.source": "pixels", "ion_tof.minimum": 0}
            job = self.submit_form(ion_momenta, store, label="Plot positions and TOF")
            self.assertEqual(job.name, "Ion momenta")
            self.assertEqual(job.cmd[2], "vmi_analysis.ion_momenta")
            options = dict(zip(job.cmd[3::2], job.cmd[4::2]))
            self.assertEqual(options["--source"], "both")
            self.assertEqual(options["--reference"], "TDC2Falling")
            for flag, value in (("--bin-size", 10), ("--minimum", 8000), ("--maximum", 12000)):
                self.assertEqual(float(options[flag]), value)
            access = job_access(job)
            self.assertIn(canonical_path(source), access.reads)
            self.assertIn(canonical_path(job.extra["histogram"]), access.writes)

    def test_pes_abel_options_are_folded_by_default(self):
        from nicegui import Client, app, ui
        from vmi_analysis.gui.panels import pes

        manager = Mock()
        manager.jobs.return_value = []
        client = Client(ui.page("/_test_pes_abel_options"))
        try:
            with client, patch.object(app.storage, "_general", {MIGRATED_KEY: True}):
                pes.build(manager)
            expansion = next(
                element for element in client.elements.values()
                if element.__class__.__name__ == "Expansion"
                and "Abel inversion options" in getattr(element, "text", "")
            )
            self.assertFalse(expansion.value)
        finally:
            client.delete()

    def test_pes_submits_selected_method_order_and_image_override(self):
        from vmi_analysis.gui.panels import pes
        from vmi_analysis.gui.resources import canonical_path

        source = str(canonical_path("chosen image.npy", pes.REPO_ROOT))
        store = {MIGRATED_KEY: True, "pes.images": source, "pes.brightest_center": False,
                 "pes.abel_method": "three_point", "pes.beta_order": 6,
                 "pes.image_geometry": json.dumps({source: {"center": [23, 25], "angle": 90, "resolution": 2}})}
        job = self.submit_form(pes, store, label="Generate spectra")
        self.assertEqual(job.cmd[3:17], ["--abel-method", "three_point", "--beta-order", "6",
                                         "--quadrants", "ur,ul,ll,lr", "--symmetry-axis", "none",
                                         "--image", source, "23", "25", "90", "2"])
        self.assertEqual(job.reads, [source])
        self.assertEqual(job.writes, [job.extra["pes_spectra"]])

    def test_pes_has_shared_x_ranges_and_individual_y_ranges_with_aligned_beta_zeros(self):
        import numpy as np
        from nicegui import Client, app, ui
        from vmi_analysis.centroiding.plot_data import save_plot_data
        from vmi_analysis.gui.jobs import Job
        from vmi_analysis.gui.panels import pes

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pes.npz"
            for alpha, units in ((None, "pixel²"), (0.5, "eV")):
                with self.subTest(alpha=alpha):
                    q = np.array([1, 4, 9])
                    x = q * (alpha or 1)
                    save_plot_data(path, {"alpha": alpha, "beta_order": 4, "images": [{"path": "image.npy"}]},
                                   r2_0=q, x_0=x, y_0=[2, 2, 2], beta_x_0=x,
                                   moments_0=[[2., 2., 2.], [1., 1., 1.], [-2., -2., -2.]])
                    job = Job("PES", status="done", extra={"pes_spectra": str(path)})
                    client = Client(ui.page("/_test_pes_uniform_grid"))
                    callbacks = {}
                    try:
                        with client, patch.object(app.storage, "_general", {MIGRATED_KEY: True}), \
                                patch.object(ui.button, "on_click", lambda button, handler:
                                             callbacks.__setitem__(button.text, handler)), \
                                patch.object(pes, "plot_options", wraps=pes.plot_options) as plot_options, \
                                patch.object(ui, "notify") as notify:
                            pes.build(Mock(jobs=Mock(return_value=[job])))
                            callbacks["View latest spectra"]()
                            native_spectrum, native_beta = [call.args[0] for call in plot_options.call_args_list[-2:]]
                            self.assertEqual(native_beta.layout.yaxis2.overlaying, "y")
                            self.assertEqual([trace.xaxis for trace in native_beta.data], ["x", "x"])
                            self.assertEqual(native_spectrum.layout.xaxis.range, native_beta.layout.xaxis.range)
                            self.assertEqual(native_spectrum.layout.margin.l, native_beta.layout.margin.l)
                            self.assertEqual(native_spectrum.layout.margin.r, native_beta.layout.margin.r)
                            fields = {element.label: element for element in client.elements.values()
                                      if isinstance(element, ui.number)}
                            fields["β2 Y min (blank = auto)"].value = -1
                            fields["β2 Y max (blank = auto)"].value = 2
                            callbacks["Apply β axes"]()
                            np.testing.assert_allclose(plot_options.call_args.args[0].layout.yaxis.range, [-1, 2])
                            fields["β4 Y min (blank = auto)"].value = -10
                            fields["β4 Y max (blank = auto)"].value = 20
                            callbacks["Apply β axes"]()
                            native_axes = plot_options.call_args.args[0].layout
                            np.testing.assert_allclose(native_axes.yaxis.range, [-1, 2])
                            np.testing.assert_allclose(native_axes.yaxis2.range, [-10, 20])
                            fields["Spectrum X min (blank = auto)"].value = x[0]
                            fields["Spectrum X max (blank = auto)"].value = x[-1]
                            callbacks["Apply axes"]()
                            for call in plot_options.call_args_list[-2:]:
                                self.assertEqual(call.args[0].layout.xaxis.range, (x[0], x[-1]))
                            section = next(element for element in client.elements.values()
                                           if isinstance(element, ui.expansion) and element.text == "Uniform-grid spectra")
                            self.assertFalse(section.value)
                            section.open()
                            fields = {element.label: element for element in client.elements.values()
                                      if isinstance(element, ui.number)}
                            width = fields[f"Bin width ({units})"]
                            coordinate = "Kinetic energy" if alpha else "Squared detector radius"
                            x_min, x_max = (fields[f"{coordinate} {bound} ({units})"] for bound in ("min", "max"))
                            width.value = 3 * (alpha or 1)
                            x_min.value, x_max.value = x[0], x[-1]
                            fields["Bin-average intensity Y min"].value = -3
                            fields["Bin-average intensity Y max"].value = 7
                            fields["Resampled β2 Y min"].value = -1
                            fields["Resampled β2 Y max"].value = 2
                            fields["Resampled β4 Y min"].value = -10
                            fields["Resampled β4 Y max"].value = 5
                            apply_settings = callbacks["Apply plot settings"]
                            apply_settings()
                            fig = plot_options.call_args.args[0]
                            intensity, beta2, beta4 = fig.data
                            np.testing.assert_allclose(intensity.x, x[0] + width.value * np.array([0.5, 1.5, 2.5]))
                            np.testing.assert_allclose(intensity.y, [2, 2, 4 / 3])
                            np.testing.assert_allclose(beta2.x, intensity.x)
                            np.testing.assert_allclose(beta2.y, [0.5, 0.5, 0.5])
                            np.testing.assert_allclose(beta4.y, [-1, -1, -1])
                            self.assertEqual([beta2.xaxis, beta4.xaxis], ["x2", "x2"])
                            self.assertEqual([beta2.yaxis, beta4.yaxis], ["y2", "y3"])
                            self.assertEqual(fig.layout.yaxis3.overlaying, "y2")
                            self.assertLess(fig.layout.yaxis3.shift, 0)
                            self.assertEqual(beta4.line.color, fig.layout.yaxis3.tickfont.color)
                            self.assertIn(units, fig.layout.xaxis2.title.text)
                            self.assertEqual(fig.layout.xaxis.range, (x[0], x[-1]))
                            self.assertEqual(fig.layout.xaxis2.range, (x[0], x[-1]))
                            self.assertEqual(fig.layout.yaxis.range, (-3, 7))
                            beta_ranges = np.array([fig.layout.yaxis2.range, fig.layout.yaxis3.range])
                            self.assertTrue(np.all(beta_ranges[:, 0] <= [-1, -10]))
                            self.assertTrue(np.all(beta_ranges[:, 1] >= [2, 5]))
                            zero_positions = -beta_ranges[:, 0] / np.ptp(beta_ranges, axis=1)
                            self.assertAlmostEqual(*zero_positions)

                            x_min.value = x[-1]
                            apply_settings()
                            notify.assert_called_once_with("Axis minima must be below maxima.", type="warning")
                            self.assertIs(plot_options.call_args.args[0], fig)

                            x_min.value = None
                            apply_settings()
                            axis = plot_options.call_args.args[0].layout.xaxis2
                            self.assertTrue(axis.autorange)
                            self.assertEqual(axis.autorangeoptions.maxallowed, x[-1])
                            x_max.value = None
                            apply_settings()
                            axis = plot_options.call_args.args[0].layout.xaxis2
                            self.assertTrue(axis.autorange)
                            self.assertIsNone(axis.autorangeoptions.maxallowed)
                    finally:
                        client.delete()

    def test_centroid_frame_limit_sanitizes_on_blur(self):
        from nicegui import Client, app, ui
        from vmi_analysis.gui.panels import centroid

        key = storage_key("processing", "frame_limit")
        store = {MIGRATED_KEY: True, key: 5}
        manager = Mock()
        manager.jobs.return_value = []
        client = Client(ui.page("/_test_frame_limit"))
        try:
            with client, patch.object(app.storage, "_general", store):
                centroid.build(manager)
                frame_limit = next(element for element in client.elements.values()
                                   if isinstance(element, ui.number)
                                   and element.label == "Frame limit (0 = all)")
                frame_limit.sanitize()
                self.assertEqual(frame_limit.value, 5)
                frame_limit.value = -1
                frame_limit.sanitize()
                self.assertEqual(frame_limit.value, 0)
                self.assertEqual(store[key], 0)
        finally:
            client.delete()

    def test_centroid_median_subtraction_is_disabled_by_dark_subtraction(self):
        from nicegui import Client, app, ui
        from vmi_analysis.gui.panels import centroid

        key = storage_key("processing", "enable_median_subtraction")
        store = {MIGRATED_KEY: True}
        manager = Mock()
        manager.jobs.return_value = []
        client = Client(ui.page("/_test_median_subtraction"))
        try:
            with client, patch.object(app.storage, "_general", store):
                centroid.build(manager)
                checks = {element.text: element for element in client.elements.values()
                          if isinstance(element, ui.checkbox)}
                median = checks["Per-frame median subtraction"]
                dark = checks["Dark subtraction"]
                self.assertTrue(median.enabled)
                self.assertTrue(median.value)
                median.value = False
                dark.value = True
                self.assertFalse(median.enabled)
                dark.value = False
                self.assertTrue(median.enabled)
                self.assertFalse(median.value)
                self.assertFalse(store[key])
        finally:
            client.delete()

        config = parse_centroid_args(["--h5-dir", "data", "--no-median-subtraction"]).settings
        self.assertFalse(config["vmi_picasso"]["processing"]["enable_median_subtraction"])

    def test_gmd_map_form_submits_hdf5_pattern(self):
        from vmi_analysis.gui.panels import gmd_map
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            (root / "chosen data").mkdir()
            (root / "gmd.txt").touch()
            store = {MIGRATED_KEY: True,
                     "gmd_map.h5_dir": str(root / "chosen data"),
                     "gmd_map.h5_pattern": "*RAW-*.h5",
                     "gmd_map.gmd_path": str(root / "gmd.txt"),
                     "gmd_map.output_path": str(root / "map.txt"),
                     "gmd_map.n_files": 0}
            job = self.submit_form(gmd_map, store)
            self.assertEqual(job.cmd[job.cmd.index("--h5-dir") + 1], store["gmd_map.h5_dir"])
            self.assertEqual(job.cmd[job.cmd.index("--h5-pattern") + 1], "*RAW-*.h5")

    def test_binning_histogram_bins_start_blank_for_auto_binning(self):
        from nicegui import Client, app, ui
        from vmi_analysis.gui.panels import binning
        manager = Mock()
        manager.jobs.return_value = []
        client = Client(ui.page("/_test_binning_bins"))
        labels = {"GMD bin min", "GMD bin max", "GMD bin size",
                  "Weight bin min", "Weight bin max", "Weight bin size"}
        try:
            with client, patch.object(app.storage, "_general", {MIGRATED_KEY: True}):
                binning.build(manager)
            fields = {element.label: element.value
                      for element in client.elements.values()
                      if getattr(element, "label", None) in labels}
            self.assertEqual(set(fields), labels)
            self.assertTrue(all(value is None for value in fields.values()))
        finally:
            client.delete()

    def test_binning_uses_explicit_paths_and_dark_preserves_zero_frames(self):
        from vmi_analysis.gui.panels import binning, dark
        store = {MIGRATED_KEY: True, "binning.checkpoint": "chosen.h5",
                 "binning.output_dir": "chosen out", "binning.gmd_map": "chosen.txt",
                 "dark.frames": 0}
        job = self.submit_form(binning, store)
        self.assertNotIn("--config", job.cmd)
        self.assertEqual(job.cmd[job.cmd.index("--checkpoint-path") + 1], "chosen.h5")
        self.assertEqual(Path(job.extra["output_dir"]), ROOT / "centroiding/chosen out")
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "RAW-dark.h5").touch()
            job = self.submit_form(dark, {**store, "dark.dark_dir": folder})
        self.assertNotIn("--config", job.cmd)
        self.assertEqual(job.cmd[job.cmd.index("--frames") + 1], "0")

    def test_binning_weight_preview_honors_zero_and_keeps_plot_on_invalid_bins(self):
        from nicegui import Client, app, ui
        from vmi_analysis.gui.jobs import JobManager
        from vmi_analysis.gui.panels import binning

        async def check(source):
            callbacks = {}
            store = {MIGRATED_KEY: True, "binning.gmd_map": str(source), "binning.weight_power": 0}
            client = Client(ui.page("/_test_binning_weight_preview"))
            try:
                with client, patch.object(app.storage, "_general", store), \
                        patch.object(ui.button, "on_click", lambda button, handler:
                                     callbacks.__setitem__(button.text, handler)), \
                        patch.object(binning, "plot_options", wraps=binning.plot_options) as plot_options, \
                        patch.object(ui, "notify") as notify:
                    binning.build(JobManager())
                    await callbacks["Histogram weights"]()
                    plot, = (element for element in client.elements.values() if isinstance(element, ui.plotly))
                    trace = plot_options.call_args.args[0].data[0]
                    self.assertEqual(list(trace.x), [1.0])
                    self.assertEqual(list(trace.y), [2])
                    numbers = {element.label: element for element in client.elements.values()
                               if isinstance(element, ui.number)}
                    numbers["Weight bin min"].value = 4
                    numbers["Weight bin max"].value = 1
                    await callbacks["Histogram weights"]()
                    self.assertIn("X min must be less", notify.call_args.args[0])
                    self.assertIs(client.elements.get(plot.id), plot)
            finally:
                client.delete()

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "map.txt"
            source.write_text("0 a.h5 0 0 1 2 0\n1 a.h5 1 1 3 4 0\n", encoding="utf-8")
            asyncio.run(check(source))


if __name__ == "__main__":
    unittest.main()
