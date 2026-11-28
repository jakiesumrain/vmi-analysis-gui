"""Check the subprocess contract for the two Julia PES estimators."""

from pathlib import Path
from subprocess import CalledProcessError
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from vmi_analysis import pes_julia
from vmi_analysis import pes_julia_fit


class JuliaPESBridgeTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.basis = self.root / "basis.h5"
        with h5py.File(self.basis, "w") as file:
            file["geometry"] = "half_plane"
        self.output = self.root / "spectra.csv"
        runtime = patch.object(pes_julia, "julia_runtime", return_value=("julia", self.root))
        runtime.start()
        self.addCleanup(runtime.stop)

    def test_count_image_preserves_detector_rows_and_columns(self):
        counts = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint64)
        source = self.root / "counts.npy"
        np.save(source, counts)

        def julia(command, *, check, env):
            self.assertTrue(check)
            mode, data, basis, output, *options = command[4:]
            self.assertEqual((mode, basis), ("counts", str(self.basis)))
            self.assertEqual(options[:9], ["0.5", "0.0", "90", "", "", "2.0", "2", "3", "half_plane"])
            self.assertEqual(Path(options[9]).read_text(encoding="ascii"), "0.1\t0.2\n0.2\t0.3\n")
            self.assertEqual(options[10], "uniform")
            self.assertEqual(len(options), 11)
            actual = np.fromfile(data, dtype=np.float64).reshape((2, 3), order="F")
            np.testing.assert_array_equal(actual, counts)
            Path(output).write_text("completed", encoding="utf-8")

        with patch.object(pes_julia.subprocess, "run", side_effect=julia):
            pes_julia.run("counts", source, self.basis, self.output, alpha=.5, resolution=2, rotation=90,
                          e_min=.1, e_max=.3, delta_e=.1)
        self.assertEqual(self.output.read_text(encoding="utf-8"), "completed")
        np.testing.assert_array_equal(np.load(source), counts)

    def test_shot_checkpoint_passes_through_and_a_failed_run_preserves_results(self):
        source = self.root / "checkpoint.h5"
        source.touch()
        self.output.write_text("previous", encoding="utf-8")

        def julia(command, *, check, env):
            self.assertTrue(check)
            mode, data, basis, output, *options = command[4:]
            self.assertEqual((mode, data, basis), ("shots", str(source), str(self.basis)))
            self.assertEqual(options[:9], ["1", "0.0", "0", "", "", "20", "0.95", "2", "42"])
            self.assertEqual(Path(options[9]).read_text(encoding="ascii"), "1.0\t3.0\n")
            self.assertEqual(options[10], "legacy")
            self.assertEqual(options[11], "8")
            Path(output).write_text("partial", encoding="utf-8")
            raise CalledProcessError(1, command)

        with patch.object(pes_julia.subprocess, "run", side_effect=julia):
            with self.assertRaises(CalledProcessError):
                pes_julia.run("shots", source, self.basis, self.output,
                              windows=[(1, 3)], replicates=20, block_length=2, window_batch=8)
        self.assertEqual(self.output.read_text(encoding="utf-8"), "previous")

    def test_rejects_interpolated_images_and_missing_bootstrap_windows(self):
        source = self.root / "interpolated.npy"
        np.save(source, np.ones((2, 3), dtype=float))
        with patch.object(pes_julia.subprocess, "run") as launch:
            with self.assertRaisesRegex(ValueError, "nonnegative integers"):
                pes_julia.run("counts", source, self.basis, self.output)
            with self.assertRaisesRegex(ValueError, "at least one energy window"):
                pes_julia.run("shots", source, self.basis, self.output)
        launch.assert_not_called()

    def test_uniform_grid_validates_exact_coverage_and_migrates_adjacent_windows(self):
        self.assertEqual(pes_julia.uniform_windows(30000, 40000, 1000)[-1], (39000, 40000))
        with self.assertRaisesRegex(ValueError, "whole multiple"):
            pes_julia.uniform_windows(30000, 40000, 3000)
        self.assertEqual(pes_julia.legacy_energy_grid("1, 2\n2, 3"),
                         {"e_min": 1, "e_max": 3, "delta_e": 1})
        self.assertEqual(pes_julia.legacy_energy_grid("1, 2\n3, 4"), {})

    def test_bootstrap_batch_requires_a_positive_integer(self):
        for batch in (None, 0, -2, 1.5, True):
            with self.subTest(batch=batch), self.assertRaisesRegex(ValueError, "Windows per batch"):
                pes_julia.analysis_options("shots", windows=[(1, 3)], window_batch=batch)

    def test_fitted_fwhm_windows_drive_both_error_estimators(self):
        spectrum = self.root / "native.csv"
        fit = self.root / "peak-fit.json"
        energy = np.linspace(0, 40, 161)
        intensity = 3 + 10 * np.exp(-((energy - 10) / 2)**2) + 7 * np.exp(-((energy - 26) / 3)**2)
        spectrum.write_text("kind,quantity,coordinate,order,value\n" + "".join(
            f"native,spectrum,{x},0,{y}\n" for x, y in zip(energy, intensity)), encoding="utf-8")
        pes_julia_fit.run(spectrum, fit, region=(4, 34), peaks=[], baseline="Linear")
        record = pes_julia_fit.load_fit(fit)
        self.assertEqual(record["guesses"], [])
        self.assertEqual(record["auto_profile"], "Gaussian")
        np.testing.assert_allclose([p["center"] for p in record["peaks"]], [10, 26], atol=1e-6)
        for peak in record["peaks"]:
            self.assertEqual(peak["window"], [peak["center"] - peak["fwhm"] / 2,
                                               peak["center"] + peak["fwhm"] / 2])
        counts, shots = self.root / "counts.npy", self.root / "shots.h5"
        np.save(counts, np.ones((2, 3), dtype=int))
        shots.touch()

        def julia(command, *, check, env):
            window_file, kind = command[17:19]
            self.assertEqual(kind, "peaks")
            self.assertEqual([[float(bound) for bound in line.split()] for line in
                              Path(window_file).read_text().splitlines()],
                             [p["window"] for p in record["peaks"]])
            Path(command[7]).write_text("complete")

        with patch.object(pes_julia.subprocess, "run", side_effect=julia):
            for mode, source in (("counts", counts), ("shots", shots)):
                pes_julia.run(mode, source, self.basis, self.output, fit_record=fit)


class JuliaCheckoutTests(unittest.TestCase):
    def test_cached_environments_follow_the_selected_checkout_and_its_dependencies(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            legacy_repo, selected = root / "old", root / "selected"
            checkout_project = f'name = "PBASEX"\nuuid = "{pes_julia.PBASEX_UUID}"\n'
            for repo in (legacy_repo, selected):
                (repo / "src").mkdir(parents=True)
                (repo / "src" / "PBASEX.jl").touch()
                (repo / "Project.toml").write_text(checkout_project, encoding="utf-8")
            cache = root / "cache"
            legacy = cache / "pbasex-julia"
            legacy.mkdir(parents=True)

            def link(environment, directory):
                (environment / "Project.toml").write_text(
                    f'[deps]\nPBASEX = "{pes_julia.PBASEX_UUID}"\n', encoding="utf-8")
                (environment / "Manifest.toml").write_text(
                    f'[[deps.PBASEX]]\nuuid = "{pes_julia.PBASEX_UUID}"\npath = "{directory.as_posix()}"\n',
                    encoding="utf-8")

            link(legacy, legacy_repo)

            def setup(command, *, check, env):
                self.assertTrue(check)
                staged = Path(command[2].removeprefix("--project="))
                self.assertTrue(staged.is_relative_to(cache))
                link(staged, Path(command[-1]))

            with (patch.object(pes_julia, "GENERATED_DIR", cache),
                  patch.object(pes_julia, "julia_executable", return_value="julia"),
                  patch.object(pes_julia.subprocess, "run", side_effect=setup) as launch):
                _, migrated = pes_julia.julia_runtime()
                self.assertNotEqual(migrated, legacy)
                self.assertEqual(pes_julia.julia_runtime(legacy_repo), ("julia", migrated))
                binary, environment = pes_julia.julia_runtime(selected)
                self.assertEqual(pes_julia.julia_runtime(selected), (binary, environment))
                self.assertEqual(launch.call_count, 2)
                (selected / "Project.toml").write_text(
                    checkout_project + '\n[deps]\nSparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"\n',
                    encoding="utf-8")
                _, refreshed = pes_julia.julia_runtime(selected)
                self.assertNotEqual(refreshed, environment)
                self.assertEqual(pes_julia.julia_runtime(selected), (binary, refreshed))
                self.assertEqual(launch.call_count, 3)
            self.assertEqual(pes_julia._linked_repo(legacy), legacy_repo)
            self.assertEqual(pes_julia._linked_repo(migrated), legacy_repo)
            self.assertEqual(pes_julia._linked_repo(environment), selected)
            self.assertEqual(pes_julia._linked_repo(refreshed), selected)
            self.assertFalse((selected / "Manifest.toml").exists())
