import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock, patch

from nicegui import Client, app, ui

from vmi_analysis.gui import app as gui, shared, workflow
from vmi_analysis.gui.jobs import Job, JobManager


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_active_jobs_labels_both_pes_backends(self):
        manager = JobManager()
        self.addCleanup(manager.shutdown)
        running = [Job(name, status="running") for name in
                   (gui.pes.JOB_NAME, gui.pes_julia.JOB_NAME, "Circularize")]
        client = Client(ui.page("/_test_active_pes_jobs"))
        try:
            with (client, patch.object(gui, "MANAGER", manager),
                  patch.object(manager, "running", return_value=running)):
                gui.index()
            labels = [element.text for element in client.elements.values() if isinstance(element, ui.label)]
            self.assertIn("PES (PyAbel) · PES (PBASEX) · Circularize", labels)
        finally:
            client.delete()

    async def test_manual_only_page_omits_agent_and_handoff_buttons(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            datasets = root / "datasets"
            datasets.mkdir()
            server = shared.SharedServer({
                "mode": "manual-only", "datasets": str(datasets), "workspaces": str(root / "users"),
                "accounts": {"alice": {"password_hash": shared.password_hash("test-password")}},
            })
            account = server.accounts["alice"]
            client = Client(ui.page("/_test_manual_only"))
            try:
                with (client, patch.object(shared, "SERVER", server),
                      patch.object(shared, "current_account", return_value=account),
                      patch.object(gui.agent, "build") as agent_panel,
                      patch.object(gui.ion_live, "build", wraps=gui.ion_live.build) as live_panel):
                    gui.index()
                agent_panel.assert_not_called()
                self.assertIsNone(live_panel.call_args.kwargs["ask_agent"])
                names = {element._props.get("name") for element in client.elements.values()}
                self.assertIn("Ion TOF", names)
                self.assertNotIn("Agent", names)
            finally:
                client.delete()

    async def test_settings_round_trip_panel_forms_without_starting_jobs(self):
        manager = JobManager()
        self.addCleanup(manager.shutdown)
        store = {
            "centroid.settings_migrated": True,
            "movie.h5_dir": "dataset-a",
            "calibration.session": "previous-fit.session.json",
            "calibration.abel": False,
            "calibration.placement": "raw",
            "centroid.raw_colormap": "Magma",
            "centroid.spectrum_plot.x_max": 90,
            "ion_live.endpoint": "tcp://acquisition-host:5556",
            "ion_live.tof_bin_width": 2.5,
            "ion_live.gmd2_min": 1,
            "ion_live.gmd2_max": 10,
            "ion_live.gmd2_bin_width": 0.5,
            "pes.beta_6_y_max": 2,
            "pes.uniform_ev_bin_width": 0.1,
            "pes.uniform_ev_x_min": 21,
            "pes.uniform_ev_intensity_y_max": 200,
            "pes.uniform_ev_beta_4_y_min": -2,
            "pes.uniform_pixel2_bin_width": 5,
            "pes_julia.mode": "shots",
            "pes_julia.shots": "corrected.h5",
            "pes_julia.windows": "1, 2\n2, 3",
            "pes_julia.window_batch": 16,
            "pes_julia.pbasex_repo": "checkout",
            gui.KEEP_ALIVE_KEY: True,
        }
        callbacks = {}

        def capture(button, callback):
            callbacks[button.text] = callback

        with TemporaryDirectory() as folder:
            path = Path(folder) / "experiment.json"
            page = ui.page("/_test_workflow")
            client = Client(page)
            try:
                with client, patch.object(gui, "MANAGER", manager), patch.object(app.storage, "_general", store), \
                        patch.object(ui.button, "on_click", capture), \
                        patch.object(gui, "LocalFilePicker", new=AsyncMock(return_value=str(path))), \
                        patch.object(ui.navigate, "reload") as reload, patch.object(ui, "notify") as notify:
                    gui.index()
                    publisher = next(element for element in client.elements.values()
                                     if isinstance(element, ui.input) and element.label == "Acquisition publisher")
                    publisher.value = "tcp://user:test-secret@acquisition-host:5556"
                    await callbacks["Save workflow"]()
                    saved = json.loads(path.read_text(encoding="utf-8"))
                    values = saved["settings"]
                    self.assertEqual(saved["schema"], workflow.SCHEMA)
                    self.assertEqual({key.split(".", 1)[0] for key in values},
                                     {"gmd_map", "movie", "dark", "accumulate", "centroid", "binning",
                                      "calibration", "circularize", "pes", "pes_julia", "ion_luna",
                                      "ion_tof", "ion_momenta", "ion_pipeline", "ion_live", "ion_unpack", "ion_tof_fallback",
                                      "ion_pipeline_fallback"})
                    self.assertEqual(values["accumulate.max_workers"], 6)
                    self.assertIsNone(values["accumulate.frame_index"])
                    self.assertEqual(values["calibration.session"], "previous-fit.session.json")
                    self.assertEqual(values["calibration.hold_order"], 1)
                    self.assertEqual(values["circularize.raw_color_min_pct"], 1)
                    self.assertEqual(values["circularize.circularized_color_max_pct"], 99.5)
                    self.assertEqual(values["centroid.raw_colormap"], "Magma")
                    self.assertEqual(values["centroid.tuning_colormap"], "Gray")
                    self.assertEqual(values["centroid.spectrum_plot.x_max"], 90)
                    for key in store:
                        if key.startswith(("ion_live.", "pes.")):
                            self.assertEqual(values[key], store[key])
                    self.assertEqual(values["pes_julia.mode"], "shots")
                    self.assertEqual(values["pes_julia.shots"], "corrected.h5")
                    self.assertEqual(values["pes_julia.window_batch"], 16)
                    self.assertEqual(values["pes_julia.pbasex_repo"], "checkout")
                    self.assertIsNone(values["pes_julia.uniform_pixel2_bin_width"])
                    self.assertEqual(values["ion_live.endpoint"], "tcp://acquisition-host:5556")
                    self.assertNotIn(gui.KEEP_ALIVE_KEY, values)

                    store.update({"movie.h5_dir": "dataset-b", "calibration.abel": True,
                                  "calibration.placement": "post_abel", "ion_live.tof_bin_width": 5,
                                  "pes.uniform_ev_bin_width": 0.2, "pes_julia.mode": "counts",
                                  "pes_julia.uniform_pixel2_bin_width": 7, "pes_julia.window_batch": 2,
                                  gui.KEEP_ALIVE_KEY: False})
                    values["accumulate.max_workers"] = "invalid"
                    path.write_text(json.dumps(saved), encoding="utf-8")
                    await callbacks["Load workflow"]()
                    self.assertEqual(store["movie.h5_dir"], "dataset-b")
                    self.assertIn("Invalid workflow value", notify.call_args.args[0])
                    reload.assert_not_called()

                    values["accumulate.max_workers"] = 6
                    values["calibration.hold_order"] = 0
                    values["centroid.spectrum_plot.x_max"] = 120
                    values["pipeline.h5_dir"] = "legacy-dataset"
                    values["calibration.color_min_pct"] = 10
                    del values["calibration.raw_color_min_pct"]
                    values["calibration.processed_color_min_pct"] = 25
                    del values["pes_julia.uniform_pixel2_bin_width"]
                    values["pes_julia.windows"] = "1, 2\n2, 3"
                    path.write_text(json.dumps(saved), encoding="utf-8")
                    await callbacks["Load workflow"]()
                    reload.assert_called_once_with()
                    self.assertEqual(store["movie.h5_dir"], "dataset-a")
                    self.assertNotIn("pipeline.h5_dir", store)
                    self.assertEqual(store["calibration.placement"], "raw")
                    self.assertEqual(store["calibration.hold_order"], 0)
                    self.assertEqual(store["centroid.spectrum_plot.x_max"], 120)
                    self.assertEqual(store["calibration.raw_color_min_pct"], 10)
                    self.assertEqual(store["calibration.processed_color_min_pct"], 25)
                    self.assertEqual(store["ion_live.tof_bin_width"], 2.5)
                    self.assertEqual(store["pes.uniform_ev_bin_width"], 0.1)
                    self.assertEqual(store["pes_julia.mode"], "shots")
                    self.assertEqual(store["pes_julia.window_batch"], 16)
                    self.assertEqual(store["pes_julia.pbasex_repo"], "checkout")
                    self.assertEqual(store["pes_julia.uniform_pixel2_bin_width"], 1)
                    self.assertFalse(store[gui.KEEP_ALIVE_KEY])
                    self.assertEqual(manager.jobs(), [])
                reloaded = Client(page)
                try:
                    with reloaded, patch.object(gui, "MANAGER", manager), patch.object(app.storage, "_general", store):
                        gui.index()
                    self.assertTrue(any(isinstance(element, ui.input) and element.value == "dataset-a"
                                        for element in reloaded.elements.values()))
                    settings = {element.label: element.value for element in reloaded.elements.values()
                                if isinstance(element, (ui.input, ui.number))}
                    self.assertEqual(settings["Acquisition publisher"], "tcp://acquisition-host:5556")
                    self.assertIn(2.5, [element.value for element in reloaded.elements.values()
                                        if isinstance(element, ui.number) and element.label == "TOF bin size (ns)"])
                finally:
                    reloaded.delete()
            finally:
                client.delete()

    async def test_workflow_save_preserves_other_json_files(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "calibration.json"
            original = '{"schema": "vmi-trig-calibration/2"}'
            path.write_text(original, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "VMI workflow"):
                workflow.save_workflow(path, {})
            self.assertEqual(path.read_text(encoding="utf-8"), original)
