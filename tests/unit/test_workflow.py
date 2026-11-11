import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock, patch

from nicegui import Client, app, ui

from vmi_analysis.gui import app as gui, workflow
from vmi_analysis.gui.jobs import JobManager


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_settings_round_trip_panel_forms_without_starting_jobs(self):
        manager = JobManager()
        self.addCleanup(manager.shutdown)
        store = {
            "centroid.settings_migrated": True,
            "pipeline.h5_dir": "dataset-a",
            "calibration.session": "previous-fit.session.json",
            "calibration.abel": False,
            "calibration.placement": "raw",
            "centroid.raw_colormap": "Magma",
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
                    await callbacks["Save workflow"]()
                    saved = json.loads(path.read_text(encoding="utf-8"))
                    values = saved["settings"]
                    self.assertEqual(saved["schema"], workflow.SCHEMA)
                    self.assertEqual({key.split(".", 1)[0] for key in values},
                                     {"gmd_map", "movie", "dark", "accumulate", "centroid", "binning",
                                      "calibration", "circularize", "pes", "pipeline"})
                    self.assertEqual(values["accumulate.max_workers"], 6)
                    self.assertIsNone(values["accumulate.frame_index"])
                    self.assertTrue(values["pipeline.skip_existing"])
                    self.assertEqual(values["calibration.session"], "previous-fit.session.json")
                    self.assertEqual(values["calibration.hold_order"], 1)
                    self.assertEqual(values["circularize.raw_color_min_pct"], 1)
                    self.assertEqual(values["circularize.circularized_color_max_pct"], 99.5)
                    self.assertEqual(values["centroid.raw_colormap"], "Magma")
                    self.assertEqual(values["centroid.tuning_colormap"], "Gray")
                    self.assertNotIn(gui.KEEP_ALIVE_KEY, values)

                    store.update({"pipeline.h5_dir": "dataset-b", "calibration.abel": True,
                                  "calibration.placement": "post_abel", gui.KEEP_ALIVE_KEY: False})
                    values["accumulate.max_workers"] = "invalid"
                    path.write_text(json.dumps(saved), encoding="utf-8")
                    await callbacks["Load workflow"]()
                    self.assertEqual(store["pipeline.h5_dir"], "dataset-b")
                    self.assertIn("Invalid workflow value", notify.call_args.args[0])
                    reload.assert_not_called()

                    values["accumulate.max_workers"] = 6
                    values["calibration.hold_order"] = 0
                    values["calibration.color_min_pct"] = 10
                    del values["calibration.raw_color_min_pct"]
                    values["calibration.processed_color_min_pct"] = 25
                    path.write_text(json.dumps(saved), encoding="utf-8")
                    await callbacks["Load workflow"]()
                    reload.assert_called_once_with()
                    self.assertEqual(store["pipeline.h5_dir"], "dataset-a")
                    self.assertEqual(store["calibration.placement"], "raw")
                    self.assertEqual(store["calibration.hold_order"], 0)
                    self.assertEqual(store["calibration.raw_color_min_pct"], 10)
                    self.assertEqual(store["calibration.processed_color_min_pct"], 25)
                    self.assertFalse(store[gui.KEEP_ALIVE_KEY])
                    self.assertEqual(manager.jobs(), [])
                reloaded = Client(page)
                try:
                    with reloaded, patch.object(gui, "MANAGER", manager), patch.object(app.storage, "_general", store):
                        gui.index()
                    self.assertTrue(any(isinstance(element, ui.input) and element.value == "dataset-a"
                                        for element in reloaded.elements.values()))
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
