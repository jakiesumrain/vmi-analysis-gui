import unittest
from unittest.mock import patch

from nicegui import Client, app, ui
from vmi_analysis.gui import app as gui
from vmi_analysis.gui.jobs import JobManager


class ConcurrencyControlsTests(unittest.TestCase):
    def test_limit_control_updates_scheduler_and_persists_for_reload(self):
        manager = JobManager()
        self.addCleanup(manager.shutdown)
        store = {'centroid.settings_migrated': True}
        with patch.object(gui, 'MANAGER', manager), patch.object(app.storage, '_general', store):
            for value in (4, 1):
                client = Client(ui.page('/_test_concurrency'))
                try:
                    with client:
                        gui.index()
                    control = next(e for e in client.elements.values()
                                   if getattr(e, 'label', None) == 'Maximum concurrent jobs')
                    self.assertEqual(control.__class__.__name__, 'Select')
                    self.assertEqual(control.options, list(range(1, 10)))
                    self.assertEqual(control.value, manager.max_concurrent)
                    control.value = value
                    self.assertEqual(manager.max_concurrent, value)
                    self.assertEqual(store['jobs.max_concurrent'], value)
                finally:
                    client.delete()

    def test_restart_restores_limit_and_rejects_invalid_saved_values(self):
        for saved, expected in ((5, 5), (0, 2), ('invalid', 2)):
            manager = JobManager()
            self.addCleanup(manager.shutdown)
            with patch.object(gui, 'MANAGER', manager), \
                    patch.object(app.storage, '_general', {'jobs.max_concurrent': saved}), \
                    patch('vmi_analysis.paths.prepare_workspace'), \
                    patch.object(gui, 'init_zoom_overlay'), patch.object(ui, 'run'):
                gui.main()
            self.assertEqual(manager.max_concurrent, expected)
