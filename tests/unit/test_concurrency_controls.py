import os
import unittest
from unittest.mock import patch

from nicegui import Client, app, ui
from vmi_analysis.gui import app as gui
from vmi_analysis.gui.jobs import JobManager


class ConcurrencyControlsTests(unittest.TestCase):
    def test_server_timers_reject_nonfinite_and_invalid_values(self):
        settings = (('VMI_GUI_SHUTDOWN_GRACE', gui.DEFAULT_SHUTDOWN_GRACE, 0.1),
                    ('VMI_GUI_RECONNECT_TIMEOUT', gui.DEFAULT_RECONNECT_TIMEOUT, 4.0))
        for name, default, minimum in settings:
            for value, expected in (('nan', default), ('inf', default), ('-inf', default),
                                    ('invalid', default), ('0', minimum), ('7.5', 7.5)):
                with self.subTest(name=name, value=value), patch.dict(os.environ, {name: value}):
                    self.assertEqual(gui._env_float(name, default, minimum), expected)

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
                    self.assertTrue(any(
                        element.__class__.__name__ == 'Label'
                        and getattr(element, 'text', '') == 'Maximum concurrent jobs'
                        for element in client.elements.values()))
                    control = next(e for e in client.elements.values()
                                   if e.__class__.__name__ == 'Select'
                                   and 'vmi-setting-concurrency' in getattr(e, '_classes', []))
                    self.assertEqual(control.options,
                                     list(range(1, len(gui.TABS) + 1)))
                    self.assertEqual(control.value, manager.max_concurrent)
                    control.value = value
                    self.assertEqual(manager.max_concurrent, value)
                    self.assertEqual(store['jobs.max_concurrent'], value)
                finally:
                    client.delete()

    def test_keep_alive_toggle_is_global_and_persisted(self):
        manager = JobManager()
        self.addCleanup(manager.shutdown)
        store = {'centroid.settings_migrated': True}
        with patch.object(gui, 'MANAGER', manager), \
                patch.object(app.storage, '_general', store):
            client = Client(ui.page('/_test_keep_alive'))
            try:
                with client:
                    gui.index()
                self.assertTrue(any(
                    element.__class__.__name__ == 'Button'
                    and getattr(element, 'text', '') == 'Settings'
                    for element in client.elements.values()))
                self.assertTrue(any(
                    element.__class__.__name__ == 'Label'
                    and getattr(element, 'text', '') == 'Keep running on disconnect'
                    for element in client.elements.values()))
                toggle = next(element for element in client.elements.values()
                              if element.__class__.__name__ == 'Checkbox'
                              and 'vmi-setting-keep-alive' in getattr(element, '_classes', []))
                self.assertTrue(toggle.value)
                toggle.value = False
                self.assertFalse(store[gui.KEEP_ALIVE_KEY])
            finally:
                client.delete()

    def test_restart_restores_limit_and_rejects_invalid_saved_values(self):
        for saved, expected in ((5, 5), (0, 2), ('invalid', 2)):
            manager = JobManager()
            self.addCleanup(manager.shutdown)
            with patch.object(gui, 'MANAGER', manager), \
                    patch.object(app.storage, '_general', {'jobs.max_concurrent': saved}), \
                    patch('vmi_analysis.paths.prepare_workspace'), \
                    patch.object(gui, 'startup_port', return_value=3157), \
                    patch.object(gui, 'init_zoom_overlay'), patch.object(ui, 'run'):
                gui.main()
            self.assertEqual(manager.max_concurrent, expected)
