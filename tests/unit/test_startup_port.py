import io
import os
import socket
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest.mock import patch

from vmi_analysis.gui import startup


class StartupPortTests(unittest.TestCase):
    def test_occupied_port_prints_copyable_powershell_commands(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            listener.listen()
            port = listener.getsockname()[1]
            output = io.StringIO()
            with patch.dict(os.environ, VMI_GUI_PORT=str(port)), redirect_stderr(output):
                with self.assertRaises(SystemExit) as exit:
                    startup.startup_port()
            self.assertEqual(exit.exception.code, 1)
            message = output.getvalue()
            self.assertIn(f'port {port} is unavailable', message)
            self.assertIn('$env:VMI_GUI_PORT = "', message)
            self.assertIn('uv run vmi-analysis', message)
            suggestion = int(message.split('$env:VMI_GUI_PORT = "')[1].split('"')[0])
            self.assertNotEqual(suggestion, port)
            self.assertIsNone(startup._bind_error(suggestion))

    def test_default_and_explicit_available_ports(self):
        with patch.dict(os.environ), patch.object(startup, '_bind_error', return_value=None) as probe:
            os.environ.pop('VMI_GUI_PORT', None)
            self.assertEqual(startup.startup_port(), 13517)
            probe.assert_called_with(13517, '0.0.0.0')
            os.environ['VMI_GUI_PORT'] = '3158'
            self.assertEqual(startup.startup_port(), 3158)

    def test_skips_unavailable_alternatives(self):
        output = io.StringIO()
        with patch.dict(os.environ, VMI_GUI_PORT='3157'), redirect_stderr(output), \
                patch.object(startup, '_bind_error',
                             side_effect=lambda port, host: None if port == 3159 else OSError('busy')):
            with self.assertRaises(SystemExit):
                startup.startup_port()
        self.assertIn('$env:VMI_GUI_PORT = "3159"', output.getvalue())

    def test_invalid_values_and_no_free_candidate_are_reported_cleanly(self):
        for raw in ('invalid', '', '0', '65536'):
            output = io.StringIO()
            with patch.dict(os.environ, VMI_GUI_PORT=raw), redirect_stderr(output), \
                    patch.object(startup, '_bind_error', return_value=OSError('unavailable')):
                with self.assertRaises(SystemExit):
                    startup.startup_port()
            self.assertIn('Invalid VMI_GUI_PORT', output.getvalue())
            self.assertIn('Set VMI_GUI_PORT to an available port', output.getvalue())

    def test_spawned_helpers_do_not_probe_the_parent_server_port(self):
        with patch.dict(os.environ, VMI_GUI_PORT='3157'), \
                patch.object(startup.multiprocessing, 'current_process', return_value=SimpleNamespace(name='SyncManager')), \
                patch.object(startup, '_bind_error') as probe:
            self.assertEqual(startup.startup_port(), 3157)
            probe.assert_not_called()
