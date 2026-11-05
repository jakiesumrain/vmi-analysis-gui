"""Exercise the real server lifecycle through the same socket handshake as the GUI."""

import ast
import asyncio
import ctypes
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import uuid

import httpx
import socketio

# These routes exist only in the isolated test server, never in gui.app.
SERVER_CODE = """
import sys
from nicegui import app
from gui import app as gui
from gui.jobs import MANAGER, Job

@app.post('/_test/jobs')
def start_jobs(marker: str):
    MANAGER.submit(Job(name='running', cmd=[sys.executable, '-c', 'import time; time.sleep(60)']))
    MANAGER.submit(Job(name='queued', cmd=[sys.executable, '-c',
        'from pathlib import Path; import sys; Path(sys.argv[1]).touch()', marker]))

@app.get('/_test/jobs')
def get_jobs():
    return [{'status': j.status, 'pid': j._proc.pid if j._proc else None} for j in MANAGER.jobs()]

@app.post('/_test/shutdown')
def stop_server():
    app.shutdown()

gui.main()
"""


class GuiShutdownTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self.log = tempfile.TemporaryFile()
        self.clients = []
        self.http = httpx.AsyncClient(timeout=2, trust_env=False)
        self.proc = subprocess.Popen(
            [sys.executable, "-c", SERVER_CODE],
            cwd=Path(__file__).resolve().parents[1],
            env=dict(os.environ, VMI_GUI_PORT=str(port), VMI_GUI_NO_SHOW="1",
                     PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8"),
            stdout=self.log, stderr=subprocess.STDOUT,
        )
        self.addAsyncCleanup(self.cleanup_server)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                response = await self.http.get(self.url)
                response.raise_for_status()
                return
            except httpx.TransportError:
                if self.proc.poll() is not None:
                    break
                await asyncio.sleep(0.1)
        self.fail("Server failed to start: " + self.logs())

    def logs(self):
        self.log.seek(0)
        return self.log.read().decode("utf-8", errors="replace")

    async def cleanup_server(self):
        for client in self.clients:
            if client.connected:
                await client.disconnect()
        if self.proc.poll() is None:
            self.proc.terminate()
        await asyncio.to_thread(self.proc.wait, 10)
        await self.http.aclose()
        self.log.close()

    async def connect(self):
        response = await self.http.get(self.url)
        query = ast.literal_eval(re.search(r"query: (\{.*?\}),", response.text)[1])
        client = socketio.AsyncClient(reconnection=False)
        self.clients.append(client)
        await client.connect(self.url, socketio_path="_nicegui_ws/socket.io", transports=["websocket"])
        query.update(tab_id=uuid.uuid4().hex, document_id=uuid.uuid4().hex)
        self.assertTrue(await client.call("handshake", query, timeout=3))
        return client

    async def wait_stopped(self, timeout=5):
        try:
            await asyncio.to_thread(self.proc.wait, timeout)
        except subprocess.TimeoutExpired:
            self.fail("Server did not exit: " + self.logs())
        self.assertEqual(self.proc.returncode, 0, self.logs())

    async def test_last_tab_stops_server_promptly(self):
        client = await self.connect()
        start = time.monotonic()
        await client.disconnect()
        await self.wait_stopped()
        self.assertLess(time.monotonic() - start, 3)

    async def test_other_tab_keeps_server_alive(self):
        first, second = await self.connect(), await self.connect()
        await first.disconnect()
        await asyncio.sleep(1.4)
        self.assertIsNone(self.proc.poll())
        await second.disconnect()
        await self.wait_stopped()

    async def test_reload_and_new_disconnect_get_fresh_grace_period(self):
        first = await self.connect()
        await first.disconnect()
        await asyncio.sleep(0.8)
        replacement = await self.connect()
        # Disconnect again before the original shutdown deadline: its stale
        # task must not shorten the replacement's grace period.
        await replacement.disconnect()
        await asyncio.sleep(0.5)
        self.assertIsNone(self.proc.poll())
        third = await self.connect()
        await asyncio.sleep(1.3)
        self.assertIsNone(self.proc.poll())
        await third.disconnect()
        await self.wait_stopped()

    async def check_running_job_cleanup(self, close_tab):
        client = await self.connect()
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / "queued-job-started"
            response = await self.http.post(self.url + "/_test/jobs", params={"marker": str(marker)})
            response.raise_for_status()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                jobs = (await self.http.get(self.url + "/_test/jobs")).json()
                if jobs[0]["pid"] is not None:
                    break
                await asyncio.sleep(0.01)
            self.assertIsNotNone(jobs[0]["pid"])
            self.assertEqual(jobs[1]["status"], "queued")
            handle = None
            if sys.platform == "win32":
                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel.OpenProcess.restype = ctypes.c_void_p
                kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
                kernel.CloseHandle.argtypes = [ctypes.c_void_p]
                handle = kernel.OpenProcess(0x00100000, False, jobs[0]["pid"])
                self.assertTrue(handle)
            try:
                if close_tab:
                    await client.disconnect()
                else:
                    # Exercise the app shutdown hook independently of on_disconnect,
                    # through the same app.shutdown() path used by normal server exit.
                    response = await self.http.post(self.url + "/_test/shutdown")
                    response.raise_for_status()
                await self.wait_stopped()
                self.assertFalse(marker.exists(), "Shutdown launched the queued job")
                if handle:
                    self.assertEqual(kernel.WaitForSingleObject(handle, 1000), 0)
                else:
                    with self.assertRaises(ProcessLookupError):
                        os.kill(jobs[0]["pid"], 0)
                self.assertNotIn("cleanup error", self.logs())
            finally:
                if handle:
                    kernel.CloseHandle(handle)

    async def test_close_tab_cleans_up_running_and_queued_jobs(self):
        await self.check_running_job_cleanup(close_tab=True)

    async def test_server_shutdown_hook_cleans_up_jobs(self):
        await self.check_running_job_cleanup(close_tab=False)


if __name__ == "__main__":
    unittest.main()
