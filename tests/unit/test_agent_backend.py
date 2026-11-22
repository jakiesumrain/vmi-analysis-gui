import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from vmi_analysis.agent_backend import (
    AgentManager,
    AgentRequest,
    AgentSession,
    ContainerLauncher,
    SubprocessLauncher,
    process_options,
    validate_request,
    write_models_config,
)


class FakeProcess:
    def __init__(self, stdout=b"", stderr=b""):
        self.pid = 12345
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self):
        self.returncode = -9


class FakeLauncher:
    def __init__(self, process=None):
        self.process = process or FakeProcess()
        self.request = None
        self.stop_calls = 0

    def launch(self, request, config_dir, session_id):
        self.request = request
        return self.process

    def request_stop(self, process, session_id):
        self.stop_calls += 1
        self.process.returncode = 0


class AgentBackendTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def request(self, key="test-secret"):
        return AgentRequest(
            owner="alice",
            workspace=self.root / "workspace",
            config_root=self.root / "config",
            datasets=self.root / "datasets",
            base_url="https://llm.example/v1",
            model="model-a",
            api_key=key,
        )

    def test_models_config_uses_environment_interpolation_not_literal_key(self):
        request = self.request()
        config_dir = self.root / "config" / "session"
        path = write_models_config(config_dir, request)
        raw = path.read_text(encoding="utf-8")
        config = json.loads(raw)
        provider = config["providers"]["vmi-custom"]
        self.assertEqual(provider["apiKey"], "$VMI_AGENT_API_KEY")
        self.assertEqual(provider["baseUrl"], request.base_url)
        self.assertNotIn(request.api_key, raw)

    def test_connection_validation_rejects_credentials_in_url(self):
        request = self.request()
        for url in ("https://user:secret@llm.example/v1", "https://llm.example/v1?key=secret",
                    "file:///tmp/model"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_request(AgentRequest(**{**request.__dict__, "base_url": url}))
        with self.assertRaises(ValueError):
            validate_request(AgentRequest(**{**request.__dict__, "api": "unknown"}))

    def test_container_command_has_no_key_and_mounts_only_allowed_data(self):
        request = self.request()
        launcher = ContainerLauncher(runtime="podman", image="agent-image")
        command = launcher.command(request, self.root / "session", "abc123")
        joined = " ".join(command)
        self.assertNotIn(request.api_key, joined)
        self.assertEqual(command[:2], ["podman", "run"])
        self.assertIn("--read-only", command)
        self.assertIn("no-new-privileges", command)
        self.assertIn(f"{request.workspace}:/workspace:rw", command)
        self.assertIn(f"{request.datasets}:/datasets:ro", command)
        self.assertEqual(command[command.index("--name") + 1], "vmi-agent-abc123")
        self.assertNotIn("--privileged", command)

    def test_subprocess_receives_key_only_in_environment(self):
        request = self.request()
        process = FakeProcess()
        with patch("vmi_analysis.agent_backend.subprocess.Popen", return_value=process) as popen, \
                patch("vmi_analysis.agent_backend.shutil.which", return_value="pi-test.CMD") as which:
            SubprocessLauncher(executable="pi-test").launch(request, self.root / "session", "abc123")
        args, kwargs = popen.call_args
        which.assert_called_once_with("pi-test")
        self.assertEqual(args[0][0], "pi-test.CMD")
        self.assertNotIn(request.api_key, " ".join(args[0]))
        self.assertEqual(kwargs["env"]["VMI_AGENT_API_KEY"], request.api_key)
        self.assertEqual(kwargs["env"]["PI_CODING_AGENT_DIR"], str(self.root / "session"))
        self.assertEqual(kwargs["cwd"], request.workspace)

    def test_prompt_correlation_streaming_and_settled_state(self):
        request = self.request()
        session = AgentSession(request, FakeLauncher())
        session.process = FakeProcess()
        session.status = "ready"
        session.prompt("Inspect the dataset")
        command = json.loads(session.process.stdin.getvalue())
        self.assertEqual(command["type"], "prompt")
        self.assertEqual(command["message"], "Inspect the dataset")
        self.assertTrue(command["id"])
        session.handle_record({"id": command["id"], "type": "response", "command": "prompt",
                               "success": True, "data": {"disposition": "started"}})
        self.assertEqual(session.status, "running")
        session.prompt("Summarize afterwards")
        second = json.loads(session.process.stdin.getvalue().splitlines()[1])
        self.assertEqual(second["streamingBehavior"], "followUp")
        session.handle_record({"type": "message_update",
                               "assistantMessageEvent": {"type": "text_delta", "delta": "A result"}})
        session.handle_record({"type": "agent_settled", "aborted": False})
        self.assertEqual(session.status, "ready")
        self.assertEqual(session.snapshot()["entries"][-1]["text"], "A result")

    def test_second_prompt_before_first_response_is_queued(self):
        session = AgentSession(self.request(), FakeLauncher())
        session.process = FakeProcess()
        session.status = "ready"
        session.prompt("First")
        session.prompt("Second")
        commands = [json.loads(line) for line in session.process.stdin.getvalue().splitlines()]
        self.assertNotIn("streamingBehavior", commands[0])
        self.assertEqual(commands[1]["streamingBehavior"], "followUp")
        session.handle_record({"type": "response", "command": "prompt", "success": False,
                               "error": "First prompt rejected"})
        self.assertEqual(session.status, "ready")

    def test_reader_splits_only_on_lf_and_cleans_up_at_exit(self):
        records = [
            {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "one\u2028two"}},
            {"type": "agent_settled", "aborted": False},
        ]
        payload = b"".join(json.dumps(record, ensure_ascii=False).encode() + b"\n" for record in records)
        session = AgentSession(self.request(), FakeLauncher())
        session.process = FakeProcess()
        session.config_dir.mkdir(parents=True)
        (session.config_dir / "models.json").write_text("disposable", encoding="utf-8")
        session._read_stdout(io.BytesIO(payload))
        self.assertEqual(session.status, "stopped")
        self.assertEqual(session.snapshot()["entries"][0]["text"], "one\u2028two")
        self.assertFalse(session.config_dir.exists())
        self.assertEqual(session.request.api_key, "")
        self.assertEqual(session._api_key, "")

    def test_malformed_record_is_actionable_and_does_not_stop_reader(self):
        payload = b"not json\n" + json.dumps({"type": "agent_settled"}).encode() + b"\n"
        session = AgentSession(self.request(), FakeLauncher())
        session.process = FakeProcess()
        session._read_stdout(io.BytesIO(payload))
        self.assertEqual(session.status, "stopped")
        self.assertIn("Invalid agent protocol record", session.snapshot()["entries"][0]["text"])

    def test_stderr_and_external_errors_redact_the_key(self):
        session = AgentSession(self.request(), FakeLauncher())
        session._read_stderr(io.BytesIO(b"authentication failed for test-secret\n"))
        self.assertNotIn("test-secret", session.snapshot()["diagnostics"][0])
        session._append("error", "provider rejected test-secret")
        self.assertNotIn("test-secret", session.snapshot()["entries"][0]["text"])

    def test_late_stderr_is_redacted_before_credentials_are_discarded(self):
        process = FakeProcess()
        read_fd, write_fd = os.pipe()
        process.stderr = os.fdopen(read_fd, "rb")
        writer = os.fdopen(write_fd, "wb")
        session = AgentSession(self.request(), FakeLauncher(process)).start()
        try:
            deadline = time.monotonic() + 2
            while session.status != "stopped" and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(session.status, "stopped")
            self.assertEqual(session._api_key, "test-secret")
            self.assertTrue(session.config_dir.exists())

            writer.write(b"provider diagnostic: test-secret\n")
            writer.flush()
            writer.close()

            deadline = time.monotonic() + 2
            while (session._api_key or session.config_dir.exists()) and time.monotonic() < deadline:
                time.sleep(0.01)
            diagnostics = session.snapshot()["diagnostics"]
            self.assertEqual(diagnostics, ["provider diagnostic: [redacted]"])
            self.assertNotIn("test-secret", diagnostics)
            self.assertEqual(session._api_key, "")
            self.assertEqual(session.request.api_key, "")
            self.assertFalse(session.config_dir.exists())
        finally:
            writer.close()

    def test_stop_removes_configuration_and_all_in_memory_key_copies(self):
        launcher = FakeLauncher()
        session = AgentSession(self.request(), launcher)
        session.process = launcher.process
        session.status = "ready"
        session.config_dir.mkdir(parents=True)
        (session.config_dir / "models.json").write_text("disposable", encoding="utf-8")
        session.stop()
        self.assertEqual(session.status, "cancelled")
        self.assertEqual(launcher.stop_calls, 1)
        self.assertFalse(session.config_dir.exists())
        self.assertEqual(session.request.api_key, "")
        self.assertEqual(session._api_key, "")

    def test_start_failure_removes_temporary_configuration(self):
        class FailingLauncher:
            def launch(self, request, config_dir, session_id):
                raise FileNotFoundError("pi is not installed")

        manager = AgentManager(FailingLauncher())
        with self.assertRaisesRegex(RuntimeError, "pi is not installed"):
            manager.start(self.request(), shared_mode=False)
        session = manager.current("alice")
        self.assertEqual(session.status, "failed")
        self.assertEqual(session.request.api_key, "")
        self.assertFalse(session.config_dir.exists())

    def test_concurrent_start_for_same_account_is_rejected(self):
        entered = threading.Event()
        release = threading.Event()

        class BlockingLauncher(FakeLauncher):
            def launch(self, request, config_dir, session_id):
                entered.set()
                release.wait(timeout=2)
                return super().launch(request, config_dir, session_id)

        manager = AgentManager(BlockingLauncher())
        errors = []
        first = threading.Thread(target=lambda: self._capture_error(errors, manager.start, self.request()))
        first.start()
        self.assertTrue(entered.wait(timeout=1))
        with self.assertRaisesRegex(RuntimeError, "Stop the current"):
            manager.start(self.request(), shared_mode=False)
        release.set()
        first.join(timeout=2)
        self.assertEqual(errors, [])
        manager.shutdown()

    def test_shutdown_during_startup_cleans_the_late_process(self):
        entered = threading.Event()
        release = threading.Event()

        class BlockingLauncher(FakeLauncher):
            def launch(self, request, config_dir, session_id):
                entered.set()
                release.wait(timeout=2)
                return super().launch(request, config_dir, session_id)

        manager = AgentManager(BlockingLauncher())
        errors = []
        first = threading.Thread(target=lambda: self._capture_error(errors, manager.start, self.request()))
        first.start()
        self.assertTrue(entered.wait(timeout=1))
        manager.begin_shutdown()
        release.set()
        first.join(timeout=2)
        session = manager.current("alice")
        self.assertFalse(session.active)
        self.assertIsNotNone(session.launcher.process.returncode)
        self.assertEqual(session.request.api_key, "")
        self.assertTrue(errors)

    @staticmethod
    def _capture_error(errors, function, *args, **kwargs):
        try:
            function(*args, shared_mode=False, **kwargs)
        except RuntimeError as exc:
            errors.append(str(exc))

    def test_real_subprocess_rpc_round_trip_and_stop(self):
        script = self.root / "fake_pi.py"
        script.write_text(
            "import json, sys\n"
            "for line in sys.stdin.buffer:\n"
            "    command = json.loads(line)\n"
            "    response = {'id': command.get('id'), 'type': 'response', "
            "'command': command['type'], 'success': True, 'data': {'disposition': 'started'}}\n"
            "    print(json.dumps(response), flush=True)\n"
            "    print(json.dumps({'type': 'message_update', 'assistantMessageEvent': "
            "{'type': 'text_delta', 'delta': 'pong'}}), flush=True)\n"
            "    print(json.dumps({'type': 'agent_settled', 'aborted': False}), flush=True)\n",
            encoding="utf-8",
        )

        class ScriptLauncher:
            def launch(self, request, config_dir, session_id):
                env = os.environ.copy()
                env["VMI_AGENT_API_KEY"] = request.api_key
                env["PI_CODING_AGENT_DIR"] = str(config_dir)
                return subprocess.Popen(
                    [sys.executable, str(script)], env=env,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    bufsize=0, **process_options())

            def request_stop(self, process, session_id):
                return None

        session = AgentSession(self.request(), ScriptLauncher()).start()
        try:
            session.prompt("ping")
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                snapshot = session.snapshot()
                if snapshot["status"] == "ready" and any(
                        entry["text"] == "pong" for entry in snapshot["entries"]):
                    break
                time.sleep(0.01)
            self.assertEqual(session.status, "ready")
            self.assertTrue(any(entry["text"] == "pong" for entry in session.entries))
        finally:
            session.stop()
        self.assertEqual(session.status, "cancelled")
        self.assertFalse(session.config_dir.exists())


if __name__ == "__main__":
    unittest.main()
