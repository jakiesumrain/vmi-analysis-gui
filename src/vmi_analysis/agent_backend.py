"""Per-account provider connections and isolated Pi conversations."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, replace
from http.client import HTTPConnection, HTTPException, HTTPSConnection
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlsplit

from .agent_history import AgentHistory
from .paths import GENERATED_DIR

API_FORMATS = {
    "openai-completions": "OpenAI compatible",
    "openai-responses": "OpenAI responses",
    "anthropic-messages": "Anthropic messages",
}
ACTIVE = {"starting", "ready", "running", "stopping"}
MAX_RECORD_BYTES = 4 * 1024 * 1024
PROVIDER = "vmi-custom"
PANEL_REQUEST_TITLE = "VMI panel operation"
PANEL_REQUEST_TIMEOUT = 300
SLASH_CONTROLS = {
    "new": ("new_session", "Start a new conversation"),
    "compact": ("compact", "Compact the current conversation"),
    "session": ("get_session_stats", "Show session statistics"),
}
PANEL_EXTENSION = files("vmi_analysis.gui.assets") / "agent_panels.js"


@dataclass(frozen=True)
class AgentRequest:
    owner: str
    workspace: Path
    config_root: Path
    base_url: str
    model: str
    api_key: str = field(repr=False)
    api: str = "openai-completions"
    datasets: Path | None = None
    additional_models: tuple[str, ...] = ()
    image_models: tuple[str, ...] = ()
    analysis: Path | None = None
    session_dir: Path | None = None
    working_directory: Path | None = None

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.model, *self.additional_models)))


def _validate_base_url(base_url):
    if not base_url:
        raise ValueError("Enter the model endpoint's base URL.")
    if len(base_url) > 2048:
        raise ValueError("Keep the base URL below 2048 characters.")
    try:
        parsed = urlsplit(base_url)
        parsed.port
    except ValueError as exc:
        raise ValueError("Enter a valid base URL.") from exc
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("The base URL must use http or https and include a host.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("The base URL cannot contain credentials, a query, or a fragment.")
    return parsed


def _validate_api_key(api_key):
    if not api_key or len(api_key) > 16384 or any(c in api_key for c in "\r\n\x00"):
        raise ValueError("Enter an API key without line breaks.")


def _valid_model_id(model):
    return isinstance(model, str) and bool(model.strip()) and len(model) <= 256 and not any(
        char in model for char in "\r\n\x00"
    )


def validate_request(request: AgentRequest) -> None:
    _validate_base_url(request.base_url)
    if not all(map(_valid_model_id, request.model_ids)):
        raise ValueError("Enter valid model IDs.")
    if set(request.image_models) - set(request.model_ids):
        raise ValueError("Choose image-capable models from the configured model IDs.")
    _validate_api_key(request.api_key)
    if request.api not in API_FORMATS:
        raise ValueError("Choose a supported API format.")


def list_provider_models(base_url: str, api_key: str) -> list[str]:
    """Read model IDs from an OpenAI-compatible endpoint, without following redirects."""
    endpoint = _validate_base_url(base_url)
    _validate_api_key(api_key)
    connection_type = HTTPSConnection if endpoint.scheme == "https" else HTTPConnection
    connection = connection_type(endpoint.hostname, endpoint.port, timeout=8)
    try:
        connection.request("GET", f"{endpoint.path.rstrip('/')}/models",
                           headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError(f"The provider's model list returned HTTP {response.status}.")
        raw = response.read(MAX_RECORD_BYTES + 1)
    except (OSError, HTTPException, UnicodeError) as exc:
        raise ValueError("Could not load the model list from this endpoint.") from exc
    finally:
        connection.close()
    if len(raw) > MAX_RECORD_BYTES:
        raise ValueError("The provider's model list is too large.")
    try:
        payload = json.loads(raw)
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise ValueError("Missing models.")
    except (ValueError, UnicodeError) as exc:
        raise ValueError("The provider did not return a model list.") from exc
    models = list(dict.fromkeys(item["id"] for item in data
                               if isinstance(item, dict) and _valid_model_id(item.get("id"))
                               and "," not in item["id"]))
    if not models:
        raise ValueError("The provider returned no model IDs.")
    return models


def write_models_config(config_dir: Path, request: AgentRequest) -> Path:
    """Write endpoint settings with no literal credential in the file."""
    config_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    path = config_dir / "models.json"
    config = {
        "providers": {
            PROVIDER: {
                "baseUrl": request.base_url,
                "api": request.api,
                "apiKey": "$VMI_AGENT_API_KEY",
                "models": [
                    {"id": model, "name": model, "input": ["text", "image"] if model in request.image_models else ["text"]}
                    for model in request.model_ids
                ],
            }
        }
    }
    with path.open("x", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


def agent_arguments(model: str, extension: str, session_dir=None) -> list[str]:
    storage = ["--no-session"] if session_dir is None else ["--session-dir", str(session_dir), "--continue"]
    return ["--mode", "rpc", *storage, "--no-extensions", "--extension", extension,
            "--provider", PROVIDER, "--model", model]


def process_options() -> dict:
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    return {"start_new_session": True}


class SubprocessLauncher:
    """Local-development launcher; shared servers use a container instead."""

    def __init__(self, executable: str | None = None):
        self.executable = executable or os.environ.get("VMI_AGENT_COMMAND", "pi")

    def launch(self, request: AgentRequest, config_dir: Path, session_id: str) -> subprocess.Popen:
        executable = shutil.which(self.executable)
        if executable is None:
            raise RuntimeError(f"Agent executable '{self.executable}' was not found. "
                               "Install pi or set VMI_AGENT_COMMAND.")
        env = os.environ.copy()
        env["VMI_AGENT_API_KEY"] = request.api_key
        env["PI_CODING_AGENT_DIR"] = str(config_dir)
        try:
            return subprocess.Popen(
                [executable, *agent_arguments(request.model, str(PANEL_EXTENSION), request.session_dir)],
                cwd=request.working_directory or request.workspace,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                **process_options(),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Agent executable '{self.executable}' was not found. "
                               "Install pi or set VMI_AGENT_COMMAND.") from exc

    def request_stop(self, process: subprocess.Popen, session_id: str) -> None:
        return None


class ContainerLauncher:
    """Run pi with only the account workspace writable and datasets read-only."""

    def __init__(self, runtime: str | None = None, image: str | None = None):
        self.runtime = runtime or os.environ.get("VMI_AGENT_CONTAINER_RUNTIME", "docker")
        self.image = image or os.environ.get("VMI_AGENT_IMAGE", "vmi-analysis-agent:latest")

    @staticmethod
    def container_name(session_id: str) -> str:
        return f"vmi-agent-{session_id}"

    def command(self, request: AgentRequest, config_dir: Path, session_id: str,
                *, rootless: bool = False) -> list[str]:
        command = [
            self.runtime, "run", "--rm", "--interactive",
            "--name", self.container_name(session_id),
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--read-only",
            "--tmpfs", "/tmp:rw,nosuid,size=268435456",
            "--pids-limit", "256",
            "--memory", "1g",
            "--cpus", "1",
            "--env", "VMI_AGENT_API_KEY",
            "--volume", f"{config_dir}:/agent:ro",
            "--volume", f"{request.workspace}:/workspace:rw",
        ]
        if request.datasets is not None:
            command.extend(["--volume", f"{request.datasets}:/datasets:ro"])
        if request.analysis is not None:
            command.extend(["--volume", f"{request.analysis}:/analysis:ro"])
        if request.session_dir is not None:
            command.extend(["--volume", f"{request.session_dir}:/history:rw"])
        if rootless and Path(self.runtime).stem == "podman":
            command.append("--userns=keep-id")
        elif not rootless and sys.platform != "win32" and hasattr(os, "getuid"):
            command.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
        workdir = "/workspace"
        if request.working_directory is not None and request.working_directory != request.workspace:
            workdir += "/" + request.working_directory.relative_to(request.workspace).as_posix()
        command.extend([
            "--workdir", workdir,
            self.image,
            *agent_arguments(request.model, "/agent/agent_panels.js",
                             "/history" if request.session_dir is not None else None),
        ])
        return command

    def launch(self, request: AgentRequest, config_dir: Path, session_id: str) -> subprocess.Popen:
        # Keep the extension and its history redaction in sync with the server.
        (config_dir / "agent_panels.js").write_bytes(PANEL_EXTENSION.read_bytes())
        env = os.environ.copy()
        env["VMI_AGENT_API_KEY"] = request.api_key
        try:
            rootless = False
            if sys.platform == "linux":
                podman = Path(self.runtime).stem == "podman"
                template = "{{.Host.Security.Rootless}}" if podman else "{{json .SecurityOptions}}"
                info = subprocess.run([self.runtime, "info", "--format", template],
                                      capture_output=True, text=True, timeout=10, check=False)
                if info.returncode:
                    raise RuntimeError(f"Could not query the {self.runtime} service: {info.stderr.strip()}")
                rootless = info.stdout.strip() == "true" if podman else "rootless" in info.stdout
            return subprocess.Popen(
                self.command(request, config_dir, session_id, rootless=rootless),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                **process_options(),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Container runtime '{self.runtime}' was not found. "
                               "Install Docker or Podman and build the agent image.") from exc

    def request_stop(self, process: subprocess.Popen, session_id: str) -> None:
        if process.poll() is not None:
            return
        subprocess.run(
            [self.runtime, "stop", "--time", "3", self.container_name(session_id)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=8,
            check=False,
        )


def terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW,
                check=False,
            )
        else:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    finally:
        try:
            process.kill()
        except ProcessLookupError:
            pass


class AgentSession:
    def __init__(self, request: AgentRequest, launcher, *, clock=time.monotonic, history=None, chat=None):
        validate_request(request)
        self.id = uuid.uuid4().hex
        self.request = request
        self.launcher = launcher
        self.clock = clock
        self.status = "starting"
        self.error = ""
        self.created = time.time()
        self.history = history
        self.chat = chat
        self.entries = deque(history.entries(chat) if chat is not None else (), maxlen=1000)
        self.diagnostics = deque(maxlen=500)
        self.process: subprocess.Popen | None = None
        self.config_dir = request.config_root / self.id
        self._api_key = request.api_key
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._stderr_done = threading.Event()
        self._stderr_done.set()
        self._finalized = False
        self._assistant_index: int | None = None
        self._run_started = False
        self._stop_thread: threading.Thread | None = None
        self._command_counter = 0
        self._panel_requests = {}
        self._activity = ""
        self._model = request.model
        self._pending_model: tuple[str, str] | None = None
        self._commands = ()
        self._command_catalog_request = None
        self._command_catalog_ready = threading.Event()
        self._pending_control = None
        self._transcript_revision = 0
        self._active_tools = {}

    @property
    def active(self) -> bool:
        return self.status in ACTIVE

    @property
    def running(self) -> bool:
        return self.status == "running"

    def start(self) -> "AgentSession":
        try:
            write_models_config(self.config_dir, self.request)
            process = self.launcher.launch(self.request, self.config_dir, self.id)
            self.request = replace(self.request, api_key="")
            with self._lock:
                stopped_while_starting = self._stop_requested.is_set() or self._finalized
                if not stopped_while_starting:
                    self.process = process
                    self.status = "ready"
            if stopped_while_starting:
                try:
                    process.stdin.close()
                except (OSError, AttributeError):
                    pass
                self.launcher.request_stop(process, self.id)
                terminate_process_tree(process)
                process.wait(timeout=5)
                raise RuntimeError("The agent session stopped during startup.")
            self._stderr_done.clear()
            threading.Thread(target=self._read_stdout, args=(process.stdout,),
                             name=f"agent-rpc-{self.id[:8]}", daemon=True).start()
            threading.Thread(target=self._read_stderr, args=(process.stderr,),
                             name=f"agent-stderr-{self.id[:8]}", daemon=True).start()
            return self
        except Exception as exc:
            self.error = self._redact(str(exc))
            self.status = "failed"
            self._discard_credentials()
            raise RuntimeError(self.error or "Could not start the agent.") from exc

    def _redact(self, value) -> str:
        text = str(value)
        return text.replace(self._api_key, "[redacted]") if self._api_key else text

    def _append(self, kind: str, text: str = "", detail: str = "") -> None:
        with self._lock:
            self.entries.append({"kind": kind, "text": self._redact(text),
                                 "detail": self._redact(detail)})
            if kind != "assistant":
                self._assistant_index = None

    def _append_assistant(self, delta: str) -> None:
        with self._lock:
            if (self._assistant_index is None or self._assistant_index >= len(self.entries)
                    or len(self.entries) == self.entries.maxlen):
                self.entries.append({"kind": "assistant", "text": "", "detail": ""})
                self._assistant_index = len(self.entries) - 1
            entry = self.entries[self._assistant_index]
            entry["text"] = self._redact(entry["text"] + delta)

    def slash_commands(self):
        with self._lock:
            if (not self._command_catalog_ready.is_set() and self._command_catalog_request is None
                    and self.status in ("ready", "running")):
                self._command_catalog_request = self.send({"type": "get_commands"})
        self._command_catalog_ready.wait(timeout=2)
        with self._lock:
            return self._commands

    def run_slash_control(self, name: str, argument: str):
        if name == "new" and self.chat is not None:
            raise ValueError("Use New chat in the sidebar to keep this conversation in history.")
        command, _ = SLASH_CONTROLS[name]
        payload = {"type": command}
        if argument:
            if name != "compact":
                raise ValueError(f"/{name} takes no argument.")
            payload["customInstructions"] = argument
        with self._lock:
            if self.status != "ready" or self._pending_model or self._pending_control:
                raise RuntimeError("Wait for the Agent to finish before running this command.")
            self._pending_control = (self.send(payload), name)

    def _control_response(self, record):
        with self._lock:
            pending = self._pending_control
            if pending is None or record.get("id") != pending[0]:
                return
            self._pending_control = None
            if record.get("success") is not True:
                self._append("error", record.get("error", "The Agent rejected the command."))
                return
            name = pending[1]
            data = record.get("data") or {}
            if name == "new" and not data.get("cancelled"):
                self.entries.clear()
                self._assistant_index = None
                self._transcript_revision += 1
            elif name == "compact":
                self._append("status", "Conversation compacted.")
            elif name == "session":
                summary = {"Model": self._model, "Messages": data.get("totalMessages"), "Tokens": data.get("tokens")}
                self._append("status", "Session statistics", json.dumps(summary, ensure_ascii=False))

    def set_model(self, model: str) -> None:
        with self._lock:
            if self.status != "ready" or self._pending_model is not None or self._pending_control:
                raise RuntimeError("Wait for the Agent to finish before changing the model.")
            if model not in self.request.model_ids:
                raise ValueError("Choose a model configured for this session.")
            if model != self._model:
                request_id = self.send({"type": "set_model", "provider": PROVIDER, "modelId": model})
                self._pending_model = (request_id, model)

    def prompt(self, message: str, *, model_context: str = "", attachments: tuple[str, ...] = (),
               images: tuple[dict, ...] = (), linked_run: str = "") -> None:
        if not message.strip():
            raise ValueError("Enter an instruction first.")
        if len(message) > 100_000:
            raise ValueError("Keep each instruction below 100,000 characters.")
        context = model_context
        if attachments:
            context += ("\n\n[Files and folders selected by the user. These paths are available to the Agent.]\n"
                        + json.dumps(attachments, ensure_ascii=False))
        command = {"type": "prompt", "message": self._redact(message + context)}
        if images:
            command["images"] = images
        with self._lock:
            if self.status not in ("ready", "running"):
                raise RuntimeError("Start an agent session before sending an instruction.")
            if self._pending_model is not None:
                raise RuntimeError("Wait for the model change to finish before sending an instruction.")
            if self._pending_control is not None:
                raise RuntimeError("Wait for the current command to finish.")
            if images and self._model not in self.request.image_models:
                raise ValueError("This model is configured for text only. Select an image-capable model or remove the images.")
            self._append("user", message, "\n".join(filter(None, (*attachments, linked_run))))
            follow_up = self.status == "running"
            if follow_up:
                command["streamingBehavior"] = "followUp"
            else:
                self.status = "running"
                self._activity = "Waiting for the model…"
        try:
            self.send(command)
        except RuntimeError:
            with self._lock:
                if not follow_up and not self._run_started and self.status == "running":
                    self.status = "ready"
            raise

    def abort(self) -> None:
        with self._lock:
            if self.status != "running":
                return
            self._panel_requests.clear()
            self._activity = "Stopping the reply…"
        self.send({"type": "abort"})
        self._append("status", "Abort requested; waiting for the current run to stop.")

    def send(self, command: dict) -> str:
        with self._lock:
            process = self.process
            if process is None or process.poll() is not None or self._stop_requested.is_set():
                raise RuntimeError("The agent process is not available.")
            self._command_counter += 1
            request_id = f"{self.id[:8]}-{self._command_counter}"
        self._write_record(process, dict(command, id=request_id))
        return request_id

    def _write_record(self, process, record):
        payload = memoryview(json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n")
        with self._write_lock:
            assert process.stdin is not None
            try:
                while payload:
                    written = process.stdin.write(payload)
                    if not written:
                        raise BrokenPipeError()
                    payload = payload[written:]
                process.stdin.flush()
            except (OSError, ValueError) as exc:
                raise RuntimeError("The agent process closed its input.") from exc

    def panel_request(self):
        with self._lock:
            now = self.clock()
            expired = [key for key, (_, deadline) in self._panel_requests.items() if deadline <= now]
            for key in expired:
                del self._panel_requests[key]
            if self.running and self._panel_requests:
                key = next(iter(self._panel_requests))
                return key, self._panel_requests[key][0]
            return None

    def take_panel_request(self, key):
        with self._lock:
            request = self._panel_requests.pop(key, None)
            return (request is not None and self.running and request[1] > self.clock()
                    and not self._stop_requested.is_set())

    def answer_panel_request(self, key, value):
        with self._lock:
            process = self.process
            if process is None or process.poll() is not None or self._stop_requested.is_set():
                return
        self._write_record(process, {"type": "extension_ui_response", "id": key,
                                     "value": json.dumps(value, ensure_ascii=False)})

    def _record_panel_request(self, record):
        if record.get("method") != "input" or record.get("title") != PANEL_REQUEST_TITLE:
            return
        key, value = record.get("id"), record.get("placeholder")
        if not isinstance(key, str) or len(key) > 128 or not isinstance(value, str) or len(value) > 8192:
            return
        try:
            request = json.loads(value)
        except ValueError:
            return
        with self._lock:
            if isinstance(request, dict) and self.running and len(self._panel_requests) < 8:
                self._panel_requests.setdefault(key, (request, self.clock() + PANEL_REQUEST_TIMEOUT))

    def handle_record(self, record: dict) -> None:
        kind = record.get("type")
        if kind == "extension_ui_request":
            self._record_panel_request(record)
            return
        if kind == "response":
            command = record.get("command")
            if command == "get_commands":
                with self._lock:
                    if record.get("id") == self._command_catalog_request:
                        if record.get("success") is True:
                            self._commands = tuple(
                                (entry["name"], str(entry.get("description") or ""))
                                for entry in (record.get("data") or {}).get("commands", [])
                                if isinstance(entry, dict) and isinstance(entry.get("name"), str)
                            )
                        self._command_catalog_ready.set()
                return
            if command in (control[0] for control in SLASH_CONTROLS.values()):
                self._control_response(record)
                return
            if command == "set_model":
                with self._lock:
                    if self._pending_model is None or record.get("id") != self._pending_model[0]:
                        return
                    if record.get("success") is True:
                        self._model = self._pending_model[1]
                    self._pending_model = None
            if record.get("success") is False:
                if record.get("command") == "prompt":
                    with self._lock:
                        if not self._stop_requested.is_set() and not self._run_started:
                            self.status = "ready"
                self._append("error", record.get("error", "The agent rejected a command."))
                return
            if record.get("command") == "prompt":
                disposition = (record.get("data") or {}).get("disposition")
                with self._lock:
                    if not self._stop_requested.is_set():
                        self.status = "ready" if disposition == "handled" else "running"
            return
        if kind == "agent_start":
            with self._lock:
                self._run_started = True
                if not self._stop_requested.is_set():
                    self.status = "running"
                    self._activity = "Waiting for the model…"
        elif kind == "agent_settled":
            with self._lock:
                self._run_started = False
                self._panel_requests.clear()
                self._active_tools.clear()
                self._activity = ""
                self._save_transcript()
                if not self._stop_requested.is_set():
                    self.status = "ready"
            self._assistant_index = None
        elif kind == "message_update":
            update = record.get("assistantMessageEvent") or {}
            if update.get("type") == "text_delta":
                with self._lock:
                    self._activity = "Writing a reply…"
                self._append_assistant(str(update.get("delta", "")))
            elif update.get("type") == "error":
                self._append("error", update.get("error", "The model stream failed."))
        elif kind == "tool_execution_start":
            name = str(record.get("toolName", "tool"))
            with self._lock:
                self._active_tools[str(record.get("toolCallId", name))] = self._redact(name)[:80]
            args = record.get("args") or {}
            detail = ""
            if isinstance(args, dict):
                detail = str(args.get("command") or args.get("path") or args.get("file_path") or "")
            self._append("tool", name, detail)
        elif kind == "tool_execution_end":
            name = str(record.get("toolName", "tool"))
            with self._lock:
                self._active_tools.pop(str(record.get("toolCallId", name)), None)
                self._activity = "Waiting for the model…"
            outcome = "failed" if record.get("isError") else "finished"
            self._append("status", f"{name} {outcome}.")
        elif kind in ("auto_retry_start", "compaction_start"):
            message = record.get("errorMessage") or record.get("reason") or kind
            self._append("status", str(message))
        elif kind == "extension_error":
            self._append("error", record.get("error", "An agent extension failed."))

    def _read_stdout(self, stream) -> None:
        try:
            assert stream is not None
            while True:
                line = stream.readline(MAX_RECORD_BYTES + 1)
                if not line:
                    break
                if len(line) > MAX_RECORD_BYTES and not line.endswith(b"\n"):
                    while line and not line.endswith(b"\n"):
                        line = stream.readline(MAX_RECORD_BYTES + 1)
                    self._append("error", "The agent emitted an oversized protocol record.")
                    continue
                try:
                    record = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    self._append("error", f"Invalid agent protocol record: {exc}")
                    continue
                if isinstance(record, dict):
                    self.handle_record(record)
                else:
                    self._append("error", "The agent emitted a non-object protocol record.")
            assert self.process is not None
            returncode = self.process.wait()
            self._finalize("cancelled" if self._stop_requested.is_set() else
                           "stopped" if returncode == 0 else "failed",
                           "" if returncode == 0 else f"Agent process exited with code {returncode}.")
        except Exception as exc:
            self._append("error", f"Agent reader stopped: {exc}")
            self._finalize("failed", str(exc))
        finally:
            try:
                stream.close()
            except OSError:
                pass

    def _read_stderr(self, stream) -> None:
        try:
            assert stream is not None
            for raw in iter(stream.readline, b""):
                text = raw.decode("utf-8", errors="replace").rstrip()
                if text:
                    with self._lock:
                        self.diagnostics.append(self._redact(text))
        except OSError as exc:
            with self._lock:
                self.diagnostics.append(self._redact(f"Could not read agent diagnostics: {exc}"))
        finally:
            try:
                stream.close()
            except OSError:
                pass
            self._stderr_done.set()
            if self._finalized:
                self._discard_credentials()

    def _save_transcript(self):
        if self.chat is None:
            return
        try:
            entries = [dict(entry, text=self._redact(entry["text"]), detail=self._redact(entry["detail"]))
                       for entry in self.entries]
            self.chat = self.history.save_entries(self.chat, entries)
        except OSError:
            self._append("error", "Could not save the Agent transcript.")

    def _finalize(self, status: str, error: str = "") -> None:
        with self._lock:
            if self._finalized:
                return
            self._finalized = True
            self._pending_model = None
            self._pending_control = None
            self._command_catalog_ready.set()
            self.status = status
            self._panel_requests.clear()
            self.error = self._redact(error)
            self._assistant_index = None
            self._save_transcript()
        if error:
            self._append("error", error)
        if self._stderr_done.is_set():
            self._discard_credentials()

    def _discard_credentials(self) -> None:
        self._api_key = ""
        self.request = replace(self.request, api_key="")
        shutil.rmtree(self.config_dir, ignore_errors=True)

    def stop_async(self) -> None:
        with self._lock:
            if self._stop_thread is not None and self._stop_thread.is_alive():
                return
            self._stop_thread = threading.Thread(target=self.stop, name=f"agent-stop-{self.id[:8]}",
                                                 daemon=True)
            self._stop_thread.start()

    def stop_for_switch(self, *, busy_error="Finish or stop the reply in the active chat before sending here.") -> None:
        with self._lock:
            if self.active:
                if self.status != "ready" or self._pending_model or self._pending_control:
                    raise RuntimeError(busy_error)
                self._stop_requested.set()
                self.status = "stopping"
        self.stop()

    def stop(self) -> None:
        with self._stop_lock:
            with self._lock:
                if self._finalized:
                    if self._stderr_done.is_set():
                        self._discard_credentials()
                    return
                self._stop_requested.set()
                self.status = "stopping"
                process = self.process
            if process is None:
                self._finalize("cancelled")
                return
            try:
                if process.stdin is not None:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass
                self.launcher.request_stop(process, self.id)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    terminate_process_tree(process)
                    process.wait(timeout=5)
                self._finalize("cancelled")
            except Exception as exc:
                self._append("error", f"Agent cleanup required forced termination: {exc}")
                terminate_process_tree(process)
                self._finalize("cancelled", str(exc))
            finally:
                if self._stderr_done.is_set():
                    self._discard_credentials()

    def join_stop(self, timeout: float) -> None:
        thread = self._stop_thread
        if thread is not None:
            thread.join(timeout=timeout)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "status": self.status,
                "transcript_id": f"{self.id}:{self._transcript_revision}",
                "control_pending": self._pending_control is not None,
                "activity": (
                    f"Running {next(iter(self._active_tools.values()))}…"
                    if self._active_tools else self._activity
                ),
                "error": self.error,
                "model": self._model,
                "models": self.request.model_ids,
                "image_models": self.request.image_models,
                "model_pending": self._pending_model is not None,
                "base_url": self.request.base_url,
                "entries": [dict(entry) for entry in self.entries],
                "diagnostics": list(self.diagnostics),
            }


class AgentManager:
    def __init__(self, launcher=None):
        self.launcher = launcher
        self._sessions: dict[str, AgentSession] = {}
        self._connections: dict[str, tuple[AgentRequest, bool, AgentHistory | None]] = {}
        self._owners = {}
        self._lock = threading.Lock()
        self._stopping = False

    def _launcher(self, shared_mode: bool):
        if self.launcher is not None:
            return self.launcher
        if shared_mode or os.environ.get("VMI_AGENT_BACKEND") == "container":
            return ContainerLauncher()
        return SubprocessLauncher()

    def start(self, request: AgentRequest, *, shared_mode: bool,
              history=None, chat_id=None, project_id=None) -> AgentSession:
        validate_request(request)
        lock = self._owner_lock(request.owner)
        if not lock.acquire(blocking=False):
            raise RuntimeError("Stop the current agent session before starting another.")
        try:
            session = self._start(request, shared_mode=shared_mode, history=history,
                                  chat_id=chat_id, project_id=project_id)
            with self._lock:
                if self._stopping:
                    raise RuntimeError("The server is shutting down.")
                self._connections[request.owner] = (request, shared_mode, history)
            return session
        finally:
            lock.release()

    def _owner_lock(self, owner):
        with self._lock:
            return self._owners.setdefault(owner, threading.Lock())

    def connect_provider(self, request: AgentRequest, *, shared_mode: bool, history=None) -> None:
        validate_request(request)
        if history is not None and history.workspace != request.workspace.resolve():
            raise ValueError("Agent history belongs to a different workspace.")
        with self._owner_lock(request.owner), self._lock:
            if self._stopping:
                raise RuntimeError("The server is shutting down.")
            self._connections.setdefault(request.owner, (request, shared_mode, history))

    def connection(self, owner) -> AgentRequest | None:
        with self._lock:
            connection = self._connections.get(owner)
            return replace(connection[0], api_key="") if connection is not None else None

    def prompt_chat(self, owner, message, *, chat_id=None, project_id=None, model=None,
                    **prompt_options) -> AgentSession:
        with self._owner_lock(owner):
            with self._lock:
                connection = self._connections.get(owner)
                if self._stopping:
                    raise RuntimeError("The server is shutting down.")
            if connection is None:
                raise RuntimeError("Connect a provider in Settings before sending a message.")
            request, shared_mode, history = connection
            current = self.current(owner)
            if chat_id is not None and current is not None and current.active and (
                    current.chat is not None and current.chat.id == chat_id):
                session = current
            else:
                if model is not None:
                    if model not in request.model_ids:
                        raise ValueError("Choose a model configured for this provider.")
                    request = replace(request, model=model,
                                      additional_models=tuple(item for item in request.model_ids if item != model))
                if history is not None:
                    project = history.resumable_chat(chat_id).project_id if chat_id is not None else project_id
                    history.project_folder(project)
                if current is not None and current.active:
                    current.stop_for_switch()
                session = self._start(request, shared_mode=shared_mode, history=history,
                                      chat_id=chat_id, project_id=project_id)
            session.prompt(message, **prompt_options)
            return session

    def _start(self, request, *, shared_mode, history, chat_id, project_id):
        with self._lock:
            if self._stopping:
                raise RuntimeError("The server is shutting down.")
            current = self._sessions.get(request.owner)
            if current is not None and current.active:
                raise RuntimeError("Stop the current agent session before starting another.")
            chat = None
            if history is not None:
                if history.workspace != request.workspace.resolve():
                    raise ValueError("Agent history belongs to a different workspace.")
                chat = history.resumable_chat(chat_id) if chat_id is not None else history.create_chat(project_id)
                folder = history.project_folder(chat.project_id)
                directory = history.session_dir(chat)
                directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                request = replace(request, workspace=history.workspace, session_dir=directory, working_directory=folder)
            session = AgentSession(request, self._launcher(shared_mode), history=history, chat=chat)
            self._sessions[request.owner] = session
        try:
            session.start()
        except Exception as exc:
            session._append("error", f"Could not start the agent: {session.error or exc}")
            raise
        with self._lock:
            if self._stopping:
                session.stop_async()
                raise RuntimeError("The server is shutting down.")
        return session

    def current(self, owner: str) -> AgentSession | None:
        with self._lock:
            return self._sessions.get(owner)

    def set_archived(self, owner, history, *, chat_id=None, project_id=None, archived):
        with self._owner_lock(owner):
            session = self.current(owner)
            chat = session.chat if session is not None else None
            closes_chat = archived and chat is not None and (
                chat.id == chat_id if chat_id is not None else chat.project_id == project_id)
            if closes_chat and session.active:
                session.stop_for_switch(busy_error="Finish or stop the active reply before archiving.")
            history.set_archived(chat_id=chat_id, project_id=project_id, archived=archived)
            if closes_chat:
                with self._lock:
                    self._sessions.pop(owner, None)

    def delete_archived(self, owner, history, *, chat_id=None, project_id=None):
        if (chat_id is None) == (project_id is None):
            raise ValueError("Choose one archived chat or project to delete.")
        with self._owner_lock(owner):
            session = self.current(owner)
            chat = session.chat if session is not None else None
            is_current = chat is not None and (chat.id == chat_id if chat_id is not None else chat.project_id == project_id)
            if is_current:
                if session.active:
                    raise RuntimeError("Stop this chat before deleting it.")
                session.stop()
            if chat_id is not None:
                history.delete_archived_chat(chat_id)
            else:
                history.delete_archived_project(project_id)
            if is_current:
                with self._lock:
                    self._sessions.pop(owner, None)

    def stop(self, owner: str) -> None:
        """Disconnect the account's provider and stop its active chat."""
        with self._owner_lock(owner):
            with self._lock:
                self._connections.pop(owner, None)
                session = self._sessions.get(owner)
            if session is not None:
                session.stop()

    def begin_shutdown(self) -> None:
        with self._lock:
            self._stopping = True
            self._connections.clear()
            sessions = list(self._sessions.values())
        for session in sessions:
            if session.active:
                session.stop_async()

    def shutdown(self) -> None:
        self.begin_shutdown()
        deadline = time.monotonic() + 15
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            session.join_stop(max(0, deadline - time.monotonic()))
        for session in sessions:
            if session.active:
                raise RuntimeError("Agent process cleanup did not finish within 15 seconds")


AGENTS = AgentManager()


def local_request(*, owner: str, workspace: Path, base_url: str, model: str,
                  api_key: str, api: str, datasets: Path | None = None) -> AgentRequest:
    return AgentRequest(owner=owner, workspace=workspace,
                        config_root=GENERATED_DIR / "agent", base_url=base_url,
                        model=model, api_key=api_key, api=api, datasets=datasets)
