"""Review, run, and inspect user-directed Python analyses."""

from collections import deque
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import re
import sys
from tempfile import NamedTemporaryFile

from nicegui import ui

from . import analysis_definition, shared
from .agent_panels import PanelOperation, _container_path, resolve_path
from .jobs import ACTIVE, Job
from .panels.common import file_url, num_input, path_input
from .workspace_io import open_regular_file

JOB_NAME = "Python analysis"
SCHEMA = "vmi.analysis-run/v1"
MAX_SCRIPT_BYTES = 64 * 1024
MAX_LOG_BYTES = 128 * 1024
MAX_ARTIFACTS = 100
RUN_ID = re.compile(r"[0-9a-f]{8}\Z")


def _inside_workspace(path):
    root = shared.workspace_root().resolve()
    path = shared.checked_path(path, cwd=root)
    if not path.is_relative_to(root):
        raise ValueError("Choose a file inside this workspace.")
    return path


def available():
    return not shared.SERVER or sys.platform == "linux"


def _analysis_home():
    account = shared.current_account()
    return (shared.analysis_root(account) if account else
            _inside_workspace(shared.workspace_root() / ".agent" / "analysis"))


def _record_path(path):
    root = _analysis_home()
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Choose a file inside your analysis records.")
    return path


def _run_root():
    return _record_path(_analysis_home() / "runs")


def _panel_root():
    return _record_path(_analysis_home() / "panels")


def _download(path):
    if shared.SERVER:
        ui.download.from_url(file_url(path) + "&download=true", filename=path.name)
    else:
        ui.download(path)


def _saved_panel(panel_id):
    if not isinstance(panel_id, str) or not RUN_ID.fullmatch(panel_id):
        raise ValueError("Choose a saved analysis panel ID.")
    path = _record_path(_panel_root() / f"{panel_id}.json")
    if not path.is_file() or path.stat().st_size > 512 * 1024:
        raise ValueError("This saved panel is unavailable.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("id") != panel_id:
        raise ValueError("This saved panel is invalid.")
    panel = analysis_definition.definition(data.get("panel"))
    source = data.get("source")
    if not isinstance(source, str) or len(source.encode("utf-8")) > MAX_SCRIPT_BYTES:
        raise ValueError("This saved panel contains an invalid Python snapshot.")
    if sha256(source.encode("utf-8")).hexdigest() != data.get("script_sha256"):
        raise ValueError("The saved Python snapshot has changed.")
    inputs, parameters = data.get("inputs"), data.get("parameters")
    if not isinstance(inputs, dict) or not isinstance(parameters, dict) or any(
            not isinstance(entry, dict) or not isinstance(entry.get("path"), str) for entry in inputs.values()):
        raise ValueError("The saved panel contains invalid field values.")
    analysis_definition.validate_values(panel, inputs, parameters)
    return data, panel, path


def _source(path):
    with path.open("rb") as handle:
        return _decode_source(handle.read(MAX_SCRIPT_BYTES + 1))


def _workspace_source(path, root):
    with open_regular_file(root, path) as handle:
        return _decode_source(handle.read(MAX_SCRIPT_BYTES + 1))


def _decode_source(raw):
    if len(raw) > MAX_SCRIPT_BYTES:
        raise ValueError("Keep Python scripts below 64 KiB.")
    try:
        return raw, raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Save the Python script as UTF-8.") from exc


def _bounded_json(value):
    try:
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("Use standard JSON values for analysis parameters.") from exc
    if len(raw.encode("utf-8")) > 16 * 1024:
        raise ValueError("Keep analysis parameters below 16 KiB.")
    return json.loads(raw)


def _write_atomic(path, content):
    with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        pending = Path(handle.name)
        try:
            handle.write(content)
        except OSError:
            handle.close()
            pending.unlink(missing_ok=True)
            raise
    try:
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def _record(path, record):
    _write_atomic(path, (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def _artifact_paths(folder):
    # The output contract is a flat directory of regular files.
    if folder.is_symlink() or not folder.is_dir():
        return [], False
    artifacts = []
    for path in folder.iterdir():
        if not path.is_symlink() and path.is_file():
            artifacts.append(path)
            if len(artifacts) > MAX_ARTIFACTS:
                return sorted(artifacts[:MAX_ARTIFACTS]), True
    return sorted(artifacts), False


def _fingerprint(path):
    info = path.stat()
    return {"size": info.st_size, "mtime_ns": info.st_mtime_ns}


def prepare(jobs, params):
    if not available():
        raise ValueError("The shared analysis worker requires a Linux server.")
    account = shared.current_account()
    workspace = shared.workspace_root()
    if ("script" in params) == ("saved_panel" in params):
        raise ValueError("Provide either a Python script or a saved panel ID.")
    panel = None
    if "saved_panel" in params:
        if "panel_definition" in params:
            raise ValueError("A saved panel uses its saved declaration.")
        bundle, panel, script = _saved_panel(params["saved_panel"])
        source = bundle["source"]
        raw = source.encode("utf-8")
    else:
        script = resolve_path(params["script"])
        if not script.is_relative_to(workspace) and (not account or not script.is_relative_to(_analysis_home())):
            raise ValueError("Choose a Python file inside this workspace.")
        if script.suffix.lower() != ".py" or not script.is_file():
            raise ValueError("Choose an existing .py file from this workspace.")
        with jobs.preview_read_access([script]):
            raw, source = (_workspace_source(script, workspace) if account and script.is_relative_to(workspace)
                           else _source(script))
        if "panel_definition" in params:
            panel = analysis_definition.definition(params["panel_definition"])
    title = params.get("title", panel["title"] if panel else "Python analysis")
    if not isinstance(title, str) or not title.strip() or len(title) > 100 or not title.isprintable():
        raise ValueError("Enter an analysis title of 1 to 100 characters.")
    raw_inputs = params.get("inputs", {})
    values = params.get("parameters", {})
    if not isinstance(raw_inputs, dict) or len(raw_inputs) > 32 or not isinstance(values, dict):
        raise ValueError("Provide up to 32 named input files and a JSON object of parameter values.")
    values = _bounded_json(values)
    if panel:
        analysis_definition.validate_values(panel, raw_inputs, values)
    inputs = {}
    for name, value in raw_inputs.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,63}", name):
            raise ValueError("Name each input with letters, digits, or underscores, starting with a letter.")
        path = resolve_path(value) if account else _inside_workspace(resolve_path(value))
        if not path.is_file():
            raise ValueError(f"Input {name} must be an existing file.")
        inputs[name] = {"path": str(path), **_fingerprint(path)}

    digest = sha256(raw).hexdigest()
    job = Job(JOB_NAME)
    folder = _run_root() / job.id
    manifest = folder / "run.json"
    code = folder / "script.py"
    output = folder / "outputs"
    logs = folder / "run.log"
    job.cwd = str(workspace)
    job.reads = [item["path"] for item in inputs.values()]
    worker_inputs = None
    if account:
        job.cmd = [sys.executable, "-m", "vmi_analysis.analysis_container", str(folder),
                   str(workspace), str(shared.SERVER.datasets)]
        job._launches_isolated_worker = True
        job.request_process_stop = lambda: (folder / ".cancel").touch()
        worker_inputs = {name: {**item, "path": _container_path(item["path"])} for name, item in inputs.items()}
    else:
        job.cmd = [sys.executable, "-m", "vmi_analysis.analysis_worker", str(manifest)]
        job.writes = [str(folder)]
    job.extra.update(source_review=source, script_sha256=digest)

    try:
        installed_version = version("vmi-analysis-gui")
    except PackageNotFoundError:
        installed_version = "source checkout"
    runtime = ({"execution": "isolated container", "configured_image": os.environ.get("VMI_AGENT_IMAGE", "vmi-analysis-agent:latest")}
               if account else {"execution": "local", "python": platform.python_version(), "executable": sys.executable})
    metadata = {"schema": SCHEMA, "id": job.id, "actor": account.name if account else "local",
                "title": title.strip(), "script": str(script), "script_sha256": digest,
                "inputs": inputs, "parameters": values, "runtime": {**runtime, "vmi_analysis": installed_version}}
    lockfile = Path(__file__).resolve().parents[3] / "uv.lock"
    if lockfile.is_file():
        metadata["runtime"]["project_lock_sha256"] = sha256(lockfile.read_bytes()).hexdigest()
    if panel:
        metadata["panel"] = panel

    def admit(record):
        if folder.resolve() != folder or record.owner != (account.name if account else None):
            raise ValueError("The run workspace changed. Prepare the analysis again.")
        folder.mkdir(parents=True, exist_ok=False, mode=0o700)
        output.mkdir(mode=0o700)
        code.write_bytes(raw)
        _record(manifest, {**metadata, "status": "queued", "created": record.created, "artifacts": []})
        if account:
            _record(folder / "worker.json", {
                "schema": SCHEMA, "id": record.id, "title": title, "script_sha256": digest,
                "parameters": values, "inputs": worker_inputs,
            })

    def finish(record, status):
        if (folder.resolve() != folder or any(path.is_symlink() for path in (manifest, logs, code))
                or sha256(_source(code)[0]).hexdigest() != digest):
            raise ValueError("The approved run record or Python snapshot changed.")
        lines = deque()
        size = 0
        for line in reversed(tuple(record.log)):
            data = (line + "\n").encode("utf-8", errors="replace")
            if size + len(data) > MAX_LOG_BYTES:
                break
            size += len(data)
            lines.appendleft(data)
        _write_atomic(logs, b"".join(lines))
        artifacts, truncated = _artifact_paths(output)
        if account and (image := folder / "worker-image.txt").is_file():
            metadata["runtime"]["image_id"] = image.read_text(encoding="utf-8").strip()
        _record(manifest, {**metadata, "status": status, "created": record.created, "started": record.started,
                           "finished": record.finished, "returncode": record.returncode,
                           "artifacts": [path.name for path in artifacts], "artifacts_truncated": truncated})
        record.extra.update(manifest=str(manifest), script=str(code), run_log=str(logs))
        record.extra.update({f"output/{path.name}": str(path) for path in artifacts})

    job.on_admit, job.on_terminal = admit, finish
    return job, {"Title": title.strip(), "Script": str(script), "SHA-256": digest,
                 "Panel declaration": panel,
                 "Inputs": {name: item["path"] for name, item in inputs.items()}, "Parameters": values,
                 "Run folder": str(folder),
                 "Execution": ("Isolated worker with read-only inputs and no network." if account else
                               "Local Python using the app environment and your local account permissions.")}


def _render_source(job):
    ui.label("Review the code before running it.").classes("vmi-caption")
    ui.code(job.extra["source_review"], language="python").classes("w-full max-h-80 overflow-auto")


def operation():
    return PanelOperation("analysis_python", {
        "panel": JOB_NAME,
        "summary": "Run a reviewed workspace Python script with the app environment. "
                   "The script receives the run manifest path as sys.argv[1]. Read inputs by alias from "
                   "manifest['inputs'][alias]['path'] and values from manifest['parameters']. "
                   "Write up to 100 generated files directly to the current output directory. "
                   "The run records the script, settings, logs, and output files. An optional panel_definition "
                   "builds a reusable form for the approved script. Use saved_panel to reopen a saved version.",
        "parameters": {
            "script": {"type": "path", "description": "UTF-8 Python file inside the workspace"},
            "saved_panel": {"type": "string", "description": "Saved panel ID in place of script"},
            "panel_definition": {"type": "object", "description": "Optional vmi.analysis-panel/v1 declaration. "
                                 "Provide schema, title, fields keyed by name with type (file, number, text, boolean) "
                                 "and label, and optional results keyed by filename with label and view (download or image)."},
            "title": {"type": "string", "description": "Title shown in Analysis runs"},
            "inputs": {"type": "object", "description": "Optional map of names to workspace files or shared datasets"},
            "parameters": {"type": "object", "description": "Optional JSON object supplied to the script"},
        },
    }, prepare, (JOB_NAME,), artifacts=lambda job: tuple(
        key for key in job.extra if key in ("manifest", "script", "run_log") or key.startswith("output/")),
        render_review=_render_source)


def _history():
    root = _run_root()
    if not root.is_dir():
        return []
    records = []
    for directory in root.iterdir():
        if not RUN_ID.fullmatch(directory.name) or directory.is_symlink():
            continue
        path = directory / "run.json"
        try:
            if not path.is_file() or path.is_symlink() or path.stat().st_size > 128 * 1024:
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            if (record["schema"] == SCHEMA and record["id"] == directory.name
                    and isinstance(record.get("title"), str) and type(record.get("created")) in (float, int)):
                records.append((record, directory))
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return sorted(records, key=lambda entry: entry[0].get("created", 0), reverse=True)[:100]


def _save_panel(record, folder):
    raw, source = _source(_record_path(folder / "script.py"))
    if sha256(raw).hexdigest() != record["script_sha256"]:
        raise ValueError("The approved Python snapshot has changed.")
    root = _panel_root()
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{record['id']}.json"
    if target.exists() or target.is_symlink():
        raise ValueError("This version has already been saved.")
    bundle = {key: record[key] for key in ("id", "panel", "inputs", "parameters", "script_sha256")}
    _record(target, {**bundle, "source": source})
    return target


def _panel_options():
    if not (root := _panel_root()).is_dir():
        return {}
    options = {}
    for path in sorted(root.glob("*.json")):
        try:
            _, panel, _ = _saved_panel(path.stem)
        except (OSError, ValueError, TypeError):
            continue
        options[path.stem] = f"{panel['title']} · {path.stem}"
        if len(options) == 100:
            break
    return options


def _open_panel(jobs, history_dialog, *, record=None, folder=None, saved_id=None):
    try:
        if saved_id is not None:
            record, panel, source_path = _saved_panel(saved_id)
            source = record["source"]
        else:
            panel = analysis_definition.definition(record["panel"])
            source_path = _record_path(folder / "script.py")
            raw, source = _source(source_path)
            if sha256(raw).hexdigest() != record["script_sha256"]:
                raise ValueError("The approved Python snapshot has changed.")
    except (OSError, ValueError, KeyError) as exc:
        ui.notify(str(exc), type="warning")
        return

    def run():
        inputs = {key: widget.value for key, widget in fields.items() if panel["fields"][key]["type"] == "file"}
        values = {key: widget.value for key, widget in fields.items() if key not in inputs}
        entry = ({"saved_panel": saved_id} if saved_id else
                 {"script": str(source_path), "panel_definition": panel})
        try:
            job, _ = prepare(jobs, {**entry, "inputs": inputs, "parameters": values})
            if job.extra["script_sha256"] != record["script_sha256"]:
                raise ValueError("The approved Python snapshot has changed.")
            jobs.submit(job)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            ui.notify(str(exc), type="warning")
            return
        dialog.close()
        history_dialog.close()
        open_history(jobs, selected_run=job.id)

    def save():
        try:
            target = _save_panel(record, folder)
        except (OSError, ValueError) as exc:
            ui.notify(str(exc), type="warning")
            return
        dialog.close()
        history_dialog.close()
        open_history(jobs)
        ui.notify(f"Saved panel {target.stem}.", type="positive")

    def discard():
        try:
            _record_path(folder / "draft-discarded").touch()
        except (OSError, ValueError) as exc:
            ui.notify(str(exc), type="warning")
            return
        dialog.close()
        history_dialog.close()
        open_history(jobs)

    with ui.dialog() as dialog, ui.card().classes("w-full max-w-3xl"):
        ui.label(panel["title"]).classes("text-lg")
        fields = {}
        for key, field in panel["fields"].items():
            kind, label = field["type"], field["label"]
            if kind == "file":
                widget = path_input(label, record["inputs"][key]["path"], browse="file")
            else:
                value = record["parameters"][key]
                if kind == "number":
                    widget = num_input(label, value, step="any")
                elif kind == "boolean":
                    widget = ui.checkbox(label, value=value)
                else:
                    widget = ui.input(label, value=value).props("outlined dense")
            fields[key] = widget.classes("w-full")
        with ui.expansion("Python source").classes("w-full"):
            ui.code(source, language="python").classes("max-h-64 overflow-auto")
        with ui.row().classes("items-center"):
            ui.button("Run panel", on_click=run)
            if saved_id:
                ui.button("Download panel definition", icon="download",
                          on_click=lambda: _download(source_path)).props("flat")
            else:
                ui.button("Save panel", on_click=save).props("flat")
                ui.button("Discard draft", on_click=discard).props("flat")
            ui.button("Close", on_click=dialog.close).props("flat")
    dialog.on("hide", lambda: dialog.delete())
    dialog.open()


def open_history(jobs, *, selected_run=None):
    """Show recent run records and live control in an Agent dialog."""
    if not available():
        return
    with ui.context.client, ui.dialog() as dialog, ui.card().classes("w-full max-w-5xl"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Analysis runs").classes("text-lg")
            ui.button("Close", on_click=dialog.close).props("flat")
        try:
            records = {entry[0]["id"]: entry for entry in _history()}
            saved = _panel_options()
        except (OSError, ValueError) as exc:
            records, saved = {}, {}
            ui.label(str(exc)).props('role="alert"')
        if saved:
            with ui.row().classes("w-full items-center"):
                panel_picker = ui.select(saved, value=next(iter(saved)), label="Saved panels").props("outlined dense")
                ui.button("Open saved panel",
                          on_click=lambda: _open_panel(jobs, dialog, saved_id=panel_picker.value)).props("flat")
        options = {key: f"{item['title']} · {key}" for key, (item, _) in records.items()}
        if not options:
            ui.label("Ask the Agent to prepare a Python analysis. Approved runs appear here.").classes("vmi-subtitle")
        else:
            picker = ui.select(options, value=selected_run if selected_run in options else next(iter(options)),
                               label="Run").props("outlined dense").classes("w-full")
            details = ui.column().classes("w-full")
            signature = None

            def refresh():
                nonlocal signature
                job = jobs.get(picker.value)
                folder = records[picker.value][1]
                try:
                    path = _record_path(folder / "run.json")
                    record = json.loads(path.read_text(encoding="utf-8"))
                    state = job.status if job else record["status"]
                    if job is None and state in ACTIVE:
                        state = "interrupted"
                    cursor = (picker.value, state, len(job.log), job.log[-1] if job.log else "") if job else (picker.value, state)
                    if cursor == signature:
                        return
                    signature = cursor
                    details.clear()
                    with details:
                        with ui.row().classes("items-center"):
                            ui.label(f"{record['title']} · {state}")
                            if job and state in ACTIVE:
                                ui.button("Cancel run", on_click=lambda: jobs.cancel(job.id)).props("flat")
                        if state in ACTIVE:
                            ui.label(job.wait_reason).classes("vmi-caption")
                        elif record.get("panel") and not (folder / "draft-discarded").exists():
                            ui.button("Open temporary panel", on_click=lambda: _open_panel(
                                jobs, dialog, record=record, folder=folder)).props("flat")
                        with ui.row().classes("items-center"):
                            for name, label in (("run.json", "Run record"), ("script.py", "Python source"), ("run.log", "Run log")):
                                target = folder / name
                                if target.is_file() and not target.is_symlink():
                                    ui.button(label, icon="download", on_click=lambda target=target: _download(target)).props("flat")
                            output = folder / "outputs"
                            artifacts = []
                            if state not in ACTIVE and not output.is_symlink():
                                artifacts = [output / name for name in record.get("artifacts", [])[:MAX_ARTIFACTS]
                                             if Path(name).name == name and (output / name).is_file()
                                             and not (output / name).is_symlink()]
                            views = analysis_definition.definition(record["panel"])["results"] if record.get("panel") else {}
                            for artifact in artifacts:
                                view = views.get(artifact.name, {})
                                ui.button(view.get("label", artifact.name), icon="download",
                                          on_click=lambda artifact=artifact: _download(artifact)).props("flat")
                        if record.get("artifacts_truncated"):
                            ui.label(f"Showing {MAX_ARTIFACTS} output files.").classes("vmi-caption")
                        for artifact in artifacts:
                            view = views.get(artifact.name, {})
                            if view.get("view") == "image":
                                ui.image(file_url(artifact)).props(f'alt={json.dumps(view["label"])}').classes("w-full")
                        if job and job.log:
                            ui.code("\n".join(tuple(job.log)[-60:]), language="text").classes("w-full max-h-56 overflow-auto")
                except (OSError, ValueError, KeyError) as exc:
                    details.clear()
                    with details:
                        ui.label(str(exc)).props('role="alert"')

            refresh()
            timer = shared.workspace_timer(0.5, refresh)
            dialog.on("hide", lambda: timer.cancel())
    dialog.on("hide", lambda: dialog.delete())
    dialog.open()
