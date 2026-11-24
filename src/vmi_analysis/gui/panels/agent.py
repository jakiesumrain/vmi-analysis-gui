"""Interactive per-user agent sessions for shared data and private workspaces."""

import asyncio
from importlib.resources import files
import json
import os
from pathlib import PurePosixPath
import re
from uuid import uuid4

from nicegui import ui

from vmi_analysis.agent_backend import (AGENTS, API_FORMATS, SLASH_CONTROLS, AgentRequest, ContainerLauncher,
                                        list_provider_models)
from vmi_analysis.paths import GENERATED_DIR

from .. import shared, theme
from ..agent_attachments import (MAX_ATTACHMENT_BYTES, MAX_ATTACHMENT_FILES, MAX_BATCH_BYTES,
                                 attachment_root)
from ..agent_completion import path_completions
from ..agent_panels import PanelView
from ..jobs import TERMINAL
from .common import JobView, field_grid
from .picker import LocalFilePicker


MODEL_SETTINGS = "\0model-settings"
SLASH_BUILTINS = {
    "model": "Choose a model",
    "settings": "Open Agent connection settings",
    **{name: description for name, (_, description) in SLASH_CONTROLS.items()},
}
AGENT_CHAT_JS = files("vmi_analysis.gui.assets") / "agent_chat.js"


class AgentStatus:
    def __init__(self):
        self.badge = None
        self.spinner = None

    def place_status(self):
        self.badge = theme.status_badge("idle")
        self.spinner = ui.spinner("dots", size="sm", color="primary")
        self.spinner.visible = False

    def sync(self, session):
        status = session.status if session is not None else "idle"
        badge_status = {
            "starting": "queued",
            "ready": "done",
            "running": "running",
            "stopping": "cancelled",
            "failed": "failed",
        }.get(status, "idle")
        theme.set_badge(self.badge, badge_status)
        self.badge.text = status
        self.spinner.visible = status in ("starting", "running", "stopping")


def _workflow_card(jobs, name, targets, navigate):
    view = JobView(jobs, job_name=name)
    with theme.panel_card() as card:
        theme.form_section(name)
        with ui.row().classes("items-center gap-4"):
            steps = ui.label().classes("vmi-caption")
            stop = ui.button("Stop workflow", icon="stop", on_click=view.cancel).props("flat")
        with ui.row().classes("items-center gap-2"):
            for target in targets:
                ui.button(f"Open {target}", on_click=lambda target=target: navigate(target)).props("flat")
        view.place_console("Workflow console")
    card.classes("vmi-agent-workflow")
    card.set_visibility(False)

    def refresh():
        job = view.job
        card.set_visibility(job is not None)
        if job is not None:
            steps.text = " · ".join(f"{stage.name} {stage.status}" for stage in job.stages)
            stop.set_visibility(job.status in ("queued", "running"))
            stop.set_enabled(not job.cancel_requested)

    shared.workspace_timer(0.4, refresh)
    return view, refresh


def _attachment_reference(value, *, container):
    path = shared.checked_path(value)
    if not (path.is_file() or path.is_dir()):
        raise ValueError(f"Attachment is no longer available: {path.name}")
    if not container:
        return str(path)
    workspace = shared.workspace_root()
    if path.is_relative_to(workspace):
        return str(PurePosixPath("/workspace", *path.relative_to(workspace).parts))
    if shared.SERVER and path.is_relative_to(shared.SERVER.datasets):
        return str(PurePosixPath("/datasets", *path.relative_to(shared.SERVER.datasets).parts))
    raise ValueError("Choose an attachment inside the Agent workspace.")


def build(jobs, *, registry, navigate):
    with ui.column().classes("vmi-agent-page w-full gap-0"):
        return _build_page(jobs, registry=registry, navigate=navigate)


def _build_page(jobs, *, registry, navigate):
    account = shared.current_account()
    owner = account.name if account else "local"
    status = AgentStatus()
    connection_button = None

    def place_connection():
        nonlocal connection_button
        connection_button = ui.button("Connection", icon="tune").props(
            'flat dense no-caps aria-label="Connection" aria-expanded="false"')
        connection_button.classes("vmi-agent-connection-toggle")

    with ui.column().classes("vmi-agent-header w-full gap-0"):
        theme.panel_header(
            "Agent",
            "Inspect shared data and work in your workspace with your own model endpoint.",
            view=status,
            actions=place_connection,
        )

    with ui.column().classes("vmi-agent-connection w-full") as connection:
        with field_grid():
            base_url = ui.input("Base URL", placeholder="https://provider.example/v1") \
                .classes("w-full").props("outlined dense autocomplete=off")
            model = ui.input("Model IDs", placeholder="model-id, another-model-id") \
                .classes("w-full").props("outlined dense autocomplete=off")
        with field_grid():
            api = ui.select(API_FORMATS, value="openai-completions", label="API format") \
                .classes("w-full").props("outlined dense")
            api_key = ui.input("API key", password=True, password_toggle_button=True) \
                .classes("w-full").props("outlined dense autocomplete=off")
        ui.label("The key is used only for this session and is never saved or included in workflows. "
                 "Stopping the session removes its temporary configuration.").classes("vmi-subtitle")
        if shared.SERVER:
            ui.label("The isolated agent can read Shared datasets and change files only in My workspace. "
                     "Prompts and data content are sent to the endpoint you choose.").classes("vmi-subtitle")
        else:
            ui.label("Local development starts pi directly unless VMI_AGENT_BACKEND=container is set. "
                     "The agent can change files in this workspace.").classes("vmi-subtitle")
        with ui.row().classes("vmi-action-bar w-full items-center gap-2"):
            session_button = theme.run_button("Start session")
            load_models_button = ui.button("Load models", icon="sync").props("flat")
            load_models_button.visible = not shared.SERVER and os.environ.get("VMI_AGENT_BACKEND") != "container"
            load_models_note = ui.label("").classes("vmi-caption").props('role="status"')
        theme.form_section("Diagnostics")
        diagnostics_log = theme.log_console().classes("vmi-agent-diagnostics")
    connection.set_visibility(False)
    error_label = ui.label("").classes("vmi-agent-error vmi-agent-notice w-full") \
        .props('role="alert" tabindex="-1"')
    error_label.visible = False

    with ui.column().classes("vmi-agent-timeline w-full gap-0"):
        entries_column = ui.column().classes("vmi-agent-entries w-full")
    approval_view = ui.column().classes("vmi-agent-approval w-full gap-2")

    workflow_views = {name: _workflow_card(jobs, name, targets, navigate)
                      for name, targets in registry.workflows().items()}

    with ui.column().classes("vmi-agent-composer w-full") as composer:
        ui.button("Latest messages", icon="arrow_downward").props(
            'flat dense no-caps hidden').classes("vmi-agent-latest")
        context_note = ui.label().classes("vmi-agent-context")
        context_note.visible = False
        upload_status = ui.label("").classes("vmi-agent-upload-status vmi-caption").props('role="status"')
        upload_status.visible = False
        attachments_row = ui.row().classes("vmi-agent-attachments w-full items-center gap-2")
        attachments_row.visible = False
        prompt = (
            ui.textarea("Instruction", placeholder="Message the Agent")
            .classes("vmi-agent-prompt w-full")
            .props('borderless dense autogrow rows=1 aria-label="Instruction" autocomplete=off '
                   'aria-description="Paste or drop files to attach them"')
        )
        completion_event = f"agent-completion-{prompt.id}"
        upload_event = f"agent-upload-{prompt.id}"
        composer.props(
            f'data-agent-upload-event="{upload_event}" data-agent-max-files="{MAX_ATTACHMENT_FILES}" '
            f'data-agent-max-file-bytes="{MAX_ATTACHMENT_BYTES}" data-agent-max-batch-bytes="{MAX_BATCH_BYTES}" '
            'aria-busy="false"')
        ui.element("div").classes("vmi-agent-completions").props(
            f'role="listbox" aria-label="Suggestions" hidden '
            f'data-agent-event="{completion_event}" data-agent-prompt="{prompt.id}"')
        with ui.row().classes("vmi-agent-composer-actions w-full items-center no-wrap"):
            with ui.button(icon="add").props(
                    'flat round aria-label="Attach file or folder"').classes("vmi-agent-attach"):
                ui.tooltip("Attach file or folder · paste or drop files")
                with ui.menu().props('anchor="top left" self="bottom left" auto-close'):
                    attach_file = ui.menu_item("Attach file")
                    attach_folder = ui.menu_item("Attach folder")
            with ui.row().classes("vmi-agent-composer-status items-center gap-2 no-wrap"):
                activity_spinner = ui.spinner("dots", size="sm", color="primary").props('aria-hidden="true"')
                activity_label = ui.label().classes("vmi-caption").props('role="status"')
            model_select = ui.select({}, label="Model").props(
                'borderless dense hide-dropdown-icon aria-label="Model"').classes("vmi-agent-model")
            with ui.button(icon="arrow_upward").props(
                    'round color=primary unelevated aria-label="Send"').classes("vmi-agent-send") as send_button:
                send_tooltip = ui.tooltip("Send (Enter)")
        with ui.row().classes("vmi-agent-drop-label items-center justify-center gap-2").props('aria-hidden="true"'):
            ui.icon("upload_file")
            ui.label("Drop files to attach")

    rendered = []
    rendered_session = None
    diagnostics_cursor = 0
    empty_state = None
    approval = None
    view_context = None
    active_review = None
    attached_paths = []
    updating_models = False
    stop_visible = False
    loading_models = False
    uploading = 0
    upload_note = ""

    def configured_models():
        values = (model.value or "").split(",")
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    def update_models(snapshot, active, session_status):
        nonlocal updating_models
        models = snapshot.get("models", ()) if active else configured_models()
        selected = snapshot.get("model") if active else model_select.value
        if selected not in models:
            selected = next(iter(models), None)
        options = {model_id: model_id for model_id in models}
        options[MODEL_SETTINGS] = "Configure models…"
        if options != model_select.options or selected != model_select.value:
            updating_models = True
            try:
                model_select.set_options(options, value=selected)
            finally:
                updating_models = False
        model_select.set_enabled(not active or session_status == "ready" and not snapshot.get("model_pending"))

    def request_model_change(event):
        if updating_models or not event.value:
            return
        if event.value == MODEL_SETTINGS:
            if not connection.visible:
                toggle_connection()
            refresh()
        elif (session := AGENTS.current(owner)) is not None and session.active:
            return change_model(session, event.value)

    async def load_models():
        nonlocal loading_models
        session = AGENTS.current(owner)
        if container_mode() or loading_models or (session is not None and session.active):
            return
        if api.value not in ("openai-completions", "openai-responses"):
            return
        clear_error()
        endpoint, key = base_url.value.strip(), api_key.value or ""
        loading_models = True
        load_models_note.text = "Loading models…"
        refresh()
        try:
            discovered = await asyncio.to_thread(list_provider_models, endpoint, key)
            if endpoint == base_url.value.strip() and key == (api_key.value or ""):
                current = configured_models()
                model.value = ", ".join(dict.fromkeys((*current, *discovered)))
                load_models_note.text = f"Loaded {len(discovered)} model IDs."
            else:
                load_models_note.text = ""
        except ValueError as exc:
            load_models_note.text = str(exc)
        finally:
            loading_models = False
            refresh()

    async def change_model(session, value):
        try:
            await asyncio.to_thread(session.set_model, value)
        except (RuntimeError, ValueError) as exc:
            show_error(exc)
        refresh()

    def container_mode():
        session = AGENTS.current(owner)
        return bool(shared.SERVER or os.environ.get("VMI_AGENT_BACKEND") == "container"
                    or isinstance(getattr(session, "launcher", None), ContainerLauncher))

    def refresh_attachments():
        attachments_row.clear()
        attachments_row.visible = bool(attached_paths)
        with attachments_row:
            for path in attached_paths:
                with ui.row().classes("vmi-agent-attachment items-center no-wrap"):
                    ui.label(path.name or str(path)).classes("ellipsis").tooltip(str(path))
                    ui.button(icon="close", on_click=lambda path=path: remove_attachment(path)).props(
                        'flat dense round').tooltip(f"Remove {path.name or path}")

    def remove_attachment(path):
        if path in attached_paths:
            attached_paths.remove(path)
            refresh_attachments()

    def add_attachment(selected):
        path = shared.checked_path(selected)
        _attachment_reference(path, container=container_mode())
        if path not in attached_paths:
            attached_paths.append(path)
            refresh_attachments()

    def add_uploaded(paths):
        if not isinstance(paths, list) or not paths or len(paths) > MAX_ATTACHMENT_FILES:
            raise ValueError("The upload returned an invalid attachment list.")
        root = attachment_root()
        selected = []
        for value in paths:
            if not isinstance(value, str):
                raise ValueError("The upload returned an invalid attachment path.")
            path = shared.checked_path(value)
            if not path.is_relative_to(root):
                raise ValueError("The upload is outside the Agent attachment workspace.")
            _attachment_reference(path, container=container_mode())
            if path not in attached_paths and path not in selected:
                selected.append(path)
        attached_paths.extend(selected)
        refresh_attachments()

    def receive_upload(event):
        nonlocal uploading, upload_note
        payload = event.args
        if not isinstance(payload, dict):
            return
        action = payload.get("action")
        if action == "start":
            clear_error()
            uploading += 1
            count = payload.get("count")
            upload_note = f"Uploading {count} attachment{'s' if count != 1 else ''}…"
        elif action == "add":
            try:
                add_uploaded(payload.get("paths"))
            except (OSError, RuntimeError, ValueError) as exc:
                show_error(exc)
        elif action == "error":
            show_error(str(payload.get("message") or "Could not upload the attachments."))
        elif action == "finish":
            uploading = max(0, uploading - 1)
            if not uploading:
                upload_note = ""
        upload_status.text = upload_note
        upload_status.visible = bool(upload_note)
        refresh()

    async def attach_path(mode):
        clear_error()
        try:
            selected = await LocalFilePicker(start=shared.workspace_root(), mode=mode)
            if selected:
                add_attachment(selected)
        except (OSError, RuntimeError, ValueError) as exc:
            show_error(exc)

    def completion_roots():
        workspace = (("workspace", shared.workspace_root()),)
        return (*workspace, ("datasets", shared.SERVER.datasets)) if shared.SERVER else workspace

    async def complete(event):
        payload = event.args
        if not isinstance(payload, dict):
            return
        kind, query, request_id = payload.get("kind"), payload.get("query"), payload.get("id")
        if type(request_id) is not int or not isinstance(query, str) or len(query) > 256:
            return
        if kind == "@":
            try:
                roots = completion_roots()
                items = await asyncio.to_thread(path_completions, query, roots)
            except (OSError, RuntimeError, ValueError):
                items = []
        elif kind == "/":
            commands = dict(SLASH_BUILTINS)
            session = AGENTS.current(owner)
            if session is not None and session.active:
                try:
                    for name, detail in await asyncio.to_thread(session.slash_commands):
                        if re.fullmatch(r"[\w:.-]+", name):
                            commands.setdefault(name, detail)
                except RuntimeError:
                    pass
            items = [{"label": f"/{name}", "detail": detail, "value": f"/{name}"}
                     for name, detail in commands.items() if name.casefold().startswith(query.casefold())]
        else:
            return
        ui.run_javascript(f"window.vmiAgentCompletionResults?.({prompt.id}, {request_id}, {json.dumps(items)})")

    def toggle_connection():
        connection.visible = not connection.visible
        connection_button.props(f'aria-expanded="{str(connection.visible).lower()}"')
        connection_button.classes(remove="vmi-agent-connection-open")
        if connection.visible:
            connection_button.classes("vmi-agent-connection-open")

    def open_view_context(view: PanelView):
        nonlocal view_context
        if active_review is not None:
            ui.notify("Approve or decline the existing Agent draft first.", type="warning")
            return
        view_context = (uuid4().hex, view)
        context_note.text = view.label
        context_note.visible = True
        navigate("Agent")

    def clear_review():
        nonlocal active_review
        if active_review is not None and not active_review.is_deleted:
            active_review.clear()
        active_review = None

    async def answer(session, key, value):
        try:
            await asyncio.to_thread(session.answer_panel_request, key, value)
        except RuntimeError as exc:
            show_error(exc)

    async def decide(session, key, job, approved, *, rebuild=None):
        nonlocal approval, view_context
        with approval_view:
            if approved and rebuild is not None:
                try:
                    job, _ = rebuild()
                except (OSError, TypeError, ValueError) as exc:
                    ui.notify(str(exc), type="warning")
                    return
            if AGENTS.current(owner) is not session or not session.take_panel_request(key):
                show_error("This approval request is no longer pending.")
                return
            approval = None
            approval_view.clear()
            if active_review is not None:
                view_context = None
                context_note.text = ""
                context_note.visible = False
            clear_review()
            if not approved:
                await answer(session, key, {"status": "declined"})
                return
            try:
                submitted = jobs.submit(job)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                show_error(exc)
                await answer(session, key, {"error": str(exc)})
                return
            if submitted.name in workflow_views:
                view, refresh = workflow_views[submitted.name]
                view.attach(submitted)
                refresh()
            ui.notify(f"{submitted.name} submitted · {submitted.id}", type="positive")
            await answer(session, key, {"job_id": submitted.id, "panel": submitted.name, "status": submitted.status})

    async def refresh_panel():
        nonlocal approval, active_review
        session = AGENTS.current(owner)
        pending = session.panel_request() if session else None
        if pending is None:
            if approval is not None:
                approval = None
                approval_view.clear()
                clear_review()
            return
        key, request = pending
        if approval == (session.id, key):
            return
        approval = None
        approval_view.clear()
        clear_review()
        view_id = request.get("view_id")
        if view_id is not None and (view_context is None or view_context[0] != view_id):
            return
        try:
            match request.get("action"):
                case "describe":
                    value = registry.describe(request.get("operation"))
                case "status" | "wait":
                    value = registry.status(jobs, request.get("job_id"))
                    if request["action"] == "wait" and value["status"] not in TERMINAL:
                        return
                case "propose":
                    operation = request.get("operation")
                    parameters = request.get("parameters", {})
                    view = view_context[1] if view_id is not None else None
                    if view is not None:
                        if operation != view.operation or not isinstance(parameters, dict):
                            raise ValueError(f"Use {view.operation} with this view.")
                        parameters = view.bind(parameters)
                    job, inputs = registry.prepare(jobs, operation, parameters)
                    if view is not None:
                        async def approve(edits):
                            await decide(session, key, job, True, rebuild=lambda: registry.prepare(
                                jobs, operation, view.bind({**parameters, **edits})
                            ))

                        async def decline():
                            await decide(session, key, job, False)

                        view.render_review(job, inputs, approve, decline)
                        approval = (session.id, key)
                        active_review = view.review_container
                        with approval_view:
                            ui.label(f"The Agent draft is ready to review in {view.panel}.")
                            with ui.row():
                                ui.button(f"Review on {view.panel}",
                                          on_click=lambda: navigate(view.panel)).props("flat")
                                ui.button("Decline draft", on_click=decline).props("flat")
                        navigate(view.panel)
                        return
                    approval = (session.id, key)
                    with approval_view, theme.panel_card():
                        theme.form_section(f"Review {job.name} run")
                        for label, value in inputs.items():
                            if value is not None:
                                ui.label(f"{label} · {value}").classes("w-full break-all")
                        with ui.row().classes("items-center gap-2"):
                            ui.button("Approve and run", icon="play_arrow",
                                      on_click=lambda: decide(session, key, job, True))
                            ui.button("Decline", on_click=lambda: decide(session, key, job, False)).props("flat")
                    return
                case _:
                    raise ValueError("Choose describe, propose, status, or wait.")
        except KeyError as exc:
            value = {"error": f"Unknown operation or missing parameter {exc.args[0]}."}
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            value = {"error": str(exc)}
        if session.take_panel_request(key):
            await answer(session, key, value)

    def show_error(error):
        error_label.text = str(error)
        error_label.visible = True
        ui.run_javascript(f"getHtmlElement({error_label.id}).focus()")

    def clear_error():
        error_label.text = ""
        error_label.visible = False

    def reset_transcript(session_id):
        nonlocal rendered, rendered_session, diagnostics_cursor, empty_state
        rendered = []
        rendered_session = session_id
        diagnostics_cursor = 0
        diagnostics_log.clear()
        entries_column.clear()
        with entries_column:
            with ui.row().classes("vmi-agent-empty w-full items-center justify-center gap-2") as empty:
                ui.icon("forum").props('aria-hidden="true"')
                ui.label("Start a session to begin a conversation.").classes("vmi-caption")
            empty_state = empty

    def add_entry(entry):
        with entries_column, ui.column().classes(f"vmi-agent-entry vmi-agent-entry-{entry['kind']} w-full"):
            if entry["kind"] == "user":
                ui.label("You").classes("vmi-overline")
                element = ui.label(entry["text"]).classes("vmi-agent-message vmi-agent-user")
                if entry["detail"]:
                    ui.label(entry["detail"]).classes("vmi-agent-tool-detail")
            elif entry["kind"] == "assistant":
                ui.label("Agent").classes("vmi-overline")
                element = ui.markdown(entry["text"]).classes("vmi-agent-message")
            elif entry["kind"] == "tool":
                with ui.row().classes("vmi-agent-tool items-center gap-2"):
                    ui.icon("construction").props('aria-hidden="true"')
                    element = ui.label(entry["text"]).classes("vmi-caption")
                if entry["detail"]:
                    ui.label(entry["detail"]).classes("vmi-agent-tool-detail")
            elif entry["kind"] == "error":
                element = ui.label(entry["text"]).classes("vmi-agent-error").props('role="alert"')
            else:
                element = ui.label(entry["text"]).classes("vmi-caption")
        rendered.append((entry["kind"], element))

    def refresh():
        nonlocal diagnostics_cursor, stop_visible
        session = AGENTS.current(owner)
        status.sync(session)
        snapshot = session.snapshot() if session is not None else None
        session_status = snapshot["status"] if snapshot else "idle"
        waiting_for_approval = session_status == "running" and approval is not None
        model_pending = bool(snapshot and snapshot.get("model_pending"))
        control_pending = bool(snapshot and snapshot.get("control_pending"))
        activity_text = {
            "starting": "Connecting…",
            "stopping": "Closing session…",
            "failed": "Open Connection for diagnostics.",
        }.get(session_status, "")
        if model_pending:
            activity_text = "Changing model…"
        elif session_status == "running":
            activity_text = "Waiting for approval…" if waiting_for_approval else (
                snapshot["activity"] or "Agent working…")
        activity_label.text = activity_text
        activity_spinner.visible = model_pending or (
            session_status in ("starting", "running", "stopping") and not waiting_for_approval)
        active = session_status in ("starting", "ready", "running", "stopping")
        session_button.text = ("Starting…" if session_status == "starting" else
                               "Stopping…" if session_status == "stopping" else
                               "Stop session" if active else "Start session")
        session_button.props("color=negative" if active else "color=primary")
        session_button.set_enabled(not loading_models and session_status not in ("starting", "stopping"))
        load_models_button.set_enabled(not active and not loading_models
                                       and api.value in ("openai-completions", "openai-responses"))
        for field in (base_url, model, api, api_key):
            field.set_enabled(not active)
        update_models(snapshot, active, session_status)
        stop_visible = session_status == "running"
        if stop_visible:
            send_button.props('icon=stop aria-label="Stop reply"')
            send_button.classes(add="vmi-agent-stop")
        else:
            send_button.props('icon=arrow_upward aria-label="Send"')
            send_button.classes(remove="vmi-agent-stop")
        send_tooltip.text = "Stop reply" if stop_visible else "Send (Enter)"
        send_button.set_enabled(stop_visible or not uploading and session_status == "ready"
                                and not (model_pending or control_pending))
        if snapshot is None:
            if rendered_session is not None:
                reset_transcript(None)
            return
        transcript_id = snapshot.get("transcript_id", snapshot["id"])
        if transcript_id != rendered_session:
            reset_transcript(transcript_id)
        empty_state.visible = not snapshot["entries"]
        for index, entry in enumerate(snapshot["entries"]):
            if index >= len(rendered):
                add_entry(entry)
            elif entry["kind"] in ("assistant", "user"):
                kind, element = rendered[index]
                content = element.content if kind == "assistant" else element.text
                if kind == entry["kind"] and content != entry["text"]:
                    if kind == "assistant":
                        element.content = entry["text"]
                    else:
                        element.text = entry["text"]
        diagnostics = snapshot["diagnostics"]
        if diagnostics_cursor > len(diagnostics):
            diagnostics_log.clear()
            diagnostics_cursor = 0
        while diagnostics_cursor < len(diagnostics):
            diagnostics_log.push(diagnostics[diagnostics_cursor])
            diagnostics_cursor += 1

    async def toggle_session():
        clear_error()
        session = AGENTS.current(owner)
        if session is not None and session.active:
            session_button.set_enabled(False)
            try:
                await asyncio.to_thread(AGENTS.stop, owner)
            except Exception as exc:
                show_error(f"Could not stop the agent cleanly: {exc}")
            finally:
                api_key.value = ""
                refresh()
            return
        models = configured_models()
        selected = model_select.value if model_select.value in models else next(iter(models), "")
        request = AgentRequest(
            owner=owner,
            workspace=shared.workspace_root(),
            config_root=account.root.parent / ".agent" if account else GENERATED_DIR / "agent",
            datasets=shared.SERVER.datasets if shared.SERVER else None,
            base_url=base_url.value.strip(),
            model=selected,
            additional_models=tuple(item for item in models if item != selected),
            api_key=api_key.value or "",
            api=api.value,
        )
        session_button.set_enabled(False)
        try:
            await asyncio.to_thread(AGENTS.start, request, shared_mode=shared.SERVER is not None)
            api_key.value = ""
            clear_error()
        except (OSError, RuntimeError, ValueError) as exc:
            api_key.value = ""
            show_error(f"Could not start the session: {exc}")
        finally:
            refresh()

    async def run_slash(message, session):
        token, _, argument = message.strip().partition(" ")
        name = token[1:] if token.startswith("/") else ""
        argument = argument.strip()
        if name == "settings":
            if not connection.visible:
                toggle_connection()
            return True
        if name == "model":
            if argument and session is not None and session.active:
                await asyncio.to_thread(session.set_model, argument)
            elif argument:
                if argument not in configured_models():
                    raise ValueError("Choose a model configured in Connection.")
                model_select.value = argument
            else:
                model_select.run_method("showPopup")
            return True
        if name in SLASH_CONTROLS:
            if session is None or not session.active:
                raise RuntimeError("Start an Agent session first.")
            await asyncio.to_thread(session.run_slash_control, name, argument)
            return True
        return False

    async def send():
        clear_error()
        session = AGENTS.current(owner)
        try:
            if uploading:
                raise RuntimeError("Wait for the attachments to finish uploading.")
            message = prompt.value or ""
            if await run_slash(message, session):
                if prompt.value == message:
                    prompt.value = ""
                refresh()
                return
            if session is None:
                raise RuntimeError("Start an agent session first.")
            paths = tuple(attached_paths)
            references = tuple(_attachment_reference(path, container=container_mode()) for path in paths)
            model_context = ""
            if view_context is not None:
                identifier, view = view_context
                model_context = (f"\n\n[App context from {view.panel}. Call vmi_panel with action propose, "
                                 f"operation {view.operation}, and this view_id. {view.instruction}]\n"
                                 + json.dumps({"view_id": identifier, "parameters": view.settings}))
            await asyncio.to_thread(session.prompt, message, model_context=model_context, attachments=references)
            if prompt.value == message:
                prompt.value = ""
            for path in paths:
                if path in attached_paths:
                    attached_paths.remove(path)
            if paths:
                refresh_attachments()
        except (OSError, RuntimeError, ValueError) as exc:
            show_error(exc)
        refresh()

    async def abort():
        clear_error()
        session = AGENTS.current(owner)
        try:
            if session is not None:
                await asyncio.to_thread(session.abort)
        except RuntimeError as exc:
            show_error(exc)
        refresh()

    async def send_or_stop():
        if stop_visible:
            await abort()
        else:
            await send()

    async def browse_file():
        await attach_path("file")

    async def browse_folder():
        await attach_path("dir")

    connection_button.on_click(toggle_connection)
    session_button.on_click(toggle_session)
    attach_file.on_click(browse_file)
    attach_folder.on_click(browse_folder)
    model.on_value_change(lambda _: refresh())
    api.on_value_change(lambda _: refresh())
    load_models_button.on_click(load_models)
    model_select.on_value_change(request_model_change)
    send_button.on_click(send_or_stop)
    ui.on(completion_event, complete)
    ui.on(upload_event, receive_upload)
    prompt.on(
        "keydown", send,
        js_handler="(event) => { if (!event.defaultPrevented && event.key === 'Enter' && !event.shiftKey && !event.isComposing) "
                   "{ event.preventDefault(); emit(event); } }",
    )
    ui.context.client.on_connect(lambda: ui.run_javascript(AGENT_CHAT_JS.read_text(encoding="utf-8")))
    reset_transcript(None)
    refresh()
    shared.workspace_timer(0.4, refresh)
    shared.workspace_timer(0.4, refresh_panel)
    return open_view_context
