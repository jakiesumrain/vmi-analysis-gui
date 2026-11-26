"""Interactive per-user agent sessions for shared data and private workspaces."""

import asyncio
from base64 import b64encode
from importlib.resources import files
import json
import os
from pathlib import Path, PurePosixPath
import re
from uuid import uuid4

from nicegui import ui

from vmi_analysis.agent_backend import (AGENTS, API_FORMATS, SLASH_CONTROLS, AgentRequest, ContainerLauncher,
                                        list_provider_models)
from vmi_analysis.agent_history import conversation_title as _conversation_title
from vmi_analysis.paths import GENERATED_DIR

from .. import shared, theme
from ..agent_attachments import (IMAGE_SUFFIXES, MAX_ATTACHMENT_BYTES, MAX_ATTACHMENT_FILES, MAX_BATCH_BYTES,
                                 MAX_PROMPT_IMAGES, attachment_root, prompt_image)
from ..agent_completion import path_completions
from ..agent_panels import PanelInspection, PanelView, RunCapture
from ..jobs import ACTIVE
from .common import JobView, checkpoint_view, field_grid
from .picker import LocalFilePicker


MODEL_SETTINGS = "\0model-settings"
SLASH_BUILTINS = {
    "model": "Choose a model",
    "settings": "Open Agent connection settings",
    **{name: description for name, (_, description) in SLASH_CONTROLS.items()},
}
AGENT_CHAT_JS = files("vmi_analysis.gui.assets") / "agent_chat.js"
WELCOME_ICON = "data:image/svg+xml;base64," + b64encode(
    (files("vmi_analysis.gui.assets") / "electron-spectrum-welcome.svg").read_bytes()
).decode("ascii")


def _render_review_details(details):
    with ui.element("dl").classes("vmi-agent-review-details"):
        for label, value in details.items():
            with ui.element("dt"):
                ui.label(str(label))
            with ui.element("dd"):
                _render_review_value(value)


def _render_review_value(value):
    if isinstance(value, dict) and value:
        _render_review_details(value)
    elif isinstance(value, (list, tuple)) and value:
        if all(isinstance(item, (int, float)) or (isinstance(item, str) and item.isalnum()) for item in value):
            ui.label(", ".join(map(str, value)))
        elif len(value) == 1:
            _render_review_value(value[0])
        else:
            with ui.element("ul"):
                for item in value:
                    with ui.element("li"):
                        _render_review_value(item)
    else:
        ui.label("None" if value is None or isinstance(value, (dict, list, tuple)) else str(value))


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
        show_checkpoint = checkpoint_view(jobs)
        view.place_console("Workflow console")
    card.classes("vmi-agent-workflow")
    card.set_visibility(False)

    def refresh():
        job = view.job
        card.set_visibility(job is not None)
        show_checkpoint(job)
        if job is not None:
            steps.text = " · ".join(f"{stage.name} {stage.status}" for stage in job.stages)
            stop.set_visibility(job.status in ACTIVE)
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
    if shared.SERVER:
        for root, mount in ((shared.SERVER.datasets, "/datasets"),
                            (shared.analysis_root(shared.current_account()), "/analysis")):
            if path.is_relative_to(root):
                return str(PurePosixPath(mount, *path.relative_to(root).parts))
    raise ValueError("Choose an attachment inside the Agent workspace.")


def build(jobs, *, registry, navigate, settings_dialog, connection_settings, history_navigation):
    with ui.column().classes("vmi-agent-page w-full gap-0"):
        return _build_page(jobs, registry=registry, navigate=navigate, settings_dialog=settings_dialog,
                           connection_settings=connection_settings, history_navigation=history_navigation)


def _build_page(jobs, *, registry, navigate, settings_dialog, connection_settings, history_navigation):
    account = shared.current_account()
    owner = account.name if account else "local"
    workspace = shared.workspace_root()
    history = shared.agent_history(account)
    with ui.element("h1").classes("vmi-agent-header w-full"):
        title = ui.label(_conversation_title(()))
    with ui.row().classes("vmi-agent-resume w-full items-center") as connection_action:
        connect_chat = ui.button("Resume chat", icon="play_arrow").props("flat no-caps")
    connection_action.visible = False

    with connection_settings, ui.column().classes("vmi-agent-connection w-full"):
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
        image_models = ui.select({}, label="Image-capable models", multiple=True).classes("w-full")
        image_models.props("outlined dense use-chips")
        image_models.tooltip("Select models whose endpoint accepts images before starting the session.")
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
        connection_error = ui.label().classes("vmi-agent-error").props('role="alert"')
        connection_error.visible = False
        theme.form_section("Diagnostics")
        diagnostics_log = theme.log_console().classes("vmi-agent-diagnostics")
    error_label = ui.label("").classes("vmi-agent-error vmi-agent-notice w-full") \
        .props('role="alert" tabindex="-1"')
    error_label.visible = False

    with ui.column().classes("vmi-agent-timeline w-full gap-0"):
        with ui.column().classes("vmi-agent-entries w-full"):
            entries_column = ui.column().classes("vmi-agent-transcript w-full")
            with ui.row().classes("vmi-agent-activity items-center gap-2") as activity_row:
                activity_label = ui.label().props('role="status"')
                activity_spinner = ui.spinner("dots", size="sm", color="primary").props('aria-hidden="true"')
            activity_row.visible = False
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
    linked_run = None
    updating_models = False
    stop_visible = False
    loading_models = False
    connecting = False
    uploading = 0
    upload_note = ""
    selected_chat = None
    selected_project = None
    follow_current = True
    navigation_state = None
    expanded_projects = set()

    def visible_session():
        session = AGENTS.current(owner)
        if session is not None and (follow_current or (
                selected_chat is not None and session.chat is not None and session.chat.id == selected_chat.id)):
            return session
        return None

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
            open_connection_settings()
            refresh()
        elif (session := visible_session()) is not None and session.active:
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
        attachments_row.visible = bool(attached_paths or linked_run)
        with attachments_row:
            if linked_run is not None:
                with ui.row().classes("vmi-agent-attachment items-center no-wrap"):
                    ui.icon("link").props('aria-hidden="true"')
                    ui.label(f"{linked_run.name} · {linked_run.id}").classes("ellipsis")
                    ui.button(icon="close", on_click=unlink_run).props(
                        'flat dense round aria-label="Remove linked run"').tooltip("Remove linked run")
            for path in attached_paths:
                with ui.row().classes("vmi-agent-attachment items-center no-wrap"):
                    ui.label(path.name or str(path)).classes("ellipsis").tooltip(str(path))
                    ui.button(icon="close", on_click=lambda path=path: remove_attachment(path)).props(
                        'flat dense round').tooltip(f"Remove {path.name or path}")

    def unlink_run():
        nonlocal linked_run
        linked_run = None
        refresh_attachments()

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
            session = visible_session()
            folder = session.request.working_directory if session is not None else None
            selected = await LocalFilePicker(start=folder or workspace, mode=mode)
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
            session = visible_session()
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

    def open_connection_settings():
        connection_settings.open()
        settings_dialog.open()

    def open_view_context(view: PanelView | PanelInspection | RunCapture):
        nonlocal view_context, linked_run
        if active_review is not None:
            ui.notify("Approve or decline the existing Agent draft first.", type="warning")
            return
        if isinstance(view, RunCapture):
            try:
                run = registry.evidence_run(jobs, view.job_id)
                add_uploaded(view.paths)
            except (OSError, RuntimeError, ValueError) as exc:
                ui.notify(str(exc), type="warning")
                return
            linked_run = run
            view_context = None
            context_note.visible = False
            clear_error()
            refresh_attachments()
            navigate("Agent")
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

    async def decide(session, key, job, approved, *, rebuild=None, interactive=None):
        nonlocal approval, view_context
        with approval_view:
            if approved and rebuild is not None:
                try:
                    job, _ = rebuild()
                except (OSError, TypeError, ValueError) as exc:
                    ui.notify(str(exc), type="warning")
                    return
            if visible_session() is not session or not session.take_panel_request(key):
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
                if interactive is not None:
                    await answer(session, key, await interactive())
                    return
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
        session = visible_session()
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
                    if request["action"] == "wait" and value["status"] in ("queued", "running"):
                        return
                case "measure":
                    value = registry.measure(jobs, request.get("job_id"), request.get("parameters"))
                case "inspect":
                    view = view_context[1] if view_id is not None else None
                    if (not isinstance(view, PanelInspection) or request.get("operation") != view.operation
                            or request.get("parameters", {}) != {}):
                        raise ValueError("Open the panel's Ask Agent control to read its preview.")
                    value = view.read_snapshot()
                case "propose":
                    operation = request.get("operation")
                    parameters = request.get("parameters", {})
                    view = view_context[1] if view_id is not None else None
                    if isinstance(view, PanelInspection):
                        raise ValueError("Use inspect to read this panel's preview.")
                    if view is not None:
                        if operation != view.operation or not isinstance(parameters, dict):
                            raise ValueError(f"Use {view.operation} with this view.")
                        parameters = view.bind(parameters)
                    interactive = view.run_interactive if view is not None else None
                    job, inputs = (None, parameters) if interactive is not None else registry.prepare(
                        jobs, operation, parameters)
                    if view is not None:
                        async def approve(edits):
                            if interactive is not None:
                                async def run():
                                    return await interactive(view.bind({**parameters, **edits}))

                                await decide(session, key, None, True, interactive=run)
                                return
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
                    with approval_view, theme.panel_card().classes("vmi-agent-approval-card"):
                        theme.form_section(f"Review {job.name} run")
                        _render_review_details({label: value for label, value in inputs.items() if value is not None})
                        registry.render_review(job)
                        with ui.row().classes("items-center gap-2"):
                            ui.button("Approve and run", icon="play_arrow",
                                      on_click=lambda: decide(session, key, job, True))
                            ui.button("Decline", on_click=lambda: decide(session, key, job, False)).props("flat")
                    return
                case _:
                    raise ValueError("Choose describe, inspect, measure, propose, status, or wait.")
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
            with ui.column().classes("vmi-agent-empty w-full items-center justify-center") as empty:
                ui.element("img").props(f'src="{WELCOME_ICON}" alt="" width="76" height="76"')
                ui.label("What would you like to analyze?")
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

    def select_conversation(chat=None, project=None):
        nonlocal selected_chat, selected_project, follow_current, view_context, linked_run
        if connecting or uploading:
            show_error("Wait for the current connection or upload before switching chats.")
            return
        if approval is not None:
            show_error("Approve or decline the current draft before switching chats.")
            return
        current = visible_session()
        if view_context is not None and current is not None and current.running:
            show_error("Wait for the current Workbench request before switching chats.")
            return
        follow_current = False
        selected_chat = chat
        selected_project = project
        if project is not None:
            expanded_projects.add(project.id)
        view_context = None
        linked_run = None
        context_note.visible = False
        attached_paths.clear()
        prompt.value = ""
        refresh_attachments()
        clear_error()
        navigate("Agent")
        refresh()
        refresh_history()

    def select_saved(chat, project):
        try:
            select_conversation(chat, project)
        except (OSError, ValueError) as exc:
            show_error(exc)

    def project_dialog():
        with ui.dialog() as dialog, ui.card().classes("w-[28rem] max-w-[92vw]"):
            ui.label("New project").classes("vmi-subtitle")
            name = ui.input("Project name").props("outlined dense maxlength=80").classes("w-full")
            folder = workspace
            folder_label = ui.label("My workspace").classes("vmi-caption break-all")
            issue = ui.label("").classes("vmi-agent-error").props('role="alert"')

            async def choose_folder():
                nonlocal folder
                if chosen := await LocalFilePicker(start=folder, mode="dir"):
                    candidate = Path(chosen).resolve()
                    if not candidate.is_relative_to(workspace.resolve()):
                        issue.text = "Choose a folder inside your workspace."
                        return
                    issue.text = ""
                    folder = candidate
                    folder_label.text = candidate.relative_to(workspace.resolve()).as_posix()

            def save_project():
                try:
                    project = history.create_project(name.value or "", folder)
                except (OSError, ValueError) as exc:
                    issue.text = str(exc)
                    return
                dialog.close()
                select_conversation(project=project)

            ui.button("Choose workspace folder", icon="folder_open", on_click=choose_folder).props("flat no-caps")
            with ui.row().classes("w-full justify-end"):
                ui.button("Cancel", on_click=dialog.close).props("flat no-caps")
                ui.button("Create project", on_click=save_project).props("unelevated no-caps")
        dialog.open()

    def refresh_history():
        nonlocal navigation_state
        projects = history.projects()
        chats = history.chats()
        current = visible_session()
        selected = current.chat if current is not None else selected_chat
        active_id = selected.id if selected is not None else None
        state = (projects, chats, active_id)
        if state == navigation_state:
            return
        if navigation_state is None and selected is not None and selected.project_id is not None:
            expanded_projects.add(selected.project_id)
        navigation_state = state
        chat_groups = {}
        for chat in chats:
            chat_groups.setdefault(chat.project_id, []).append(chat)

        def chat_button(chat, project=None):
            button = ui.button(chat.title, on_click=lambda: select_saved(chat, project)) \
                .props("flat dense no-caps").classes("vmi-agent-history-chat").tooltip(chat.title)
            if chat.id == active_id:
                button.props('aria-current="true"')

        def remember_expansion(event, project_id):
            if event.value:
                expanded_projects.add(project_id)
            else:
                expanded_projects.discard(project_id)

        history_navigation.clear()
        with history_navigation:
            ui.button("New chat", icon="add", on_click=select_conversation).props("flat dense no-caps")
            for chat in chat_groups.get(None, ()):
                chat_button(chat)
            with ui.row().classes("vmi-agent-project-heading w-full no-wrap items-center justify-between"):
                ui.label("Projects").classes("vmi-caption")
                ui.button(icon="create_new_folder", on_click=project_dialog).props(
                    'flat dense round aria-label="New project"').tooltip("New project")
            for project in projects:
                with ui.expansion(project.name, icon="folder_open", value=project.id in expanded_projects,
                                  on_value_change=lambda event, key=project.id: remember_expansion(event, key)) \
                        .props("dense").classes("vmi-agent-project w-full"):
                    ui.button("New project chat", icon="add",
                              on_click=lambda project=project: select_conversation(project=project)) \
                        .props("flat dense no-caps")
                    for chat in chat_groups.get(project.id, ()):
                        chat_button(chat, project)

    def refresh():
        nonlocal diagnostics_cursor, stop_visible
        session = AGENTS.current(owner)
        visible = visible_session()
        live_snapshot = session.snapshot() if session is not None else None
        snapshot = live_snapshot if visible is session else (visible.snapshot() if visible else None)
        session_status = live_snapshot["status"] if live_snapshot else "idle"
        waiting_for_approval = session_status == "running" and approval is not None
        model_pending = bool(snapshot and snapshot.get("model_pending"))
        control_pending = bool(snapshot and snapshot.get("control_pending"))
        activity_text = {
            "starting": "Connecting…",
            "stopping": "Closing session…",
            "failed": "Open Agent connection in Settings for diagnostics.",
        }.get(session_status, "")
        if model_pending:
            activity_text = "Changing model…"
        elif session_status == "running":
            activity_text = "Waiting for approval…" if waiting_for_approval else "Working"
        activity_label.text = activity_text
        activity_row.visible = visible is not None and bool(activity_text)
        activity_spinner.visible = model_pending or (
            session_status in ("starting", "running", "stopping") and not waiting_for_approval)
        active = session_status in ("starting", "ready", "running", "stopping")
        resuming = selected_chat is not None or (visible is not None and visible.chat is not None)
        session_button.text = ("Starting…" if session_status == "starting" else
                               "Stopping…" if session_status == "stopping" else
                               "Stop session" if active else "Resume chat" if resuming else "Start session")
        session_button.props("color=negative" if active else "color=primary")
        session_button.set_enabled(not connecting and not loading_models and session_status not in ("starting", "stopping"))
        load_models_button.set_enabled(not active and not loading_models
                                       and api.value in ("openai-completions", "openai-responses"))
        for field in (base_url, model, api, api_key, image_models):
            field.set_enabled(not active)
        available = {name: name for name in configured_models()}
        if available != image_models.options:
            image_models.set_options(available, value=[value for value in image_models.value or [] if value in available])
        update_models(live_snapshot, active, session_status)
        stop_visible = visible is not None and session_status == "running"
        if stop_visible:
            send_button.props('icon=stop aria-label="Stop reply"')
            send_button.classes(add="vmi-agent-stop")
        else:
            send_button.props('icon=arrow_upward aria-label="Send"')
            send_button.classes(remove="vmi-agent-stop")
        send_tooltip.text = "Stop reply" if stop_visible else "Send (Enter)"
        send_button.set_enabled(stop_visible or visible is not None and not uploading and session_status == "ready"
                                and not (model_pending or control_pending))
        connect_chat.text = ("Manage active session" if active and visible is None else
                             "Resume chat" if resuming else "Start chat")
        connection_action.visible = not (visible is not None and active) and (
            not follow_current or visible is not None)
        if snapshot is None and selected_chat is not None:
            archive_id = f"saved:{selected_chat.id}"
            if rendered_session != archive_id:
                reset_transcript(archive_id)
                try:
                    selected = history.chat(selected_chat.id)
                    entries = history.entries(selected)
                except (OSError, ValueError) as exc:
                    show_error(exc)
                    entries = []
                    selected = selected_chat
                title.text = selected.title
                empty_state.visible = not entries
                for entry in entries:
                    add_entry(entry)
            return
        if snapshot is None:
            title.text = selected_project.name if selected_project is not None else _conversation_title(())
            if rendered_session is not None:
                reset_transcript(None)
            return
        transcript_id = snapshot.get("transcript_id", snapshot["id"])
        if transcript_id != rendered_session:
            reset_transcript(transcript_id)
        chat = visible.chat
        title.text = (chat.title if chat is not None and chat.title != "New chat" else
                      _conversation_title(snapshot["entries"]))
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
        nonlocal selected_chat, follow_current, connecting
        clear_error()
        connection_error.visible = False
        session = AGENTS.current(owner)
        if session is not None and session.active:
            session_button.set_enabled(False)
            try:
                await asyncio.to_thread(AGENTS.stop, owner)
            except Exception as exc:
                connection_error.text = f"Could not stop the agent cleanly: {exc}"
                connection_error.visible = True
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
            analysis=shared.analysis_root(account) if account else None,
            base_url=base_url.value.strip(),
            model=selected,
            additional_models=tuple(item for item in models if item != selected),
            image_models=tuple(image_models.value or ()),
            api_key=api_key.value or "",
            api=api.value,
        )
        connecting = True
        session_button.set_enabled(False)
        try:
            resume = selected_chat or (session.chat if session is not None and follow_current else None)
            started = await asyncio.to_thread(
                AGENTS.start, request, shared_mode=shared.SERVER is not None, history=history,
                chat_id=resume.id if resume is not None else None,
                project_id=selected_project.id if selected_project is not None else None)
            selected_chat = started.chat
            follow_current = False
            api_key.value = ""
            clear_error()
        except (OSError, RuntimeError, ValueError) as exc:
            api_key.value = ""
            connection_error.text = f"Could not start the session: {exc}"
            connection_error.visible = True
        finally:
            connecting = False
            refresh()
            refresh_history()

    async def run_slash(message, session):
        token, _, argument = message.strip().partition(" ")
        name = token[1:] if token.startswith("/") else ""
        argument = argument.strip()
        if name == "settings":
            open_connection_settings()
            return True
        if name == "new":
            if argument:
                raise ValueError("/new takes no argument.")
            current = visible_session()
            chat = current.chat if current is not None else selected_chat
            project = history.project(chat.project_id) if chat and chat.project_id else selected_project
            select_conversation(project=project)
            return True
        if name == "model":
            if argument and session is not None and session.active:
                await asyncio.to_thread(session.set_model, argument)
            elif argument:
                if argument not in configured_models():
                    raise ValueError("Choose a model configured under Agent connection in Settings.")
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
        nonlocal linked_run
        clear_error()
        session = visible_session()
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
            image_paths = [path for path in paths if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file()]
            if len(image_paths) > MAX_PROMPT_IMAGES:
                raise ValueError(f"Attach up to {MAX_PROMPT_IMAGES} images per message.")
            if image_paths:
                snapshot = session.snapshot()
                if snapshot.get("model") not in snapshot.get("image_models", ()):
                    raise ValueError("Select an image-capable model. Configure image support under Agent connection "
                                     "in Settings before starting the session.")
            run = registry.evidence_run(jobs, linked_run.id) if linked_run is not None else None
            references = tuple(_attachment_reference(path, container=container_mode()) for path in paths)
            model_context = ""
            if view_context is not None:
                identifier, view = view_context
                if isinstance(view, PanelInspection):
                    model_context = (f"\n\n[App context from {view.panel}. Call vmi_panel with action inspect, "
                                     f"operation {view.operation}, and this view_id to read the current preview.]\n"
                                     + json.dumps({"view_id": identifier}))
                else:
                    model_context = (f"\n\n[App context from {view.panel}. Call vmi_panel with action propose, "
                                     f"operation {view.operation}, and this view_id. {view.instruction}]\n"
                                     + json.dumps({"view_id": identifier, "parameters": view.settings,
                                                   **({"context": view.context} if view.context is not None else {})}))
            if image_paths:
                model_context += ("\n\n[Use the attached image for visible features. Label visual readings as estimates. "
                                  "Distinguish them from values verified using saved data.]")
            if run:
                model_context += ("\n\n[The user linked this run. Use vmi_panel with action measure and this job_id "
                                  "and the supplied measurement parameters to verify features against the saved data. "
                                  "Report the measurement method returned by the tool.]\n"
                                  + json.dumps(registry.measurement_context(run)))
            prompt_options = {}
            if image_paths:
                account = shared.current_account()
                roots = (account.root, shared.SERVER.datasets, shared.analysis_root(account)) if account else ()
                prompt_options["images"] = await asyncio.to_thread(
                    lambda: tuple(prompt_image(path, roots=roots) for path in image_paths)
                )
            if run:
                prompt_options["linked_run"] = f"Linked run · {run.name} · {run.id}"
            await asyncio.to_thread(session.prompt, message, model_context=model_context, attachments=references,
                                    **prompt_options)
            if prompt.value == message:
                prompt.value = ""
            for path in paths:
                if path in attached_paths:
                    attached_paths.remove(path)
            if run is linked_run:
                linked_run = None
            if paths or run:
                refresh_attachments()
        except (OSError, RuntimeError, ValueError) as exc:
            show_error(exc)
        refresh()

    async def abort():
        clear_error()
        session = visible_session()
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

    session_button.on_click(toggle_session)
    connect_chat.on_click(open_connection_settings)
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
    refresh_history()
    shared.workspace_timer(0.4, refresh)
    shared.workspace_timer(0.4, refresh_panel)
    shared.workspace_timer(2.0, refresh_history)
    return open_view_context
