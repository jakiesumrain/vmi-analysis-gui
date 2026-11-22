"""Interactive per-user agent sessions for shared data and private workspaces."""

import asyncio
import json
from uuid import uuid4

from nicegui import ui

from vmi_analysis.agent_backend import AGENTS, API_FORMATS, AgentRequest
from vmi_analysis.paths import GENERATED_DIR

from .. import shared, theme
from ..agent_panels import PanelView
from ..jobs import TERMINAL
from .common import JobView, field_grid


class AgentStatus:
    def __init__(self):
        self.badge = None
        self.spinner = None
        self.wait_label = None

    def place_status(self):
        self.badge = theme.status_badge("idle")
        self.spinner = ui.spinner("dots", size="sm", color="primary")
        self.spinner.visible = False
        self.wait_label = ui.label("").classes("vmi-caption")
        self.wait_label.visible = False

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
        self.wait_label.text = "The session remains available after a browser reload." if status in (
            "ready", "running") else ""
        self.wait_label.visible = bool(self.wait_label.text)


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


def build(jobs, *, registry, navigate):
    account = shared.current_account()
    owner = account.name if account else "local"
    status = AgentStatus()
    theme.panel_header(
        "Agent",
        "Inspect shared data and work in your workspace with your own model endpoint.",
        view=status,
    )

    with theme.panel_card():
        with ui.expansion("Connection", icon="tune", value=False) \
                .classes("w-full vmi-agent-connection") as connection:
            with field_grid():
                base_url = ui.input("Base URL", placeholder="https://provider.example/v1") \
                    .classes("w-full").props("outlined dense autocomplete=off")
                model = ui.input("Model ID", placeholder="provider-model-name") \
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
        error_label = ui.label("").classes("vmi-agent-error w-full") \
            .props('role="alert" tabindex="-1"')
        error_label.visible = False

        theme.form_section("Conversation")
        with ui.scroll_area().classes("vmi-agent-transcript w-full") as transcript:
            entries_column = ui.column().classes("w-full gap-3")
        approval_view = ui.column().classes("w-full gap-2")
        context_note = ui.label().classes("vmi-caption")
        prompt = ui.textarea("Instruction", placeholder="Describe the analysis or file operation to perform.") \
            .classes("w-full").props("outlined autogrow rows=3")
        with ui.row().classes("w-full justify-between items-center gap-3"):
            with ui.column().classes("gap-0"):
                with ui.row().classes("items-center gap-2"):
                    activity_spinner = ui.spinner("dots", size="sm", color="primary").props('aria-hidden="true"')
                    activity_label = ui.label().classes("vmi-caption").props('role="status"')
                ui.label("Enter to send · Shift+Enter for a new line").classes("vmi-caption")
            with ui.row().classes("items-center gap-2"):
                abort_button = ui.button("Abort run", icon="stop").props("flat color=negative")
                send_button = ui.button("Send", icon="send").props("color=primary unelevated")

    with ui.expansion("Diagnostics", icon="terminal", value=False).classes("w-full vmi-console"):
        ui.label("Process diagnostics appear here; protocol messages stay in the conversation.") \
            .classes("vmi-caption")
        diagnostics_log = theme.log_console()

    workflow_views = {name: _workflow_card(jobs, name, targets, navigate)
                      for name, targets in registry.workflows().items()}

    rendered = []
    rendered_session = None
    diagnostics_cursor = 0
    empty_state = None
    approval = None
    view_context = None
    active_review = None

    def open_view_context(view: PanelView):
        nonlocal view_context
        if active_review is not None:
            ui.notify("Approve or decline the existing Agent draft first.", type="warning")
            return
        view_context = (uuid4().hex, view)
        context_note.text = view.label
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
            with ui.row().classes("vmi-results-empty w-full items-center justify-center gap-2") as empty:
                ui.icon("forum").props('aria-hidden="true"')
                ui.label("Start a session to begin a conversation.").classes("vmi-caption")
            empty_state = empty

    def add_entry(entry):
        with entries_column:
            if entry["kind"] == "user":
                ui.label("You").classes("vmi-overline")
                element = ui.label(entry["text"]).classes("vmi-agent-message vmi-agent-user")
            elif entry["kind"] == "assistant":
                ui.label("Agent").classes("vmi-overline")
                element = ui.markdown(entry["text"]).classes("vmi-agent-message")
            elif entry["kind"] == "tool":
                with ui.row().classes("items-center gap-2"):
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
        nonlocal diagnostics_cursor
        session = AGENTS.current(owner)
        status.sync(session)
        snapshot = session.snapshot() if session is not None else None
        session_status = snapshot["status"] if snapshot else "idle"
        waiting_for_approval = session_status == "running" and approval is not None
        activity_text = {
            "idle": "Start a session to send a message.",
            "starting": "Starting the Agent session…",
            "ready": "Ready for an instruction.",
            "stopping": "Stopping the Agent session…",
            "failed": "The Agent session failed. See Diagnostics.",
        }.get(session_status, "The Agent session has ended.")
        if session_status == "running":
            activity_text = "Waiting for your approval." if waiting_for_approval else (
                snapshot["activity"] or "Agent working…")
        activity_label.text = activity_text
        activity_spinner.visible = session_status in ("starting", "running", "stopping") and not waiting_for_approval
        active = session_status in ("starting", "ready", "running", "stopping")
        session_button.text = ("Starting…" if session_status == "starting" else
                               "Stopping…" if session_status == "stopping" else
                               "Stop session" if active else "Start session")
        session_button.props("color=negative" if active else "color=primary")
        session_button.set_enabled(session_status not in ("starting", "stopping"))
        for field in (base_url, model, api, api_key):
            field.set_enabled(not active)
        abort_button.visible = session_status == "running"
        send_button.set_enabled(session_status in ("ready", "running"))
        if snapshot is None:
            if rendered_session is not None:
                reset_transcript(None)
            return
        if snapshot["id"] != rendered_session:
            reset_transcript(snapshot["id"])
        empty_state.visible = not snapshot["entries"]
        changed = len(snapshot["entries"]) != len(rendered)
        for index, entry in enumerate(snapshot["entries"]):
            if index >= len(rendered):
                add_entry(entry)
                changed = True
            elif entry["kind"] in ("assistant", "user"):
                kind, element = rendered[index]
                content = element.content if kind == "assistant" else element.text
                if kind == entry["kind"] and content != entry["text"]:
                    if kind == "assistant":
                        element.content = entry["text"]
                    else:
                        element.text = entry["text"]
                    changed = True
        if changed:
            ui.run_javascript(
                f"const e=getHtmlElement({transcript.id}).querySelector('.q-scrollarea__container');"
                "if(e)e.scrollTop=e.scrollHeight;")
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
        request = AgentRequest(
            owner=owner,
            workspace=shared.workspace_root(),
            config_root=account.root.parent / ".agent" if account else GENERATED_DIR / "agent",
            datasets=shared.SERVER.datasets if shared.SERVER else None,
            base_url=base_url.value.strip(),
            model=model.value.strip(),
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

    async def send():
        clear_error()
        session = AGENTS.current(owner)
        try:
            if session is None:
                raise RuntimeError("Start an agent session first.")
            message = prompt.value or ""
            model_context = ""
            if view_context is not None:
                identifier, view = view_context
                model_context = (f"\n\n[App context from {view.panel}. Call vmi_panel with action propose, "
                                 f"operation {view.operation}, and this view_id. {view.instruction}]\n"
                                 + json.dumps({"view_id": identifier, "parameters": view.settings}))
            await asyncio.to_thread(session.prompt, message, model_context=model_context)
            if prompt.value == message:
                prompt.value = ""
        except (RuntimeError, ValueError) as exc:
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

    session_button.on_click(toggle_session)
    send_button.on_click(send)
    prompt.on(
        "keydown", send,
        js_handler="(event) => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) "
                   "{ event.preventDefault(); emit(event); } }",
    )
    abort_button.on_click(abort)
    reset_transcript(None)
    refresh()
    shared.workspace_timer(0.4, refresh)
    shared.workspace_timer(0.4, refresh_panel)
    return open_view_context
