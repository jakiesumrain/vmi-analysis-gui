"""Interactive per-user agent sessions for shared data and private workspaces."""

import asyncio

from nicegui import ui

from vmi_analysis.agent_backend import AGENTS, API_FORMATS, AgentRequest
from vmi_analysis.paths import GENERATED_DIR

from .. import shared, theme
from .common import field_grid


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


def build(jobs):
    del jobs  # Interactive sessions do not occupy a finite analysis-job slot.
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
                abort_button = ui.button("Abort run", icon="stop") \
                    .props("flat color=negative")
                abort_button.visible = False
        error_label = ui.label("").classes("vmi-agent-error w-full") \
            .props('role="alert" tabindex="-1"')
        error_label.visible = False

        theme.form_section("Conversation")
        with ui.scroll_area().classes("vmi-agent-transcript w-full") as transcript:
            entries_column = ui.column().classes("w-full gap-3")
        prompt = ui.textarea("Instruction", placeholder="Describe the analysis or file operation to perform.") \
            .classes("w-full").props("outlined autogrow rows=3")
        with ui.row().classes("w-full justify-between items-center gap-3"):
            ui.label("Enter to send · Shift+Enter for a new line").classes("vmi-caption")
            send_button = ui.button("Send", icon="send").props("color=primary unelevated")

    with ui.expansion("Diagnostics", icon="terminal", value=False).classes("w-full vmi-console"):
        ui.label("Process diagnostics appear here; protocol messages stay in the conversation.") \
            .classes("vmi-caption")
        diagnostics_log = theme.log_console()

    rendered = []
    rendered_session = None
    diagnostics_cursor = 0
    empty_state = None

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
            config_root=account.root / ".cache" / "agent" if account else GENERATED_DIR / "agent",
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

    def send():
        clear_error()
        session = AGENTS.current(owner)
        try:
            if session is None:
                raise RuntimeError("Start an agent session first.")
            session.prompt(prompt.value or "")
            prompt.value = ""
        except (RuntimeError, ValueError) as exc:
            show_error(exc)
        refresh()

    def abort():
        clear_error()
        session = AGENTS.current(owner)
        try:
            if session is not None:
                session.abort()
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
