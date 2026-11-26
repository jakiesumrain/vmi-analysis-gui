"""Account-local project metadata and saved Agent conversations."""

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import UUID, uuid4


@dataclass(frozen=True)
class AgentProject:
    id: str
    name: str
    folder: str


@dataclass(frozen=True)
class SavedChat:
    id: str
    project_id: str | None
    title: str = "New chat"


def conversation_title(entries):
    for entry in entries:
        if entry["kind"] == "user" and (title := " ".join(entry["text"].split())):
            return title if len(title) <= 72 else title[:71].rstrip() + "…"
    return "New chat"


def _record_path(folder, identifier):
    if not isinstance(identifier, str) or len(identifier) != 32:
        raise ValueError("Invalid Agent history ID.")
    try:
        if UUID(identifier).hex != identifier:
            raise ValueError
    except ValueError:
        raise ValueError("Invalid Agent history ID.") from None
    return folder / f"{identifier}.json"


def _write_json(path, content):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(content, stream, ensure_ascii=False)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class AgentHistory:
    def __init__(self, root: Path, workspace: Path):
        self.root = root
        self.workspace = workspace.resolve()

    def _list(self, folder, kind):
        records = []
        for path in (self.root / folder).glob("*.json"):
            try:
                value = kind(**_read_json(path))
                if _record_path(path.parent, value.id) == path:
                    records.append((path.stat().st_mtime_ns, value))
            except (OSError, TypeError, ValueError):
                continue
        return [value for _, value in sorted(records, key=lambda record: record[0], reverse=True)]

    def projects(self):
        return sorted(self._list("projects", AgentProject), key=lambda item: item.name.casefold())

    def chats(self):
        return self._list("chats", SavedChat)

    def project(self, identifier):
        path = _record_path(self.root / "projects", identifier)
        try:
            return AgentProject(**_read_json(path))
        except FileNotFoundError:
            raise ValueError("This project is no longer available.") from None

    def chat(self, identifier):
        path = _record_path(self.root / "chats", identifier)
        try:
            return SavedChat(**_read_json(path))
        except FileNotFoundError:
            raise ValueError("This chat is no longer available.") from None

    def project_folder(self, project_id):
        if project_id is None:
            return self.workspace
        folder = (self.workspace / self.project(project_id).folder).resolve()
        if not folder.is_relative_to(self.workspace) or not folder.is_dir():
            raise ValueError("The project folder must be an existing folder inside your workspace.")
        return folder

    def create_project(self, name, folder):
        name = name.strip()
        folder = Path(folder).resolve()
        if not name or len(name) > 80:
            raise ValueError("Enter a project name of up to 80 characters.")
        if not folder.is_relative_to(self.workspace) or not folder.is_dir():
            raise ValueError("Choose a folder inside your workspace.")
        project = AgentProject(uuid4().hex, name, folder.relative_to(self.workspace).as_posix())
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        _write_json(_record_path(self.root / "projects", project.id), asdict(project))
        return project

    def create_chat(self, project_id):
        self.project_folder(project_id)
        chat = SavedChat(uuid4().hex, project_id)
        self._save_chat(chat)
        return chat

    def _save_chat(self, chat):
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        _write_json(_record_path(self.root / "chats", chat.id), asdict(chat))

    def session_dir(self, chat):
        return self.root / "sessions" / chat.id

    def entries(self, chat):
        path = _record_path(self.root / "transcripts", chat.id)
        try:
            return _read_json(path)
        except FileNotFoundError:
            return []

    def save_entries(self, chat, entries):
        messages = [entry for entry in entries if entry["kind"] in ("user", "assistant") and entry["text"]]
        if not messages:
            return chat
        _write_json(_record_path(self.root / "transcripts", chat.id), messages)
        title = conversation_title(messages)
        if chat.title == "New chat" and title != chat.title:
            chat = replace(chat, title=title)
            self._save_chat(chat)
        return chat
