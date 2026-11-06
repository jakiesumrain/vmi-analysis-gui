"""Local file/folder picker dialog — browses the server's filesystem.

Adapted for Windows: drive letters are offered via a dropdown, dotfiles and
hidden-looking entries are skipped, and long listings are truncated.
"""

from __future__ import annotations

import os
import string
from pathlib import Path

from nicegui import ui
from ..jobs import MANAGER, ResourceBusyError

_DRIVES = [f"{d}:/" for d in string.ascii_uppercase if os.path.exists(f"{d}:/")]
_MAX_ENTRIES = 1000

_TITLES = {"file": "Select file", "dir": "Select folder", "save": "Save as"}


def _create_directory(parent: Path, name: str) -> Path:
    """Create one named child, without replacing an existing file or folder."""
    name = name.strip()
    if (not name or name in (".", "..") or name.endswith(".")
            or any(c in '<>:"/\\|?*' or ord(c) < 32 for c in name)):
        raise ValueError("Enter a folder name without path separators or special characters.")
    stem = name.split(".", 1)[0].upper()
    if stem in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10))}:
        raise ValueError("This folder name is reserved by Windows.")
    target = parent / name
    with MANAGER.write_access([target]):
        target.mkdir()
    return target


def _nearest_dir(start) -> Path:
    p = Path(str(start)).expanduser() if start else None
    while p is not None and not p.is_dir():
        p = None if p.parent == p else p.parent
    return p or Path.home()


class LocalFilePicker(ui.dialog):
    """Awaitable dialog; resolves to the chosen path string or None.

    mode: "file" (pick an existing file), "dir" (pick a directory),
    "save" (navigate + file name, may not exist yet).
    """

    def __init__(self, start=None, mode: str = "file",
                 filters: list[str] | None = None, default_name: str = ""):
        super().__init__()
        self.mode = mode
        self.filters = {f.lower() for f in filters} if filters else None
        self.path = _nearest_dir(start)

        with self, ui.card().classes("w-[38rem] max-w-[95vw] gap-2"):
            ui.label(_TITLES.get(mode, "Select")).classes("vmi-caption")
            with ui.row().classes("w-full items-center gap-1 no-wrap"):
                ui.button(icon="arrow_upward", on_click=self._up) \
                    .props("flat dense round color=primary")
                self.path_label = ui.label() \
                    .classes("grow vmi-caption").style("word-break: break-all")
                if _DRIVES:
                    ui.select(_DRIVES, value=None, on_change=self._drive) \
                        .props("dense outlined options-dense label=Drive") \
                        .classes("w-28")
            self.entries = ui.list().props("bordered separator") \
                .classes("w-full h-72 overflow-auto")
            with ui.row().classes("w-full items-center justify-end gap-2"):
                ui.button("New folder", icon="create_new_folder", on_click=self._new_folder) \
                    .props("flat color=primary").classes("mr-auto")
                if mode == "save":
                    self.name_input = ui.input("File name", value=default_name) \
                        .props("outlined dense").classes("grow")
                    ui.button("Save", on_click=self._save) \
                        .props("unelevated color=primary")
                elif mode == "dir":
                    ui.button("Select this folder", on_click=self._choose_dir) \
                        .props("unelevated color=primary")
                ui.button("Cancel", on_click=lambda: self.submit(None)) \
                    .props("flat")
        self._refresh()

    # -- navigation -------------------------------------------------------

    def _new_folder(self):
        parent = self.path
        with ui.dialog() as dialog, ui.card().classes("w-96 max-w-[95vw] gap-3"):
            ui.label("New folder").classes("vmi-caption")
            ui.label(str(parent)).classes("vmi-subtitle").style("word-break: break-all")
            name = ui.input("Folder name").props("outlined dense autofocus").classes("w-full")

            def create():
                try:
                    target = _create_directory(parent, name.value or "")
                except FileExistsError:
                    ui.notify("A file or folder with this name already exists.", type="warning")
                    return
                except (ValueError, OSError, ResourceBusyError) as exc:
                    ui.notify(f"Could not create folder: {exc}", type="warning")
                    return
                self._nav(target)
                dialog.close()

            name.on("keydown.enter", create)
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Create", icon="create_new_folder", on_click=create).props("unelevated color=primary")
                ui.button("Cancel", on_click=dialog.close).props("flat")
        dialog.open()

    def _up(self):
        parent = self.path.parent
        if parent != self.path:
            self.path = parent
            self._refresh()

    def _drive(self, e):
        if e.value:
            self.path = Path(e.value)
            self._refresh()

    def _nav(self, p: Path):
        self.path = p
        self._refresh()

    # -- selection --------------------------------------------------------

    def _choose(self, p: Path):
        if self.mode == "save":
            self.name_input.value = p.name
        else:
            self.submit(str(p))

    def _choose_dir(self):
        self.submit(str(self.path))

    def _save(self):
        name = (self.name_input.value or "").strip()
        if name:
            self.submit(str(self.path / name))

    # -- listing ----------------------------------------------------------

    def _visible(self, p: Path) -> bool:
        if p.name.startswith("."):
            return False
        if p.is_dir():
            return True
        if self.mode == "dir":
            return False
        return self.filters is None or p.suffix.lower() in self.filters

    def _refresh(self):
        self.path_label.text = str(self.path)
        self.entries.clear()
        try:
            children = sorted(
                (p for p in self.path.iterdir() if self._visible(p)),
                key=lambda p: (not p.is_dir(), p.name.lower()))
            accessible = True
        except (PermissionError, OSError):
            children, accessible = [], False
        with self.entries:
            if not children:
                if not accessible:
                    msg = "(not accessible)"
                elif self.mode == "dir":
                    msg = "(no subfolders)"
                else:
                    msg = "(no matching files)"
                ui.item_label(msg).classes("text-grey-6 italic")
            for p in children[:_MAX_ENTRIES]:
                is_dir = p.is_dir()
                handler = (lambda p=p: self._nav(p)) if is_dir \
                    else (lambda p=p: self._choose(p))
                with ui.item(on_click=handler).props("clickable dense"):
                    with ui.item_section().props("avatar"):
                        ui.icon("folder" if is_dir else "description",
                                color="primary" if is_dir else "grey-6")
                    with ui.item_section():
                        ui.item_label(p.name)
            if len(children) > _MAX_ENTRIES:
                ui.item_label(f"… {len(children) - _MAX_ENTRIES} more not shown") \
                    .classes("text-grey-6 italic")
