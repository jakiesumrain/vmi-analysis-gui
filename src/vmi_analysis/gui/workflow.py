"""Portable snapshots of the GUI's panel settings."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable

from nicegui import ui

from .centroid_plots import COLORMAPS, RESULT_COLORMAP_DEFAULTS

SCHEMA = "vmi-workflow/1"
Value = str | bool | int | float | None


@dataclass(frozen=True)
class Field:
    read: Callable[[], Value]
    accepts: Callable[[Value], bool]


_fields: ContextVar[dict[str, Field] | None] = ContextVar("workflow_fields", default=None)


@contextmanager
def collect_fields(fields):
    token = _fields.set(fields)
    try:
        yield
    finally:
        _fields.reset(token)


def result_fields(store):
    """Include colormap settings whose controls are built only with results."""
    return {key: Field(lambda key=key, default=default: store.get(key, default), COLORMAPS.__contains__)
            for key, default in RESULT_COLORMAP_DEFAULTS.items()}


def register_field(key, widget):
    fields = _fields.get()
    if fields is None:
        return
    if isinstance(widget, ui.select):
        # Capture choices before another field narrows them (e.g. filter placement).
        choices = tuple(widget.options)
        accepts = choices.__contains__
    elif isinstance(widget, ui.checkbox):
        accepts = lambda value: type(value) is bool
    elif isinstance(widget, (ui.number, ui.slider)):
        accepts = lambda value: (
            (value is None and isinstance(widget, ui.number))
            or (type(value) is int and abs(value) < 2**53)
            or (type(value) is float and isfinite(value))
        )
    else:
        accepts = lambda value: isinstance(value, str)
    fields[key] = Field(lambda: widget.value, accepts)


def _read(path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(doc, dict) or doc.get("schema") != SCHEMA
            or not isinstance(doc.get("settings"), dict)):
        raise ValueError("Choose a VMI workflow JSON file.")
    return doc["settings"]


def save_workflow(path, fields):
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() != ".json":
        raise ValueError("Choose a workflow file ending in .json.")
    if path.exists():
        _read(path)  # Only replace an existing workflow file.
    settings = {key: field.read() for key, field in fields.items()}
    data = json.dumps({"schema": SCHEMA, "settings": settings},
                      indent=2, sort_keys=True, allow_nan=False)
    temporary = NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                   suffix=".tmp", delete=False)
    temp_path = Path(temporary.name)
    try:
        with temporary:
            temporary.write(data + "\n")
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def load_workflow(path, fields):
    settings = {key: value for key, value in _read(Path(path)).items() if key in fields}
    if not settings:
        raise ValueError("This workflow has no recognized panel settings.")
    for key, value in settings.items():
        if not fields[key].accepts(value):
            raise ValueError(f"Invalid workflow value for {key}.")
    return settings
