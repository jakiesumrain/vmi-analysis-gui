"""Declarative form validation for reusable workspace analyses."""

from math import isfinite
import re

SCHEMA = "vmi.analysis-panel/v1"
FIELD_NAME = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{0,63}\Z")
OUTPUT_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,99}\Z")
FIELD_TYPES = {"file", "number", "text", "boolean"}


def _label(value):
    if not isinstance(value, str) or not value.strip() or not value.isprintable() or len(value) > 100:
        raise ValueError("Enter labels of 1 to 100 printable characters.")
    return value.strip()


def definition(value):
    if (not isinstance(value, dict) or value.get("schema") != SCHEMA
            or value.keys() - {"schema", "title", "fields", "results"}):
        raise ValueError(f"Use a panel definition with schema {SCHEMA}, title, fields, and optional results.")
    fields, results = value.get("fields", {}), value.get("results", {})
    if not isinstance(fields, dict) or not isinstance(results, dict) or max(len(fields), len(results)) > 32:
        raise ValueError("Use up to 32 named fields and 32 named results.")
    normalized = {"schema": SCHEMA, "title": _label(value.get("title")), "fields": {}, "results": {}}
    for key, field in fields.items():
        if (not isinstance(key, str) or not FIELD_NAME.fullmatch(key) or not isinstance(field, dict)
                or field.keys() - {"type", "label"} or field.get("type") not in FIELD_TYPES):
            raise ValueError("Use named file, number, text, or boolean fields with a label.")
        normalized["fields"][key] = {"type": field["type"], "label": _label(field.get("label", key))}
    for name, view in results.items():
        if (not isinstance(name, str) or not OUTPUT_NAME.fullmatch(name) or not isinstance(view, dict)
                or view.keys() - {"label", "view"} or view.get("view", "download") not in ("download", "image")):
            raise ValueError("Identify each result by its output filename and use a download or image view.")
        kind = view.get("view", "download")
        if kind == "image" and not name.lower().endswith((".png", ".jpg", ".jpeg")):
            raise ValueError("Use PNG or JPEG for an image result view.")
        normalized["results"][name] = {"label": _label(view.get("label", name)), "view": kind}
    return normalized


def validate_values(panel, inputs, parameters):
    files = {key for key, field in panel["fields"].items() if field["type"] == "file"}
    if set(inputs) != files or set(parameters) != panel["fields"].keys() - files:
        raise ValueError("Supply one value for every declared panel field.")
    for key, value in parameters.items():
        field = panel["fields"][key]
        kind = field["type"]
        valid = ((kind == "text" and isinstance(value, str))
                 or (kind == "boolean" and type(value) is bool)
                 or (kind == "number" and type(value) in (int, float) and isfinite(value)))
        if not valid:
            raise ValueError(f"{field['label']} requires a {kind} value.")
