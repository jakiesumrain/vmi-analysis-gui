"""Theme helpers — applies the dark dashboard tokens in theme.css."""

import os
from nicegui import ui

from importlib.resources import files

THEME_CSS = files("vmi_analysis.gui.assets") / "theme.css"

_FONTS_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?'
    "family=Geist:wght@400;500;600;700"
    "&family=Geist+Mono:wght@400;500"
    '&display=swap" rel="stylesheet">'
)


def apply_theme() -> None:
    """Quasar palette + fonts + token CSS. Call once at app startup."""
    ui.colors(
        primary="#ededed",
        secondary="#a3a3a3",
        accent="#3b82f6",
        dark="#000000",
        dark_page="#000000",
        positive="#22c55e",
        negative="#ef4444",
        info="#38bdf8",
        warning="#f59e0b",
    )
    if os.environ.get("VMI_GUI_OFFLINE") != "1":
        ui.add_head_html(_FONTS_LINK)
    ui.add_head_html(f"<style>{THEME_CSS.read_text(encoding='utf-8')}</style>")


# --------------------------------------------------------------------------
# Component helpers (DESIGN.md component vocabulary)
# --------------------------------------------------------------------------

def section_title(text: str, subtitle: str | None = None) -> None:
    """Compact sans panel heading."""
    ui.label(text).classes("vmi-title vmi-title-sm")
    if subtitle:
        ui.label(subtitle).classes("vmi-subtitle")


def panel_card() -> ui.card:
    """Quiet bordered panel for one script's controls/results."""
    return ui.card().classes("vmi-panel w-full shadow-none").props("flat")


def run_button(label: str = "Run", **kwargs) -> ui.button:
    return ui.button(label, **kwargs).props("color=primary unelevated")


def status_badge(text: str = "idle") -> ui.badge:
    return ui.badge(text).classes("vmi-badge-idle")


def set_badge(badge: ui.badge, status: str) -> None:
    for cls in ("idle", "queued", "running", "done", "failed", "cancelled", "skipped"):
        badge.classes(remove=f"vmi-badge-{cls}")
    badge.classes(f"vmi-badge-{status}")
    badge.text = status


def log_console() -> ui.log:
    """Monospaced console on the near-black canvas."""
    return ui.log(max_lines=2000).classes("vmi-log w-full").style("height: 340px")
