"""Theme helpers — applies the DESIGN.md token spec to the NiceGUI app."""

import os
from nicegui import ui

from importlib.resources import files

THEME_CSS = files("vmi_analysis.gui.assets") / "theme.css"

_FONTS_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?'
    "family=Cormorant+Garamond:wght@400;500;700"
    "&family=Inter:wght@400;500"
    '&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">'
)


def apply_theme() -> None:
    """Quasar palette + fonts + token CSS. Call once at app startup."""
    ui.colors(
        primary="#cc785c",
        secondary="#6c6a64",
        accent="#e8a55a",
        dark="#181715",
        dark_page="#1f1e1b",
        positive="#5db872",
        negative="#c64545",
        info="#5db8a6",
        warning="#d4a017",
    )
    if os.environ.get("VMI_GUI_OFFLINE") != "1":
        ui.add_head_html(_FONTS_LINK)
    ui.add_head_html(f"<style>{THEME_CSS.read_text(encoding='utf-8')}</style>")


# --------------------------------------------------------------------------
# Component helpers (DESIGN.md component vocabulary)
# --------------------------------------------------------------------------

def section_title(text: str, subtitle: str | None = None) -> None:
    """Serif panel heading (Copernicus substitute: Cormorant Garamond)."""
    ui.label(text).classes("vmi-title vmi-title-sm")
    if subtitle:
        ui.label(subtitle).classes("vmi-subtitle")


def panel_card() -> ui.card:
    """Cream feature-card container for one script's controls/results."""
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
    """Dark code-window-card log console."""
    return ui.log(max_lines=2000).classes("vmi-log w-full").style("height: 340px")
