"""Standalone decoding controls for the Ion Pipeline verification branch."""

from nicegui import ui

from vmi_analysis.tpx3_analysis import UNWRAP_MODE

from .. import theme


def options_form(save):
    theme.form_section("TPX3 decoding")
    unwrap = save(ui.select({"signed": "Signed (default)", "go4": "Go4"}, value=UNWRAP_MODE,
                               label="Timestamp unwrapping").props("outlined dense"), "unwrap")
    ui.label("Uses the standalone TPX3 decoder and each source frame’s TDC1 rising edge."
             ).classes("vmi-subtitle")
    return lambda: {"unwrap": unwrap.value}
