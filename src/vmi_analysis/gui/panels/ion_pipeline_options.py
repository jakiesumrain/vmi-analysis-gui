"""LUNA controls for the Ion Pipeline."""

from nicegui import ui

from .. import shared, theme
from .common import field_grid, num_input


LUNA_DEFAULTS = {
    "source": "pixels", "timewalk": False, "clustering_algorithm": "DFSCluster",
    "eps_t": "100ns", "eps_s": 1, "kuttruff_buffer_size": 10, "ctot_cut": 500,
    "disable_clustering": False,
}
WSL_FIELDS = ("wsl_distribution", "wsl_executable")


def options_form(save, *, fallback=False):
    if fallback:
        from vmi_analysis.tpx3_analysis import UNWRAP_MODE

        theme.form_section("Standalone decoding")
        save(ui.select({"signed": "Signed (default)", "go4": "Go4"}, value=UNWRAP_MODE,
                                label="Timestamp unwrapping").props("outlined dense"), "unwrap")
        ui.label("Decodes raw pixel hits.").classes("vmi-subtitle")
        return

    theme.form_section("LUNA decoding")
    with field_grid():
        save(ui.select({"pixels": "Raw pixel hits", "clusters": "Clustered events"},
                                value=LUNA_DEFAULTS["source"], label="Count source").props("outlined dense"), "source")
        timewalk = save(ui.checkbox("Apply timewalk correction", value=LUNA_DEFAULTS["timewalk"]), "timewalk")
    ui.label("Uses the 32-byte acquisition prefix and the TDC1 rising edge. "
             "Runs LUNA separately for each detected acquisition.").classes("vmi-subtitle")
    with ui.expansion("LUNA processing options", icon="tune").classes("w-full"):
        with field_grid(3):
            algorithm = save(ui.select(["DFSCluster", "Kuttruff2024"], value=LUNA_DEFAULTS["clustering_algorithm"],
                                       label="Clustering algorithm").props("outlined dense"), "clustering_algorithm")
            save(ui.input("Clustering time gap", value=LUNA_DEFAULTS["eps_t"]).props("outlined dense"), "eps_t")
            save(num_input("Clustering space gap (pixels)", LUNA_DEFAULTS["eps_s"], min=0, step=1), "eps_s")
            buffer = save(num_input("Kuttruff buffer size", LUNA_DEFAULTS["kuttruff_buffer_size"], min=1, step=1), "kuttruff_buffer_size")
            buffer.bind_visibility_from(algorithm, "value", lambda selected: selected == "Kuttruff2024")
            cut = save(num_input("Cluster ToT cut (ns)", LUNA_DEFAULTS["ctot_cut"], min=0, step=25), "ctot_cut")
            cut.bind_visibility_from(timewalk, "value")
            save(ui.checkbox("Disable clustering", value=LUNA_DEFAULTS["disable_clustering"]), "disable_clustering")
    if not shared.SERVER:
        with ui.expansion("Windows WSL2 settings", icon="settings").classes("w-full"), field_grid():
            save(ui.input("WSL distribution (blank = default)").props("outlined dense"), "wsl_distribution")
            save(ui.input("LUNA executable in WSL (blank = PATH)").props("outlined dense"), "wsl_executable")
