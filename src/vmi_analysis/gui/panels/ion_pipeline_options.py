"""LUNA controls for the Ion Pipeline."""

from nicegui import ui

from .. import shared, theme
from .common import field_grid, num_input


def options_form(save):
    theme.form_section("LUNA decoding")
    with field_grid():
        source = save(ui.select({"pixels": "Raw pixel hits", "clusters": "Clustered events"},
                                value="pixels", label="Count source").props("outlined dense"), "source")
        timewalk = save(ui.checkbox("Apply timewalk correction", value=False), "timewalk")
    ui.label("Uses the 32-byte acquisition prefix and the TDC1 rising edge. "
             "Runs LUNA separately for each detected acquisition.").classes("vmi-subtitle")
    with ui.expansion("LUNA processing options", icon="tune").classes("w-full"):
        with field_grid(3):
            algorithm = save(ui.select(["DFSCluster", "Kuttruff2024"], value="DFSCluster",
                                       label="Clustering algorithm").props("outlined dense"), "clustering_algorithm")
            time_gap = save(ui.input("Clustering time gap", value="100ns").props("outlined dense"), "eps_t")
            space_gap = save(num_input("Clustering space gap (pixels)", 1, min=0, step=1), "eps_s")
            buffer = save(num_input("Kuttruff buffer size", 10, min=1, step=1), "kuttruff_buffer_size")
            buffer.bind_visibility_from(algorithm, "value", lambda selected: selected == "Kuttruff2024")
            cut = save(num_input("Cluster ToT cut (ns)", 500, min=0, step=25), "ctot_cut")
            cut.bind_visibility_from(timewalk, "value")
            disable = save(ui.checkbox("Disable clustering", value=False), "disable_clustering")
    distro = executable = None
    if not shared.SERVER:
        with ui.expansion("Windows WSL2 settings", icon="settings").classes("w-full"), field_grid():
            distro = save(ui.input("WSL distribution (blank = default)").props("outlined dense"), "wsl_distribution")
            executable = save(ui.input("LUNA executable in WSL (blank = PATH)").props("outlined dense"),
                              "wsl_executable")

    def values():
        if disable.value and (timewalk.value or source.value == "clusters"):
            raise ValueError("Enable clustering for timewalk correction or clustered counts.")
        if not time_gap.value.strip():
            raise ValueError("Enter a LUNA clustering time gap.")
        for field in (space_gap, buffer, cut):
            if field.value is None or not float(field.value).is_integer():
                raise ValueError("Use whole numbers for LUNA pixel-gap, buffer-size, and ToT settings.")
        if timewalk.value and cut.value % 25:
            raise ValueError("Cluster ToT cut must be a multiple of 25 ns.")
        return dict(source=source.value, timewalk=timewalk.value, clustering_algorithm=algorithm.value,
                    eps_t=time_gap.value.strip(), eps_s=int(space_gap.value), kuttruff_buffer_size=int(buffer.value),
                    ctot_cut=int(cut.value), disable_clustering=disable.value,
                    wsl_distribution=distro.value.strip() if distro else "",
                    wsl_executable=executable.value.strip() if executable else "")

    return values
