"""VMI Analysis GUI — one tab per script plus the pipeline.

Launch from the repository root:  uv run python -m gui.app
"""

from nicegui import ui

from . import theme
from .jobs import MANAGER
from .panels import (accumulate, binning, centroid, circularize, gmd_map,
                     movie, pes, pipeline)

TABS = [
    ("GMD Map", gmd_map),
    ("Movie", movie),
    ("Accumulate", accumulate),
    ("Centroiding", centroid),
    ("Binning", binning),
    ("Circularize", circularize),
    ("PES", pes),
    ("Pipeline", pipeline),
]


def main() -> None:
    theme.apply_theme()

    with ui.header().classes("items-center justify-between px-6 py-3") \
            .style("background: #faf9f5; border-bottom: 1px solid #e6dfd8"):
        with ui.row().classes("items-baseline gap-3"):
            ui.label("VMI Analysis").classes("vmi-title vmi-title-md")
            ui.label("photoelectron imaging toolbox").classes("vmi-subtitle")
        queue_label = ui.label("").classes("vmi-caption")

    def refresh_queue():
        cur = MANAGER.current()
        n_queued = sum(1 for j in MANAGER.jobs() if j.status == "queued")
        if cur is None:
            queue_label.text = "idle"
        else:
            queue_label.text = (f"running: {cur.name}"
                                + (f"  ·  {n_queued} queued" if n_queued else ""))

    ui.timer(1.0, refresh_queue)

    with ui.column().classes("w-full max-w-6xl mx-auto gap-4 p-6"):
        with ui.tabs().classes("w-full") as tabs:
            tab_objs = [ui.tab(name) for name, _ in TABS]
        with ui.tab_panels(tabs, value=tab_objs[0]).classes("w-full bg-transparent"):
            for tab, (_, module) in zip(tab_objs, TABS):
                with ui.tab_panel(tab):
                    module.build(MANAGER)

    with ui.footer().classes("px-6 py-2") \
            .style("background: #181715"):
        ui.label("vmi-analysis-gui — jobs run sequentially; "
                 "close the browser tab to stop watching "
                 "(jobs continue on the server).") \
            .classes("vmi-subtitle").style("color: #a09d96")

    ui.run(title="VMI Analysis", port=8080, reload=False, show=True)


if __name__ == "__main__":
    main()
