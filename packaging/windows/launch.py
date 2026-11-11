"""Start the installed app with its private Python runtime."""

import os
from pathlib import Path
import sys


def main():
    launcher = Path(__file__).resolve()
    install_root = launcher.parents[1]
    cache = install_root / "vmi-analysis-cache"
    os.environ.setdefault("VMI_WORKSPACE_DIR", str(Path.home() / "VMI Analysis"))
    os.environ.setdefault("NICEGUI_STORAGE_PATH", str(install_root / "vmi-analysis-settings"))
    os.environ.setdefault("VMI_CACHE_DIR", str(cache / "previews"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache / "numba"))
    os.environ["VMI_GUI_OFFLINE"] = "1"

    from vmi_analysis.gui.app import main as run_gui

    executable = sys.executable.replace("'", "''")
    script = str(launcher).replace("'", "''")
    run_gui(host="127.0.0.1", launch_command=f"& '{executable}' '{script}'")


if __name__ == "__main__":
    main()
