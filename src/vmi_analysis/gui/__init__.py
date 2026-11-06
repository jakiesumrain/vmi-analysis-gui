import os

from vmi_analysis.paths import STORAGE_DIR

# Preserve existing form values even when the installed app is launched elsewhere.
os.environ.setdefault("NICEGUI_STORAGE_PATH", str(STORAGE_DIR))
