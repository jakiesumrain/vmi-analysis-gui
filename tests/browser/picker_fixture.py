"""File-picker browser fixture; all file creation stays in the supplied test root."""

import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--port', type=int, default=18771)
args = parser.parse_args()
args.root = args.root.resolve()
args.root.mkdir(parents=True, exist_ok=True)
(args.root / 'Existing').mkdir(exist_ok=True)
(args.root / 'note.txt').write_text('existing data', encoding='utf-8')
os.environ['VMI_WORKSPACE_DIR'] = str(args.root)
os.environ['NICEGUI_STORAGE_PATH'] = str(args.root / 'storage')

from nicegui import ui
from vmi_analysis.gui import theme
from vmi_analysis.gui.panels.picker import LocalFilePicker


@ui.page('/')
def index():
    theme.apply_theme()
    result = ui.label('No selection').props('id=picker-result')

    async def choose(mode):
        chosen = await LocalFilePicker(start=args.root, mode=mode, filters=['.txt'], default_name='output.txt')
        result.text = chosen or 'Cancelled'

    ui.button('Choose folder', on_click=lambda: choose('dir'))
    ui.button('Choose file', on_click=lambda: choose('file'))
    ui.button('Save file', on_click=lambda: choose('save'))


ui.run(host='127.0.0.1', port=args.port, reload=False, show=False)
