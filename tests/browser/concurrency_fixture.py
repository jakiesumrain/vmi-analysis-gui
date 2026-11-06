"""Isolated real GUI with harmless jobs; test routes never enter production."""

import argparse
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=18770)
parser.add_argument('--workspace', type=Path, required=True)
args = parser.parse_args()
args.workspace.mkdir(parents=True, exist_ok=True)
os.environ.update(VMI_GUI_PORT=str(args.port), VMI_GUI_NO_SHOW='1',
                  VMI_WORKSPACE_DIR=str(args.workspace.resolve()),
                  NICEGUI_STORAGE_PATH=str(args.workspace.resolve() / 'storage'))

from nicegui import app
from vmi_analysis.gui import app as gui
from vmi_analysis.gui.jobs import Job, MANAGER


def sleeper(name, **kwargs):
    return Job(name=name, cmd=[sys.executable, '-c', 'import time; time.sleep(120)'], **kwargs)


@app.post('/_test/start')
def start():
    shared = str(args.workspace.resolve() / 'dark.npy')
    MANAGER.submit(sleeper('Dark frame', writes=[shared]))
    MANAGER.submit(sleeper('Accumulate', reads=[shared]))
    MANAGER.submit(sleeper('Frame movie'))
    MANAGER.submit(sleeper('Centroiding'))


@app.post('/_test/add-map')
def add_map():
    MANAGER.submit(sleeper('GMD map'))


@app.get('/_test/state')
def state():
    return {'limit': MANAGER.max_concurrent, 'jobs': [
        {'name': j.name, 'status': j.status, 'wait': j.wait_reason} for j in MANAGER.jobs()]}


gui.main()
