"""Generate an ignored standalone fixture using the production image overlay.

Run from the repository root, then serve gui/generated/ over localhost and
run zoom_overlay.browser.js with playwright-cli. No dataset is needed.
"""

import json
from pathlib import Path
import sys
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gui.panels.common import _OVERLAY_HTML


def main():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="800">'
           '<rect width="1000" height="800" fill="white"/>'
           '<path d="M0 400H1000M500 0V800" stroke="#888"/>'
           '<rect x="200" y="150" width="250" height="200" fill="#cc785c"/>'
           '<text x="220" y="240" font-size="28">Inspect this region</text>'
           '<circle cx="700" cy="550" r="120" fill="#5db8a6"/></svg>')
    source = json.dumps('data:image/svg+xml,' + quote(svg))
    path = ROOT / 'gui/generated/zoom_check.html'
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text('<!DOCTYPE html><html><head><meta charset="utf-8">'
                    '<link rel="icon" href="data:,"><title>Area zoom check</title></head><body>'
                    + _OVERLAY_HTML + '<script>window.openFixture = () => vmiZoomOpen('
                    + source + '); openFixture();</script></body></html>', encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
