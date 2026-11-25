from pathlib import Path
from shutil import which
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from vmi_analysis.gui.job_sandbox import panel_command


@unittest.skipUnless(sys.platform == "linux" and which("bwrap"), "Requires Linux and Bubblewrap")
class SharedJobSandboxTests(unittest.TestCase):
    def test_child_process_has_only_the_owner_data_mounts(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            workspace = root / "users" / "alice" / "files"
            analysis = workspace.parent / "analysis"
            sibling = root / "users" / "bob" / "files"
            datasets = root / "datasets"
            for path in (workspace, analysis, sibling, datasets):
                path.mkdir(parents=True)
                (path / "input.txt").write_text("fixture", encoding="utf-8")
            script = """
from pathlib import Path
import subprocess
import sys

import vmi_analysis

workspace, datasets, analysis, sibling = map(Path, sys.argv[1:])
assert Path.cwd() == workspace
for directory in (workspace, datasets, analysis):
    assert (directory / "input.txt").read_text() == "fixture"
(workspace / "result.txt").write_text("done")
assert not (sibling / "input.txt").exists()
for directory in (datasets, analysis, sibling):
    try:
        (directory / "output.txt").write_text("blocked")
    except OSError:
        pass
    else:
        raise AssertionError("Unexpected write access")
subprocess.run(
    [sys.executable, "-c", "import sys, pathlib; assert not pathlib.Path(sys.argv[1]).exists()",
     str(sibling / "input.txt")],
    check=True,
)
"""
            command = [sys.executable, "-c", script, workspace, datasets, analysis, sibling]
            result = subprocess.run(
                panel_command(command, workspace, workspace, datasets, analysis),
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((workspace / "result.txt").read_text(), "done")
