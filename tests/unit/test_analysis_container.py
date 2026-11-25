"""Regression coverage for shared analysis job ownership and worker configuration."""

from contextlib import contextmanager
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from vmi_analysis.analysis_container import create_command
from vmi_analysis.gui import analysis_runs, shared
from vmi_analysis.gui.jobs import JobManager, TERMINAL
from vmi_analysis.gui.panel_operations import build_registry


class WorkerCommandTests(unittest.TestCase):
    def test_worker_mounts_only_the_owner_roots_and_its_output(self):
        path = Path("/srv/vmi/alice")
        runs = path / "analysis" / "runs"
        job = runs / "01234567"
        for runtime, rootless in (("docker", False), ("podman", True)):
            with self.subTest(runtime=runtime):
                cmd = create_command(runtime, "sha256:example", "unique-name", job, path / "files",
                                     Path("/srv/shared"), rootless=rootless, uid=1001, gid=1001)
                for flag in ("--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges"):
                    self.assertIn(flag, cmd)
                self.assertNotIn("VMI_AGENT_API_KEY", " ".join(cmd))
                self.assertNotIn("/srv/vmi/bob", " ".join(cmd))
                self.assertIn("/job/01234567/worker.json", cmd)
                if runtime == "docker":
                    mounts = [cmd[i + 1] for i, flag in enumerate(cmd) if flag == "--mount"]
                    self.assertEqual(sum("readonly" in mount for mount in mounts), 4)
                    self.assertTrue(all("bind-propagation=rprivate" in mount for mount in mounts[:-1]))
                    self.assertIn("target=/job/01234567/outputs", mounts[-1])
                    self.assertNotIn("readonly", mounts[-1])
                    self.assertEqual(cmd[cmd.index("--user") + 1], "1001:1001")
                else:
                    mounts = [cmd[i + 1] for i, flag in enumerate(cmd) if flag == "--volume"]
                    self.assertTrue(all(mount.endswith(":ro") for mount in mounts[:-1]))
                    self.assertTrue(mounts[-1].endswith(":rw"))
                    self.assertIn("--userns=keep-id", cmd)

    @unittest.skipUnless(sys.platform == "linux", "requires the Linux descriptor-based source reader")
    def test_workspace_source_rejects_symlink_substitution(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "actual.py").write_text("data", encoding="utf-8")
            (root / "linked.py").symlink_to("actual.py")
            self.assertEqual(analysis_runs._workspace_source(root / "actual.py", root)[0], b"data")
            with self.assertRaises(OSError):
                analysis_runs._workspace_source(root / "linked.py", root)


class SharedAnalysisTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name).resolve()
        dataset = root / "datasets"
        dataset.mkdir()
        (dataset / "centers.txt").write_text("100\n109\n", encoding="utf-8")
        password = shared.password_hash("test")
        self.server = shared.SharedServer({
            "datasets": str(dataset), "workspaces": str(root / "users"),
            "accounts": {name: {"password_hash": password} for name in ("alice", "bob")},
        })
        self.enterContext(patch.object(shared, "SERVER", self.server))
        self.enterContext(patch.object(analysis_runs, "available", return_value=True))
        if sys.platform != "linux":
            self.enterContext(patch.object(analysis_runs, "_workspace_source",
                                          side_effect=lambda path, _: analysis_runs._source(path)))
        self.manager = JobManager(max_concurrent=2)
        self.addCleanup(self.manager.shutdown)
        self.registry = build_registry()
        self.fake = root / "fake_supervisor.py"
        self.fake.write_text(
            """import json, sys, time
from pathlib import Path
folder, workspace, datasets = map(Path, sys.argv[1:])
run = json.loads((folder/'worker.json').read_text(encoding='utf-8'))
if run['parameters']['wait']:
    print('waiting', flush=True)
    while not (folder/'.release').exists():
        if (folder/'.cancel').exists():
            raise SystemExit(1)
        time.sleep(.025)
path = run['inputs']['input']['path']
assert path == '/datasets/centers.txt'
a, b = map(float, (datasets/'centers.txt').read_text().splitlines())
(folder/'outputs'/'result.txt').write_text(f'{b-a:g}', encoding='utf-8')
""", encoding="utf-8",
        )

    @contextmanager
    def login(self, name):
        actor = self.server.accounts[name]
        self.server.sessions[name] = actor, time.monotonic() + 300
        token = shared.SESSION_TOKEN.set(name)
        try:
            yield actor
        finally:
            shared.SESSION_TOKEN.reset(token)

    def wait_for(self, condition):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if condition():
                return
            time.sleep(.025)
        self.fail("Worker did not reach the expected state")

    def submit(self, params):
        job, _ = self.registry.prepare(self.manager, "analysis_python", params)
        self.assertEqual(job.cmd[2], "vmi_analysis.analysis_container")
        folder = Path(job.cmd[3])
        job.cmd = [sys.executable, str(self.fake), *job.cmd[3:]]
        self.manager.submit(job)
        return job, folder

    def test_saved_panel_runs_and_cancel_are_owned_per_account(self):
        definition = {"schema": "vmi.analysis-panel/v1", "title": "Ion TOF drift", "fields": {
            "input": {"type": "file"}, "wait": {"type": "boolean"},
        }}
        values = {"inputs": {"input": "/datasets/centers.txt"}}
        saved = {}
        for name in ("alice", "bob"):
            with self.login(name) as actor:
                (actor.root / "drift.py").write_text("print('approved script')\n", encoding="utf-8")
                prototype, folder = self.submit({**values, "script": "/workspace/drift.py",
                                                  "panel_definition": definition, "parameters": {"wait": False}})
                self.wait_for(lambda: prototype.status in TERMINAL)
                self.assertEqual(prototype.status, "done", "\n".join(prototype.log))
                self.assertEqual((folder / "outputs/result.txt").read_text(), "9")
                self.assertIn("output/result.txt", self.registry.status(self.manager, prototype.id)["files"])
                record = json.loads((folder / "run.json").read_text(encoding="utf-8"))
                self.assertEqual(record["actor"], name)
                saved[name] = analysis_runs._save_panel(record, folder).stem

        with self.login("alice"):
            a, folder_a = self.submit({**values, "saved_panel": saved["alice"], "parameters": {"wait": True}})
        with self.login("bob"):
            b, folder_b = self.submit({**values, "saved_panel": saved["bob"], "parameters": {"wait": True}})
            self.wait_for(lambda: "waiting" in b.log and "waiting" in a.log)
            self.assertIsNone(self.manager.get(a.id))
            self.assertNotIn(a, self.manager.jobs())
            with self.assertRaises(shared.WorkspaceDenied):
                shared.checked_path(folder_a / "run.json")
            with self.assertRaises(shared.WorkspaceDenied):
                shared.checked_path(folder_b / "run.json", write=True)
            self.manager.cancel(a.id)
            self.assertFalse(a.cancel_requested)
        with self.login("alice"):
            self.manager.cancel(a.id)
        self.wait_for(lambda: a.status in TERMINAL)
        self.assertEqual(a.status, "cancelled", "\n".join(a.log))
        self.assertEqual(b.status, "running")
        self.assertFalse((folder_b / ".cancel").exists())
        (folder_b / ".release").touch()
        self.wait_for(lambda: b.status in TERMINAL)
        self.assertEqual(b.status, "done", "\n".join(b.log))
        self.assertEqual((folder_b / "outputs/result.txt").read_text(), "9")
