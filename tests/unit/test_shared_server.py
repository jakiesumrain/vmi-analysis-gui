from contextlib import contextmanager
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vmi_analysis.gui import shared
from vmi_analysis.gui.jobs import Job, JobManager


class SharedServerTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(TemporaryDirectory())
        self.root = Path(directory)
        self.datasets = self.root / "datasets"
        self.datasets.mkdir()
        self.config = {
            "datasets": str(self.datasets),
            "workspaces": str(self.root / "workspaces"),
            "accounts": {
                name: {"password_hash": shared.password_hash(f"password-for-{name}")}
                for name in ("alice", "bob")
            },
        }
        self.server = shared.SharedServer(self.config)
        self.alice = self.server.accounts["alice"]
        self.bob = self.server.accounts["bob"]
        self.enterContext(patch.object(shared, "SERVER", self.server))

    @contextmanager
    def account(self, account):
        self.server.sessions[account.name] = (account, shared.monotonic() + 60)
        binding = shared.SESSION_TOKEN.set(account.name)
        try:
            yield
        finally:
            shared.SESSION_TOKEN.reset(binding)

    def test_settings_are_independent_and_survive_restart(self):
        with self.account(self.alice):
            shared.settings_store()["movie.interval"] = 31
        with self.account(self.bob):
            self.assertNotIn("movie.interval", shared.settings_store())
            shared.settings_store()["movie.interval"] = 27
        restored = shared.SharedServer(self.config)
        self.assertEqual(restored.accounts["alice"].settings["movie.interval"], 31)
        self.assertEqual(restored.accounts["bob"].settings["movie.interval"], 27)

    def test_traversal_other_user_and_shared_writes_are_rejected(self):
        with self.account(self.alice):
            self.assertEqual(shared.checked_path("result.txt", write=True), self.alice.root / "result.txt")
            self.assertEqual(shared.checked_path(self.datasets / "input.h5"), self.datasets / "input.h5")
            with self.assertRaises(shared.WorkspaceDenied):
                shared.checked_path(self.datasets / "input.h5", write=True)
            for forbidden in ("../../bob/files/result.txt", self.bob.root / "result.txt", "../settings.json"):
                with self.subTest(path=forbidden), self.assertRaises(shared.WorkspaceDenied):
                    shared.checked_path(forbidden)
            with self.assertRaises(shared.WorkspaceDenied):
                shared.check_command(["python", "-m", "module", "--pattern", "../*.h5"])

    def test_job_visibility_cancel_and_file_permissions_follow_owner(self):
        manager = JobManager()
        with patch.object(manager, "_schedule_locked"):
            with self.account(self.alice):
                alice_job = manager.submit(Job("Movie", ["python"], cwd=str(self.alice.root),
                                               reads=[self.datasets / "input.h5"]))
            with self.account(self.bob):
                bob_job = manager.submit(Job("Movie", ["python"], cwd=str(self.bob.root),
                                             reads=[self.datasets / "input.h5"]))
                self.assertEqual(manager.jobs(), [bob_job])
                self.assertIsNone(manager.get(alice_job.id))
                manager.cancel(alice_job.id)
                with self.assertRaises(shared.WorkspaceDenied):
                    manager.submit(Job("Write", ["python"], writes=[self.alice.root / "result.txt"]))
        self.assertFalse(alice_job.cancel_requested)
        self.assertIsNone(alice_job._access.conflict(bob_job._access))

    def test_standard_job_launch_passes_owner_roots_to_sandbox(self):
        manager = JobManager()
        with self.account(self.alice), patch.object(manager, "_schedule_locked"):
            job = manager.submit(Job("Ion TOF", [sys.executable, "-c", "print('sandboxed')"]))
        with patch("vmi_analysis.gui.jobs.panel_command", return_value=job.cmd) as command:
            self.assertEqual(manager._run_cmd(job, job.cmd, job.cwd), 0)
        command.assert_called_once_with(job.cmd, str(self.alice.root), self.alice.root,
                                        self.datasets, shared.analysis_root(self.alice))
        self.assertIn("sandboxed", job.log)

    def test_shared_job_does_not_start_if_sandbox_is_unavailable(self):
        manager = JobManager()
        with self.account(self.alice), patch.object(manager, "_schedule_locked"):
            job = manager.submit(Job("Ion TOF", [sys.executable, "-c", "pass"]))
        with patch("vmi_analysis.gui.jobs.panel_command", side_effect=OSError("sandbox unavailable")), \
                patch("vmi_analysis.gui.jobs.subprocess.Popen") as launch:
            with self.assertRaisesRegex(OSError, "sandbox unavailable"):
                manager._run_cmd(job, job.cmd, job.cwd)
        launch.assert_not_called()

    def test_symlink_cannot_expose_another_workspace_or_write_shared_data(self):
        other_file = self.bob.root / "secret.h5"
        other_file.touch()
        hidden_file = self.datasets / "hidden.h5"
        try:
            hidden_file.symlink_to(other_file)
        except OSError as exc:
            self.skipTest(f"Cannot create test symlinks: {exc}")
        with self.account(self.alice):
            with self.assertRaises(shared.WorkspaceDenied):
                shared.checked_path(hidden_file)
            with self.assertRaises(shared.WorkspaceDenied):
                self.server.check_directory_entries(self.alice, self.datasets)
            raw_file = self.datasets / "raw.h5"
            raw_file.touch()
            link = self.alice.root / "raw.h5"
            link.symlink_to(raw_file)
            self.assertEqual(shared.checked_path(link), raw_file)
            with self.assertRaises(shared.WorkspaceDenied):
                shared.checked_path(link, write=True)

    def test_http_authentication_downloads_and_logout(self):
        config = self.root / "shared.json"
        config.write_text(json.dumps(self.config), encoding="utf-8")
        web = FastAPI()
        with patch.object(shared, "app", web), patch.object(shared, "protect_socket_events"):
            server = shared.install(config)
        self.addCleanup(setattr, shared, "SERVER", self.server)
        (server.accounts["alice"].root / "preview.txt").write_text("alice", encoding="utf-8")
        (self.datasets / "raw.txt").write_text("shared", encoding="utf-8")
        protected = shared.analysis_root(server.accounts["alice"]) / "run.log"
        protected.write_text("alice analysis", encoding="utf-8")
        alice = TestClient(web, base_url="http://127.0.0.1:13517")
        bob = TestClient(web, base_url="http://127.0.0.1:13517")
        private = "/workspace-file/private/preview.txt"
        self.assertEqual(alice.get(private).status_code, 401)
        download = private + "?download=true"
        self.assertEqual(alice.get(download).status_code, 401)
        for name, client in (("alice", alice), ("bob", bob)):
            login = client.post("/login", data={"username": name, "password": f"password-for-{name}"},
                                follow_redirects=False)
            self.assertEqual(login.status_code, 303)
            self.assertIn("HttpOnly", login.headers["set-cookie"])
            self.assertEqual(client.get("/workspace-file/datasets/raw.txt").text, "shared")
        self.assertEqual(alice.get(private).text, "alice")
        self.assertEqual(bob.get(private).status_code, 404)
        result = alice.get(download)
        self.assertEqual(result.text, "alice")
        self.assertEqual(result.headers["content-disposition"], 'attachment; filename="preview.txt"')
        self.assertEqual(bob.get(download).status_code, 404)
        analysis = "/workspace-file/analysis/run.log?download=true"
        self.assertEqual(alice.get(analysis).text, "alice analysis")
        self.assertEqual(bob.get(analysis).status_code, 404)
        other_analysis = "/workspace-file/analysis/%2E%2E/%2E%2E/alice/analysis/run.log"
        self.assertEqual(bob.get(other_analysis).status_code, 404)
        self.assertEqual(alice.get("/workspace-file/datasets/raw.txt?download=true").status_code, 404)
        self.assertEqual(bob.get("/workspace-file/private/%2E%2E/%2E%2E/alice/files/preview.txt?download=true").status_code,
                         404)
        self.assertEqual(bob.get("/workspace-file/private/%2E%2E/%2E%2E/alice/files/preview.txt").status_code,
                         404)
        self.assertEqual(alice.post("/logout", headers={"Origin": "http://external.invalid"}).status_code, 403)
        self.assertEqual(alice.post("/logout", follow_redirects=False).status_code, 303)
        self.assertEqual(alice.get(private).status_code, 401)
