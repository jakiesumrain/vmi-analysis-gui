import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vmi_analysis.gui import desktop_client, shared


class DesktopClientTests(TestCase):
    def setUp(self):
        directory = self.enterContext(TemporaryDirectory())
        root = Path(directory)
        datasets = root / "datasets"
        datasets.mkdir()
        self.server = shared.SharedServer({
            "datasets": str(datasets),
            "workspaces": str(root / "accounts"),
            "accounts": {name: {"password_hash": shared.password_hash(name)} for name in ("alice", "bob")},
        })
        self.enterContext(patch.object(shared, "SERVER", self.server))
        self.links = desktop_client.DesktopLinks(self.server)
        self.alice = self.server.accounts["alice"]
        self.server.sessions["alice-session"] = (self.alice, desktop_client.monotonic() + 3600)
        app = FastAPI()
        app.add_middleware(shared.AccountMiddleware)
        desktop_client.install(app, self.links)
        self.client = self.enterContext(TestClient(app, base_url="http://127.0.0.1:13517"))

    def pair(self):
        code = self.links.issue_code(self.alice, "alice-session")
        data = {"account": "alice", "code": code}
        response = self.client.post("/client/v1/pair", json=data)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.client.post("/client/v1/pair", json=data).status_code, 403)
        return response.json()["credential"]

    def launch(self, credential, **profile):
        return self.client.post("/client/v1/launch", json={"account": "alice", **profile},
                                headers={"Authorization": f"Bearer {credential}"})

    def test_saved_credential_and_one_use_browser_ticket(self):
        credential = self.pair()
        stored = (self.alice.root.parent / "desktop-client.json").read_text(encoding="utf-8")
        self.assertNotIn(credential, stored)
        self.assertEqual(set(json.loads(stored)), {"credential"})
        launch = self.launch(credential)
        self.assertEqual(launch.status_code, 200)
        ticket = launch.json()["ticket"]
        data = {"account": "alice", "code": ticket}
        self.assertEqual(self.client.post("/client/v1/consume",
                                          json={**data, "account": "bob"}).status_code, 403)
        response = self.client.post("/client/v1/consume", json=data)
        self.assertEqual(response.status_code, 204)
        self.assertIn("HttpOnly", response.headers["set-cookie"])
        self.assertIs(self.server.session(self.client.cookies[shared.COOKIE]), self.alice)
        self.assertEqual(self.client.post("/client/v1/consume", json=data).status_code, 403)

    def test_revoke_and_expiry(self):
        credential = self.pair()
        ticket = self.launch(credential).json()["ticket"]
        self.links.revoke(self.alice)
        self.assertEqual(self.launch(credential).status_code, 401)
        self.assertEqual(self.client.post("/client/v1/consume", json={
            "account": "alice", "code": ticket,
        }).status_code, 403)
        code = self.links.issue_code(self.alice, "alice-session")
        with patch.object(desktop_client, "monotonic", return_value=desktop_client.monotonic() + 301):
            self.assertEqual(self.client.post("/client/v1/pair", json={
                "account": "alice", "code": code,
            }).status_code, 403)

    def test_agent_starts_under_the_authenticated_account(self):
        credential = self.pair()
        profile = {"base_url": "https://example.test/v1", "api": "openai-completions",
                   "model": "model", "api_key": "test-only-secret"}
        with patch.object(desktop_client.AGENTS, "current", return_value=None), patch.object(
                desktop_client.AGENTS, "start") as start:
            response = self.launch(credential, agent=profile)
            self.assertEqual(response.status_code, 200)
            self.assertNotIn(profile["api_key"], response.text)
            request = start.call_args.args[0]
            self.assertEqual(request.owner, "alice")
            self.assertEqual(request.workspace, self.alice.root)
            self.assertEqual(request.api_key, profile["api_key"])
            self.assertTrue(start.call_args.kwargs["shared_mode"])
        with patch.object(desktop_client.AGENTS, "current", return_value=Mock(active=True)), patch.object(
                desktop_client.AGENTS, "start") as start:
            self.assertEqual(self.launch(credential, agent=profile).status_code, 200)
            start.assert_not_called()

    def test_browser_launch_does_not_replace_another_signed_in_account(self):
        credential = self.pair()
        ticket = self.launch(credential).json()["ticket"]
        self.server.sessions["bob-session"] = (
            self.server.accounts["bob"], desktop_client.monotonic() + 3600,
        )
        response = self.client.post("/client/v1/consume", json={"account": "alice", "code": ticket},
                                    headers={"Cookie": f"{shared.COOKIE}=bob-session"})
        self.assertEqual(response.status_code, 409)
        self.assertNotIn("set-cookie", response.headers)
