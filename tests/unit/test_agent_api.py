from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from unittest import TestCase
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vmi_analysis.gui import agent_api, shared


class AgentApiTests(TestCase):
    def setUp(self):
        root = Path(self.enterContext(TemporaryDirectory()))
        datasets = root / "datasets"
        datasets.mkdir()
        self.server = shared.SharedServer({
            "datasets": str(datasets),
            "workspaces": str(root / "accounts"),
            "accounts": {"alice": {"password_hash": shared.password_hash("alice")}},
        })
        self.enterContext(patch.object(shared, "SERVER", self.server))
        self.alice = self.server.accounts["alice"]
        self.server.sessions["session"] = (self.alice, monotonic() + 3600)
        app = FastAPI()
        app.add_middleware(shared.AccountMiddleware)
        app.post("/api/agent/session")(agent_api.start_session)
        self.client = self.enterContext(TestClient(app, base_url="http://127.0.0.1:13517"))
        self.connection = {"base_url": "https://example.test/v1", "api": "openai-completions",
                           "model": "test-model", "api_key": "test-provider-key"}

    def start(self, **settings):
        return self.client.post("/api/agent/session", json={**self.connection, **settings},
                                headers={"Cookie": f"{shared.COOKIE}=session"})

    def test_session_is_required_and_manual_only_cannot_start_agent(self):
        self.assertEqual(self.client.post("/api/agent/session", json=self.connection).status_code, 401)
        self.server.manual_only = True
        with patch.object(agent_api.AGENTS, "start") as start:
            self.assertEqual(self.start().status_code, 403)
            start.assert_not_called()

    def test_start_uses_authenticated_account_and_existing_agent(self):
        with patch.object(agent_api.AGENTS, "current", return_value=None), patch.object(
                agent_api.AGENTS, "start") as start:
            response = self.start(owner="different-account")
            self.assertEqual(response.json(), {"state": "started"})
            request = start.call_args.args[0]
            self.assertEqual(request.owner, "alice")
            self.assertEqual(request.workspace, self.alice.root)
            self.assertEqual(request.api_key, self.connection["api_key"])
            self.assertTrue(start.call_args.kwargs["shared_mode"])
        with patch.object(agent_api.AGENTS, "current", return_value=Mock(active=True)), patch.object(
                agent_api.AGENTS, "start") as start:
            self.assertEqual(self.start().json(), {"state": "active"})
            start.assert_not_called()

    def test_invalid_request_does_not_reflect_the_key(self):
        with patch.object(agent_api.AGENTS, "current", return_value=None):
            response = self.start(api_key={"secret": self.connection["api_key"]})
        self.assertEqual(response.status_code, 400)
        self.assertNotIn(self.connection["api_key"], response.text)

    def test_local_desktop_requires_loopback_and_uses_the_local_agent(self):
        app = FastAPI()
        app.head("/api/local-client")(agent_api.local_desktop)
        app.post("/api/local-client")(agent_api.start_local_session)
        base = "http://127.0.0.1:13517"
        with patch.object(shared, "SERVER", None), TestClient(
                app, base_url=base, client=("127.0.0.1", 50000)) as client:
            self.assertEqual(client.head("/api/local-client").headers["X-VMI-Local-Client"], "1")
            with patch.object(agent_api.AGENTS, "current", return_value=None), patch.object(
                    agent_api.AGENTS, "start") as start:
                for headers in ({"Origin": "http://evil.test"}, {"Sec-Fetch-Site": "same-origin"},
                                {"Host": "example.test"}):
                    denied = client.post("/api/local-client", json=self.connection, headers=headers)
                    self.assertEqual(denied.status_code, 403)
                start.assert_not_called()
                self.assertEqual(client.post("/api/local-client", json=self.connection).json(), {"state": "started"})
                self.assertEqual(start.call_args.args[0].owner, "local")
                self.assertFalse(start.call_args.kwargs["shared_mode"])
        with patch.object(shared, "SERVER", None), TestClient(
                app, base_url=base, client=("192.0.2.10", 50000)) as client:
            self.assertEqual(client.head("/api/local-client").status_code, 403)
