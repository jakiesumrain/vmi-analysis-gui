from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from unittest import TestCase
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vmi_analysis.agent_backend import AgentManager
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
        self.manager = self.enterContext(patch.object(agent_api, "AGENTS", AgentManager()))
        self.alice = self.server.accounts["alice"]
        self.server.sessions["session"] = (self.alice, monotonic() + 3600)
        app = FastAPI()
        app.add_middleware(shared.AccountMiddleware)
        app.post("/api/agent/session")(agent_api.connect_provider)
        self.client = self.enterContext(TestClient(app, base_url="http://127.0.0.1:13517"))
        self.connection = {"base_url": "https://example.test/v1", "api": "openai-completions",
                           "model": "test-model", "api_key": "test-provider-key"}

    def connect(self, **settings):
        return self.client.post("/api/agent/session", json={**self.connection, **settings},
                                headers={"Cookie": f"{shared.COOKIE}=session"})

    def test_session_is_required_and_manual_only_cannot_connect_agent(self):
        self.assertEqual(self.client.post("/api/agent/session", json=self.connection).status_code, 401)
        self.server.manual_only = True
        self.assertEqual(self.connect().status_code, 403)
        self.assertIsNone(self.manager.connection("alice"))

    def test_connect_is_idempotent_and_creates_no_chat(self):
        response = self.connect(owner="different-account")
        self.assertEqual(response.json(), {"state": "connected"})
        self.assertEqual(self.connect(model="other-model").json(), {"state": "connected"})
        provider = self.manager.connection("alice")
        self.assertEqual(provider.workspace, self.alice.root)
        self.assertEqual(provider.model, self.connection["model"])
        self.assertEqual(provider.api_key, "")
        self.assertIsNone(self.manager.connection("different-account"))
        self.assertIsNone(self.manager.current("alice"))
        self.assertEqual(shared.agent_history(self.alice).chats(), [])

    def test_invalid_request_does_not_reflect_the_key(self):
        response = self.connect(api_key={"secret": self.connection["api_key"]})
        self.assertEqual(response.status_code, 400)
        self.assertNotIn(self.connection["api_key"], response.text)
