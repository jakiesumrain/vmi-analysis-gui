from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from vmi_analysis.agent_backend import AgentManager, AgentRequest, AgentSession, ContainerLauncher
from vmi_analysis.agent_history import AgentHistory


class AgentHistoryTests(TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory())).resolve()
        self.workspace = self.root / "alice" / "workspace"
        self.folder = self.workspace / "Xe calibration"
        self.folder.mkdir(parents=True)
        self.history = AgentHistory(self.workspace.parent / ".agent-history", self.workspace)
        self.project = self.history.create_project("Xe calibration", self.folder)
        self.request = AgentRequest("alice", self.workspace, self.workspace.parent / ".agent",
                                    "https://example.test/v1", "test-model", "test-provider-secret")

    def test_reopens_the_same_chat_with_new_credentials_and_no_pending_approval(self):
        launcher = ContainerLauncher()
        with patch.object(AgentSession, "start", autospec=True, side_effect=lambda session: session):
            manager = AgentManager(launcher)
            original = manager.start(self.request, shared_mode=True, history=self.history, project_id=self.project.id)
            original._append("user", "Check the Xe fit")
            original._append_assistant("A partial match, token test-provider-")
            original._append_assistant("secret")
            original._panel_requests["approval"] = ({"action": "propose"}, float("inf"))
            original.handle_record({"type": "agent_settled"})
            with self.assertRaisesRegex(RuntimeError, "Stop the current"):
                manager.start(self.request, shared_mode=True, history=self.history)
            reopened_history = AgentHistory(self.history.root, self.workspace)
            reopened = AgentManager(launcher).start(
                self.request, shared_mode=True, history=reopened_history, chat_id=original.chat.id)

        self.assertEqual(reopened.request.working_directory, self.folder)
        self.assertEqual(reopened.request.session_dir, original.request.session_dir)
        self.assertEqual(reopened.chat.title, "Check the Xe fit")
        self.assertIsNone(reopened.panel_request())
        self.assertEqual(reopened.snapshot()["entries"][-1]["text"], "A partial match, token [redacted]")
        command = launcher.command(reopened.request, reopened.config_dir, reopened.id)
        self.assertIn("/workspace/Xe calibration", command)
        self.assertIn(f"{reopened.request.session_dir}:/history:rw", command)
        self.assertIn("--continue", command)
        self.assertNotIn("--no-session", command)
        self.assertNotIn(self.request.api_key, " ".join(command))

    def test_other_accounts_and_outside_folders_are_out_of_scope(self):
        chat = self.history.create_chat(self.project.id)
        other = AgentHistory(self.root / "bob" / ".agent-history", self.root / "bob" / "workspace")
        with self.assertRaisesRegex(ValueError, "chat is no longer available"):
            other.chat(chat.id)
        with self.assertRaisesRegex(ValueError, "Invalid Agent history ID"):
            self.history.chat("../settings")
        with self.assertRaisesRegex(ValueError, "inside your workspace"):
            self.history.create_project("Outside", self.root)
        self.assertFalse(self.history.session_dir(chat).is_relative_to(self.workspace))
