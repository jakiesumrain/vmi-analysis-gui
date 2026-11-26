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

    def test_one_connection_starts_and_resumes_chats_in_different_projects(self):
        manager = AgentManager()
        supplied_keys = []

        def launch(session):
            supplied_keys.append(session.request.api_key)
            session.status = "ready"
            return session

        with patch.object(AgentSession, "start", autospec=True, side_effect=launch), patch.object(
                AgentSession, "prompt", autospec=True):
            initial = manager.start(self.request, shared_mode=True, history=self.history)
            self.assertEqual(manager.connection("alice").api_key, "")
            self.assertIsNone(manager.connection("bob"))
            project_chat = manager.prompt_chat("alice", "Check the fit", project_id=self.project.id)
            self.assertFalse(initial.active)
            self.assertEqual(initial.request.api_key, "")
            self.assertEqual(project_chat.request.working_directory, self.folder)
            resumed = manager.prompt_chat("alice", "Continue", chat_id=initial.chat.id)
            self.assertFalse(project_chat.active)
            self.assertEqual(resumed.chat.id, initial.chat.id)
            self.assertEqual(resumed.request.session_dir, initial.request.session_dir)
            self.assertEqual(supplied_keys, [self.request.api_key] * 3)
            manager.stop("alice")
            self.assertIsNone(manager.connection("alice"))
            with self.assertRaisesRegex(RuntimeError, "Connect a provider"):
                manager.prompt_chat("alice", "Try again", chat_id=initial.chat.id)

    def test_a_running_reply_keeps_its_chat_when_another_chat_sends(self):
        manager = AgentManager()
        with patch.object(AgentSession, "start", autospec=True, side_effect=lambda session: session):
            current = manager.start(self.request, shared_mode=True, history=self.history)
            current.status = "running"
            with self.assertRaisesRegex(RuntimeError, "Finish or stop the reply"):
                manager.prompt_chat("alice", "Start another chat", project_id=self.project.id)
            with self.assertRaisesRegex(RuntimeError, "before archiving"):
                manager.set_archived("alice", self.history, chat_id=current.chat.id, archived=True)
            self.assertFalse(self.history.chat(current.chat.id).archived)
            self.assertIs(manager.current("alice"), current)
            self.assertEqual(len(self.history.chats()), 1)

    def test_archiving_a_project_preserves_its_chats_and_provider(self):
        manager = AgentManager()
        with patch.object(AgentSession, "start", autospec=True, side_effect=lambda session: session):
            current = manager.start(self.request, shared_mode=True, history=self.history, project_id=self.project.id)
            current._append("user", "Check the fit")
            current.handle_record({"type": "agent_settled"})
            other = self.history.create_chat(self.project.id)
            manager.set_archived("alice", self.history, chat_id=other.id, archived=True)
            manager.set_archived("alice", self.history, project_id=self.project.id, archived=True)
            self.assertFalse(current.active)
            self.assertIsNotNone(manager.connection("alice"))
            self.assertIsNone(manager.current("alice"))
            with self.assertRaisesRegex(ValueError, "Restore the project"):
                manager.prompt_chat("alice", "Continue", chat_id=current.chat.id)

            manager.set_archived("alice", self.history, project_id=self.project.id, archived=False)
            self.assertEqual(self.history.project_folder(self.project.id), self.folder)
            self.assertEqual(self.history.entries(current.chat)[0]["text"], "Check the fit")
            self.assertTrue(self.history.chat(other.id).archived)
            with self.assertRaisesRegex(ValueError, "Restore the chat"):
                manager.prompt_chat("alice", "Continue", chat_id=other.id)
            manager.set_archived("alice", self.history, chat_id=other.id, archived=False)
            self.assertEqual(self.history.resumable_chat(other.id).project_id, self.project.id)
