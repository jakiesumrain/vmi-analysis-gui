import asyncio
from io import BytesIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from PIL import Image
from starlette.datastructures import UploadFile

from vmi_analysis.gui import agent_attachments, shared


class AgentAttachmentTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(TemporaryDirectory())
        self.root = Path(directory)
        datasets = self.root / "datasets"
        datasets.mkdir()
        self.server = shared.SharedServer({
            "datasets": str(datasets),
            "workspaces": str(self.root / "workspaces"),
            "accounts": {
                name: {"password_hash": shared.password_hash(f"password-for-{name}")}
                for name in ("alice", "bob")
            },
        })
        self.enterContext(patch.object(shared, "SERVER", self.server))
        web = FastAPI()
        web.post("/agent-attachments")(agent_attachments.upload_agent_attachments)
        web.add_middleware(shared.AccountMiddleware)
        self.client = self.enterContext(TestClient(web, base_url="http://127.0.0.1:13517"))
        self.alice = self.server.accounts["alice"]
        self.bob = self.server.accounts["bob"]

    def sign_in(self, account):
        session = f"session-for-{account.name}"
        self.server.sessions[session] = (account, shared.monotonic() + 60)
        self.client.cookies.set(shared.COOKIE, session)

    def test_uploads_require_an_account_and_same_origin(self):
        self.assertEqual(self.client.post("/agent-attachments").status_code, 401)
        self.sign_in(self.alice)
        response = self.client.post("/agent-attachments", headers={"Origin": "http://external.invalid"})
        self.assertEqual(response.status_code, 403)
        self.assertFalse((self.alice.root / agent_attachments.ATTACHMENT_DIR).exists())

    def test_manual_only_rejects_authenticated_attachment_uploads(self):
        self.server.manual_only = True
        self.sign_in(self.alice)
        response = self.client.post("/agent-attachments", files={"files": ("note.txt", b"sample", "text/plain")})
        self.assertEqual(response.status_code, 404)
        self.assertFalse((self.alice.root / agent_attachments.ATTACHMENT_DIR).exists())

    @unittest.skipUnless(sys.platform == "linux", "Shared attachment uploads run on Linux")
    def test_multiple_files_are_stored_in_the_current_accounts_workspace(self):
        self.sign_in(self.alice)
        response = self.client.post("/agent-attachments", files=[
            ("files", ("notes.txt", b"alice notes", "text/plain")),
            ("files", ("plot.png", b"png bytes", "image/png")),
        ])
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["cache-control"], "no-store")
        paths = [Path(value) for value in response.json()["paths"]]
        self.assertEqual([path.read_bytes() for path in paths], [b"alice notes", b"png bytes"])
        for path in paths:
            self.assertTrue(path.is_relative_to(self.alice.root / agent_attachments.ATTACHMENT_DIR))
            self.assertFalse(path.is_relative_to(self.bob.root))

        self.sign_in(self.bob)
        response = self.client.post("/agent-attachments",
                                    files={"files": ("private.txt", b"bob only", "text/plain")})
        self.assertEqual(response.status_code, 200, response.text)
        path = Path(response.json()["paths"][0])
        self.assertTrue(path.is_relative_to(self.bob.root / agent_attachments.ATTACHMENT_DIR))
        self.assertEqual(path.read_bytes(), b"bob only")

    @unittest.skipUnless(sys.platform == "linux", "Shared attachment uploads run on Linux")
    def test_unsafe_names_are_sanitized_and_collisions_are_deduplicated(self):
        self.sign_in(self.alice)
        response = self.client.post("/agent-attachments", files=[
            ("files", ("../notes.txt", b"first", "text/plain")),
            ("files", ("folder/notes.txt", b"second", "text/plain")),
            ("files", (r"C:\temp\bad?.txt", b"third", "text/plain")),
        ])
        self.assertEqual(response.status_code, 200, response.text)
        paths = [Path(value) for value in response.json()["paths"]]
        self.assertEqual([path.name for path in paths], ["notes.txt", "notes-2.txt", "bad_.txt"])
        self.assertEqual([path.read_bytes() for path in paths], [b"first", b"second", b"third"])

    def test_windows_device_names_get_safe_filenames(self):
        for source, expected in (("NUL.txt", "_NUL.txt"), ("COM1", "_COM1"),
                                 ("LPT².dat", "_LPT².dat"), ("CONOUT$", "_CONOUT$")):
            with self.subTest(source=source):
                self.assertEqual(agent_attachments.attachment_name(source), expected)

    def test_local_upload_batch_supports_duplicate_names(self):
        root = self.root / "local"
        uploads = [UploadFile(BytesIO(content), filename="a.txt") for content in (b"one", b"two")]
        paths = agent_attachments._store_uploads(root, None, uploads)
        self.assertEqual([path.name for path in paths], ["a.txt", "a-2.txt"])
        self.assertEqual([path.read_bytes() for path in paths], [b"one", b"two"])

    def test_streamed_request_stops_before_parsing_a_batch_over_the_limit(self):
        parts = iter((
            (b'--vmi\r\nContent-Disposition: form-data; name="files"; filename="large.txt"'
             b"\r\nContent-Type: text/plain\r\n\r\n"),
            b"x" * 200,
            b"\r\n--vmi--\r\n",
        ))
        received = []

        async def receive():
            body = next(parts)
            received.append(body)
            return {"type": "http.request", "body": body, "more_body": len(received) < 3}

        request = Request({"type": "http", "headers": [
            (b"content-type", b"multipart/form-data; boundary=vmi"),
        ]}, receive)
        with (patch.object(agent_attachments, "MAX_BATCH_BYTES", 8),
              patch.object(agent_attachments, "MAX_MULTIPART_OVERHEAD_BYTES", 128),
              patch.object(shared, "current_account", return_value=self.alice)):
            with self.assertRaises(HTTPException) as error:
                asyncio.run(agent_attachments.upload_agent_attachments(request))
        self.assertEqual(error.exception.status_code, 413)
        self.assertEqual(len(received), 2)
        self.assertFalse((self.alice.root / agent_attachments.ATTACHMENT_DIR).exists())

    @unittest.skipUnless(sys.platform == "linux", "Shared attachment uploads run on Linux")
    def test_rejected_batches_do_not_leave_partial_files(self):
        self.sign_in(self.alice)
        with patch.object(agent_attachments, "MAX_ATTACHMENT_FILES", 1):
            response = self.client.post("/agent-attachments", files=[
                ("files", ("one.txt", b"one", "text/plain")),
                ("files", ("two.txt", b"two", "text/plain")),
            ])
        self.assertEqual(response.status_code, 400)
        with patch.object(agent_attachments, "MAX_ATTACHMENT_BYTES", 4):
            response = self.client.post("/agent-attachments",
                                        files={"files": ("large.txt", b"12345", "text/plain")})
        self.assertEqual(response.status_code, 413)
        with (patch.object(agent_attachments, "MAX_ATTACHMENT_BYTES", 10),
              patch.object(agent_attachments, "MAX_BATCH_BYTES", 5)):
            response = self.client.post("/agent-attachments", files=[
                ("files", ("one.txt", b"1234", "text/plain")),
                ("files", ("two.txt", b"567", "text/plain")),
            ])
        self.assertEqual(response.status_code, 413)
        root = self.alice.root / agent_attachments.ATTACHMENT_DIR
        self.assertFalse(root.exists() and any(root.iterdir()))

    @unittest.skipUnless(sys.platform == "linux", "Shared attachment uploads run on Linux")
    def test_shared_upload_requires_real_attachment_directories(self):
        self.sign_in(self.alice)
        other = self.bob.root / "attachments"
        other.mkdir()
        prefix = self.alice.root / ".agent"
        prefix.mkdir()
        (prefix / "attachments").symlink_to(other, target_is_directory=True)
        response = self.client.post("/agent-attachments",
                                    files={"files": ("notes.txt", b"sample", "text/plain")})
        self.assertEqual(response.status_code, 400)
        self.assertFalse(any(other.iterdir()))

    @unittest.skipUnless(sys.platform == "linux", "Shared attachment reads run on Linux")
    def test_prompt_image_is_read_from_the_authorized_workspace(self):
        image = self.alice.root / "plot.png"
        Image.new("RGB", (8, 8)).save(image)
        self.assertEqual(agent_attachments.prompt_image(image, roots=(self.alice.root,))["mimeType"],
                         "image/png")
        with self.assertRaisesRegex(ValueError, "Choose an image"):
            agent_attachments.prompt_image(image, roots=(self.bob.root,))


if __name__ == "__main__":
    unittest.main()
