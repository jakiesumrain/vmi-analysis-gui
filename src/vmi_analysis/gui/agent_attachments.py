"""Private workspace uploads for pasted and dropped Agent attachments."""

import asyncio
from contextlib import aclosing
from pathlib import Path
import re
import shutil
from uuid import uuid4

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from nicegui import app
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from . import shared

MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024
MAX_ATTACHMENT_FILES = 10
MAX_BATCH_BYTES = 100 * 1024 * 1024
MAX_MULTIPART_OVERHEAD_BYTES = 1024 * 1024
ATTACHMENT_DIR = Path(".agent") / "attachments"


def attachment_root() -> Path:
    return shared.checked_path(shared.workspace_root() / ATTACHMENT_DIR, write=True)


def attachment_name(value: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value.replace("\\", "/").split("/")[-1])
    name = name.strip(" .")
    if not name:
        raise ValueError("The attachment needs a file name.")
    name = name[-128:]
    stem = name.split(".", 1)[0].rstrip(" ").upper()
    if re.fullmatch(r"CON|PRN|AUX|NUL|CONIN\$|CONOUT\$|(?:COM|LPT)[1-9¹²³]", stem):
        name = f"_{name}"
    return name


def _write_attachment(batch: Path, name: str, content: bytes) -> Path:
    target = batch / name
    stem, suffix = target.stem, target.suffix
    index = 2
    while target.exists():
        target = batch / f"{stem}-{index}{suffix}"
        index += 1
    target.write_bytes(content)
    return target


class UploadTooLarge(MultiPartException):
    pass


async def _bounded_stream(stream, limit: int):
    size = 0
    async for chunk in stream:
        size += len(chunk)
        if size > limit:
            raise UploadTooLarge("Keep each attachment batch below 100 MiB.")
        yield chunk


@app.post("/agent-attachments")
async def upload_agent_attachments(request: Request):
    origin = request.headers.get("origin")
    if origin and origin != f"{request.url.scheme}://{request.headers.get('host', '')}":
        raise HTTPException(403, "Cross-origin uploads are not allowed.")
    content_type = request.headers.get("content-type", "").partition(";")[0].strip().lower()
    if content_type != "multipart/form-data":
        raise HTTPException(400, "Choose one or more files to attach.")
    limit = MAX_BATCH_BYTES + MAX_MULTIPART_OVERHEAD_BYTES
    length = request.headers.get("content-length", "")
    if length.isdigit() and int(length) > limit:
        raise HTTPException(413, "Keep each attachment batch below 100 MiB.")
    try:
        async with aclosing(request.stream()) as stream:
            form = await MultiPartParser(
                request.headers, _bounded_stream(stream, limit),
                max_fields=0, max_files=MAX_ATTACHMENT_FILES, max_part_size=MAX_ATTACHMENT_BYTES,
            ).parse()
        try:
            items = list(form.multi_items())
            uploads = [value for key, value in items if key == "files" and isinstance(value, UploadFile)]
            if not uploads or len(uploads) != len(items):
                raise HTTPException(400, "Choose one or more files to attach.")
            root = attachment_root()
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
            batch = root / uuid4().hex
            batch.mkdir(mode=0o700)
            paths = []
            total = 0
            try:
                for upload in uploads:
                    try:
                        name = attachment_name(upload.filename or "")
                    except ValueError as exc:
                        raise HTTPException(400, str(exc)) from exc
                    content = await upload.read(MAX_ATTACHMENT_BYTES + 1)
                    if len(content) > MAX_ATTACHMENT_BYTES:
                        raise HTTPException(413, "Each attachment must be 25 MiB or smaller.")
                    total += len(content)
                    if total > MAX_BATCH_BYTES:
                        raise HTTPException(413, "Keep each attachment batch below 100 MiB.")
                    paths.append(await asyncio.to_thread(_write_attachment, batch, name, content))
            except Exception:
                await asyncio.to_thread(shutil.rmtree, batch, True)
                raise
        finally:
            await form.close()
    except UploadTooLarge as exc:
        raise HTTPException(413, exc.message) from exc
    except MultiPartException as exc:
        raise HTTPException(400, "Check the number and size of the attachments.") from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse({"paths": [str(path) for path in paths]},
                        headers={"Cache-Control": "no-store"})
