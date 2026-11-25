"""Private workspace uploads for pasted and dropped Agent attachments."""

from base64 import b64encode
from contextlib import aclosing, nullcontext
from io import BytesIO
import os
from pathlib import Path
import re
import shutil
from uuid import uuid4

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from nicegui import app
from PIL import Image, ImageOps
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from . import shared
from .workspace_io import directory_at, open_regular_file

MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024
MAX_ATTACHMENT_FILES = 10
MAX_BATCH_BYTES = 100 * 1024 * 1024
MAX_MULTIPART_OVERHEAD_BYTES = 1024 * 1024
ATTACHMENT_DIR = Path(".agent") / "attachments"
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"})
MAX_PROMPT_IMAGES = 4
MAX_IMAGE_PIXELS = 24_000_000
MAX_ENCODED_IMAGE_BYTES = 512 * 1024


def prompt_image(path: Path, *, roots=()) -> dict:
    """Decode a checked attachment and send a bounded, metadata-free image to Pi."""
    if roots:
        root = next((root for root in roots if path.is_relative_to(root)), None)
        if root is None:
            raise ValueError("Choose an image in your workspace or the shared datasets.")
        opened = open_regular_file(root, path)
    else:
        opened = path.open("rb")
    with opened as handle:
        raw = handle.read(MAX_ATTACHMENT_BYTES + 1)
    if len(raw) > MAX_ATTACHMENT_BYTES:
        raise ValueError(f"{path.name} exceeds the 25 MiB image limit.")
    try:
        with Image.open(BytesIO(raw)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError(f"{path.name} exceeds the 24-megapixel image limit.")
            image = ImageOps.exif_transpose(source)
            image.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
            if "A" in image.getbands() or "transparency" in image.info:
                rgba = image.convert("RGBA")
                clean = Image.new("RGB", rgba.size, "white")
                clean.paste(rgba, mask=rgba.getchannel("A"))
            else:
                clean = Image.new("RGB", image.size)
                clean.paste(image.convert("RGB"))
        output = BytesIO()
        clean.save(output, format="PNG")
        mime = "image/png"
        if output.tell() > MAX_ENCODED_IMAGE_BYTES:
            output = BytesIO()
            clean.save(output, format="JPEG", quality=90)
            mime = "image/jpeg"
        if output.tell() > MAX_ENCODED_IMAGE_BYTES:
            raise ValueError(f"{path.name} is too detailed to send. Crop it and attach the area you want to inspect.")
    except Image.DecompressionBombError as exc:
        raise ValueError(f"{path.name} exceeds the 24-megapixel image limit.") from exc
    except OSError as exc:
        raise ValueError(f"Could not read {path.name} as an image.") from exc
    return {"type": "image", "mimeType": mime, "data": b64encode(output.getvalue()).decode("ascii")}


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


def _write_attachment(batch: Path, name: str, content: bytes, directory: int | None) -> Path:
    original = target = batch / name
    index = 2
    while True:
        try:
            if directory is None:
                handle = target.open("xb")
            else:
                fd = os.open(target.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=directory)
                handle = os.fdopen(fd, "wb")
        except FileExistsError:
            target = batch / f"{original.stem}-{index}{original.suffix}"
            index += 1
            continue
        with handle:
            handle.write(content)
        return target


def _store_uploads(root, workspace, uploads):
    if workspace is not None:
        if os.name != "posix":
            raise OSError("Shared attachment uploads require a Linux host.")
        parent = directory_at(workspace, ATTACHMENT_DIR.parts, create=True)
    else:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        parent = nullcontext()
    batch = root / uuid4().hex
    with parent as directory:
        if directory is None:
            batch.mkdir(mode=0o700)
            opened = nullcontext()
        else:
            os.mkdir(batch.name, mode=0o700, dir_fd=directory)
            opened = directory_at(directory, (batch.name,))
        try:
            paths = []
            total = 0
            with opened as batch_fd:
                for upload in uploads:
                    name = attachment_name(upload.filename or "")
                    content = upload.file.read(MAX_ATTACHMENT_BYTES + 1)
                    if len(content) > MAX_ATTACHMENT_BYTES:
                        raise HTTPException(413, "Each attachment must be 25 MiB or smaller.")
                    total += len(content)
                    if total > MAX_BATCH_BYTES:
                        raise HTTPException(413, "Keep each attachment batch below 100 MiB.")
                    paths.append(_write_attachment(batch, name, content, batch_fd))
            return paths
        except BaseException:
            if directory is None:
                shutil.rmtree(batch, ignore_errors=True)
            else:
                shutil.rmtree(batch.name, dir_fd=directory, ignore_errors=True)
            raise


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
            account = shared.current_account()
            workspace = account.root if account else None
            root = workspace / ATTACHMENT_DIR if workspace is not None else attachment_root()
            paths = await run_in_threadpool(_store_uploads, root, workspace, uploads)
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
