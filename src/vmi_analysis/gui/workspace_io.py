"""Open shared workspace files relative to a trusted Linux directory."""

from contextlib import contextmanager, ExitStack
import os
import stat

from starlette.responses import FileResponse, JSONResponse


@contextmanager
def directory_at(root, parts=(), *, create=False):
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory = os.dup(root) if isinstance(root, int) else os.open(root, flags)
    try:
        for name in parts:
            if name in ("", ".", "..") or "/" in name:
                raise ValueError("Choose a path inside the workspace.")
            if create:
                try:
                    os.mkdir(name, mode=0o700, dir_fd=directory)
                except FileExistsError:
                    pass
            child = os.open(name, flags, dir_fd=directory)
            os.close(directory)
            directory = child
        yield directory
    finally:
        os.close(directory)


@contextmanager
def open_regular_file(root, path):
    parts = path.relative_to(root).parts
    if not parts or parts[-1] == "..":
        raise OSError("Choose a regular file.")
    with directory_at(root, parts[:-1]) as directory:
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=directory)
        with os.fdopen(fd, "rb") as handle:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise OSError("Choose a regular file.")
            yield handle


class WorkspaceFileResponse(FileResponse):
    def __init__(self, root, path, **kwargs):
        super().__init__(path, **kwargs)
        self.root = root

    async def __call__(self, scope, receive, send):
        with ExitStack() as stack:
            try:
                handle = stack.enter_context(open_regular_file(self.root, self.path))
            except (OSError, ValueError):
                await JSONResponse({"detail": "Not Found"}, status_code=404)(scope, receive, send)
                return
            self.stat_result = os.fstat(handle.fileno())
            self.set_stat_headers(self.stat_result)
            self.path = f"/proc/self/fd/{handle.fileno()}"
            # Keep this process responsible for streaming while the descriptor
            # is open, including for range requests.
            extensions = dict(scope.get("extensions", {}))
            extensions.pop("http.response.pathsend", None)
            await super().__call__({**scope, "extensions": extensions}, receive, send)
