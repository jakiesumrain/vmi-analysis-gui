"""Completions for paths visible from the embedded Agent workspace."""

from collections import deque
from itertools import islice
import os
from pathlib import Path
from time import monotonic


def _match_rank(name: str, query: str) -> int:
    name, query = name.casefold(), query.casefold()
    if name.startswith(query):
        return 0
    if query in name:
        return 1
    letters = iter(name)
    return 2 if all(letter in letters for letter in query) else 3


def path_completions(query: str, roots: tuple[tuple[str, Path], ...]) -> list[dict]:
    query = query.replace("\\", "/")
    qualifier = next((name for name, _ in roots if query.startswith(f"/{name}/")), "")
    if qualifier:
        query = query[len(qualifier) + 2:]
    prefix, _, pattern = query.rpartition("/")
    results = []
    for name, root in roots:
        if qualifier and qualifier != name:
            continue
        start = (root / prefix).resolve()
        if not start.is_relative_to(root):
            continue
        directories = deque([start])
        deadline = monotonic() + 0.12
        remaining = 4000
        while directories and remaining and monotonic() < deadline:
            directory = directories.popleft()
            try:
                with os.scandir(directory) as children:
                    for entry in islice(children, remaining):
                        remaining -= 1
                        if monotonic() >= deadline:
                            break
                        if entry.name.startswith(".") or entry.is_symlink():
                            continue
                        path = Path(entry.path)
                        folder = entry.is_dir(follow_symlinks=False)
                        if folder and not path.resolve().is_relative_to(root):
                            continue
                        if folder and pattern and not prefix:
                            directories.append(path)
                        rank = _match_rank(entry.name, pattern)
                        if rank < 3:
                            relative = path.relative_to(root)
                            reference = f"/{name}/{relative.as_posix()}" if len(roots) > 1 else relative.as_posix()
                            if folder:
                                reference += "/"
                            results.append((rank, len(relative.parts), reference.casefold(), {
                                "label": reference,
                                "detail": "Folder" if folder else "File",
                                "value": reference,
                            }))
            except OSError:
                continue
    results.sort(key=lambda choice: choice[:3])
    return [choice[-1] for choice in results[:12]]
