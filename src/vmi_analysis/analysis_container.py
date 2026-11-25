"""Launch one approved analysis in a disposable Linux container."""

import argparse
import csv
import io
import os
from pathlib import Path
import subprocess
import sys
from threading import Thread
from time import sleep
from uuid import uuid4


PYTHON = "/opt/vmi-agent/.venv/bin/python"
RUN_ROOT = "/job"
MAX_WORKER_LOG = 1024 * 1024


def _mount(source, target, *, readonly, podman):
    if podman:
        # Podman's :ro bind is non-recursive unless rbind is also requested.
        return ["--volume", f"{source}:{target}:{'ro' if readonly else 'rw'}"]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="")
    fields = ["type=bind", f"source={source}", f"target={target}"]
    if readonly:
        fields.extend(("readonly", "bind-recursive=readonly", "bind-propagation=rprivate"))
    writer.writerow(fields)
    return ["--mount", buffer.getvalue()]


def create_command(runtime, image, name, folder, workspace, datasets, *, rootless, uid, gid):
    podman = Path(runtime).stem == "podman"
    command = [
        runtime, "create", "--pull=never", "--name", name,
        "--label", "org.vmi-analysis.worker=python",
        "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
        "--pids-limit=256", "--memory=1g", "--cpus=1",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=268435456",
        "--env", "HOME=/tmp", "--env", "MPLBACKEND=Agg", "--env", "PYTHONUNBUFFERED=1",
        "--env", "PYTHONSAFEPATH=1", "--env", "VMI_WORKSPACE_DIR=/workspace",
        "--env", "VMI_CACHE_DIR=/tmp/vmi-cache",
    ]
    if podman and rootless:
        command.append("--userns=keep-id")
    elif not rootless and uid is not None:
        command.extend(("--user", f"{uid}:{gid}"))
    target = f"{RUN_ROOT}/{folder.name}"
    for path, mount, readonly in (
        (workspace, "/workspace", True), (datasets, "/datasets", True),
        (folder.parent.parent, "/analysis", True), (folder, target, True),
        (folder / "outputs", f"{target}/outputs", False),
    ):
        command.extend(_mount(path, mount, readonly=readonly, podman=podman))
    command.extend(("--workdir", f"{target}/outputs", "--entrypoint", PYTHON, image,
                    "-m", "vmi_analysis.analysis_worker", f"{target}/worker.json"))
    return command


def _query(args, *, timeout=10):
    result = subprocess.run(args, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"{args[0]} {args[1]} failed.")
    return result.stdout.strip()


def _forward_logs(stream):
    remaining = MAX_WORKER_LOG
    for line in iter(lambda: stream.readline(4096), b""):
        if remaining:
            sys.stdout.buffer.write(line[:remaining])
            if not line.endswith(b"\n"):
                sys.stdout.buffer.write(b"\n")
            remaining -= min(len(line), remaining)
            if remaining == 0:
                print("\n[worker] Further container logs were omitted.")
            sys.stdout.flush()


def run(folder, workspace, datasets):
    runtime = os.environ.get("VMI_AGENT_CONTAINER_RUNTIME", "docker")
    image = os.environ.get("VMI_AGENT_IMAGE", "vmi-analysis-agent:latest")
    podman = Path(runtime).stem == "podman"
    name = f"vmi-analysis-worker-{uuid4().hex}"
    parent = os.getppid()
    stop = folder / ".cancel"

    def cancelled():
        return stop.exists() or os.getppid() != parent

    if cancelled():
        return 1
    template = "{{.Host.Security.Rootless}}" if podman else "{{json .SecurityOptions}}"
    info = _query([runtime, "info", "--format", template]) if os.name == "posix" else ""
    if cancelled():
        return 1
    image_id = _query([runtime, "image", "inspect", "--format", "{{.Id}}", image])
    if cancelled():
        return 1
    (folder / "worker-image.txt").write_text(image_id + "\n", encoding="utf-8")
    print(f"Isolated worker {name} · {image} · {image_id}", flush=True)
    args = create_command(runtime, image_id, name, folder, workspace, datasets,
                          rootless=info == "true" if podman else "rootless" in info,
                          uid=os.getuid() if hasattr(os, "getuid") else None,
                          gid=os.getgid() if hasattr(os, "getgid") else None)
    process = reader = None
    created = False
    try:
        _query(args, timeout=25)
        created = True
        if cancelled():
            return 1
        process = subprocess.Popen([runtime, "start", "--attach", name], stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        reader = Thread(target=_forward_logs, args=(process.stdout,), daemon=True)
        reader.start()
        while process.poll() is None:
            if cancelled():
                return 1
            sleep(0.1)
        state = _query([runtime, "inspect", "--format", "{{.State.Status}} {{.State.ExitCode}}", name], timeout=8)
        status, code = state.split()
        return int(code) if status == "exited" else (process.returncode or 1)
    finally:
        try:
            removed = subprocess.run([runtime, "rm", "--force", name], stdin=subprocess.DEVNULL,
                                     capture_output=True, text=True, timeout=10)
            if removed.returncode and created:
                raise RuntimeError(f"Could not remove {name}: {removed.stderr.strip()}")
        finally:
            if process is not None and process.poll() is None:
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            if reader is not None:
                reader.join(timeout=2)
                process.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("datasets", type=Path)
    args = parser.parse_args()
    try:
        raise SystemExit(run(args.folder, args.workspace, args.datasets))
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f"Analysis worker timed out waiting for the container runtime ({exc.cmd[1]}).") from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Analysis worker failed: {exc}") from exc


if __name__ == "__main__":
    main()
