"""Filesystem boundary for standard panel commands on a shared Linux server."""

from pathlib import Path
from shutil import which
import sys


# Keep runtime mounts narrow. In particular, the service account's home and
# application checkout can also contain its configuration and private data.
SYSTEM_RUNTIME = (
    "/usr", "/bin", "/sbin", "/lib", "/lib64",
    "/etc/ld.so.cache", "/etc/fonts", "/etc/localtime",
)
PACKAGE = Path(__file__).absolute().parents[1]


def panel_command(command, cwd, workspace, datasets, analysis):
    """Launch using the host's tools with a private mount namespace."""
    if sys.platform != "linux" or not (sandbox := which("bwrap")):
        raise OSError("Shared panel jobs require Bubblewrap (bwrap) on the Linux server.")

    workspace, datasets, analysis = map(Path, (workspace, datasets, analysis))
    private = workspace.parent.parent
    protected = (private, datasets)
    for root in (workspace, datasets, analysis):
        # The service owns the parents of these mount points.
        if root.resolve(strict=True) != root:
            raise OSError("A shared storage root changed since server startup.")

    args = [
        sandbox, "--unshare-user", "--unshare-pid", "--unshare-net",
        "--unshare-ipc", "--unshare-uts", "--cap-drop", "ALL",
        "--die-with-parent", "--dev", "/dev", "--proc", "/proc",
        "--tmpfs", "/dev/shm", "--tmpfs", "/tmp",
        "--setenv", "HOME", str(workspace), "--setenv", "TMPDIR", "/tmp",
        "--setenv", "PYTHONNOUSERSITE", "1",
    ]

    def mount_runtime(path, destination=None):
        path = Path(path).absolute()
        source = path.resolve(strict=True)
        destination = Path(destination) if destination else path
        if any(root.is_relative_to(candidate) or candidate.is_relative_to(root)
               for root in protected for candidate in (source, destination)):
            raise OSError("Install analysis tools outside shared datasets and private workspaces.")
        args.extend(("--ro-bind", str(source), str(destination)))

    for path in map(Path, SYSTEM_RUNTIME):
        if path.exists():
            mount_runtime(path)
    # uv can point a virtualenv at a version alias as well as the resolved
    # installation. The interpreter and pyvenv.cfg need the alias at its
    # original location inside the namespace.
    interpreter_home = Path(sys._base_executable).parent.parent
    for path in dict.fromkeys((Path(sys.base_prefix), interpreter_home, Path(sys.prefix), PACKAGE)):
        if not path.is_relative_to("/usr"):
            mount_runtime(path)
    if binary := which("tpx3dump"):
        mount_runtime(binary, "/run/vmi-tools/tpx3dump")
    args.extend(("--setenv", "PATH", "/run/vmi-tools:/usr/local/bin:/usr/bin:/bin"))
    args.extend((
        "--ro-bind", str(datasets), str(datasets),
        "--ro-bind", str(analysis), str(analysis),
        "--bind", str(workspace), str(workspace),
        "--chdir", str(cwd), "--", *map(str, command),
    ))
    return args
