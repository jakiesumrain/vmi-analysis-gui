"""Check the requested listening port and print actionable startup errors."""

import multiprocessing
import os
import socket
import sys

DEFAULT_PORT = 3157


def _bind_error(port, host="0.0.0.0"):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if sys.platform == 'win32':
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((host, port))
    except OSError as exc:
        return exc
    return None


def _fail(message, port, host, launch_command):
    print(f'[gui] {message}', file=sys.stderr, flush=True)
    candidates = dict.fromkeys([*range(port + 1, min(port + 21, 65536)),
                               *range(DEFAULT_PORT, DEFAULT_PORT + 20)])
    for candidate in candidates:
        if candidate != port and _bind_error(candidate, host) is None:
            print(f'[gui] Port {candidate} is available. In PowerShell, run:\n'
                  f'$env:VMI_GUI_PORT = "{candidate}"\n'
                  f'{launch_command}\n'
                  f'[gui] Then open http://localhost:{candidate}', file=sys.stderr, flush=True)
            break
    else:
        print(f'[gui] Set VMI_GUI_PORT to an available port, then run: {launch_command}',
              file=sys.stderr, flush=True)
    raise SystemExit(1)


def startup_port(*, host="0.0.0.0", launch_command="uv run vmi-analysis"):
    raw = os.environ.get('VMI_GUI_PORT', str(DEFAULT_PORT))
    try:
        port = int(raw)
        if not 1 <= port <= 65535:
            raise ValueError
    except ValueError:
        _fail(f'Invalid VMI_GUI_PORT={raw!r}; enter a whole number from 1 to 65535.',
              DEFAULT_PORT - 1, host, launch_command)
    # NiceGUI initializes spawned helper processes too, but only MainProcess
    # starts a listener. A helper must not reject its parent's occupied port.
    if multiprocessing.current_process().name == 'MainProcess':
        error = _bind_error(port, host)
        if error is not None:
            _fail(f'Cannot start: port {port} is unavailable ({error}).', port, host, launch_command)
    return port
