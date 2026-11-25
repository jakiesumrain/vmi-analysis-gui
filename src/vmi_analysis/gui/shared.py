"""Account identity and workspace policy for the optional shared server."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import MutableMapping
from contextvars import ContextVar
from dataclasses import dataclass
from getpass import getpass
import hashlib
from hmac import compare_digest
from http.cookies import SimpleCookie, CookieError
from importlib.resources import files
import json
import os
from pathlib import Path
import re
import secrets
import sys
from tempfile import NamedTemporaryFile
from time import monotonic
from urllib.parse import parse_qs

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from nicegui import app, ui
from nicegui.client import Client
from starlette.datastructures import Headers

from vmi_analysis.paths import WORKSPACE_ROOT, GENERATED_DIR
from .workspace_io import WorkspaceFileResponse

COOKIE = 'vmi_session'
MAX_SESSION_AGE = 12 * 3600
ACCOUNT_ID = re.compile(r'[a-z][a-z0-9_-]{0,31}\Z')
LOGIN_ASSETS = {
    '/login-assets/theme.css': 'text/css',
    '/login-assets/login.css': 'text/css',
    '/login-assets/electron-spectrum-measured.svg': 'image/svg+xml',
}


class WorkspaceDenied(ValueError):
    """A requested path falls outside the account's permitted storage."""


class Settings(MutableMapping):
    def __init__(self, path):
        self.path = path
        self.data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def save(self, updated):
        with NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            try:
                json.dump(updated, handle, allow_nan=False)
            except Exception:
                handle.close()
                temporary.unlink(missing_ok=True)
                raise
        try:
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)
        self.data = updated

    def __setitem__(self, key, value):
        if key not in self.data or self.data[key] != value:
            self.save(self.data | {key: value})

    def __delitem__(self, key):
        updated = self.data.copy()
        del updated[key]
        self.save(updated)

    def update(self, *args, **kwargs):
        self.save(self.data | dict(*args, **kwargs))


@dataclass(frozen=True)
class Account:
    name: str
    root: Path
    password: str
    admin: bool
    settings: Settings


def analysis_root(account):
    return account.root.parent / 'analysis'


def password_hash(password, salt=None):
    salt = secrets.token_hex(16) if salt is None else salt
    digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=2**14, r=8, p=1)
    return f'{salt}:{digest.hex()}'


class SharedServer:
    manual_only = False

    def __init__(self, config):
        mode = config.get('mode', 'sandboxed')
        if mode not in ('sandboxed', 'manual-only'):
            raise ValueError('Shared-server mode must be sandboxed or manual-only.')
        self.manual_only = mode == 'manual-only'
        self.datasets = Path(config['datasets']).expanduser().resolve(strict=True)
        root = Path(config['workspaces']).expanduser().resolve()
        if not self.datasets.is_dir() or root.is_relative_to(self.datasets) or self.datasets.is_relative_to(root):
            raise ValueError('Shared datasets and private workspace directories must be separate.')
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.accounts = {}
        self.sessions = {}
        self.dummy_password = password_hash(secrets.token_hex(16))
        for name, entry in config['accounts'].items():
            if not ACCOUNT_ID.fullmatch(name):
                raise ValueError('Use lowercase account IDs with letters, digits, underscores or hyphens.')
            parent = root / name
            workspace = parent / 'files'
            settings = parent / 'settings.json'
            analysis = parent / 'analysis'
            if any(path.is_symlink() for path in (parent, workspace, settings, analysis)):
                raise ValueError('Private workspace and settings paths must be regular paths.')
            digest = entry['password_hash']
            if not re.fullmatch(r'[0-9a-f]{32}:[0-9a-f]{128}', digest):
                raise ValueError(f'Invalid password hash for {name}.')
            for folder in (parent, workspace, analysis, workspace / 'centroiding', workspace / 'circularization',
                           workspace / '.cache'):
                folder.mkdir(exist_ok=True, mode=0o700)
            self.accounts[name] = Account(name, workspace, digest, entry.get('admin') is True, Settings(settings))
        if not self.accounts:
            raise ValueError('Configure at least one account.')
        self.limit = config.get('max_concurrent_jobs', 2)
        self.login_slots = asyncio.Semaphore(2)
        self.failed_logins = 0
        self.retry_after = 0.0

    def authenticate(self, name, password):
        account = self.accounts.get(name)
        expected = account.password if account else self.dummy_password
        return account if compare_digest(password_hash(password, expected.split(':')[0]), expected) else None

    def session(self, token):
        session = self.sessions.get(token)
        if session and session[1] > monotonic():
            return session[0]
        self.sessions.pop(token, None)
        return None

    def path(self, account, value, *, write=False, cwd=None):
        path = (Path(cwd or account.root) / Path(value).expanduser()).resolve()
        if path.is_relative_to(account.root) or (not write and (
                path.is_relative_to(self.datasets) or path.is_relative_to(analysis_root(account)))):
            return path
        raise WorkspaceDenied('Choose a path in your workspace or the shared datasets. Shared datasets are read-only.')

    def check_directory_entries(self, account, path, *, write=False):
        """Validate links that a flat input scan or an output stage may follow."""
        if path.is_dir():
            with os.scandir(path) as entries:
                for entry in entries:
                    if entry.is_symlink() or (os.name == 'nt' and Path(entry.path).is_junction()):
                        self.path(account, entry.path, write=write)


SERVER: SharedServer | None = None
SESSION_TOKEN = ContextVar('vmi_session', default=None)


def agent_enabled():
    return SERVER is None or not SERVER.manual_only


def request_account(scope):
    if SERVER:
        return SERVER.session(scope.get('vmi_session'))
    return None


def current_account():
    if SERVER is None:
        return None
    if token := SESSION_TOKEN.get():
        if account := SERVER.session(token):
            return account
        raise WorkspaceDenied('Sign in to access your workspace.')
    request = ui.context.client.request if ui.context.slot_stack else None
    account = request_account(request.scope) if request is not None else None
    if account is None:
        raise WorkspaceDenied('Sign in to access your workspace.')
    return account


def settings_store():
    account = current_account()
    return account.settings if account else app.storage.general


def workspace_root(default=WORKSPACE_ROOT):
    account = current_account()
    return account.root if account else default


def centroid_dir(default=WORKSPACE_ROOT / 'centroiding'):
    return workspace_root(default.parent) / 'centroiding' if SERVER else default


def circularization_dir(default=WORKSPACE_ROOT / 'circularization'):
    return workspace_root(default.parent) / 'circularization' if SERVER else default


def generated_dir(default=GENERATED_DIR):
    account = current_account()
    return account.root / '.cache' if account else default


def checked_path(value, *, write=False, cwd=None):
    account = current_account()
    if account:
        return SERVER.path(account, value, write=write, cwd=cwd)
    return (Path(cwd or Path.cwd()) / Path(value).expanduser()).resolve()


def require_admin():
    account = current_account()
    if account and not account.admin:
        raise WorkspaceDenied('This control requires a server administrator.')


def workspace_timer(interval, callback):
    if not SERVER:
        return ui.timer(interval, callback)
    client = ui.context.client

    def refresh():
        if not request_account(client.request.scope):
            timer.cancel()
            return None
        return callback()

    timer = ui.timer(interval, refresh, immediate=False)
    return timer


def check_command(cmd):
    """Check flags that expand paths or invoke another tool."""
    for index, arg in enumerate(map(str, cmd)):
        flag, sep, value = arg.partition('=')
        if flag in ('--wsl-executable', '--wsl-distribution', '--config'):
            raise WorkspaceDenied('This option is unavailable on a shared server.')
        if flag in ('--h5-pattern', '--pattern'):
            value = value if sep else str(cmd[index + 1])
            if '/' in value or '\\' in value or ':' in value or value in ('.', '..'):
                raise WorkspaceDenied('The HDF5 pattern must match file names in the selected directory.')


class AccountMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] not in ('http', 'websocket'):
            return await self.app(scope, receive, send)
        headers = Headers(scope=scope)
        host = headers.get('host', '')
        if (not re.fullmatch(r'(?:127\.0\.0\.1|localhost)(?::[0-9]+)?', host)
                or headers.get('origin') not in (None, f'http://{host}', f'https://{host}')):
            if scope['type'] == 'websocket':
                return await send({'type': 'websocket.close', 'code': 4403})
            return await HTMLResponse('Forbidden', status_code=403)(scope, receive, send)
        cookies = SimpleCookie()
        try:
            cookies.load(headers.get('cookie', ''))
        except CookieError:
            pass
        scope['vmi_session'] = cookies[COOKIE].value if COOKIE in cookies else None
        public_login = scope['type'] == 'http' and (
            scope['path'] == '/login'
            or (scope['method'] == 'GET' and scope['path'] in LOGIN_ASSETS)
        )
        if not public_login and not request_account(scope):
            if scope['type'] == 'websocket':
                return await send({'type': 'websocket.close', 'code': 4401})
            if scope['path'] == '/':
                return await RedirectResponse('/login')(scope, receive, send)
            return await HTMLResponse('Sign in required', status_code=401)(scope, receive, send)
        # A Socket.IO polling request or transport upgrade can name an
        # existing Engine.IO connection. Bind it to its original cookie.
        if scope['path'].startswith('/_nicegui_ws'):
            from nicegui.nicegui import sio

            sid = parse_qs(scope['query_string']).get(b'sid', [None])[0]
            if sid is not None:
                environ = sio.environ.get(sid.decode('ascii', errors='ignore'), {})
                if environ.get('asgi.scope', {}).get('vmi_session') != scope['vmi_session']:
                    if scope['type'] == 'websocket':
                        return await send({'type': 'websocket.close', 'code': 4403})
                    return await HTMLResponse('Forbidden', status_code=403)(scope, receive, send)
        binding = SESSION_TOKEN.set(scope['vmi_session'])
        try:
            await self.app(scope, receive, send)
        finally:
            SESSION_TOKEN.reset(binding)


def install(config_path):
    global SERVER
    if sys.platform != 'linux':
        raise RuntimeError('Shared-server mode requires a Linux host.')
    SERVER = SharedServer(json.loads(Path(config_path).read_text(encoding='utf-8')))
    app.add_middleware(AccountMiddleware)
    assets = files('vmi_analysis.gui.assets')
    login_page = (assets / 'login.html').read_text(encoding='utf-8')

    @app.get('/login-assets/{name}')
    async def login_asset(name: str):
        media_type = LOGIN_ASSETS.get('/login-assets/' + name)
        if media_type is None:
            raise HTTPException(404)
        return Response((assets / name).read_bytes(), media_type=media_type,
                        headers={'X-Content-Type-Options': 'nosniff'})

    @app.get('/login')
    async def login_form():
        return HTMLResponse(login_page.replace('{message}', ''), headers={'Cache-Control': 'no-store'})

    @app.post('/login')
    async def login(request: Request):
        if monotonic() < SERVER.retry_after:
            raise HTTPException(429, 'Wait briefly before trying again.')
        if not request.headers.get('content-length', '').isdigit() or int(request.headers['content-length']) > 8192:
            raise HTTPException(400, 'Invalid sign-in details.')
        async with request.form(max_fields=2, max_files=0) as form:
            name, password = form.get('username'), form.get('password')
            if not isinstance(name, str) or not isinstance(password, str) or len(name) > 32 or len(password) > 1024:
                raise HTTPException(400, 'Invalid sign-in details.')
            async with SERVER.login_slots:
                if monotonic() < SERVER.retry_after:
                    raise HTTPException(429, 'Wait briefly before trying again.')
                account = await asyncio.to_thread(SERVER.authenticate, name, password)
        if not account:
            SERVER.failed_logins += 1
            if SERVER.failed_logins >= 5:
                SERVER.retry_after = monotonic() + 2
                SERVER.failed_logins = 0
            message = '<p class="login-error" role="alert">Check your account name and password, then try again.</p>'
            return HTMLResponse(login_page.replace('{message}', message), status_code=401,
                                headers={'Cache-Control': 'no-store'})
        for token in list(SERVER.sessions):
            SERVER.session(token)
        SERVER.sessions.pop(request.scope.get('vmi_session'), None)
        token = secrets.token_urlsafe(32)
        SERVER.sessions[token] = (account, monotonic() + MAX_SESSION_AGE)
        response = RedirectResponse('/', status_code=303)
        response.set_cookie(COOKIE, token, httponly=True, samesite='strict', max_age=MAX_SESSION_AGE)
        return response

    @app.post('/logout')
    async def logout(request: Request):
        from nicegui.nicegui import sio

        token = request.scope['vmi_session']
        SERVER.sessions.pop(token, None)
        for sid, environ in list(sio.environ.items()):
            if environ.get('asgi.scope', {}).get('vmi_session') == token:
                socket_id = sio.manager.sid_from_eio_sid(sid, '/')
                if socket_id:
                    await sio.disconnect(socket_id)
        response = RedirectResponse('/login', status_code=303)
        response.delete_cookie(COOKIE)
        return response

    @app.get('/workspace-file/{area}/{relative:path}')
    async def workspace_file(request: Request, area: str, relative: str, download: bool = False):
        account = request_account(request.scope)
        if not account or area not in ('private', 'datasets', 'analysis') or (download and area == 'datasets'):
            raise HTTPException(404)
        root = {'private': account.root, 'datasets': SERVER.datasets, 'analysis': analysis_root(account)}[area]
        try:
            path = SERVER.path(account, root / relative)
            if not path.is_relative_to(root):
                raise HTTPException(404)
        except (WorkspaceDenied, OSError, RuntimeError):
            raise HTTPException(404) from None
        return WorkspaceFileResponse(root, path, filename=path.name if download else None, headers={
            'Cache-Control': 'private, no-store',
            'Content-Security-Policy': "sandbox; default-src 'none'",
            'X-Content-Type-Options': 'nosniff',
        })

    protect_socket_events()
    return SERVER


def protect_socket_events():
    # NiceGUI's event messages supply a client ID. Verify it against the
    # authenticated socket and the original page before dispatching any event.
    from nicegui.nicegui import sio
    from socketio.exceptions import ConnectionRefusedError

    def protected(handler, *, joined):
        async def dispatch(sid, message):
            from inspect import isawaitable

            if not isinstance(message, dict):
                return False
            client_id = message.get('client_id')
            if not isinstance(client_id, str):
                return False
            client = Client.instances.get(client_id)
            scope = (sio.get_environ(sid) or {}).get('asgi.scope', {})
            token = scope.get('vmi_session')
            if (not client or not client.request or not SERVER.session(token)
                    or client.request.scope.get('vmi_session') != token
                    or (joined and client.id not in sio.rooms(sid))):
                return False
            binding = SESSION_TOKEN.set(token)
            try:
                result = handler(sid, message)
                return await result if isawaitable(result) else result
            finally:
                SESSION_TOKEN.reset(binding)
        return dispatch

    for event in ('handshake', 'event', 'javascript_response', 'ack', 'log'):
        sio.on(event, protected(sio.handlers['/'][event], joined=event != 'handshake'))

    async def connect(sid, environ, auth=None):
        query = {key: values[0] for key, values in parse_qs(environ.get('QUERY_STRING', '')).items()}
        if query.get('implicit_handshake') == 'true':
            if not await sio.handlers['/']['handshake'](sid, query):
                raise ConnectionRefusedError('Handshake failed.')

    sio.on('connect', connect)

    @app.on_startup
    async def expire_connections():
        from nicegui import background_tasks

        async def sweep():
            while True:
                for sid, environ in list(sio.environ.items()):
                    if not SERVER.session(environ.get('asgi.scope', {}).get('vmi_session')):
                        socket_id = sio.manager.sid_from_eio_sid(sid, '/')
                        if socket_id:
                            await sio.disconnect(socket_id)
                for token in list(SERVER.sessions):
                    SERVER.session(token)
                await asyncio.sleep(15)

        background_tasks.create(sweep(), name='expire-shared-sessions')


def main():
    parser = argparse.ArgumentParser(description='Generate a password hash for the shared VMI server configuration.')
    parser.parse_args()
    password = getpass('Password: ')
    if len(password) < 12 or password != getpass('Repeat password: '):
        parser.error('Enter matching passwords of at least 12 characters.')
    print(password_hash(password))


if __name__ == '__main__':
    main()
