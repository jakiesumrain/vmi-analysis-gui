"""Pair a desktop client to one account and exchange one-use browser tickets."""

from __future__ import annotations

import asyncio
from hashlib import sha256
from hmac import compare_digest
from importlib.resources import files
import re
import secrets
from time import monotonic

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from vmi_analysis.agent_backend import AGENTS, AgentRequest, validate_request
from . import shared


PREFIX = "/client/v1"
PUBLIC_ROUTES = {("POST", PREFIX + suffix) for suffix in ("/pair", "/launch", "/consume")} | {
    ("GET", PREFIX + "/open"), ("GET", PREFIX + "/open.js"),
}
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{43}\Z")
PRIVATE = {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"}
ASSETS = files("vmi_analysis.gui.assets")


def digest(token):
    return sha256(token.encode("ascii")).hexdigest()


class DesktopLinks:
    def __init__(self, server):
        self.server = server
        self.codes = {}
        self.tickets = {}

    def issue_code(self, account, session):
        code = secrets.token_urlsafe(32)
        self.codes[account.name] = (digest(code), session, monotonic() + 300)
        return code

    def _store(self, account):
        return shared.Settings(account.root.parent / "desktop-client.json")

    def _device_digest(self, account, token):
        return digest(account.password + ":" + token)

    def pair(self, account, code):
        expected, session, expires = self.codes.get(account.name, ("", None, 0))
        if expires <= monotonic() or self.server.session(session) is not account or not compare_digest(
                digest(code), expected):
            raise HTTPException(403, "Invalid or expired link code. Generate a new code in VMI Settings.")
        device = secrets.token_urlsafe(32)
        self._store(account).save({"credential": self._device_digest(account, device)})
        self.codes.pop(account.name, None)
        return device

    def revoke(self, account):
        self._store(account).save({})
        self.codes.pop(account.name, None)
        self.tickets.pop(account.name, None)

    def authorize(self, name, authorization):
        account = self.server.accounts.get(name)
        token = authorization.removeprefix("Bearer ") if authorization.startswith("Bearer ") else ""
        if account is None or not TOKEN_PATTERN.fullmatch(token):
            raise HTTPException(401, "Link this VMI account again.")
        credential = self._device_digest(account, token)
        if not self.is_current(account, credential):
            raise HTTPException(401, "Link this VMI account again.")
        return account, credential

    def is_current(self, account, credential):
        return compare_digest(self._store(account).get("credential", ""), credential)

    def issue_ticket(self, account, credential):
        if not self.is_current(account, credential):
            raise HTTPException(401, "Link this VMI account again.")
        ticket = secrets.token_urlsafe(32)
        self.tickets[account.name] = (digest(ticket), credential, monotonic() + 60)
        return ticket

    def consume_ticket(self, name, ticket):
        account = self.server.accounts.get(name)
        expected, credential, expires = self.tickets.get(name, ("", "", 0))
        if account is None or expires <= monotonic() or not compare_digest(digest(ticket), expected) or not (
                self.is_current(account, credential)):
            raise HTTPException(403, "This launch has expired. Open VMI again from the client.")
        self.tickets.pop(name, None)
        return account


class Claim(BaseModel):
    account: str = Field(max_length=32)
    code: str = Field(pattern=r"^[A-Za-z0-9_-]{43}$")


class AgentProfile(BaseModel):
    base_url: str = Field(max_length=2048)
    api: str = Field(max_length=32)
    model: str = Field(max_length=256)
    additional_models: list[str] = Field(default_factory=list, max_length=64)
    image_models: list[str] = Field(default_factory=list, max_length=64)
    api_key: str = Field(max_length=16384, repr=False)


class Launch(BaseModel):
    account: str = Field(max_length=32)
    agent: AgentProfile | None = None


async def read_payload(request, model):
    if request.headers.get("content-type", "").split(";", 1)[0] != "application/json":
        raise HTTPException(415, "Use a JSON request.")
    length = request.headers.get("content-length", "")
    if not length.isascii() or not length.isdecimal() or len(length) > 5 or int(length) > 49152:
        raise HTTPException(413, "The request is too large.")
    try:
        return model.model_validate(await request.json())
    except ValueError:
        raise HTTPException(400, "Invalid desktop client request.") from None


def install(app, links):
    @app.post(PREFIX + "/pair")
    async def pair(request: Request):
        claim = await read_payload(request, Claim)
        account = links.server.accounts.get(claim.account)
        if account is None:
            raise HTTPException(403, "Invalid or expired link code. Generate a new code in VMI Settings.")
        return JSONResponse({"credential": links.pair(account, claim.code)}, headers=PRIVATE)

    @app.post(PREFIX + "/launch")
    async def launch(request: Request):
        data = await read_payload(request, Launch)
        account, credential = links.authorize(data.account, request.headers.get("authorization", ""))
        warning = None
        session = AGENTS.current(account.name)
        if data.agent is not None and (session is None or not session.active):
            if links.server.manual_only:
                warning = "The server is in manual-only mode. Agent is unavailable."
            else:
                profile = data.agent
                agent = AgentRequest(
                    owner=account.name, workspace=account.root, config_root=account.root.parent / ".agent",
                    datasets=links.server.datasets, analysis=shared.analysis_root(account),
                    base_url=profile.base_url.strip(), api=profile.api, model=profile.model,
                    additional_models=tuple(profile.additional_models), image_models=tuple(profile.image_models),
                    api_key=profile.api_key,
                )
                try:
                    validate_request(agent)
                    await asyncio.to_thread(AGENTS.start, agent, shared_mode=True)
                except (ValueError, RuntimeError, OSError):
                    current = AGENTS.current(account.name)
                    if current is None or not current.active:
                        warning = "Agent could not start. Check the saved model profile and server diagnostics."
        ticket = links.issue_ticket(account, credential)
        return JSONResponse({"ticket": ticket, "warning": warning}, headers=PRIVATE)

    @app.get(PREFIX + "/open")
    async def open_browser(request: Request):
        if request.headers.get("sec-fetch-site") not in (None, "none", "same-origin"):
            raise HTTPException(403, "Open VMI from the desktop client.")
        return HTMLResponse((ASSETS / "desktop-client.html").read_text(encoding="utf-8"), headers={
            **PRIVATE, "Content-Security-Policy": "default-src 'none'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'",
        })

    @app.get(PREFIX + "/open.js")
    async def browser_script():
        return Response((ASSETS / "desktop-client.js").read_text(encoding="utf-8"),
                        media_type="text/javascript", headers=PRIVATE)

    @app.post(PREFIX + "/consume")
    async def consume(request: Request):
        claim = await read_payload(request, Claim)
        current = shared.request_account(request.scope)
        if current is not None and current.name != claim.account:
            raise HTTPException(409, "Sign out of the other VMI account before opening this one.")
        account = links.consume_ticket(claim.account, claim.code)
        response = Response(status_code=204, headers=PRIVATE)
        links.server.sign_in(account, request, response)
        return response
