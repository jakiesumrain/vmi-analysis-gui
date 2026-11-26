"""Start an Agent for a shared account or an opted-in local desktop."""

import asyncio

from fastapi import HTTPException, Request, Response
from pydantic import BaseModel, Field

from vmi_analysis.agent_backend import AGENTS, AgentRequest, validate_request
from vmi_analysis.paths import GENERATED_DIR
from . import shared


class ModelConnection(BaseModel):
    base_url: str = Field(max_length=2048)
    api: str = Field(max_length=32)
    model: str = Field(max_length=256)
    additional_models: list[str] = Field(default_factory=list, max_length=64)
    image_models: list[str] = Field(default_factory=list, max_length=64)
    api_key: str = Field(max_length=16384, repr=False)


async def start_session(request: Request):
    account = shared.request_account(request.scope)
    if account is None:
        raise HTTPException(401, "Sign in to VMI Analysis first.")
    if shared.SERVER.manual_only:
        raise HTTPException(403, "Agent is disabled in manual-only mode.")
    return await _start(request, account)


def _require_local_desktop(request: Request):
    host, port = request.scope.get("server") or (None, None)
    if (shared.SERVER is not None or host != "127.0.0.1" or port is None
            or request.client is None or request.client.host != "127.0.0.1"
            or request.headers.get("host") != f"127.0.0.1:{port}"
            or "origin" in request.headers or "sec-fetch-site" in request.headers):
        raise HTTPException(403, "The desktop endpoint requires a direct loopback connection.")


async def local_desktop(request: Request):
    _require_local_desktop(request)
    return Response(status_code=204, headers={"X-VMI-Local-Client": "1", "Cache-Control": "no-store"})


async def start_local_session(request: Request):
    _require_local_desktop(request)
    return await _start(request, None)


async def _start(request: Request, account):
    owner = account.name if account else "local"
    session = AGENTS.current(owner)
    if session is not None and session.active:
        return {"state": "active"}

    if request.headers.get("content-type", "").split(";", 1)[0] != "application/json":
        raise HTTPException(415, "Use a JSON request.")
    length = request.headers.get("content-length", "")
    if not length.isascii() or not length.isdecimal() or len(length) > 5 or int(length) > 49152:
        raise HTTPException(413, "The model connection request is too large.")
    try:
        profile = ModelConnection.model_validate(await request.json())
        agent = AgentRequest(
            owner=owner,
            workspace=account.root if account else shared.workspace_root(),
            config_root=account.root.parent / ".agent" if account else GENERATED_DIR / "agent",
            datasets=shared.SERVER.datasets if account else None,
            analysis=shared.analysis_root(account) if account else None,
            base_url=profile.base_url.strip(), api=profile.api, model=profile.model,
            additional_models=tuple(profile.additional_models), image_models=tuple(profile.image_models),
            api_key=profile.api_key,
        )
        validate_request(agent)
    except ValueError:
        raise HTTPException(400, "Check the model connection settings.") from None
    try:
        await asyncio.to_thread(AGENTS.start, agent, shared_mode=account is not None,
                                history=shared.agent_history(account))
    except (ValueError, RuntimeError, OSError):
        session = AGENTS.current(owner)
        if session is not None and session.active:
            return {"state": "active"}
        raise HTTPException(503, "Agent could not start. Check the server diagnostics.") from None
    return {"state": "started"}
