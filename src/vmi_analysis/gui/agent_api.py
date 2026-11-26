"""Start an Agent session for an authenticated shared account."""

import asyncio

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field

from vmi_analysis.agent_backend import AGENTS, AgentRequest, validate_request
from . import shared


class ModelConnection(BaseModel):
    base_url: str = Field(max_length=2048)
    api: str = Field(max_length=32)
    model: str = Field(max_length=256)
    additional_models: list[str] = Field(default_factory=list, max_length=64)
    image_models: list[str] = Field(default_factory=list, max_length=64)
    api_key: str = Field(max_length=16384, repr=False)


async def start_session(request: Request):
    server = shared.SERVER
    account = shared.request_account(request.scope)
    if account is None:
        raise HTTPException(401, "Sign in to VMI Analysis first.")
    if server.manual_only:
        raise HTTPException(403, "Agent is disabled in manual-only mode.")
    session = AGENTS.current(account.name)
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
            owner=account.name, workspace=account.root, config_root=account.root.parent / ".agent",
            datasets=server.datasets, analysis=shared.analysis_root(account),
            base_url=profile.base_url.strip(), api=profile.api, model=profile.model,
            additional_models=tuple(profile.additional_models), image_models=tuple(profile.image_models),
            api_key=profile.api_key,
        )
        validate_request(agent)
    except ValueError:
        raise HTTPException(400, "Check the model connection settings.") from None
    try:
        await asyncio.to_thread(AGENTS.start, agent, shared_mode=True, history=shared.agent_history(account))
    except (ValueError, RuntimeError, OSError):
        session = AGENTS.current(account.name)
        if session is not None and session.active:
            return {"state": "active"}
        raise HTTPException(503, "Agent could not start. Check the server diagnostics.") from None
    return {"state": "started"}
