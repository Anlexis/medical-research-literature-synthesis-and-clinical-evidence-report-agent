"""AgentCore Platform v1.0

Standalone HTTP entry point for the agent. Entry points are adapters only — no
business logic here. For platform-level routing the gateway calls
agent.invoke() directly.
"""

import json
import os
import re
import secrets
from typing import Any, Dict
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from framework.security.credential_detector import detect_credentials_in_value
from shared.secrets import factory as secrets_factory
from src.graph.graph import MedicalResearchLiteratureSynthesisAgent, load_runtime_config

app = FastAPI(title="Agent")

# Constructed with the runtime config so a value declared in config/config.yaml
# takes effect here exactly as it does on the platform.
agent = MedicalResearchLiteratureSynthesisAgent(config=load_runtime_config())
agent.compile()
# namespace / agent_name match the config/agent.yaml manifest values.
agent.provision_secrets(secrets_factory(namespace="hcr-c2-045", agent_name="MedicalResearchLiteratureSynthesisAgent"))

# Ceiling on the structured caller channel. The platform adapter caps request
# bodies at roughly this size; the entry point enforces its own bound so a
# direct caller cannot hand the pipeline an unbounded structure.
MAX_INPUT_CONTEXT_BYTES = 256 * 1024

# Keys the pipeline reads on the structured channel. Anything else is DROPPED
# rather than ignored: an undeclared key is not stripped by the validators, and
# every value of the initialisation result is scanned for credential patterns —
# so an unread key that happens to hold a credential-shaped string fails the
# request at the very first node, before any of this template's code runs.
_ACCEPTED_CONTEXT_KEYS = frozenset({"literature_records"})

# Field names are caller data too. A name is echoed back only when it is
# unremarkable; otherwise the position is reported instead.
_SAFE_FIELD_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class InvokeRequest(BaseModel):
    """Request body for POST /invoke."""

    input: str
    session_id: str = ""
    # Structured caller data. The fields the pipeline consumes, and the bounds
    # each is held to, are declared by the pre-processing node; this adapter
    # enforces the size ceiling, drops unread keys, and refuses credentials.
    input_context: Dict[str, Any] = Field(default_factory=dict)


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request) -> Dict[str, Any]:
    """Validate the request envelope and run the agent."""
    trust = getattr(request.state, "trust_level", TrustLevel.ANONYMOUS)
    # Standalone caller auth: when INVOKE_AUTH_TOKEN is set on the server
    # environment, callers that no upstream middleware vouched for (still
    # anonymous) must present it as a bearer token and run as a verified
    # external caller. Middleware-established trust is never demoted. This is a
    # deployment-level caller credential, not an agent secret, so the secrets
    # provider does not apply — no invocation context exists before auth.
    expected = os.environ.get("INVOKE_AUTH_TOKEN")
    if expected and trust is TrustLevel.ANONYMOUS:
        supplied = request.headers.get("authorization", "")
        # Compare bytes: compare_digest raises TypeError on non-ASCII str input
        # (headers decode as latin-1), which would 500 instead of a clean 401.
        if not secrets.compare_digest(supplied.encode(), f"Bearer {expected}".encode()):
            # Generic body on purpose — do not leak whether the token was
            # absent, malformed, or wrong.
            raise HTTPException(status_code=401, detail="Token is invalid or expired.")
        trust = TrustLevel.VERIFIED_EXTERNAL

    input_context = _accepted_context(req.input_context or {})

    if _encoded_size(input_context) > MAX_INPUT_CONTEXT_BYTES:
        # Name the limit, never the payload.
        raise HTTPException(
            status_code=413,
            detail=f"input_context exceeds the {MAX_INPUT_CONTEXT_BYTES} byte limit.",
        )

    offending = _credential_field(input_context)
    if offending is not None:
        # The request cannot succeed either way: a credential-shaped value
        # anywhere in this channel is refused at the first node of the graph,
        # with an opaque error the caller cannot act on. Refusing here converts
        # that into an actionable one. 400, not 422 — the validation layer owns
        # 422 and answers there with a list of error objects, so reusing it
        # would make client handling ambiguous.
        raise HTTPException(
            status_code=400,
            detail=f"{offending} looks like a credential and was refused. Remove it and retry.",
        )

    with bound_secrets(agent._secrets_provider):
        ctx = InvocationContext(
            session_id=req.session_id or str(uuid4()),
            caller_trust_level=trust,
            caller_id=getattr(request.state, "caller_id", ""),
        )
        result: Dict[str, Any] = agent.invoke(req.input, ctx=ctx, input_context=input_context)
        return result


def _accepted_context(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the structured-channel keys the pipeline reads."""
    return {key: value for key, value in payload.items() if key in _ACCEPTED_CONTEXT_KEYS}


def _credential_field(payload: Dict[str, Any]) -> "str | None":
    """Return a caller-safe name for the first field carrying a credential.

    Fields are iterated one at a time rather than scanned as a whole mapping so
    that the refusal can name the field. That is exactly equivalent to scanning
    the mapping: the detector's result for a mapping is defined as the union
    over its values, and using the framework's own function keeps the refused
    set identical to the set the framework blocks — no local approximation that
    could drift apart from it.
    """
    for position, (name, value) in enumerate(payload.items()):
        if detect_credentials_in_value(value):
            if _SAFE_FIELD_NAME.match(name) and not detect_credentials_in_value(name):
                return f"input_context.{name}"
            return f"input_context field #{position + 1}"
    return None


def _encoded_size(payload: Dict[str, Any]) -> int:
    """Serialized size of the caller channel, used for the entry-point cap.

    A structure that cannot be serialized at all is treated as over-limit: the
    pipeline could not have carried it anyway, and refusing is the safe
    direction.
    """
    try:
        return len(json.dumps(payload, default=str).encode("utf-8"))
    except (TypeError, ValueError):
        return MAX_INPUT_CONTEXT_BYTES + 1


@app.get("/health")
def health() -> Dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "agent": "MedicalResearchLiteratureSynthesisAgent"}
