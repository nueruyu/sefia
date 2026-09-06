"""Shared external-input values, identities, and request lifecycle."""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import glyff

from ._async import MaybeAwaitable, maybe_await
from ._session_state import execution_id_scope_key
from .exceptions import InputRequired


@dataclass(frozen=True)
class InputRequest:
    """A request for external input."""

    interaction_id: str
    prompt: str


@dataclass(frozen=True)
class InputResult:
    """A completed external input interaction."""

    interaction_id: str
    prompt: str
    value: str


InputProvider = Callable[[InputRequest], MaybeAwaitable[str | None]]
InputRequestCallback = Callable[[InputRequest], MaybeAwaitable[None]]
InputCompleteCallback = Callable[[InputResult], MaybeAwaitable[None]]
InputPromptDeltaCallback = Callable[[str, str], MaybeAwaitable[None]]


async def no_input(_: InputRequest) -> str | None:
    return None


def preview_id_for(tool_call_id: str) -> str:
    return hashlib.sha256(tool_call_id.encode("utf-8")).hexdigest()


def current_interaction_id() -> str:
    execution_id = glyff.get_context().current_execution_id
    if execution_id is None:
        raise RuntimeError("Input requires an engraved execution.")
    return execution_id_scope_key(execution_id)


async def request_input(
    prompt: str,
    provider: InputProvider,
    on_request: InputRequestCallback | None = None,
    on_complete: InputCompleteCallback | None = None,
) -> str:
    request = InputRequest(current_interaction_id(), prompt)
    value = await maybe_await(provider(request))
    if value is not None:
        if on_complete is not None:
            await maybe_await(
                on_complete(InputResult(request.interaction_id, prompt, value))
            )
        return value
    if on_request is not None:
        await maybe_await(on_request(request))
    raise InputRequired(prompt, interaction_id=request.interaction_id)
