"""Shared external-input values and request lifecycle."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass

from glyff import ExecutionId

from ._async import MaybeAwaitable, maybe_await
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


def _execution_id_to_data(execution_id: ExecutionId) -> dict[str, object]:
    parent_id = execution_id.parent_id
    return {
        "domain_id": execution_id.domain_id.value,
        "name": execution_id.name.value,
        "sequence": execution_id.sequence,
        "arguments_digest": execution_id.arguments_digest.value,
        "parent_id": _execution_id_to_data(parent_id) if parent_id else None,
    }


def interaction_id_for_execution(execution_id: ExecutionId) -> str:
    """Derive an opaque external-input identity from an engraved execution."""
    data = _execution_id_to_data(execution_id)
    stable_repr = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable_repr.encode("utf-8")).hexdigest()


async def request_input(
    request: InputRequest,
    provider: InputProvider,
    on_request: InputRequestCallback | None = None,
    on_complete: InputCompleteCallback | None = None,
) -> str:
    value = await maybe_await(provider(request))
    if value is not None:
        if on_complete is not None:
            await maybe_await(
                on_complete(InputResult(request.interaction_id, request.prompt, value))
            )
        return value
    if on_request is not None:
        await maybe_await(on_request(request))
    raise InputRequired(request.prompt, interaction_id=request.interaction_id)
