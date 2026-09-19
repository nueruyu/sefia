"""Durable, opaque JSON request/result exchanges, independent of execution."""

import hashlib
import json
from dataclasses import dataclass
from typing import TypeVar, final

from pydantic import ConfigDict, JsonValue, TypeAdapter
from typing_extensions import TypeForm

from ._interaction_context import get_interaction_channel
from .exceptions import (
    InteractionConflictError,
    InteractionRequired,
    UnknownInteractionError,
)
from .storage import SessionStorage

__all__ = [
    "JsonValue",
    "InteractionRequest",
    "InteractionResult",
    "InteractionChannel",
    "require_interaction",
]

_JSON: TypeAdapter[JsonValue] = TypeAdapter(
    JsonValue, config=ConfigDict(allow_inf_nan=False)
)
T = TypeVar("T")


@dataclass(frozen=True)
class InteractionRequest:
    interaction_id: str
    payload: JsonValue


@dataclass(frozen=True)
class InteractionResult:
    value: JsonValue


def _json(value: JsonValue) -> str:
    validated = _JSON.validate_python(value, strict=True)
    return json.dumps(validated, sort_keys=True, ensure_ascii=True, allow_nan=False)


def _key(interaction_id: str) -> str:
    return "interactions/" + hashlib.sha256(interaction_id.encode()).hexdigest()


@final
class InteractionChannel:
    """Persist immutable request/result facts in one session's storage.

    Redelivery is idempotent; contradictory facts raise InteractionConflictError.
    Result domain validation belongs to the requester. Even an invalid domain
    result is final once stored. Pending discovery is ordered by interaction ID
    and is not an atomic snapshot of concurrent requests and resolutions.
    """

    def __init__(self, storage: SessionStorage):
        self._storage = storage

    async def _record(self, key: str, value: str, interaction_id: str) -> None:
        if not await self._storage.set_if_absent(key, value, str):
            if await self._storage.get(key, str) != value:
                raise InteractionConflictError(interaction_id)

    async def request(
        self, interaction_id: str, request: JsonValue
    ) -> InteractionResult | None:
        key = _key(interaction_id)
        record: JsonValue = {"interaction_id": interaction_id, "payload": request}
        await self._record(key + "/request", _json(record), interaction_id)
        return await self._result(key)

    async def _result(self, key: str) -> InteractionResult | None:
        stored = await self._storage.get(key + "/result", str)
        return (
            None if stored is None else InteractionResult(_JSON.validate_json(stored))
        )

    async def resolve(self, interaction_id: str, result: JsonValue) -> None:
        key = _key(interaction_id)
        if await self._storage.get(key + "/request", str) is None:
            raise UnknownInteractionError(interaction_id)
        await self._record(key + "/result", _json(result), interaction_id)

    async def pending(self) -> list[InteractionRequest]:
        pending: list[InteractionRequest] = []
        for key in await self._storage.keys("interactions/"):
            if not key.endswith("/request"):
                continue
            if await self._result(key.removesuffix("/request")) is not None:
                continue
            stored = await self._storage.get(key, str)
            if stored is None:
                continue
            record = json.loads(stored)
            pending.append(
                InteractionRequest(record["interaction_id"], record["payload"])
            )
        return sorted(pending, key=lambda request: request.interaction_id)


async def require_interaction(
    interaction_id: str, request: JsonValue, result_type: TypeForm[T]
) -> T:
    """Register a request, pause if unresolved, or validate its immutable result.

    Pydantic ValidationError on replay means the externally stored result violates
    the requester's contract; it cannot be replaced through resolution.
    """
    resolved = await get_interaction_channel().request(interaction_id, request)
    if resolved is None:
        raise InteractionRequired(interaction_id, request)
    adapter: TypeAdapter[T] = TypeAdapter(result_type)
    return adapter.validate_python(resolved.value)
