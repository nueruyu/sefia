from typing import Any

from glyff.interfaces import Serializer
from glyff.stores import MemoryClient

from ..interfaces.session_store import SessionStore


class MemorySessionStore(SessionStore):
    """An in-memory metadata store backed by glyff's MemoryClient."""

    def __init__(self, client: MemoryClient, serializer: Serializer):
        self._client = client
        self._serializer = serializer

    def _prefix(self, key: str) -> str:
        return f"sefia::metadata::{key}"

    async def get(self, key: str, type_hint: type) -> Any | None:
        raw_value = await self._client.read(self._prefix(key))
        if raw_value is not None:
            return await self._serializer.deserialize(raw_value, type_hint)
        return None

    async def set(self, key: str, value: Any, type_hint: type) -> None:
        self._client.stage_write(
            self._prefix(key), await self._serializer.serialize(value, type_hint)
        )

    async def delete(self, key: str) -> None:
        self._client.stage_delete(self._prefix(key))
