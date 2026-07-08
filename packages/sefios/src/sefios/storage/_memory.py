from typing import Any

from glyff import Serializer

from ._base import SessionStorage


class MemorySessionStorage(SessionStorage):
    """An in-memory storage for session-scoped state.

    Values are serialized and held in a plain dict, and every write takes effect
    immediately — so state written before a pause survives the interrupt and is
    visible when the run resumes.
    """

    def __init__(self, serializer: Serializer):
        self._serializer = serializer
        self._data: dict[str, bytes] = {}

    async def get(self, key: str, type_hint: type) -> Any | None:
        raw_value = self._data.get(key)
        if raw_value is not None:
            return await self._serializer.deserialize(raw_value, type_hint)
        return None

    async def set(self, key: str, value: Any, type_hint: type) -> None:
        self._data[key] = await self._serializer.serialize(value, type_hint)

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)
