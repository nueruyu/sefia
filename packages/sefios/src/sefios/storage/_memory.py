import asyncio
from typing import Any

from glyff import Serializer
from typing_extensions import final, override

from ._base import SessionStorage


@final
class MemorySessionStorage(SessionStorage):
    """An in-memory storage for session-scoped state.

    Values are serialized and held in a plain dict, and every write takes effect
    immediately — so state written before a pause survives the interrupt and is
    visible when the run resumes.
    """

    def __init__(self, serializer: Serializer):
        self._serializer = serializer
        self._data: dict[str, bytes] = {}
        self._lock = asyncio.Lock()

    @override
    async def get(self, key: str, type_hint: type) -> Any | None:
        raw_value = self._data.get(key)
        if raw_value is not None:
            return await self._serializer.deserialize(raw_value, type_hint)
        return None

    @override
    async def set(self, key: str, value: Any, type_hint: type) -> None:
        serialized = await self._serializer.serialize(value, type_hint)
        async with self._lock:
            self._data[key] = serialized

    @override
    async def set_if_absent(self, key: str, value: Any, type_hint: type) -> bool:
        serialized = await self._serializer.serialize(value, type_hint)
        async with self._lock:
            if key in self._data:
                return False
            self._data[key] = serialized
            return True

    @override
    async def keys(self, prefix: str = "") -> tuple[str, ...]:
        async with self._lock:
            return tuple(sorted(key for key in self._data if key.startswith(prefix)))

    @override
    async def delete(self, key: str) -> None:
        async with self._lock:
            self._data.pop(key, None)
