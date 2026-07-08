from typing import Any

import pytest


class InMemoryKeyValueStore:
    """A dict-backed KeyValueStore for exercising the human-input core."""

    def __init__(self):
        self._data: dict[str, Any] = {}

    async def get(self, key: str, type_hint: type) -> Any | None:
        return self._data.get(key)

    async def set(self, key: str, value: Any, type_hint: type) -> None:
        self._data[key] = value

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)


@pytest.fixture
def kv_store() -> InMemoryKeyValueStore:
    return InMemoryKeyValueStore()
