from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from sefia.interfaces.session_store import SessionStore
from sefia.state_store import StateStore


@dataclass
class MyState:
    value: str = "default"
    count: int = 0


class MockSessionStore(SessionStore):
    def __init__(self):
        self.data: dict[str, Any] = {}

    async def get(self, key: str, type_hint: type) -> Any | None:
        return self.data.get(key)

    async def set(self, key: str, value: Any, type_hint: type) -> None:
        self.data[key] = value

    async def delete(self, key: str) -> None:
        self.data.pop(key, None)


@pytest.fixture
def mock_store() -> MockSessionStore:
    return MockSessionStore()


class TestStateStore:
    async def test_ensure_loads_existing_state(self, mock_store: MockSessionStore):
        state = MyState(value="existing", count=1)
        mock_store.data["my_key"] = state
        store = StateStore(mock_store, "my_key", MyState)

        loaded_state = await store.ensure()

        assert loaded_state == state

    async def test_ensure_creates_default_state_if_not_exists(
        self, mock_store: MockSessionStore
    ):
        store = StateStore(mock_store, "my_key", MyState)
        state = await store.ensure()

        assert state == MyState()

    async def test_get_returns_state_or_none(self, mock_store: MockSessionStore):
        store = StateStore(mock_store, "my_key", MyState)
        assert await store.get() is None

        state = MyState(value="existing", count=1)
        mock_store.data["my_key"] = state
        store_with_data = StateStore(mock_store, "my_key", MyState)
        assert await store_with_data.get() == state

    async def test_save_writes_to_store_and_updates_cache(
        self, mock_store: MockSessionStore
    ):
        store = StateStore(mock_store, "my_key", MyState)
        new_state = MyState(value="new", count=10)

        await store.save(new_state)

        assert mock_store.data["my_key"] == new_state
        assert store._cache == new_state
        assert store._is_loaded is True

    async def test_delete_clears_cache(self, mock_store: MockSessionStore):
        state = MyState(value="to_delete", count=1)
        mock_store.data["my_key"] = state
        store = StateStore(mock_store, "my_key", MyState)
        await store.ensure()  # Load into cache

        await store.delete()

        assert store._cache is None
        assert store._is_loaded is True
        assert "my_key" not in mock_store.data

    async def test_cache_is_used_on_subsequent_calls(
        self, mock_store: MockSessionStore, mocker
    ):
        store = StateStore(mock_store, "my_key", MyState)
        spy = mocker.spy(mock_store, "get")

        await store.ensure()  # First call, should access store
        await store.ensure()  # Second call, should use cache
        await store.get()  # Third call, should also use cache

        spy.assert_called_once()

    async def test_ensure_after_get_with_none_result_caches_new_instance(
        self, mock_store: MockSessionStore
    ):
        store = StateStore(mock_store, "my_key", MyState)

        # First, call get(), which will load and cache `None`
        result1 = await store.get()
        assert result1 is None
        assert store._is_loaded is True
        assert store._cache is None

        # Now, call ensure(). It should create a default instance and cache it.
        result2 = await store.ensure()
        assert result2 == MyState()
        assert store._cache == MyState()

        # A subsequent call to get() should return the cached default instance
        result3 = await store.get()
        assert result3 == MyState()
