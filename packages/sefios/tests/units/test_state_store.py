from __future__ import annotations

from dataclasses import dataclass

from pytest_mock import MockerFixture

from sefios import StateStore
from sefios.storage import MemorySessionStorage


@dataclass
class MyState:
    value: str = "default"
    count: int = 0


class TestStateStore:
    async def test_ensure_loads_existing_state(
        self, memory_session_storage: MemorySessionStorage
    ) -> None:
        state = MyState(value="existing", count=1)
        await memory_session_storage.set("my_key", state, MyState)
        store = StateStore(memory_session_storage, "my_key", MyState)

        loaded_state = await store.ensure()

        assert loaded_state == state

    async def test_ensure_creates_default_state_if_not_exists(
        self, memory_session_storage: MemorySessionStorage
    ) -> None:
        store = StateStore(memory_session_storage, "my_key", MyState)
        state = await store.ensure()

        assert state == MyState()

    async def test_get_returns_state_or_none(
        self, memory_session_storage: MemorySessionStorage
    ) -> None:
        store = StateStore(memory_session_storage, "my_key", MyState)
        assert await store.get() is None

        state = MyState(value="existing", count=1)
        await memory_session_storage.set("my_key", state, MyState)
        store_with_data = StateStore(memory_session_storage, "my_key", MyState)
        assert await store_with_data.get() == state

    async def test_save_writes_to_store_and_updates_cache(
        self, memory_session_storage: MemorySessionStorage, mocker: MockerFixture
    ) -> None:
        store = StateStore(memory_session_storage, "my_key", MyState)
        new_state = MyState(value="new", count=10)

        await store.save(new_state)

        assert await memory_session_storage.get("my_key", MyState) == new_state
        spy = mocker.spy(memory_session_storage, "get")
        assert await store.get() == new_state
        spy.assert_not_called()

    async def test_delete_clears_cache(
        self, memory_session_storage: MemorySessionStorage, mocker: MockerFixture
    ) -> None:
        state = MyState(value="to_delete", count=1)
        await memory_session_storage.set("my_key", state, MyState)
        store = StateStore(memory_session_storage, "my_key", MyState)
        await store.ensure()

        await store.delete()

        assert await memory_session_storage.get("my_key", MyState) is None
        spy = mocker.spy(memory_session_storage, "get")
        assert await store.get() is None
        spy.assert_not_called()

    async def test_cache_is_used_on_subsequent_calls(
        self, memory_session_storage: MemorySessionStorage, mocker: MockerFixture
    ) -> None:
        store = StateStore(memory_session_storage, "my_key", MyState)
        spy = mocker.spy(memory_session_storage, "get")

        await store.ensure()
        await store.ensure()
        await store.get()

        spy.assert_called_once()

    async def test_ensure_after_get_with_none_result_caches_new_instance(
        self, memory_session_storage: MemorySessionStorage, mocker: MockerFixture
    ) -> None:
        store = StateStore(memory_session_storage, "my_key", MyState)
        spy = mocker.spy(memory_session_storage, "get")

        result1 = await store.get()
        assert result1 is None

        result2 = await store.ensure()
        assert result2 == MyState()

        result3 = await store.get()
        assert result3 == MyState()
        spy.assert_called_once()
