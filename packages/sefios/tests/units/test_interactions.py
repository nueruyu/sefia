import asyncio
from collections.abc import Callable
from pathlib import Path

import pytest
from glyff_pydantic import PydanticSerializer
from pydantic import BaseModel, ValidationError
from sefia.testing import MockLLMClient
from sefios import SessionScope, require_interaction
from sefios.exceptions import (
    InteractionConflictError,
    InteractionRequired,
    UnknownInteractionError,
)
from sefios.interactions import (
    InteractionChannel,
    InteractionRequest,
    InteractionResult,
)
from sefios.storage import (
    FileSessionStorage,
    MemorySessionStorage,
    SQLiteSessionStorage,
)


@pytest.fixture(params=["memory", "file", "sqlite"])
def channels(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Callable[[], InteractionChannel]:
    serializer = PydanticSerializer()
    memory = MemorySessionStorage(serializer)

    def channel() -> InteractionChannel:
        if request.param == "file":
            return InteractionChannel(FileSessionStorage(tmp_path, serializer))
        if request.param == "sqlite":
            return InteractionChannel(
                SQLiteSessionStorage(tmp_path / "state.db", "session", serializer)
            )
        return InteractionChannel(memory)

    return channel


async def test_request_resolution_and_null(
    channels: Callable[[], InteractionChannel],
) -> None:
    channel = channels()
    assert await channel.request("../arbitrary/日本語", {"x": [1, None]}) is None
    assert await channels().pending() == [
        InteractionRequest("../arbitrary/日本語", {"x": [1, None]})
    ]
    assert await channels().request("../arbitrary/日本語", {"x": [1, None]}) is None
    await channels().resolve("../arbitrary/日本語", None)
    await channels().resolve("../arbitrary/日本語", None)
    assert await channels().request(
        "../arbitrary/日本語", {"x": [1, None]}
    ) == InteractionResult(None)
    assert await channels().pending() == []
    with pytest.raises(InteractionConflictError):
        await channels().request("../arbitrary/日本語", "changed")
    with pytest.raises(InteractionConflictError):
        await channels().resolve("../arbitrary/日本語", "changed")


async def test_unknown_and_pending_order(
    channels: Callable[[], InteractionChannel],
) -> None:
    with pytest.raises(UnknownInteractionError):
        await channels().resolve("unknown", 1)
    assert await channels().pending() == []
    await channels().request("z", None)
    await channels().request("a", False)
    assert [r.interaction_id for r in await channels().pending()] == ["a", "z"]


async def test_concurrent_resolution(
    channels: Callable[[], InteractionChannel],
) -> None:
    await channels().request("race", {})
    results = await asyncio.gather(
        *(channels().resolve("race", i) for i in range(10)), return_exceptions=True
    )
    assert sum(r is None for r in results) == 1
    assert sum(isinstance(r, InteractionConflictError) for r in results) == 9
    assert await channels().request("race", {}) == InteractionResult(
        results.index(None)
    )
    await asyncio.gather(
        *(channels().resolve("race", results.index(None)) for _ in range(10))
    )


class Weather(BaseModel):
    temperature: int


async def test_adapter_in_plain_scope_and_invalid_result():
    from sefios._interaction_context import get_interaction_channel

    scope = SessionScope(llm_client=MockLLMClient([]))
    async with scope.session(session_id="plain"):
        with pytest.raises(InteractionRequired) as pause:
            await require_interaction("weather", {"city": "Tokyo"}, Weather)
        assert pause.value.request == {"city": "Tokyo"}
        await get_interaction_channel().resolve("weather", {"temperature": 21})
        assert await require_interaction(
            "weather", {"city": "Tokyo"}, Weather
        ) == Weather(temperature=21)
        with pytest.raises(InteractionRequired):
            await require_interaction("bad", None, Weather)
        await get_interaction_channel().resolve("bad", "invalid")
        with pytest.raises(ValidationError):
            await require_interaction("bad", None, Weather)
        with pytest.raises(InteractionConflictError):
            await get_interaction_channel().resolve("bad", {"temperature": 22})


async def test_concurrent_requests(channels: Callable[[], InteractionChannel]) -> None:
    results = await asyncio.gather(
        *(channels().request("race", {"producer": i}) for i in range(10)),
        return_exceptions=True,
    )
    assert sum(r is None for r in results) == 1
    assert sum(isinstance(r, InteractionConflictError) for r in results) == 9
    assert await channels().pending() == [
        InteractionRequest("race", {"producer": results.index(None)})
    ]


async def test_json_validation_and_object_key_order(
    channels: Callable[[], InteractionChannel],
) -> None:
    from typing import cast

    from sefios.interactions import JsonValue

    await channels().request("json", {"a": 1, "b": True})
    await channels().request("json", {"b": True, "a": 1})
    for invalid in [
        float("nan"),
        float("inf"),
        Weather(temperature=21),
        (1, 2),
        {1: "value"},
    ]:
        with pytest.raises((ValidationError, ValueError)):
            await channels().resolve("json", cast(JsonValue, invalid))
    assert len(await channels().pending()) == 1
    await channels().resolve("json", {"b": True, "a": 1})
    await channels().resolve("json", {"a": 1, "b": True})
    with pytest.raises(InteractionConflictError):
        await channels().resolve("json", {"a": True, "b": 1})


async def test_scope_binding_isolation_and_null_result() -> None:
    from sefios._interaction_context import get_interaction_channel

    scope = SessionScope(llm_client=MockLLMClient([]))
    with pytest.raises(RuntimeError, match="active sefios session"):
        await require_interaction("null", None, type(None))
    async with scope.session(session_id="outer"):
        outer = get_interaction_channel()
        with pytest.raises(InteractionRequired):
            await require_interaction("null", None, type(None))
        async with scope.session(session_id="inner"):
            assert await get_interaction_channel().pending() == []
            with pytest.raises(UnknownInteractionError):
                await get_interaction_channel().resolve("null", None)
        assert get_interaction_channel() is outer
        await outer.resolve("null", None)
        assert await require_interaction("null", None, type(None)) is None
    with pytest.raises(RuntimeError, match="active sefios session"):
        get_interaction_channel()
