from collections.abc import Callable
from pathlib import Path

import pytest
from sefia import JsonSchemaToolEntry
from sefia.llm import LLMResponse
from sefia.testing import MockLLMClient, result_response, tool_calls_response
from sefia.tool_collectors import StaticToolCollector
from sefios import (
    MemoryPersistenceProvider,
    MemorySessionStorage,
    SessionScope,
    SQLiteSessionStorage,
    domain,
    get_session_storage,
)

infer = domain("packages.sefios.tests.units.test_scope").infer


class _Probe:
    """Receiver used to exercise the configured tool collector."""

    @infer
    async def answer(self) -> str:
        """Answer using the available tools."""
        ...


def _static_collector(name: str, calls: list[str]) -> StaticToolCollector:
    async def handler() -> str:
        calls.append(name)
        return f"{name}-result"

    return StaticToolCollector(
        [
            JsonSchemaToolEntry(
                handler,
                name=name,
                parameters={"type": "object", "properties": {}},
                description=f"the {name} tool",
            )
        ]
    )


async def test_tool_collector_default_is_used(
    make_mock_llm: Callable[[list[LLMResponse]], MockLLMClient],
) -> None:
    calls: list[str] = []
    llm = make_mock_llm([tool_calls_response(("init_tool", {})), result_response("ok")])
    scope = SessionScope(
        llm_client=llm,
        tool_collector=_static_collector("init_tool", calls),
        persistence=MemoryPersistenceProvider(),
    )

    async with scope.session(session_id="s"):
        assert await _Probe().answer() == "ok"

    assert calls == ["init_tool"]


async def test_session_tool_collector_overrides_init_default(
    make_mock_llm: Callable[[list[LLMResponse]], MockLLMClient],
) -> None:
    calls: list[str] = []
    llm = make_mock_llm([tool_calls_response(("call_tool", {})), result_response("ok")])
    scope = SessionScope(
        llm_client=llm,
        tool_collector=_static_collector("init_tool", calls),
        persistence=MemoryPersistenceProvider(),
    )

    async with scope.session(
        session_id="s", tool_collector=_static_collector("call_tool", calls)
    ):
        assert await _Probe().answer() == "ok"

    assert calls == ["call_tool"]


async def test_memory_persistence_overrides_sqlite_default(
    make_mock_llm: Callable[[list[LLMResponse]], MockLLMClient],
) -> None:
    persistence = MemoryPersistenceProvider()
    scope = SessionScope(
        llm_client=make_mock_llm([]),
        persistence=persistence,
    )

    async with scope.session(session_id="custom-store"):
        assert isinstance(get_session_storage(), MemorySessionStorage)


async def test_sqlite_persistence_is_default(
    tmp_path: Path,
    make_mock_llm: Callable[[list[LLMResponse]], MockLLMClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    scope = SessionScope(llm_client=make_mock_llm([]))

    async with scope.session(session_id="durable"):
        assert isinstance(get_session_storage(), SQLiteSessionStorage)

    assert (tmp_path / ".sessions" / "sessions.sqlite3").is_file()
