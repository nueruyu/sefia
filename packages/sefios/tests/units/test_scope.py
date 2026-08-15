from collections.abc import Callable
from pathlib import Path

from glyff_pydantic import PydanticSerializer
from sefia import JsonSchemaToolEntry
from sefia.tool_collectors import StaticToolCollector
from sefia.llm import LLMResponse
from sefia.testing import MockLLMClient, result_response, tool_calls_response

from sefios import domain, MemorySessionStorage, SessionScope, get_session_storage

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
    tmp_path: Path,
    make_mock_llm: Callable[[list[LLMResponse]], MockLLMClient],
) -> None:
    calls: list[str] = []
    llm = make_mock_llm([tool_calls_response(("init_tool", {})), result_response("ok")])
    scope = SessionScope(
        session_dir=tmp_path,
        llm_client=llm,
        tool_collector=_static_collector("init_tool", calls),
    )

    async with scope.session(session_id="s"):
        assert await _Probe().answer() == "ok"

    assert calls == ["init_tool"]


async def test_session_tool_collector_overrides_init_default(
    tmp_path: Path,
    make_mock_llm: Callable[[list[LLMResponse]], MockLLMClient],
) -> None:
    calls: list[str] = []
    llm = make_mock_llm([tool_calls_response(("call_tool", {})), result_response("ok")])
    scope = SessionScope(
        session_dir=tmp_path,
        llm_client=llm,
        tool_collector=_static_collector("init_tool", calls),
    )

    async with scope.session(
        session_id="s", tool_collector=_static_collector("call_tool", calls)
    ):
        assert await _Probe().answer() == "ok"

    assert calls == ["call_tool"]


async def test_session_storage_factory_overrides_default(
    tmp_path: Path,
    make_mock_llm: Callable[[list[LLMResponse]], MockLLMClient],
) -> None:
    captured: dict[str, MemorySessionStorage] = {}

    def factory(session_id: str) -> MemorySessionStorage:
        storage = MemorySessionStorage(serializer=PydanticSerializer())
        captured[session_id] = storage
        return storage

    scope = SessionScope(
        session_dir=tmp_path,
        llm_client=make_mock_llm([]),
        session_storage_factory=factory,
    )

    async with scope.session(session_id="custom-store"):
        assert get_session_storage() is captured["custom-store"]
