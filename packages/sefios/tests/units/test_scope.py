from glyff_pydantic import PydanticSerializer
from sefia import JsonSchemaToolEntry, infer
from sefia.tool_collectors import StaticToolCollector
from sefia.testing import result_response, tool_calls_response

from sefios import MemorySessionStorage, SessionScope, get_session_storage


class _Probe:
    """A receiver with no tools of its own, so the run's registry is exactly
    whatever collector the scope installs (the static collector ignores the
    receiver's capabilities)."""

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


async def test_tool_collector_default_is_used(tmp_path, make_mock_llm):
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


async def test_session_tool_collector_overrides_init_default(tmp_path, make_mock_llm):
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


async def test_session_storage_factory_overrides_default(tmp_path, make_mock_llm):
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
