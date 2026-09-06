import glyff
import pytest
from sefia import Domain, ToolRegistry
from sefia._tool_execution import call_tools
from sefia.event_system import EventHandler, EventPublisher
from sefia.events import ToolExecutionBound
from sefia.exceptions import PauseException
from sefia.testing import MockLLMClient, make_tool_call_request, memory_session

engrave = Domain(glyff.Domain(__name__, version="1")).engrave


class Bindings(EventHandler[ToolExecutionBound]):
    def __init__(self) -> None:
        self.events: list[ToolExecutionBound] = []

    async def handle(self, event: ToolExecutionBound) -> None:
        assert glyff.get_context().current_execution_id == event.execution_id
        self.events.append(event)


async def test_binding_precedes_pause_and_matches_body_execution() -> None:
    bindings = Bindings()
    body_ids: list[glyff.ExecutionId | None] = []

    @engrave
    async def tool() -> str:
        assert len(bindings.events) == 1
        body_ids.append(glyff.get_context().current_execution_id)
        raise PauseException()

    registry = ToolRegistry()
    registry.add(tool, name="tool")
    async with memory_session(MockLLMClient([])):
        with pytest.raises(PauseException):
            await call_tools(
                [make_tool_call_request(id="call", name="tool", arguments={})],
                registry,
                EventPublisher([bindings]),
            )
    assert len(bindings.events) == 1
    assert bindings.events[0].tool_call_id == "call"
    assert bindings.events[0].execution_id == body_ids[0]


async def test_nested_executions_do_not_bind_to_parent_tool() -> None:
    bindings = Bindings()

    @engrave
    async def child() -> str:
        return "child"

    @engrave
    async def parent(depth: int) -> str:
        return await parent(depth - 1) if depth else await child()

    registry = ToolRegistry()
    registry.add(parent, name="parent")
    async with memory_session(MockLLMClient([])):
        await call_tools(
            [
                make_tool_call_request(
                    id="parent-call", name="parent", arguments={"depth": 1}
                )
            ],
            registry,
            EventPublisher([bindings]),
        )
    assert len(bindings.events) == 1
    assert bindings.events[0].tool_call_id == "parent-call"
    assert bindings.events[0].execution_id.name.value.endswith("parent")


async def test_ordinary_parent_does_not_bind_engraved_child() -> None:
    bindings = Bindings()

    @engrave
    async def child() -> str:
        return "child"

    async def parent() -> str:
        return await child()

    registry = ToolRegistry()
    registry.add(parent, name="parent")
    async with memory_session(MockLLMClient([])):
        await call_tools(
            [make_tool_call_request(id="parent-call", name="parent", arguments={})],
            registry,
            EventPublisher([bindings]),
        )
    assert bindings.events == []
