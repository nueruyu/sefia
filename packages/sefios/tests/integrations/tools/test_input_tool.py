from collections.abc import AsyncIterator

import pytest

from sefia import ToolRegistry, Tools
from sefia._tool_execution import call_tools
from sefia.event_system import EventPublisher
from sefia.inference import Capability
from sefia.streaming import ArgEvent, StringDelta
from sefia.testing import MockLLMClient, make_tool_call_request, memory_session
from sefia.tool_collectors import DefaultToolCollector
from sefios.tools import Input


class Agent:
    """A tool's own methods are never self-exposed; hold it as a dependency."""

    _input: Tools[Input]

    def __init__(self, input_tool: Input):
        self._input = input_tool


async def _events(*events: ArgEvent) -> AsyncIterator[ArgEvent]:
    for event in events:
        yield event


async def test_input_tool_streams_prompt_deltas():
    seen: list[tuple[str, str]] = []
    agent = Agent(
        Input(on_prompt_delta=lambda call_id, text: seen.append((call_id, text)))
    )
    registry = DefaultToolCollector().collect([Capability(value=agent, declared=None)])
    registered = next(tool for tool in registry.get_all() if "get_input" in tool.name)
    assert registered.stream_handler is not None

    await registered.stream_handler(
        "call-1",
        _events(
            StringDelta(name="prompt", text="What "),
            StringDelta(name="prompt", text="topic?"),
            StringDelta(name="other", text="ignored"),
        ),
    )

    assert seen == [("call-1", "What "), ("call-1", "topic?")]


async def test_input_fails_fast_outside_tool_dispatch():
    async with memory_session(MockLLMClient([])):
        with pytest.raises(RuntimeError, match="must be invoked as a dispatched tool"):
            await Input().get_input("Name?")


async def test_nested_input_fails_instead_of_reusing_parent_call_id():
    input_tool = Input()

    async def parent() -> str:
        return await input_tool.get_input("Name?")

    registry = ToolRegistry()
    registry.add(parent, name="parent")

    async with memory_session(MockLLMClient([])):
        results = await call_tools(
            [make_tool_call_request(id="parent-call", name="parent", arguments={})],
            registry,
            EventPublisher([]),
        )

    assert results[0].tool_call_id == "parent-call"
    assert "Input.get_input() must be invoked as a dispatched tool" in results[0].result
