import pytest
from glyff.store import MemoryBackend

from sefia import ToolRegistry, Tools
from sefia._tool_execution import call_tools
from sefia.event_system import EventPublisher
from sefia.inference import Capability, ToolCallRequest
from sefia.llm._arg_stream import _ArgStreamChannel
from sefia.streaming import StringDelta
from sefia.testing import MockLLMClient, memory_session
from sefia.tool_collectors import DefaultToolCollector
from sefios import InputRequired, MemorySessionStorage
from sefios._session_state import bind_session_storage
from sefios.tools import Input, InputRequest


class Agent:
    """A tool's own methods are never self-exposed; hold it as a dependency."""

    _input: Tools[Input]

    def __init__(self, input_tool: Input):
        self._input = input_tool


async def test_input_tool_streams_prompt_deltas():
    seen: list[tuple[str, str]] = []
    agent = Agent(
        Input(on_prompt_delta=lambda call_id, text: seen.append((call_id, text)))
    )
    registry = DefaultToolCollector().collect([Capability(value=agent, declared=None)])
    registered = next(tool for tool in registry.get_all() if "get_input" in tool.name)
    assert registered.stream_handler is not None

    channel = _ArgStreamChannel()
    channel.feed(StringDelta(name="prompt", text="What "))
    channel.feed(StringDelta(name="prompt", text="topic?"))
    channel.feed(StringDelta(name="other", text="ignored"))
    channel.close()

    await registered.stream_handler("call-1", channel)

    assert seen == [("call-1", "What "), ("call-1", "topic?")]


async def test_nested_input_uses_a_stable_fallback_id_across_resume(serializer):
    seen: list[InputRequest] = []
    answers: dict[str, str] = {}

    def provide(request: InputRequest) -> str | None:
        seen.append(request)
        return answers.get(request.interaction_id)

    input_tool = Input(get_input=provide)

    async def parent() -> str:
        return await input_tool.get_input("Name?")

    registry = ToolRegistry()
    registry.add(parent, name="parent")
    call = ToolCallRequest(id="parent-call", name="parent", arguments={})
    backend = MemoryBackend()
    storage = MemorySessionStorage(serializer=serializer)

    with pytest.raises(InputRequired):
        async with memory_session(
            MockLLMClient([]), session_id="nested-input", backend=backend
        ):
            with bind_session_storage(storage):
                await call_tools([call], registry, EventPublisher([]))

    fallback_id = seen[0].interaction_id
    assert fallback_id != "parent-call"
    answers[fallback_id] = "Alice"

    async with memory_session(
        MockLLMClient([]), session_id="nested-input", backend=backend
    ):
        with bind_session_storage(storage):
            results = await call_tools([call], registry, EventPublisher([]))

    assert results[0].result == "Alice"
    assert [request.interaction_id for request in seen] == [fallback_id, fallback_id]
