import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from unittest.mock import Mock

import glyff
import sefia
from sefia import ToolDefinition, ToolFunctionInspector, Tools
from sefia.inference import FunctionInfo
from sefia.llm import (
    JsonCompatible,
    JsonMaterializer,
    LLMCompletion,
    Message,
    MessageComposer,
    MessageLayout,
    PromptRenderer,
)
from sefia.llm.result_format import ResultFormat, ResultFormatFactory
from sefia.llm.transports import PromptedDecisionTransport
from sefia.pydantic import (
    PydanticJsonMaterializer,
    PydanticResultFormatFactory,
    PydanticToolFunctionInspector,
)
from sefia.testing import (
    MockLLMClient,
    memory_session,
    result_completion,
    tool_calls_completion,
)
from sefia.tool_collectors import DefaultToolCollector, StaticToolCollector
from typing_extensions import override

infer = sefia.Domain(
    glyff.Domain(
        "packages.sefia.tests.integrations.test_session_configuration", version="1"
    )
).infer


@dataclass
class _Report:
    topic: str
    summary: str


class _Agent:
    @infer
    async def generate_report(self, topic: str) -> _Report: ...


@dataclass(frozen=True)
class _Input:
    value: int


@dataclass(frozen=True)
class _ToolOutput:
    value: int


class _Toolkit:
    def __init__(self) -> None:
        self.outputs: list[_ToolOutput] = []

    async def lookup(self, value: int) -> _ToolOutput:
        """Look up a value."""
        output = _ToolOutput(value=value)
        self.outputs.append(output)
        return output


class _CapabilityAgent:
    toolkit: Tools[_Toolkit]

    def __init__(self, toolkit: _Toolkit) -> None:
        self.toolkit = toolkit

    @infer
    async def answer(self, payload: _Input) -> str:
        """Answer using the lookup tool."""
        ...


@infer
async def _answer_without_tools(payload: _Input) -> str:
    """Answer without tools."""
    ...


class _RecordingToolFunctionInspector(ToolFunctionInspector):
    def __init__(self) -> None:
        self._delegate = PydanticToolFunctionInspector()
        self.names: list[Callable[..., Any]] = []
        self.definitions: list[Callable[..., Any]] = []
        self.bindings: list[dict[str, Any]] = []

    @override
    def tool_name(self, func: Callable[..., Any]) -> str:
        self.names.append(func)
        return "lookup"

    @override
    def definition(self, func: Callable[..., Any], *, name: str) -> ToolDefinition:
        self.definitions.append(func)
        return self._delegate.definition(func, name=name)

    @override
    def bind(
        self, func: Callable[..., Any], arguments: dict[str, Any]
    ) -> dict[str, Any]:
        self.bindings.append(arguments)
        return self._delegate.bind(func, arguments)


class _RecordingResultFormatFactory(ResultFormatFactory):
    def __init__(self) -> None:
        self._delegate = PydanticResultFormatFactory()
        self.types: list[Any] = []

    @override
    def create(self, python_type: Any) -> ResultFormat:
        self.types.append(python_type)
        return self._delegate.create(python_type)


class _RecordingJsonMaterializer(JsonMaterializer):
    def __init__(self) -> None:
        self._delegate = PydanticJsonMaterializer()
        self.values: list[object] = []

    @override
    def materialize(self, value: object) -> JsonCompatible:
        self.values.append(value)
        return self._delegate.materialize(value)


async def test_session_connects_a_custom_prompt_renderer_to_the_transport() -> None:
    client = MockLLMClient([result_completion(_Report("custom", "rendered"))])
    renderer = Mock(spec=PromptRenderer)
    renderer.render.return_value = "custom prompt"

    async with memory_session(client, prompt_renderer=renderer):
        report = await _Agent().generate_report(topic="custom")

    assert report == _Report("custom", "rendered")
    messages = client.requests[0]["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"].startswith("custom prompt\n\n## Response\n\n")


async def test_session_connects_a_prompted_decision_transport() -> None:
    client = MockLLMClient(
        [
            LLMCompletion(
                content=json.dumps(
                    {
                        "decision": "result",
                        "result": {"topic": "prompted", "summary": "decoded"},
                    }
                )
            )
        ]
    )

    async with memory_session(client, decision_transport=PromptedDecisionTransport()):
        report = await _Agent().generate_report(topic="prompted")

    assert report == _Report("prompted", "decoded")
    assert client.requests[0]["decision_spec"] is None


async def test_session_wires_independent_python_llm_capabilities() -> None:
    inspector = _RecordingToolFunctionInspector()
    result_format_factory = _RecordingResultFormatFactory()
    converter = _RecordingJsonMaterializer()
    toolkit = _Toolkit()
    payload = _Input(value=7)
    client = MockLLMClient(
        [
            tool_calls_completion(("lookup", {"value": 7})),
            result_completion("done"),
        ]
    )

    async with memory_session(
        client,
        tool_collector=DefaultToolCollector(inspector=inspector),
        result_format_factory=result_format_factory,
        json_materializer=converter,
    ):
        result = await _CapabilityAgent(toolkit).answer(payload)

    assert result == "done"
    assert inspector.names
    assert inspector.definitions
    assert inspector.bindings == [{"value": 7}]
    assert result_format_factory.types == [str, str]
    assert any(value is payload for value in converter.values)
    assert any(value is toolkit.outputs[0] for value in converter.values)
    assert {"value": 7} in converter.values


async def test_custom_tool_collector_keeps_strategy_capabilities_independent() -> None:
    result_format_factory = _RecordingResultFormatFactory()
    converter = _RecordingJsonMaterializer()
    payload = _Input(value=9)
    client = MockLLMClient([result_completion("done")])

    async with memory_session(
        client,
        tool_collector=StaticToolCollector([]),
        result_format_factory=result_format_factory,
        json_materializer=converter,
    ):
        result = await _answer_without_tools(payload)

    assert result == "done"
    assert result_format_factory.types == [str]
    assert converter.values == [payload]


class _ProfileMessage(MessageComposer):
    @override
    async def compose(
        self, function: FunctionInfo, layout: MessageLayout
    ) -> MessageLayout:
        return MessageLayout(
            before=(Message(role="developer", content="shared"), *layout.before),
            arguments=layout.arguments,
            after=layout.after,
        )


async def test_profiles_share_session_message_composers() -> None:
    @infer
    @sefia.profile("alternate")
    async def answer(topic: str) -> str:
        """Answer the task."""
        ...

    default_client = MockLLMClient([])
    profile_client = MockLLMClient([result_completion("done")])
    result_format_factory = _RecordingResultFormatFactory()
    converter = _RecordingJsonMaterializer()
    async with memory_session(
        default_client,
        profiles=[sefia.Profile(key="alternate", client=profile_client)],
        message_composers=(_ProfileMessage(),),
        result_format_factory=result_format_factory,
        json_materializer=converter,
    ):
        assert await answer("topic") == "done"

    assert not default_client.requests
    assert profile_client.requests[0]["messages"][0] == {
        "role": "developer",
        "content": "shared",
    }
    assert result_format_factory.types == [str]
    assert converter.values == ["topic"]
