from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace

from litellm import (
    ChatCompletionMessageToolCall,
    ModelResponse,
)
from pytest_mock import MockerFixture
from sefia._tool_system import ToolRegistry
from sefia.llm.json_schema import JsonSchemaDocument
from sefia.llm.step_decision import DecisionSpec
from sefia.llm.streaming import (
    OutputStreamEvent,
)
from sefia.llm.streaming import (
    Scalar as OutputScalar,
)
from sefia.llm.streaming import (
    StringEnd as OutputStringEnd,
)
from sefia.pydantic import PydanticResultFormatFactory
from sefia_litellm._schema import StructuredDecisionFormat
from sefia_litellm._schema._data_format import StructuredDataFormat
from sefia_litellm._streaming import (
    consume_completion_stream,
)

_ResponseFactory = Callable[..., ModelResponse]


def _chunk(
    *,
    content: str | None = None,
    reasoning: str | None = None,
    tool_name: str | None = None,
    tool_arguments: str | None = None,
) -> SimpleNamespace:
    tool_calls = None
    if tool_name is not None or tool_arguments is not None:
        tool_calls = [
            SimpleNamespace(
                index=0,
                function=SimpleNamespace(
                    name=tool_name,
                    arguments=tool_arguments,
                ),
            )
        ]
    delta = SimpleNamespace(
        content=content,
        reasoning_content=reasoning,
        tool_calls=tool_calls,
    )
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


async def _stream(*chunks: SimpleNamespace) -> AsyncIterator[SimpleNamespace]:
    for chunk in chunks:
        yield chunk


def _tool_call(name: str, arguments: str) -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        id="call-1",
        function={"name": name, "arguments": arguments},
        type="function",
    )


def _structured_decision_format(
    *, include_tools: bool = True
) -> StructuredDecisionFormat:
    def lookup(key: str) -> str:
        return key

    registry = ToolRegistry()
    if include_tools:
        registry.add(lookup, name="lookup")
    return StructuredDecisionFormat.from_spec(
        DecisionSpec.for_inference(
            output_type=str,
            tools=registry.get_all(),
            result_format_factory=PydanticResultFormatFactory(),
        )
    )


async def test_decodes_enveloped_structured_decision_and_streams_logical_paths(
    mocker: MockerFixture,
    make_litellm_response: _ResponseFactory,
) -> None:
    content = (
        '{"payload":{"decision":"tool_calls","tool_calls":['
        '{"name":"lookup","arguments":{"key":"item"}}]}}'
    )
    mocker.patch(
        "litellm.stream_chunk_builder",
        return_value=make_litellm_response(content=content),
    )
    events: list[OutputStreamEvent] = []

    async def collect(event: OutputStreamEvent) -> None:
        events.append(event)

    response = await consume_completion_stream(
        _stream(_chunk(content=content[:40]), _chunk(content=content[40:])),
        content_callback=None,
        output_callback=collect,
        reasoning_callback=None,
        messages=[],
        decision_format=_structured_decision_format(),
        requested_model="gpt-4o",
    )

    assert OutputStringEnd(("tool_calls", 0, "name"), "lookup") in events
    assert OutputStringEnd(("tool_calls", 0, "arguments", "key"), "item") in events
    assert response.structured_output is not None
    assert response.structured_output.tree == {
        "decision": "tool_calls",
        "tool_calls": [{"name": "lookup", "arguments": {"key": "item"}}],
    }


async def test_restores_translated_native_tool_arguments(
    mocker: MockerFixture,
    make_litellm_response: _ResponseFactory,
) -> None:
    wire_arguments = '{"labels":[{"key":"important","value":2}]}'
    mocker.patch(
        "litellm.stream_chunk_builder",
        return_value=make_litellm_response(
            finish_reason="tool_calls",
            tool_calls=[_tool_call("categorize", wire_arguments)],
        ),
    )
    data_format = StructuredDataFormat.from_generated_schema(
        JsonSchemaDocument.from_mapping(
            {
                "type": "object",
                "properties": {
                    "labels": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
                    }
                },
                "required": ["labels"],
                "additionalProperties": False,
            }
        )
    )
    events: list[OutputStreamEvent] = []

    async def collect(event: OutputStreamEvent) -> None:
        events.append(event)

    response = await consume_completion_stream(
        _stream(
            _chunk(
                tool_name="categorize",
                tool_arguments='{"labels":[{"key":"important",',
            ),
            _chunk(tool_arguments='"value":2}]}'),
        ),
        content_callback=None,
        output_callback=collect,
        reasoning_callback=None,
        messages=[],
        decision_format=None,
        tool_data_formats={"categorize": data_format},
        requested_model="gpt-4o",
    )

    assert OutputStringEnd(("tool_calls", 0, "name"), "categorize") in events
    assert (
        OutputScalar(("tool_calls", 0, "arguments", "labels", "important"), 2) in events
    )
    assert response.tool_calls[0].arguments.tree == {"labels": {"important": 2}}
