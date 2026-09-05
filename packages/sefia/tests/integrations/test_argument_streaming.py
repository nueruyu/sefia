import json
from collections.abc import Callable, Coroutine
from typing import Any
from unittest.mock import Mock

import pytest

from sefia import ToolRegistry
from sefia.event_system import EventPublisher
from sefia.inference import FunctionInfo, ToolCallsDecision
from sefia.llm import LLMCompletion, LLMInferenceStrategy, Message
from sefia.llm._client import LLMClient
from sefia.llm.step_decision import DecisionSpec, StepTool
from sefia.llm.streaming import (
    OutputStreamCallback,
    Scalar as OutputScalar,
    StringDelta as OutputStringDelta,
    StringEnd as OutputStringEnd,
)
from sefia.llm.structured_data import StructuredData
from sefia.llm.transports import (
    DecisionTransport,
    PromptedDecisionTransport,
    StructuredDecisionTransport,
)
from sefia.pydantic import PydanticModelBackend
from sefia.streaming import ArgStream, StringDelta, StringEnd


class _Collector:
    def __init__(self) -> None:
        self.events: list[Any] = []
        self.tool_call_ids: list[str] = []

    async def __call__(self, tool_call_id: str, stream: ArgStream) -> None:
        self.tool_call_ids.append(tool_call_id)
        async for event in stream:
            self.events.append(event)


class _StreamingClient(LLMClient):
    def __init__(self, content: str) -> None:
        self.content = content

    async def complete(
        self,
        messages: list[Message],
        tools: list[StepTool] | None = None,
        decision_spec: DecisionSpec | None = None,
        stream_callback: Callable[[str], Coroutine[None, None, None]] | None = None,
        output_callback: OutputStreamCallback | None = None,
        reasoning_callback: Callable[[str], Coroutine[None, None, None]] | None = None,
    ) -> LLMCompletion:
        if stream_callback is not None:
            for character in self.content:
                await stream_callback(character)
        if output_callback is not None:
            payload = json.loads(self.content)
            for index, call in enumerate(payload.get("tool_calls", [])):
                await output_callback(
                    OutputStringEnd(("tool_calls", index, "name"), call["name"])
                )
                for name, value in call["arguments"].items():
                    path = ("tool_calls", index, "arguments", name)
                    if isinstance(value, str):
                        for character in value:
                            await output_callback(OutputStringDelta(path, character))
                        await output_callback(OutputStringEnd(path, value))
                    else:
                        await output_callback(OutputScalar(path, value))
        return LLMCompletion(
            content=self.content,
            structured_output=(
                StructuredData.parse_json(self.content)
                if decision_spec is not None
                else None
            ),
        )


def _function_info() -> FunctionInfo:
    return FunctionInfo(
        qualname="step",
        instructions="do it",
        bound_arguments={},
        type_hints={},
        return_type=str,
        args=(),
        kwargs={},
    )


_TOOL_CALL_CONTENT = (
    '{"decision": "tool_calls", "tool_calls": [{"name": "ask_human", '
    '"arguments": {"question": "What is your name?"}}]}'
)


@pytest.mark.parametrize(
    ("transport", "content"),
    [
        (StructuredDecisionTransport(), _TOOL_CALL_CONTENT),
        (
            PromptedDecisionTransport(),
            f"Decision with {{irrelevant}} braces:\n```json\n{_TOOL_CALL_CONTENT}\n```",
        ),
    ],
)
async def test_arguments_stream_from_transport_through_strategy_to_tool_handler(
    transport: DecisionTransport,
    content: str,
) -> None:
    renderer = Mock()
    renderer.render.return_value = "prompt"
    strategy = LLMInferenceStrategy(
        llm_client=_StreamingClient(content),
        result_format_factory=PydanticModelBackend(),
        prompt_renderer=renderer,
        decision_transport=transport,
        stream=True,
    )

    async def ask_human(question: str) -> str:
        return question

    collector = _Collector()
    tools = ToolRegistry()
    tools.add(ask_human, name="ask_human", stream_handler=collector)

    decision = await strategy.decide_next_step(
        function_info=_function_info(),
        history=[],
        tools=tools,
        publisher=EventPublisher([]),
    )

    assert isinstance(decision, ToolCallsDecision)
    assert collector.tool_call_ids == [decision.calls[0].id]
    assert (
        "".join(e.text for e in collector.events if isinstance(e, StringDelta))
        == "What is your name?"
    )
    assert StringEnd(name="question", value="What is your name?") in collector.events
