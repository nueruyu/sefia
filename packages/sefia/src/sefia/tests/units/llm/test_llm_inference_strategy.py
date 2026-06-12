import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest

from sefia.event_system import EventPublisher
from sefia.inference import (
    FinalAnswerDecision,
    ToolCallDecision,
    ToolCallRequest,
    ToolCallResult,
)
from sefia.llm import LLMInferenceStrategy, LLMResponse
from sefia.llm.events import LLMTokenReceived
from sefia.pydantic import PydanticModelInspector
from sefia.pydantic.json_utils import pydantic_json_default


class MockEventPublisher(EventPublisher):
    def __init__(self):
        super().__init__(handlers=[])
        self.events = []

    async def publish(self, event):
        self.events.append(event)


@dataclass(frozen=True)
class MyOutput:
    name: str
    value: int


@dataclass(frozen=True)
class MyIssue:
    description: str


@pytest.fixture
def mock_llm_client():
    return AsyncMock()


DUMMY_SCHEMA: dict = {}


class TestLLMInferenceStrategy:
    def _strategy(self, llm_client, stream: bool = False):
        mock_formatter = Mock()
        mock_formatter.format_arguments.return_value = "<arguments/>"
        return LLMInferenceStrategy(
            llm_client=llm_client,
            model_inspector=PydanticModelInspector(),
            prompt_formatter=mock_formatter,
            json_default=pydantic_json_default,
            stream=stream,
        )

    def test_build_messages_correctly(self):
        strategy = self._strategy(AsyncMock())
        history = [
            ToolCallDecision(
                calls=[
                    ToolCallRequest(
                        id="1",
                        name="search",
                        arguments={"q": "日本語の検索クエリ"},
                    )
                ]
            ),
            ToolCallResult(tool_call_id="1", result="見つかりました"),
        ]

        dummy_tools = [{"function": {"name": "search"}}]
        messages = strategy._build_messages(
            "instructions", {"arg": "val"}, {}, history, DUMMY_SCHEMA, dummy_tools
        )

        assert len(messages) == 4
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert messages[2].role == "assistant"
        assert messages[3].role == "tool"
        tool_calls = messages[2].tool_calls
        assert tool_calls is not None
        tool_arguments = tool_calls[0]["function"]["arguments"]
        assert "日本語の検索クエリ" in tool_arguments
        assert "\\u65e5" not in tool_arguments
        assert json.loads(tool_arguments) == {"q": "日本語の検索クエリ"}
        assert "見つかりました" in str(messages[3].content)
        assert "\\u898b" not in str(messages[3].content)
        assert json.loads(str(messages[3].content)) == "見つかりました"

    def test_build_decision_schema_hoists_nested_definitions(self):
        strategy = self._strategy(AsyncMock())

        schema = strategy._build_llm_decision_schema(
            list[MyIssue],
            [{"type": "function"}],
        )

        assert "MyIssue" in schema["$defs"]
        final_answer_schema = schema["properties"]["final_answer"]["anyOf"][0]
        assert "$defs" not in final_answer_schema
        assert final_answer_schema["items"]["$ref"] == "#/$defs/MyIssue"
        tool_call_schema = schema["properties"]["tool_calls"]["anyOf"][0]["items"]
        assert "$defs" not in tool_call_schema
        assert tool_call_schema["$ref"] == "#/$defs/LLMToolCall"

    def test_build_decision_schema_requires_non_null_answer_without_tools(self):
        strategy = self._strategy(AsyncMock())

        schema = strategy._build_llm_decision_schema(list[MyIssue], [])

        assert schema["required"] == ["final_answer"]
        final_answer_schema = schema["properties"]["final_answer"]
        assert final_answer_schema["type"] == "array"
        assert "oneOf" not in final_answer_schema

    def test_build_decision_schema_requires_nullable_fields_with_tools(self):
        strategy = self._strategy(AsyncMock())

        schema = strategy._build_llm_decision_schema(
            list[MyIssue],
            [{"type": "function"}],
        )

        assert schema["required"] == ["final_answer", "tool_calls"]
        assert {"type": "null"} in schema["properties"]["final_answer"]["anyOf"]
        assert {"type": "null"} in schema["properties"]["tool_calls"]["anyOf"]

    def test_build_messages_tells_no_tool_agent_to_return_empty_collection(self):
        strategy = self._strategy(AsyncMock())

        messages = strategy._build_messages(
            "instructions",
            {},
            {},
            [],
            DUMMY_SCHEMA,
            [],
        )

        assert "empty collection instead of null" in str(messages[0].content)

    async def test_decide_next_step_handles_tool_calls(self, mock_llm_client):
        tool_calls_payload = json.dumps(
            {"tool_calls": [{"name": "my_tool", "arguments": {"param": 1}}]}
        )
        mock_llm_client.complete.return_value = LLMResponse(content=tool_calls_payload)
        strategy = self._strategy(mock_llm_client, stream=True)

        decision = await strategy.decide_next_step(
            "do it", {}, {}, [], [{"type": "function"}], str, MockEventPublisher()
        )

        assert isinstance(decision, ToolCallDecision)
        assert len(decision.calls) == 1
        assert decision.calls[0].name == "my_tool"
        assert decision.calls[0].arguments == {"param": 1}
        assert decision.calls[0].id.startswith("call_")

    async def test_decide_next_step_handles_final_answer_with_validation(
        self, mock_llm_client
    ):
        final_answer_payload = json.dumps(
            {"final_answer": {"name": "test", "value": 42}}
        )
        mock_llm_client.complete.return_value = LLMResponse(
            content=final_answer_payload
        )
        strategy = self._strategy(mock_llm_client, stream=True)
        publisher = MockEventPublisher()

        decision = await strategy.decide_next_step(
            "do it", {}, {}, [], [], MyOutput, publisher
        )

        assert isinstance(decision, FinalAnswerDecision)
        assert isinstance(decision.answer, MyOutput)
        assert decision.answer.name == "test"
        assert decision.answer.value == 42

        stream_callback_obj = mock_llm_client.complete.await_args.kwargs[
            "stream_callback"
        ]
        assert callable(stream_callback_obj)
        stream_callback = cast(Callable[[str], Awaitable[None]], stream_callback_obj)
        await stream_callback("tok")
        assert any(
            isinstance(event, LLMTokenReceived) and event.token == "tok"
            for event in publisher.events
        )

    async def test_decide_next_step_does_not_set_stream_callback_by_default(
        self, mock_llm_client
    ):
        mock_llm_client.complete.return_value = LLMResponse(
            content='{"final_answer": "done"}'
        )
        strategy = self._strategy(mock_llm_client)

        decision = await strategy.decide_next_step(
            "do it", {}, {}, [], [], str, MockEventPublisher()
        )

        assert isinstance(decision, FinalAnswerDecision)
        assert mock_llm_client.complete.await_args.kwargs["stream_callback"] is None

    async def test_decide_next_step_raises_on_validation_error(self, mock_llm_client):
        # final_answer is missing required 'value' field — Pydantic should reject it
        mock_llm_client.complete.return_value = LLMResponse(
            content='{"final_answer": {"name": "test"}}'
        )
        strategy = self._strategy(mock_llm_client)

        with pytest.raises(ValueError, match="LLM output failed validation"):
            await strategy.decide_next_step(
                "do it", {}, {}, [], [], MyOutput, MockEventPublisher()
            )

    async def test_decide_next_step_handles_plain_string_output(self, mock_llm_client):
        mock_llm_client.complete.return_value = LLMResponse(
            content='{"final_answer": "Hello, world!"}'
        )
        strategy = self._strategy(mock_llm_client)

        decision = await strategy.decide_next_step(
            "do it", {}, {}, [], [], str, MockEventPublisher()
        )

        assert isinstance(decision, FinalAnswerDecision)
        assert decision.answer == "Hello, world!"

    async def test_decide_next_step_raises_when_both_fields_null(self, mock_llm_client):
        mock_llm_client.complete.return_value = LLMResponse(
            content='{"final_answer": null}'
        )
        strategy = self._strategy(mock_llm_client)

        with pytest.raises(ValueError, match="must contain either"):
            await strategy.decide_next_step(
                "do it", {}, {}, [], [], str, MockEventPublisher()
            )
