import json
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from sefia.event_publisher import EventPublisher
from sefia.llm.messages import LLMResponse
from sefia.llm.strategy import LLMInferenceStrategy
from sefia.models import (
    FinalAnswerDecision,
    ToolCallDecision,
    ToolCallRequest,
    ToolCallResult,
)


class MockEventPublisher(EventPublisher):
    def __init__(self):
        super().__init__(handlers=[])
        self.events = []

    async def publish(self, event):
        self.events.append(event)


class MyOutput(BaseModel):
    name: str
    value: int


@pytest.fixture
def mock_llm_client():
    return AsyncMock()


DUMMY_SCHEMA: dict = {}


class TestLLMInferenceStrategy:
    def test_build_messages_correctly(self):
        strategy = LLMInferenceStrategy(llm_client=AsyncMock())
        history = [
            ToolCallDecision(
                calls=[ToolCallRequest(id="1", name="search", arguments={"q": "test"})]
            ),
            ToolCallResult(tool_call_id="1", result="found"),
        ]

        dummy_tools = [{"function": {"name": "search"}}]
        messages = strategy._build_messages(
            "instructions", {"arg": "val"}, history, DUMMY_SCHEMA, dummy_tools
        )

        assert len(messages) == 4
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert messages[2].role == "assistant"
        assert messages[3].role == "tool"
        assert json.loads(str(messages[3].content)) == "found"

    async def test_decide_next_step_handles_tool_calls(self, mock_llm_client):
        tool_calls_payload = json.dumps(
            {"tool_calls": [{"name": "my_tool", "arguments": {"param": 1}}]}
        )
        mock_llm_client.complete.return_value = LLMResponse(content=tool_calls_payload)
        strategy = LLMInferenceStrategy(llm_client=mock_llm_client)

        decision = await strategy.decide_next_step(
            "do it", {}, [], [{"type": "function"}], str, MockEventPublisher()
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
        strategy = LLMInferenceStrategy(llm_client=mock_llm_client)

        decision = await strategy.decide_next_step(
            "do it", {}, [], [], MyOutput, MockEventPublisher()
        )

        assert isinstance(decision, FinalAnswerDecision)
        assert isinstance(decision.answer, MyOutput)
        assert decision.answer.name == "test"
        assert decision.answer.value == 42

    async def test_decide_next_step_raises_on_validation_error(self, mock_llm_client):
        # final_answer is missing required 'value' field — Pydantic should reject it
        mock_llm_client.complete.return_value = LLMResponse(
            content='{"final_answer": {"name": "test"}}'
        )
        strategy = LLMInferenceStrategy(llm_client=mock_llm_client)

        with pytest.raises(ValueError, match="LLM output failed validation"):
            await strategy.decide_next_step(
                "do it", {}, [], [], MyOutput, MockEventPublisher()
            )

    async def test_decide_next_step_handles_plain_string_output(self, mock_llm_client):
        mock_llm_client.complete.return_value = LLMResponse(
            content='{"final_answer": "Hello, world!"}'
        )
        strategy = LLMInferenceStrategy(llm_client=mock_llm_client)

        decision = await strategy.decide_next_step(
            "do it", {}, [], [], str, MockEventPublisher()
        )

        assert isinstance(decision, FinalAnswerDecision)
        assert decision.answer == "Hello, world!"

    async def test_decide_next_step_raises_when_both_fields_null(self, mock_llm_client):
        mock_llm_client.complete.return_value = LLMResponse(
            content='{"final_answer": null}'
        )
        strategy = LLMInferenceStrategy(llm_client=mock_llm_client)

        with pytest.raises(ValueError, match="must contain either"):
            await strategy.decide_next_step(
                "do it", {}, [], [], str, MockEventPublisher()
            )
