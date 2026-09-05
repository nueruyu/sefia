"""Apply the public LLM-client contracts to the LiteLLM adapter."""

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from litellm import (
    ChatCompletionMessageToolCall,
    ModelResponse,
)
from pytest_mock import MockerFixture

from sefia.llm import LLMCompletion, ToolCall
from sefia.llm.json_schema import JsonSchemaDocument
from sefia.llm.step_decision import DecisionSpec, StepTool, ToolSchemaSource
from sefia.llm.structured_data import StructuredData
from sefia.pydantic import PydanticModelBackend
from sefia.testing import (
    LLMClientCase,
    LLMClientContract,
    StreamingLLMClientCase,
    StreamingLLMClientContract,
)
from sefia_litellm import LiteLLMClient

_ResponseFactory = Callable[..., ModelResponse]


class TestLiteLLMPlainCompletionContract(LLMClientContract):
    @pytest.fixture
    def llm_client_case(
        self,
        mock_acompletion: AsyncMock,
        make_litellm_response: _ResponseFactory,
    ) -> LLMClientCase:
        mock_acompletion.return_value = make_litellm_response(
            content="Hello", model="gpt-4o"
        )
        expected = LLMCompletion(model="gpt-4o", content="Hello", stop_reason="stop")
        return LLMClientCase(LiteLLMClient(model="gpt-4o"), expected)


class TestLiteLLMStructuredCompletionContract(LLMClientContract):
    @pytest.fixture
    def llm_client_case(
        self,
        mock_acompletion: AsyncMock,
        make_litellm_response: _ResponseFactory,
    ) -> LLMClientCase:
        content = '{"payload":{"decision":"result","result":"done"}}'
        mock_acompletion.return_value = make_litellm_response(
            content=content, model="gpt-4o"
        )
        decision_spec = DecisionSpec.for_inference(
            output_type=str,
            tools=[],
            result_format_factory=PydanticModelBackend(),
        )
        expected = LLMCompletion(
            model="gpt-4o",
            content=content,
            stop_reason="stop",
            structured_output=StructuredData.from_json(
                {"decision": "result", "result": "done"}
            ),
        )
        return LLMClientCase(
            LiteLLMClient(model="gpt-4o"),
            expected,
            decision_spec=decision_spec,
        )


class TestLiteLLMNativeToolContract(LLMClientContract):
    @pytest.fixture
    def llm_client_case(
        self,
        mock_acompletion: AsyncMock,
        make_litellm_response: _ResponseFactory,
    ) -> LLMClientCase:
        upstream_call = ChatCompletionMessageToolCall(
            id="call-1",
            function={"name": "lookup", "arguments": '{"key":"item"}'},
            type="function",
        )
        mock_acompletion.return_value = make_litellm_response(
            finish_reason="tool_calls",
            tool_calls=[upstream_call],
            model="gpt-4o",
        )
        tool = StepTool(
            name="lookup",
            description="Look up an item.",
            arguments=JsonSchemaDocument.from_mapping(
                {
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                    "additionalProperties": False,
                }
            ),
            schema_source=ToolSchemaSource.USER_DEFINED,
        )
        expected = LLMCompletion(
            model="gpt-4o",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    name="lookup",
                    arguments=StructuredData.from_json({"key": "item"}),
                )
            ],
            stop_reason="tool_calls",
        )
        return LLMClientCase(LiteLLMClient(model="gpt-4o"), expected, tools=(tool,))


@dataclass
class _Delta:
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: None = None


async def _stream(*deltas: _Delta) -> AsyncIterator[SimpleNamespace]:
    for delta in deltas:
        yield SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class TestLiteLLMStreamingContract(StreamingLLMClientContract):
    @pytest.fixture
    def streaming_llm_client_case(
        self,
        mocker: MockerFixture,
        mock_acompletion: AsyncMock,
        make_litellm_response: _ResponseFactory,
    ) -> StreamingLLMClientCase:
        mock_acompletion.return_value = _stream(
            _Delta(reasoning_content="Let me "),
            _Delta(reasoning_content="think."),
            _Delta(content="Hel"),
            _Delta(content="lo"),
        )
        mocker.patch(
            "litellm.stream_chunk_builder",
            return_value=make_litellm_response(content="Hello", model="gpt-4o"),
        )
        expected = LLMCompletion(
            model="gpt-4o",
            content="Hello",
            reasoning_content="Let me think.",
            stop_reason="stop",
        )
        return StreamingLLMClientCase(
            LiteLLMClient(model="gpt-4o"),
            expected,
            content_chunks=("Hel", "lo"),
            reasoning_chunks=("Let me ", "think."),
        )
