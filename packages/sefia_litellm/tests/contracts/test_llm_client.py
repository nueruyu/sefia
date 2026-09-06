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
from sefia.llm.streaming import StringDelta, StringEnd
from sefia.llm.structured_data import StructuredData
from sefia.pydantic import PydanticModelBackend
from sefia.testing import (
    LLMClientCase,
    LLMClientContract,
    StreamingLLMClientCase,
    StreamingLLMClientContract,
)
from sefia_litellm import LiteLLMClient
from typing_extensions import override

_ResponseFactory = Callable[..., ModelResponse]


class TestLiteLLMPlainCompletionContract(LLMClientContract):
    @pytest.fixture(autouse=True)
    def _prepare_case(
        self,
        mock_acompletion: AsyncMock,
        make_litellm_response: _ResponseFactory,
    ) -> None:
        mock_acompletion.return_value = make_litellm_response(
            content="Hello", model="gpt-4o"
        )
        expected = LLMCompletion(model="gpt-4o", content="Hello", stop_reason="stop")
        self._case = LLMClientCase(LiteLLMClient(model="gpt-4o"), expected)

    @override
    def make_llm_client_case(self) -> LLMClientCase:
        return self._case


class TestLiteLLMStructuredCompletionContract(LLMClientContract):
    @pytest.fixture(autouse=True)
    def _prepare_case(
        self,
        mock_acompletion: AsyncMock,
        make_litellm_response: _ResponseFactory,
    ) -> None:
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
        self._case = LLMClientCase(
            LiteLLMClient(model="gpt-4o"),
            expected,
            decision_spec=decision_spec,
        )

    @override
    def make_llm_client_case(self) -> LLMClientCase:
        return self._case


class TestLiteLLMNativeToolContract(LLMClientContract):
    @pytest.fixture(autouse=True)
    def _prepare_case(
        self,
        mock_acompletion: AsyncMock,
        make_litellm_response: _ResponseFactory,
    ) -> None:
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
        self._case = LLMClientCase(
            LiteLLMClient(model="gpt-4o"), expected, tools=(tool,)
        )

    @override
    def make_llm_client_case(self) -> LLMClientCase:
        return self._case


@dataclass
class _Delta:
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: None = None


async def _stream(*deltas: _Delta) -> AsyncIterator[SimpleNamespace]:
    for delta in deltas:
        yield SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class TestLiteLLMStreamingContract(StreamingLLMClientContract):
    @pytest.fixture(autouse=True)
    def _prepare_case(
        self,
        mocker: MockerFixture,
        mock_acompletion: AsyncMock,
        make_litellm_response: _ResponseFactory,
    ) -> None:
        content_chunks = (
            '{"payload":{"decision":"result",',
            '"result":"done"}}',
        )
        content = "".join(content_chunks)
        mock_acompletion.return_value = _stream(
            _Delta(reasoning_content="Let me "),
            _Delta(reasoning_content="think."),
            *(_Delta(content=chunk) for chunk in content_chunks),
        )
        mocker.patch(
            "litellm.stream_chunk_builder",
            return_value=make_litellm_response(content=content, model="gpt-4o"),
        )
        decision_spec = DecisionSpec.for_inference(
            output_type=str,
            tools=[],
            result_format_factory=PydanticModelBackend(),
        )
        expected = LLMCompletion(
            model="gpt-4o",
            content=content,
            reasoning_content="Let me think.",
            stop_reason="stop",
            structured_output=StructuredData.from_json(
                {"decision": "result", "result": "done"}
            ),
        )
        self._case = StreamingLLMClientCase(
            LiteLLMClient(model="gpt-4o"),
            expected,
            decision_spec=decision_spec,
            content_chunks=content_chunks,
            reasoning_chunks=("Let me ", "think."),
            output_events=(
                StringDelta(("decision",), "result"),
                StringEnd(("decision",), "result"),
                StringDelta(("result",), "done"),
                StringEnd(("result",), "done"),
            ),
        )

    @override
    def make_streaming_llm_client_case(self) -> StreamingLLMClientCase:
        return self._case
