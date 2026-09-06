"""Reusable pytest contract for ``DecisionTransport`` implementations."""

from collections.abc import Callable, Coroutine
from dataclasses import dataclass

from typing_extensions import override

from ..inference import ToolCallResult
from ..llm._client import LLMClient
from ..llm._messages import LLMCompletion, Message
from ..llm._prompt_renderer import DecisionPrompt, PromptRenderer
from ..llm.step_decision import DecisionSpec, StepTool
from ..llm.streaming import OutputStreamCallback, OutputStreamEvent, Scalar
from ..llm.structured_data import StructuredData
from ..llm.transports import DecisionObserver, DecisionRequest, DecisionTransport
from ..pydantic import PydanticModelBackend
from ._factories import make_decision_request


_STREAM_TEXT = '{"contract":1}'
_OUTPUT_EVENT = Scalar(("contract",), 1)


@dataclass(frozen=True)
class DecisionTransportCase:
    """A transport paired with a valid completion and its decoded decision."""

    transport: DecisionTransport
    completion: LLMCompletion
    expected_data: StructuredData


class _Renderer(PromptRenderer):
    @override
    def render(self, prompt: DecisionPrompt) -> str:
        return "contract prompt"

    @override
    def render_tool_result(self, result: ToolCallResult) -> str:
        return "contract tool result"


class _Observer(DecisionObserver):
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.response_texts: list[str] = []
        self.reasoning_texts: list[str] = []
        self.output_events: list[OutputStreamEvent] = []

    @override
    async def before_request(self, prompt: str) -> None:
        self.prompts.append(prompt)

    @override
    async def response_text(self, text: str) -> None:
        self.response_texts.append(text)

    @override
    async def reasoning_text(self, text: str) -> None:
        self.reasoning_texts.append(text)

    @override
    async def output(self, event: OutputStreamEvent) -> None:
        self.output_events.append(event)


class _CompletionClient(LLMClient):
    def __init__(self, completion: LLMCompletion) -> None:
        self.completion = completion
        self.calls = 0

    @override
    async def complete(
        self,
        messages: list[Message],
        tools: list[StepTool] | None = None,
        decision_spec: DecisionSpec | None = None,
        stream_callback: Callable[[str], Coroutine[None, None, None]] | None = None,
        output_callback: OutputStreamCallback | None = None,
        reasoning_callback: Callable[[str], Coroutine[None, None, None]] | None = None,
    ) -> LLMCompletion:
        self.calls += 1
        if stream_callback is not None:
            await stream_callback(_STREAM_TEXT)
        if output_callback is not None:
            await output_callback(_OUTPUT_EVENT)
        if reasoning_callback is not None:
            await reasoning_callback("reasoning")
        return self.completion


def _request() -> DecisionRequest:
    return make_decision_request(
        DecisionSpec.for_inference(
            output_type=str,
            tools=[],
            result_format_factory=PydanticModelBackend(),
        ),
    )


class DecisionTransportContract:
    """Shared request, decoding, and observation behavior for transports."""

    async def test_returns_decoded_data_with_the_source_completion(
        self, decision_transport_case: DecisionTransportCase
    ) -> None:
        client = _CompletionClient(decision_transport_case.completion)
        observer = _Observer()

        decoded = await decision_transport_case.transport.request_decision(
            client, _Renderer(), _request(), observer, stream=False
        )

        assert decoded.decision_data == decision_transport_case.expected_data
        assert decoded.completion is decision_transport_case.completion
        assert observer.prompts == ["contract prompt"]
        assert observer.response_texts == []
        assert observer.reasoning_texts == []
        assert observer.output_events == []
        assert client.calls == 1

    async def test_connects_stream_observation_callbacks(
        self, decision_transport_case: DecisionTransportCase
    ) -> None:
        client = _CompletionClient(decision_transport_case.completion)
        observer = _Observer()

        await decision_transport_case.transport.request_decision(
            client, _Renderer(), _request(), observer, stream=True
        )

        assert observer.response_texts == [_STREAM_TEXT]
        assert observer.reasoning_texts == ["reasoning"]
        assert observer.output_events == [_OUTPUT_EVENT]


__all__ = ["DecisionTransportCase", "DecisionTransportContract"]
