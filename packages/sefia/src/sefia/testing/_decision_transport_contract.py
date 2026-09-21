"""Reusable pytest contract for ``DecisionTransport`` implementations."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass

from typing_extensions import override

from ..llm._client import LLMClient
from ..llm._messages import LLMCompletion, Message
from ..llm._prompt_renderer import InferencePrompt, PromptRenderer
from ..llm.step_decision import DecisionSpec, StepTool
from ..llm.streaming import (
    OutputStreamCallback,
    OutputStreamEvent,
)
from ..llm.structured_data import StructuredData
from ..llm.transports import DecisionObserver, DecisionRequest, DecisionTransport


@dataclass(frozen=True)
class DecisionTransportCase:
    """A transport paired with a valid completion and its decoded decision."""

    transport: DecisionTransport
    completion: LLMCompletion
    expected_data: StructuredData
    request: DecisionRequest
    content_chunks: Sequence[str] = ()
    reasoning_chunks: Sequence[str] = ()
    client_output_events: Sequence[OutputStreamEvent] = ()
    expected_output_events: Sequence[OutputStreamEvent] = ()


class _Renderer(PromptRenderer):
    @override
    def render(self, prompt: InferencePrompt) -> str:
        return "contract prompt"


class _Observer(DecisionObserver):
    def __init__(self) -> None:
        self.requests: list[tuple[Message, ...]] = []
        self.response_texts: list[str] = []
        self.reasoning_texts: list[str] = []
        self.output_events: list[OutputStreamEvent] = []

    @override
    async def before_request(self, messages: tuple[Message, ...]) -> None:
        self.requests.append(messages)

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
    def __init__(self, case: DecisionTransportCase) -> None:
        self.case = case
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
        if reasoning_callback is not None:
            for chunk in self.case.reasoning_chunks:
                await reasoning_callback(chunk)
        if stream_callback is not None:
            for chunk in self.case.content_chunks:
                await stream_callback(chunk)
        if output_callback is not None:
            for event in self.case.client_output_events:
                await output_callback(event)
        return self.case.completion


class DecisionTransportContract(ABC):
    """Shared request, decoding, and observation behavior for transports."""

    @abstractmethod
    def make_decision_transport_case(self) -> DecisionTransportCase:
        """Return a matching request, completion, and stream script."""
        ...

    async def test_returns_decoded_data_with_the_source_completion(self) -> None:
        decision_transport_case = self.make_decision_transport_case()
        client = _CompletionClient(decision_transport_case)
        observer = _Observer()

        decoded = await decision_transport_case.transport.request_decision(
            client, _Renderer(), decision_transport_case.request, observer, stream=False
        )

        assert decoded.decision_data == decision_transport_case.expected_data
        assert decoded.completion is decision_transport_case.completion
        assert len(observer.requests) == 1
        assert len(observer.requests[0]) == 1
        assert observer.requests[0][0].role == "user"
        content = observer.requests[0][0].content
        assert isinstance(content, str)
        assert content.startswith("contract prompt\n\n## Response\n\n")
        assert observer.response_texts == []
        assert observer.reasoning_texts == []
        assert observer.output_events == []
        assert client.calls == 1

    async def test_connects_stream_observation_callbacks(self) -> None:
        decision_transport_case = self.make_decision_transport_case()
        client = _CompletionClient(decision_transport_case)
        observer = _Observer()

        decoded = await decision_transport_case.transport.request_decision(
            client, _Renderer(), decision_transport_case.request, observer, stream=True
        )

        assert decoded.decision_data == decision_transport_case.expected_data
        assert decoded.completion is decision_transport_case.completion
        assert len(observer.requests) == 1
        assert len(observer.requests[0]) == 1
        assert observer.requests[0][0].role == "user"
        content = observer.requests[0][0].content
        assert isinstance(content, str)
        assert content.startswith("contract prompt\n\n## Response\n\n")
        assert observer.response_texts == list(decision_transport_case.content_chunks)
        assert observer.reasoning_texts == list(
            decision_transport_case.reasoning_chunks
        )
        assert observer.output_events == list(
            decision_transport_case.expected_output_events
        )
        assert client.calls == 1


__all__ = ["DecisionTransportCase", "DecisionTransportContract"]
