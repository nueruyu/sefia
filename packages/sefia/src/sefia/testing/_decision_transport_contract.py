"""Reusable pytest contract for ``DecisionTransport`` implementations."""

from collections.abc import Callable, Coroutine
from dataclasses import dataclass

from ..inference import FunctionInfo, ToolCallResult
from ..llm._client import LLMClient
from ..llm._messages import LLMCompletion, Message
from ..llm._prompt_renderer import DecisionPrompt, PromptRenderer
from ..llm.step_decision import DecisionSpec, StepTool
from ..llm.streaming import OutputStreamCallback, OutputStreamEvent
from ..llm.structured_data import StructuredData
from ..llm.transports import DecisionObserver, DecisionRequest, DecisionTransport
from ..pydantic import PydanticModelBackend


@dataclass(frozen=True)
class DecisionTransportCase:
    """A transport paired with a valid completion and its decoded decision."""

    transport: DecisionTransport
    completion: LLMCompletion
    expected_data: StructuredData


class _Renderer(PromptRenderer):
    def render(self, prompt: DecisionPrompt) -> str:
        return "contract prompt"

    def render_tool_result(self, result: ToolCallResult) -> str:
        return "contract tool result"


class _Observer(DecisionObserver):
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.response_texts: list[str] = []
        self.reasoning_texts: list[str] = []

    async def before_request(self, prompt: str) -> None:
        self.prompts.append(prompt)

    async def response_text(self, text: str) -> None:
        self.response_texts.append(text)

    async def reasoning_text(self, text: str) -> None:
        self.reasoning_texts.append(text)

    async def output(self, event: OutputStreamEvent) -> None:
        pass


class _CompletionClient(LLMClient):
    def __init__(self, completion: LLMCompletion) -> None:
        self.completion = completion
        self.calls = 0

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
            await stream_callback("token")
        if reasoning_callback is not None:
            await reasoning_callback("reasoning")
        return self.completion


def _request() -> DecisionRequest:
    return DecisionRequest(
        function=FunctionInfo(
            qualname="answer",
            instructions="Return the answer.",
            bound_arguments={},
            type_hints={},
            return_type=str,
            args=(),
            kwargs={},
        ),
        decision_spec=DecisionSpec.for_inference(
            output_type=str,
            tools=[],
            result_format_factory=PydanticModelBackend(),
        ),
        history=(),
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
        assert client.calls == 1

    async def test_connects_stream_observation_callbacks(
        self, decision_transport_case: DecisionTransportCase
    ) -> None:
        client = _CompletionClient(decision_transport_case.completion)
        observer = _Observer()

        await decision_transport_case.transport.request_decision(
            client, _Renderer(), _request(), observer, stream=True
        )

        assert observer.response_texts == ["token"]
        assert observer.reasoning_texts == ["reasoning"]


__all__ = ["DecisionTransportCase", "DecisionTransportContract"]
