"""Reusable pytest contracts for ``LLMClient`` implementations."""

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..llm import LLMClient, LLMCompletion, Message
from ..llm.step_decision import DecisionSpec, StepTool
from ..llm.streaming import OutputStreamEvent


@dataclass(frozen=True)
class LLMClientCase:
    """A configured client request and its normalized completion."""

    client: LLMClient
    expected_completion: LLMCompletion
    messages: Sequence[Message] = field(
        default_factory=lambda: (Message(role="user", content="Hello"),)
    )
    tools: Sequence[StepTool] | None = None
    decision_spec: DecisionSpec | None = None


@dataclass(frozen=True)
class StreamingLLMClientCase(LLMClientCase):
    """A streaming request and the callback values it must emit."""

    content_chunks: Sequence[str] = ()
    reasoning_chunks: Sequence[str] = ()
    output_events: Sequence[OutputStreamEvent] = ()


class LLMClientContract:
    """Shared normalized-completion behavior required by every LLM client."""

    async def test_returns_the_normalized_completion(
        self, llm_client_case: LLMClientCase
    ) -> None:
        completion = await llm_client_case.client.complete(
            list(llm_client_case.messages),
            tools=(
                list(llm_client_case.tools)
                if llm_client_case.tools is not None
                else None
            ),
            decision_spec=llm_client_case.decision_spec,
        )

        assert completion == llm_client_case.expected_completion


class StreamingLLMClientContract:
    """Callback and reconstruction behavior for clients supporting streaming."""

    async def test_streams_callbacks_and_returns_the_reconstructed_completion(
        self, streaming_llm_client_case: StreamingLLMClientCase
    ) -> None:
        content_chunks: list[str] = []
        reasoning_chunks: list[str] = []
        output_events: list[OutputStreamEvent] = []

        async def on_content(text: str) -> None:
            content_chunks.append(text)

        async def on_reasoning(text: str) -> None:
            reasoning_chunks.append(text)

        async def on_output(event: OutputStreamEvent) -> None:
            output_events.append(event)

        completion = await streaming_llm_client_case.client.complete(
            list(streaming_llm_client_case.messages),
            tools=(
                list(streaming_llm_client_case.tools)
                if streaming_llm_client_case.tools is not None
                else None
            ),
            decision_spec=streaming_llm_client_case.decision_spec,
            stream_callback=on_content,
            output_callback=on_output,
            reasoning_callback=on_reasoning,
        )

        assert completion == streaming_llm_client_case.expected_completion
        assert content_chunks == list(streaming_llm_client_case.content_chunks)
        assert reasoning_chunks == list(streaming_llm_client_case.reasoning_chunks)
        assert output_events == list(streaming_llm_client_case.output_events)


__all__ = [
    "LLMClientCase",
    "LLMClientContract",
    "StreamingLLMClientCase",
    "StreamingLLMClientContract",
]
