import glyff

from sefia import Domain, Policy, Tools, preview
from sefia.event_system import EventHandler
from sefia.llm import JsonSnapshot, LLMCompletion, ToolCall
from sefia.llm.events import LLMReasoningTokenReceived, LLMTokenReceived
from sefia.llm.streaming import (
    StringDelta as OutputStringDelta,
    StringEnd as OutputStringEnd,
)
from sefia.llm.transports import NativeDecisionTransport, PromptedDecisionTransport
from sefia.streaming import ArgStream, StringDelta, StringEnd
from sefia.testing import (
    MockLLMClient,
    ScriptedCompletion,
    StreamContent,
    StreamOutput,
    StreamReasoning,
    memory_session,
    result_completion,
    tool_calls_completion,
)

infer = Domain(
    glyff.Domain("packages.sefia.tests.scenarios.test_mock_llm_streaming", version="1")
).infer


class StreamingAgent:
    @infer
    async def answer(self) -> str:
        """Return the final answer."""
        ...


class PreviewToolkit:
    def __init__(self) -> None:
        self.preview_call_ids: list[str] = []
        self.preview_events: list[StringDelta | StringEnd] = []

    async def ask(self, question: str) -> str:
        """Answer a question."""
        return question

    @preview(ask)
    async def _preview_ask(self, tool_call_id: str, stream: ArgStream) -> None:
        self.preview_call_ids.append(tool_call_id)
        async for event in stream:
            if isinstance(event, (StringDelta, StringEnd)):
                self.preview_events.append(event)


class PreviewAgent:
    _toolkit: Tools[PreviewToolkit]

    def __init__(self, toolkit: PreviewToolkit) -> None:
        self._toolkit = toolkit

    @infer
    async def run(self) -> str:
        """Ask the toolkit once, then return the final result."""
        ...


class TokenRecorder(EventHandler[LLMTokenReceived]):
    def __init__(self) -> None:
        self.tokens: list[str] = []

    async def handle(self, event: LLMTokenReceived) -> None:
        self.tokens.append(event.token)


class ReasoningRecorder(EventHandler[LLMReasoningTokenReceived]):
    def __init__(self) -> None:
        self.tokens: list[str] = []

    async def handle(self, event: LLMReasoningTokenReceived) -> None:
        self.tokens.append(event.token)


class PreviewProgressRecorder(EventHandler[LLMTokenReceived]):
    def __init__(self, toolkit: PreviewToolkit) -> None:
        self._toolkit = toolkit
        self.snapshots: list[tuple[str, tuple[StringDelta | StringEnd, ...]]] = []

    async def handle(self, event: LLMTokenReceived) -> None:
        self.snapshots.append((event.token, tuple(self._toolkit.preview_events)))


def _ask_output_events() -> tuple[StreamOutput, ...]:
    return (
        StreamOutput(OutputStringEnd(("tool_calls", 0, "name"), "PreviewToolkit_ask")),
        StreamOutput(
            OutputStringDelta(("tool_calls", 0, "arguments", "question"), "hello ")
        ),
        StreamOutput(
            OutputStringDelta(("tool_calls", 0, "arguments", "question"), "world")
        ),
        StreamOutput(
            OutputStringEnd(("tool_calls", 0, "arguments", "question"), "hello world")
        ),
    )


async def test_scripted_content_and_reasoning_publish_token_events() -> None:
    tokens = TokenRecorder()
    reasoning = ReasoningRecorder()
    client = MockLLMClient(
        [
            ScriptedCompletion(
                result_completion("done"),
                events=(
                    StreamReasoning("think"),
                    StreamReasoning("ing"),
                    StreamContent("do"),
                    StreamContent("ne"),
                ),
            )
        ]
    )

    async with memory_session(
        client,
        session_id="mock-content-stream",
        stream=True,
        policies=[Policy(handlers=lambda: [tokens, reasoning])],
    ):
        result = await StreamingAgent().answer()

    assert result == "done"
    assert tokens.tokens == ["do", "ne"]
    assert reasoning.tokens == ["think", "ing"]


async def test_scripted_output_events_run_preview_during_stream() -> None:
    toolkit = PreviewToolkit()
    progress = PreviewProgressRecorder(toolkit)
    output_events = _ask_output_events()
    client = MockLLMClient(
        [
            ScriptedCompletion(
                tool_calls_completion(
                    ("PreviewToolkit_ask", {"question": "hello world"})
                ),
                events=(
                    output_events[0],
                    output_events[1],
                    StreamContent("still generating"),
                    *output_events[2:],
                ),
            ),
            result_completion("done"),
        ]
    )

    async with memory_session(
        client,
        session_id="mock-output-stream",
        stream=True,
        policies=[Policy(handlers=lambda: [progress])],
    ):
        result = await PreviewAgent(toolkit).run()

    assert result == "done"
    assert len(toolkit.preview_call_ids) == 1
    assert toolkit.preview_call_ids[0].startswith("call_")
    assert progress.snapshots == [
        ("still generating", (StringDelta(name="question", text="hello "),))
    ]
    assert toolkit.preview_events == [
        StringDelta(name="question", text="hello "),
        StringDelta(name="question", text="world"),
        StringEnd(name="question", value="hello world"),
    ]


async def test_scripted_prompted_transport_streams_tool_arguments() -> None:
    toolkit = PreviewToolkit()
    first = tool_calls_completion(("PreviewToolkit_ask", {"question": "hello world"}))
    content = first.content
    assert content is not None
    client = MockLLMClient(
        [
            ScriptedCompletion(
                first,
                events=tuple(
                    StreamContent(content[i : i + 6])
                    for i in range(0, len(content), 6)
                ),
            ),
            result_completion("done"),
        ]
    )

    async with memory_session(
        client,
        session_id="mock-prompted-stream",
        stream=True,
        decision_transport=PromptedDecisionTransport(),
    ):
        result = await PreviewAgent(toolkit).run()

    assert result == "done"
    assert len(toolkit.preview_call_ids) == 1
    assert "".join(
        event.text for event in toolkit.preview_events if isinstance(event, StringDelta)
    ) == "hello world"
    assert StringEnd(name="question", value="hello world") in toolkit.preview_events


async def test_scripted_native_transport_streams_tool_arguments() -> None:
    toolkit = PreviewToolkit()
    client = MockLLMClient(
        [
            ScriptedCompletion(
                LLMCompletion(
                    tool_calls=[
                        ToolCall(
                            id="provider-call",
                            name="PreviewToolkit_ask",
                            arguments=JsonSnapshot.capture({"question": "hello world"}),
                        )
                    ]
                ),
                events=_ask_output_events(),
            ),
            LLMCompletion(
                tool_calls=[
                    ToolCall(
                        id="provider-result",
                        name="return_result",
                        arguments=JsonSnapshot.capture({"result": "done"}),
                    )
                ]
            ),
        ]
    )

    async with memory_session(
        client,
        session_id="mock-native-stream",
        stream=True,
        decision_transport=NativeDecisionTransport(),
    ):
        result = await PreviewAgent(toolkit).run()

    assert result == "done"
    assert len(toolkit.preview_call_ids) == 1
    assert toolkit.preview_events == [
        StringDelta(name="question", text="hello "),
        StringDelta(name="question", text="world"),
        StringEnd(name="question", value="hello world"),
    ]
