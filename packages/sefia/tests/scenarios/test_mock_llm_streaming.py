import glyff

from sefia import Domain, Policy, Tools, preview
from sefia.event_system import EventHandler
from sefia.llm.events import LLMTokenReceived
from sefia.llm.streaming import (
    StringDelta as OutputStringDelta,
    StringEnd as OutputStringEnd,
)
from sefia.streaming import ArgStream, StringDelta, StringEnd
from sefia.testing import (
    MockLLMClient,
    ScriptedCompletion,
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
    async def preview_ask(self, tool_call_id: str, stream: ArgStream) -> None:
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


async def test_scripted_content_chunks_publish_llm_token_events() -> None:
    recorder = TokenRecorder()
    client = MockLLMClient(
        [
            ScriptedCompletion(
                result_completion("done"),
                content_chunks=("do", "ne"),
            )
        ]
    )

    async with memory_session(
        client,
        session_id="mock-content-stream",
        stream=True,
        policies=[Policy(handlers=lambda: [recorder])],
    ):
        result = await StreamingAgent().answer()

    assert result == "done"
    assert recorder.tokens == ["do", "ne"]


async def test_scripted_output_events_reach_tool_preview() -> None:
    toolkit = PreviewToolkit()
    client = MockLLMClient(
        [
            ScriptedCompletion(
                tool_calls_completion(
                    ("PreviewToolkit_ask", {"question": "hello world"})
                ),
                output_events=(
                    OutputStringEnd(
                        ("tool_calls", 0, "name"),
                        "PreviewToolkit_ask",
                    ),
                    OutputStringDelta(
                        ("tool_calls", 0, "arguments", "question"),
                        "hello ",
                    ),
                    OutputStringDelta(
                        ("tool_calls", 0, "arguments", "question"),
                        "world",
                    ),
                    OutputStringEnd(
                        ("tool_calls", 0, "arguments", "question"),
                        "hello world",
                    ),
                ),
            ),
            result_completion("done"),
        ]
    )

    async with memory_session(
        client,
        session_id="mock-output-stream",
        stream=True,
    ):
        result = await PreviewAgent(toolkit).run()

    assert result == "done"
    assert len(toolkit.preview_call_ids) == 1
    assert toolkit.preview_call_ids[0].startswith("call_")
    assert toolkit.preview_events == [
        StringDelta(name="question", text="hello "),
        StringDelta(name="question", text="world"),
        StringEnd(name="question", value="hello world"),
    ]
