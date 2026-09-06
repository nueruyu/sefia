from dataclasses import dataclass

import glyff
import pytest
import sefia

from sefia import Tools
from sefia.exceptions import InvalidInferenceResponseError, UnknownToolDecisionError
from sefia.llm import LLMCompletion
from sefia.testing import (
    MockLLMClient,
    memory_session,
    result_completion,
    tool_calls_completion,
)

infer = sefia.Domain(
    glyff.Domain("packages.sefia.tests.scenarios.test_basic_inference", version="1")
).infer


@dataclass
class SearchResult:
    title: str
    url: str


class WebToolkit:
    """A simple toolkit for web operations."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def search(self, query: str) -> list[SearchResult]:
        """Search the web for a query."""
        self.calls.append(("search", query))
        if query == "sefia":
            return [
                SearchResult(title="sefia framework", url="https://example.com/sefia")
            ]
        return []

    async def fetch_content(self, url: str) -> str:
        """Fetch content from a URL."""
        self.calls.append(("fetch_content", url))
        if url == "https://example.com/sefia":
            return "Sefia is a framework for building LLM agents."
        return "Not found."


@dataclass
class Report:
    topic: str
    summary: str
    sources: list[str]


class Researcher:
    """An agent that uses WebToolkit to research topics."""

    _web: Tools[WebToolkit]

    def __init__(self, web: WebToolkit):
        self._web = web

    @infer
    async def generate_report(self, topic: str) -> Report:
        """
        Generate a report on the given topic by searching the web,
        fetching content, and summarizing it.
        """
        ...


class BrokenToolkit:
    """A toolkit where tools can fail."""

    async def always_fail(self, reason: str) -> None:
        """This tool always raises an exception."""
        raise ValueError(f"Failed because: {reason}")


class SimpleAgent:
    """An agent that has no tools."""

    @infer
    async def generate_report(self, topic: str) -> Report:
        """
        Generate a report on the given topic.
        This agent has no tools and must produce the report directly.
        """
        ...


async def test_inference_with_tool_calls():
    mock_llm = MockLLMClient(
        completions=[
            # 1. LLM decides to search
            tool_calls_completion(("WebToolkit_search", {"query": "sefia"})),
            # 2. LLM decides to fetch content based on search result
            tool_calls_completion(
                ("WebToolkit_fetch_content", {"url": "https://example.com/sefia"})
            ),
            # 3. LLM generates the final report
            result_completion(
                Report(
                    topic="sefia",
                    summary="Sefia is a framework for building LLM agents.",
                    sources=["https://example.com/sefia"],
                )
            ),
        ]
    )

    toolkit = WebToolkit()
    async with memory_session(mock_llm, session_id="basic-inference-tools"):
        researcher = Researcher(toolkit)
        report = await researcher.generate_report(topic="sefia")

    assert isinstance(report, Report)
    assert report.topic == "sefia"
    assert "framework" in report.summary
    assert report.sources == ["https://example.com/sefia"]

    assert toolkit.calls == [
        ("search", "sefia"),
        ("fetch_content", "https://example.com/sefia"),
    ]


async def test_inference_without_tool_calls():
    # Scenario: The LLM can generate the output in a single step.
    mock_llm = MockLLMClient(
        completions=[
            result_completion(
                Report(topic="direct", summary="This is a direct answer.", sources=[])
            )
        ]
    )

    async with memory_session(mock_llm, session_id="basic-inference-no-tools"):
        agent = SimpleAgent()
        report = await agent.generate_report(topic="direct")

    assert isinstance(report, Report)
    assert report.topic == "direct"
    assert "direct answer" in report.summary


async def test_inference_with_tool_exception():
    # Scenario: A tool fails, and the error is reported back to the LLM.
    mock_llm = MockLLMClient(
        completions=[
            tool_calls_completion(("BrokenToolkit_always_fail", {"reason": "test"})),
            # LLM receives the error and generates a final report about the failure.
            result_completion(
                Report(
                    topic="failure",
                    summary="The tool failed with a ValueError.",
                    sources=["error: ValueError(Failed because: test)"],
                )
            ),
        ]
    )

    class AgentWithBrokenTool:
        _kit: Tools[BrokenToolkit]

        def __init__(self, kit: BrokenToolkit):
            self._kit = kit

        @infer
        async def run_and_report(self) -> Report:
            """Run a tool and report on the outcome."""
            ...

    async with memory_session(mock_llm, session_id="tool-exception-test"):
        agent = AgentWithBrokenTool(BrokenToolkit())
        report = await agent.run_and_report()

    assert report.topic == "failure"
    assert "tool failed" in report.summary


async def test_inference_with_nonexistent_tool_call():
    # Scenario: LLM calls a tool that does not exist. This is a recoverable
    # malformed decision, not an executor-level tool failure.
    mock_llm = MockLLMClient(
        completions=[tool_calls_completion(("NonExistent_tool", {}))]
    )

    # Repair is disabled so the propagation path itself is under test.
    async with memory_session(
        mock_llm, session_id="nonexistent-tool-test", max_repair_attempts=0
    ):
        researcher = Researcher(WebToolkit())
        with pytest.raises(InvalidInferenceResponseError) as exc_info:
            await researcher.generate_report(topic="sefia")

    assert isinstance(exc_info.value.__cause__, UnknownToolDecisionError)
    assert exc_info.value.__cause__.tool_name == "NonExistent_tool"
    assert len(mock_llm.requests) == 1


async def test_inference_with_invalid_decision_model():
    # Scenario: The LLM returns a result that doesn't match the schema.
    # An invalid response is recoverable, so it is NOT engraved as a permanent
    # failure: it surfaces as an InvalidInferenceResponseError (a PauseException),
    # leaving the step resumable on re-invocation.
    mock_llm = MockLLMClient(
        completions=[result_completion({"summary": "This is missing the topic field."})]
    )

    # Repair is disabled so the propagation path itself is under test.
    async with memory_session(
        mock_llm, session_id="invalid-schema-test", max_repair_attempts=0
    ):
        agent = SimpleAgent()
        with pytest.raises(
            InvalidInferenceResponseError, match="LLM decision failed validation"
        ):
            await agent.generate_report(topic="invalid")


async def test_invalid_response_is_repaired_with_feedback():
    # Scenario: the LLM first returns an empty body (the gemini-2.5-flash-lite
    # failure from issue #35), then a valid decision once the validation error
    # is fed back. The run succeeds without ever surfacing an error.
    mock_llm = MockLLMClient(
        completions=[
            LLMCompletion(content=""),
            result_completion(Report(topic="sefia", summary="Repaired.", sources=[])),
        ]
    )

    async with memory_session(mock_llm, session_id="repair-test"):
        agent = SimpleAgent()
        report = await agent.generate_report(topic="sefia")

    assert report == Report(topic="sefia", summary="Repaired.", sources=[])
    assert len(mock_llm.requests) == 2


async def test_inference_on_standalone_function():
    """Tests that @infer works correctly on a standalone function without any tools."""

    @infer
    async def summarize_text(text: str, length: int) -> str:
        """Summarize the given text to the specified length in sentences."""
        ...

    mock_llm = MockLLMClient(completions=[result_completion("This is a summary.")])

    async with memory_session(mock_llm, session_id="standalone-function-test"):
        summary = await summarize_text(text="This is a long text...", length=1)

    assert summary == "This is a summary."
