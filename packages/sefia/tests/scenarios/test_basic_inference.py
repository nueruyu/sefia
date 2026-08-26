import glyff
import sefia
from dataclasses import dataclass

import pytest

from sefia import Policy, Tools, policy
from sefia._authoring.metadata import get_metadata
from sefia.exceptions import InvalidInferenceResponseError, UnknownToolDecisionError
from sefia.llm import LLMResponse
from sefia.llm.step_decision import StepDecisionMode, StepDecisionModel
from sefia.testing import (
    MockLLMClient,
    memory_session,
    result_response,
    tool_calls_response,
)

infer = sefia.Domain(
    glyff.Domain("packages.sefia.tests.scenarios.test_basic_inference", version="1")
).infer


@dataclass
class SearchResult:
    title: str
    url: str


@dataclass
class WebToolkit:
    """A simple toolkit for web operations."""

    async def search(self, query: str) -> list[SearchResult]:
        """Search the web for a query."""
        if query == "sefia":
            return [
                SearchResult(title="sefia framework", url="https://example.com/sefia")
            ]
        return []

    async def fetch_content(self, url: str) -> str:
        """Fetch content from a URL."""
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


@dataclass
class _PolicyFixture(Policy):
    count: int


async def test_inference_with_tool_calls():
    mock_llm = MockLLMClient(
        responses=[
            # 1. LLM decides to search
            tool_calls_response(("WebToolkit_search", {"query": "sefia"})),
            # 2. LLM decides to fetch content based on search result
            tool_calls_response(
                ("WebToolkit_fetch_content", {"url": "https://example.com/sefia"})
            ),
            # 3. LLM generates the final report
            result_response(
                Report(
                    topic="sefia",
                    summary="Sefia is a framework for building LLM agents.",
                    sources=["https://example.com/sefia"],
                )
            ),
        ]
    )

    async with memory_session(mock_llm, session_id="basic-inference-tools"):
        researcher = Researcher(WebToolkit())
        report = await researcher.generate_report(topic="sefia")

    assert isinstance(report, Report)
    assert report.topic == "sefia"
    assert "framework" in report.summary
    assert report.sources == ["https://example.com/sefia"]

    assert len(mock_llm.requests) == 3
    final_messages = mock_llm.requests[2]["messages"]
    assert any(
        "sefia framework" in str(message["content"]) for message in final_messages
    )


async def test_inference_without_tool_calls():
    # Scenario: The LLM can generate the output in a single step.
    mock_llm = MockLLMClient(
        responses=[
            result_response(
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
    assert len(mock_llm.requests) == 1


async def test_inference_with_tool_exception():
    # Scenario: A tool fails, and the error is reported back to the LLM.
    mock_llm = MockLLMClient(
        responses=[
            tool_calls_response(("BrokenToolkit_always_fail", {"reason": "test"})),
            # LLM receives the error and generates a final report about the failure.
            result_response(
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
    assert len(mock_llm.requests) == 2
    messages = mock_llm.requests[1]["messages"]
    history = "\n".join(str(message["content"]) for message in messages)
    assert "Error executing tool" in history
    assert "ValueError(Failed because: test)" in history


async def test_inference_with_nonexistent_tool_call():
    # Scenario: LLM calls a tool that does not exist. This is a recoverable
    # malformed decision, not an executor-level tool failure.
    mock_llm = MockLLMClient(responses=[tool_calls_response(("NonExistent_tool", {}))])

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
        responses=[result_response({"summary": "This is missing the topic field."})]
    )

    # Repair is disabled so the propagation path itself is under test.
    async with memory_session(
        mock_llm, session_id="invalid-schema-test", max_repair_attempts=0
    ):
        agent = SimpleAgent()
        with pytest.raises(
            InvalidInferenceResponseError, match="LLM output failed validation"
        ):
            await agent.generate_report(topic="invalid")


async def test_invalid_response_is_repaired_with_feedback():
    # Scenario: the LLM first returns an empty body (the gemini-2.5-flash-lite
    # failure from issue #35), then a valid decision once the validation error
    # is fed back. The run succeeds without ever surfacing an error.
    mock_llm = MockLLMClient(
        responses=[
            LLMResponse(content=""),
            result_response(Report(topic="sefia", summary="Repaired.", sources=[])),
        ]
    )

    async with memory_session(mock_llm, session_id="repair-test"):
        agent = SimpleAgent()
        report = await agent.generate_report(topic="sefia")

    assert report == Report(topic="sefia", summary="Repaired.", sources=[])
    assert len(mock_llm.requests) == 2
    feedback = mock_llm.requests[1]["messages"][-1]
    assert feedback["role"] == "user"
    assert "invalid" in feedback["content"]
    assert "Your previous response was empty." in feedback["content"]


async def test_inference_on_standalone_function():
    """Tests that @infer works correctly on a standalone function without any tools."""

    @infer
    async def summarize_text(text: str, length: int) -> str:
        """Summarize the given text to the specified length in sentences."""
        ...

    mock_llm = MockLLMClient(responses=[result_response("This is a summary.")])

    async with memory_session(mock_llm, session_id="standalone-function-test"):
        summary = await summarize_text(text="This is a long text...", length=1)

    assert summary == "This is a summary."
    assert len(mock_llm.requests) == 1
    decision_model = mock_llm.requests[0].get("decision_model")
    assert isinstance(decision_model, StepDecisionModel)
    assert decision_model.mode is StepDecisionMode.RESULT_ONLY
    assert decision_model.tools == ()


def test_policy_attaches_metadata():
    """`@policy` records its policy under the metadata "policies" key, no matter
    where it sits relative to @infer."""

    @infer
    @policy(_PolicyFixture(count=3))
    async def below(value: int) -> int:
        """Policy applied below @infer."""
        ...

    @policy(_PolicyFixture(count=3))
    @infer
    async def above(value: int) -> int:
        """Policy applied above @infer."""
        ...

    for fn in (below, above):
        policies = get_metadata(fn)["policies"]
        assert len(policies) == 1
        assert isinstance(policies[0], _PolicyFixture)


def test_policy_coexists_with_other_metadata():
    """A non-policies entry in the metadata must not hide a policy attached above
    @infer — regression for the metadata-present-but-no-policies-key bug."""

    async def fn(value: int) -> int:
        """A function whose metadata was already touched by another decorator."""
        ...

    setattr(fn, "__sefia_metadata__", {"other": True})

    # @policy sits above @infer, so the policy lands on the wrapper chain.
    decorated = policy(_PolicyFixture(count=2))(infer(fn))

    metadata = get_metadata(decorated)
    assert metadata.get("other") is True
    assert [type(p) for p in metadata.get("policies", [])] == [_PolicyFixture]


def test_policy_rejects_non_policy():
    """@policy raises a clear error when given a non-Policy (e.g. the class
    itself instead of an instance)."""
    with pytest.raises(TypeError):
        policy(_PolicyFixture)  # type: ignore
