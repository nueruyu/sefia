import json
from dataclasses import dataclass

import glyff
import pytest
from glyff import ArgsHasher, Serializer
from glyff.store import MemoryClient
from glyff.store import MemorySessionStore as GlyffMemoryStore

from sefia import Policy, Session, infer, policy
from sefia._metadata import get_metadata
from sefia.exceptions import InvalidInferenceResponseError, UnknownToolDecisionError
from sefia.llm import LLMResponse
from sefia.stores import MemorySessionStore as SefiaMemoryStore

from ..conftest import (
    BrokenToolkit,
    MockLLMClient,
    Report,
    Researcher,
    SimpleAgent,
    WebToolkit,
)


def _make_stores(serializer):
    client = MemoryClient()
    return (
        GlyffMemoryStore(client=client, serializer=serializer),
        SefiaMemoryStore(client=client, serializer=serializer),
    )


@dataclass
class _PolicyFixture(Policy):
    count: int


async def test_inference_with_tool_calls(
    web_toolkit: WebToolkit, serializer: Serializer, hasher: ArgsHasher
):
    mock_responses = [
        # 1. LLM decides to search
        LLMResponse(
            content=json.dumps(
                {
                    "decision": "tool_calls",
                    "tool_calls": [
                        {
                            "name": "WebToolkit_search",
                            "arguments": {"query": "sefia"},
                        }
                    ],
                }
            )
        ),
        # 2. LLM decides to fetch content based on search result
        LLMResponse(
            content=json.dumps(
                {
                    "decision": "tool_calls",
                    "tool_calls": [
                        {
                            "name": "WebToolkit_fetch_content",
                            "arguments": {"url": "https://example.com/sefia"},
                        }
                    ],
                }
            )
        ),
        # 3. LLM generates the final report
        LLMResponse(
            content=json.dumps(
                {
                    "decision": "final_answer",
                    "final_answer": {
                        "topic": "sefia",
                        "summary": "Sefia is a framework for building LLM agents.",
                        "sources": ["https://example.com/sefia"],
                    },
                }
            )
        ),
    ]

    mock_llm = MockLLMClient(responses=mock_responses)
    session_id = "basic-inference-tools"
    glyff_store, sefia_store = _make_stores(serializer)

    async with glyff.Session(id=session_id, store=glyff_store, hasher=hasher) as gs:
        async with Session(
            llm_client=mock_llm, glyff_session=gs, session_store=sefia_store
        ):
            researcher = Researcher(web_toolkit)
            report = await researcher.generate_report(topic="sefia")

    assert isinstance(report, Report)
    assert report.topic == "sefia"
    assert "framework" in report.summary
    assert report.sources == ["https://example.com/sefia"]

    # LLM was called 3 times (search, fetch, final output)
    assert len(mock_llm.requests) == 3

    # 3rd call receives 6 messages: system, user, assistant(search), tool(search result),
    # assistant(fetch), tool(fetch result)
    final_messages = mock_llm.requests[2]["messages"]
    assert len(final_messages) == 6
    assert final_messages[3]["role"] == "tool"
    assert "sefia framework" in final_messages[3]["content"]


async def test_inference_without_tool_calls(serializer: Serializer, hasher: ArgsHasher):
    # Scenario: The LLM can generate the output in a single step.
    mock_response = LLMResponse(
        content=json.dumps(
            {
                "decision": "final_answer",
                "final_answer": {
                    "topic": "direct",
                    "summary": "This is a direct answer.",
                    "sources": [],
                },
            }
        )
    )
    mock_llm = MockLLMClient(responses=[mock_response])
    session_id = "basic-inference-no-tools"
    glyff_store, sefia_store = _make_stores(serializer)

    async with glyff.Session(id=session_id, store=glyff_store, hasher=hasher) as gs:
        async with Session(
            llm_client=mock_llm, glyff_session=gs, session_store=sefia_store
        ):
            agent = SimpleAgent()
            report = await agent.generate_report(topic="direct")

    assert isinstance(report, Report)
    assert report.topic == "direct"
    assert "direct answer" in report.summary
    assert len(mock_llm.requests) == 1


async def test_inference_with_tool_exception(
    broken_toolkit: BrokenToolkit, serializer: Serializer, hasher: ArgsHasher
):
    # Scenario: A tool fails, and the error is reported back to the LLM.
    mock_responses = [
        LLMResponse(
            content=json.dumps(
                {
                    "decision": "tool_calls",
                    "tool_calls": [
                        {
                            "name": "BrokenToolkit_always_fail",
                            "arguments": {"reason": "test"},
                        }
                    ],
                }
            )
        ),
        # LLM receives the error and generates a final report about the failure.
        LLMResponse(
            content=json.dumps(
                {
                    "decision": "final_answer",
                    "final_answer": {
                        "topic": "failure",
                        "summary": "The tool failed with a ValueError.",
                        "sources": ["error: ValueError(Failed because: test)"],
                    },
                }
            )
        ),
    ]
    mock_llm = MockLLMClient(responses=mock_responses)
    session_id = "tool-exception-test"
    glyff_store, sefia_store = _make_stores(serializer)

    @dataclass
    class AgentWithBrokenTool:
        def __init__(self, kit: BrokenToolkit):
            self._kit = kit

        @infer
        async def run_and_report(self) -> Report:
            """Run a tool and report on the outcome."""
            ...

    async with glyff.Session(id=session_id, store=glyff_store, hasher=hasher) as gs:
        async with Session(
            llm_client=mock_llm, glyff_session=gs, session_store=sefia_store
        ):
            agent = AgentWithBrokenTool(broken_toolkit)
            report = await agent.run_and_report()

    assert report.topic == "failure"
    assert "tool failed" in report.summary
    assert len(mock_llm.requests) == 2
    # Check that the tool error was passed to the second LLM call
    messages = mock_llm.requests[1]["messages"]
    assert len(messages) == 4  # system, user, assistant, tool
    assert messages[3]["role"] == "tool"
    assert "Error executing tool" in json.loads(messages[3]["content"])
    assert "ValueError(Failed because: test)" in json.loads(messages[3]["content"])


async def test_inference_with_nonexistent_tool_call(
    web_toolkit: WebToolkit, serializer: Serializer, hasher: ArgsHasher
):
    # Scenario: LLM calls a tool that does not exist. This is a recoverable
    # malformed decision, not an executor-level tool failure.
    mock_responses = [
        LLMResponse(
            content=json.dumps(
                {
                    "decision": "tool_calls",
                    "tool_calls": [
                        {
                            "name": "NonExistent_tool",
                            "arguments": {},
                        }
                    ],
                }
            )
        ),
    ]
    mock_llm = MockLLMClient(responses=mock_responses)
    session_id = "nonexistent-tool-test"
    glyff_store, sefia_store = _make_stores(serializer)

    async with glyff.Session(id=session_id, store=glyff_store, hasher=hasher) as gs:
        async with Session(
            llm_client=mock_llm, glyff_session=gs, session_store=sefia_store
        ):
            researcher = Researcher(web_toolkit)
            with pytest.raises(InvalidInferenceResponseError) as exc_info:
                await researcher.generate_report(topic="sefia")

    assert isinstance(exc_info.value.__cause__, UnknownToolDecisionError)
    assert exc_info.value.__cause__.tool_name == "NonExistent_tool"
    assert len(mock_llm.requests) == 1


async def test_inference_with_invalid_output_schema(
    serializer: Serializer, hasher: ArgsHasher
):
    # Scenario: The LLM returns a final_answer that doesn't match the schema.
    # An invalid response is recoverable, so it is NOT engraved as a permanent
    # failure: it surfaces as an InvalidInferenceResponseError (a YieldException),
    # leaving the step resumable on re-invocation.
    mock_response = LLMResponse(
        content=json.dumps(
            {
                "decision": "final_answer",
                "final_answer": {"summary": "This is missing the topic field."},
            }
        )
    )
    mock_llm = MockLLMClient(responses=[mock_response])
    session_id = "invalid-schema-test"
    glyff_store, sefia_store = _make_stores(serializer)

    async with glyff.Session(id=session_id, store=glyff_store, hasher=hasher) as gs:
        async with Session(
            llm_client=mock_llm, glyff_session=gs, session_store=sefia_store
        ):
            agent = SimpleAgent()
            with pytest.raises(
                InvalidInferenceResponseError, match="LLM output failed validation"
            ):
                await agent.generate_report(topic="invalid")


async def test_inference_on_standalone_function(
    serializer: Serializer, hasher: ArgsHasher
):
    """Tests that @infer works correctly on a standalone function without any tools."""

    @infer
    async def summarize_text(text: str, length: int) -> str:
        """Summarize the given text to the specified length in sentences."""
        ...

    mock_response = LLMResponse(
        content=json.dumps(
            {"decision": "final_answer", "final_answer": "This is a summary."}
        )
    )
    mock_llm = MockLLMClient(responses=[mock_response])
    session_id = "standalone-function-test"
    glyff_store, sefia_store = _make_stores(serializer)

    async with glyff.Session(id=session_id, store=glyff_store, hasher=hasher) as gs:
        async with Session(
            llm_client=mock_llm, glyff_session=gs, session_store=sefia_store
        ):
            summary = await summarize_text(text="This is a long text...", length=1)

    assert summary == "This is a summary."
    assert len(mock_llm.requests) == 1
    # Check that the schema passed to the LLM does not include the optional `tool_calls`
    output_schema = mock_llm.requests[0].get("output_schema")
    assert output_schema is not None
    assert "tool_calls" not in output_schema.get("properties", {})


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
