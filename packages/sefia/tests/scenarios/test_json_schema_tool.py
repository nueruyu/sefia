import glyff
import sefia
from dataclasses import dataclass

from sefia import JsonSchemaToolEntry
from sefia.testing import (
    MockLLMClient,
    memory_session,
    result_completion,
    tool_calls_completion,
)
from sefia.tool_collectors import StaticToolCollector

infer = sefia.Domain(
    glyff.Domain("packages.sefia.tests.scenarios.test_json_schema_tool", version="1")
).infer

_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}


@dataclass
class Report:
    topic: str
    summary: str
    sources: list[str]


class SimpleAgent:
    """An agent that has no tools of its own."""

    @infer
    async def generate_report(self, topic: str) -> Report:
        """Generate a report on the given topic."""
        ...


async def test_json_schema_tool_reaches_the_llm_and_is_dispatched():
    """A tool registered from a raw JSON Schema (no Python signature) is exposed
    to the model and, when called, dispatched to its handler with the decoded
    arguments — issue #38 end to end."""

    calls: list[dict[str, str]] = []

    async def search_handler(query: str) -> str:
        calls.append({"query": query})
        return "Sefia is a framework for building LLM agents."

    search_tool = JsonSchemaToolEntry(
        search_handler,
        name="search",
        parameters=_SEARCH_SCHEMA,
        description="Search the corpus for a query.",
    )

    mock_llm = MockLLMClient(
        completions=[
            tool_calls_completion(("search", {"query": "sefia"})),
            result_completion(
                Report(
                    topic="sefia",
                    summary="Sefia is a framework for building LLM agents.",
                    sources=["search"],
                )
            ),
        ]
    )

    async with memory_session(
        mock_llm,
        session_id="json-schema-tool",
        tool_collector=StaticToolCollector([search_tool]),
    ):
        report = await SimpleAgent().generate_report(topic="sefia")

    assert isinstance(report, Report)
    assert report.summary == "Sefia is a framework for building LLM agents."

    # The handler was dispatched with the decoded arguments.
    assert calls == [{"query": "sefia"}]

    decision_spec = mock_llm.requests[0]["decision_spec"]
    assert decision_spec is not None
    tool = decision_spec.tools[0]
    assert tool.name == "search"
    assert tool.description == "Search the corpus for a query."
    assert tool.arguments.to_dict() == _SEARCH_SCHEMA

    history = "\n".join(
        str(message["content"]) for message in mock_llm.requests[1]["messages"]
    )
    assert "framework" in history
