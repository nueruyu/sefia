import json

import glyff
from glyff import ArgsHasher, Serializer
from glyff.store import MemoryBackend

from sefia import JsonSchemaTool, Session
from sefia.llm import LLMResponse
from sefia.tool_collectors import StaticToolCollector

from ..conftest import MockLLMClient, Report, SimpleAgent

_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}


async def test_json_schema_tool_reaches_the_llm_and_is_dispatched(
    serializer: Serializer, hasher: ArgsHasher
):
    """A tool registered from a raw JSON Schema (no Python signature) is exposed
    to the model and, when called, dispatched to its handler with the decoded
    arguments — issue #38 end to end."""

    calls: list[dict] = []

    async def search_handler(query: str) -> str:
        calls.append({"query": query})
        return "Sefia is a framework for building LLM agents."

    search_tool = JsonSchemaTool(
        search_handler,
        name="search",
        parameters=_SEARCH_SCHEMA,
        description="Search the corpus for a query.",
    )

    mock_responses = [
        LLMResponse(
            content=json.dumps(
                {
                    "decision": "tool_calls",
                    "tool_calls": [{"name": "search", "arguments": {"query": "sefia"}}],
                }
            )
        ),
        LLMResponse(
            content=json.dumps(
                {
                    "decision": "result",
                    "result": {
                        "topic": "sefia",
                        "summary": "Sefia is a framework for building LLM agents.",
                        "sources": ["search"],
                    },
                }
            )
        ),
    ]
    mock_llm = MockLLMClient(responses=mock_responses)

    async with glyff.Session(
        id="json-schema-tool",
        backend=MemoryBackend(),
        serializer=serializer,
        hasher=hasher,
    ) as gs:
        async with Session(
            llm_client=mock_llm,
            glyff_session=gs,
            tool_collector=StaticToolCollector([search_tool]),
        ):
            report = await SimpleAgent().generate_report(topic="sefia")

    assert isinstance(report, Report)
    assert report.summary == "Sefia is a framework for building LLM agents."

    # The handler was dispatched with the decoded arguments.
    assert calls == [{"query": "sefia"}]

    # The raw JSON Schema reached the model through the tool-definition path.
    system_prompt = mock_llm.requests[0]["messages"][0]["content"]
    assert '"name": "search"' in system_prompt
    assert "Search the corpus for a query." in system_prompt
    assert '"additionalProperties": false' in system_prompt

    # The result step received the tool result in history.
    tool_message = mock_llm.requests[1]["messages"][3]
    assert tool_message["role"] == "tool"
    assert "framework" in tool_message["content"]
