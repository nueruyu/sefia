import asyncio
import json
from dataclasses import dataclass
from typing import Any, Callable, Coroutine

import glyff
from glyff import ArgsHasher, Serializer
from glyff.store import MemoryClient
from glyff.store import MemorySessionStore as GlyffMemoryStore
from glyff_pydantic import PydanticArgsHasher, PydanticSerializer
from sefia import Session, infer, policy, tool
from sefia.llm import LLMClient, LLMResponse, Message
from sefia.stores import MemorySessionStore as SefiaMemoryStore
from sefios.policies import StagnationPolicy


class MockLLMClient(LLMClient):
    """A mock LLM client that returns pre-defined responses."""

    def __init__(self, responses: list[LLMResponse]):
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        output_schema: dict | None = None,
        stream_callback: Callable[[str], Coroutine[None, None, None]] | None = None,
    ) -> LLMResponse:
        self.requests.append(
            {
                "messages": [m.to_dict(exclude_none=True) for m in messages],
                "tools": tools,
                "output_schema": output_schema,
                "stream_callback": stream_callback,
            }
        )
        if not self.responses:
            raise asyncio.InvalidStateError("MockLLMClient has no more responses.")
        return self.responses.pop(0)


@dataclass
class SearchResult:
    title: str
    url: str


@dataclass
class WebToolkit:
    @tool
    async def search(self, query: str) -> list[SearchResult]:
        """Search the web for a query."""
        if query == "sefia":
            return [
                SearchResult(title="sefia framework", url="https://example.com/sefia")
            ]
        return []


@dataclass
class Report:
    topic: str
    summary: str
    sources: list[str]


@dataclass
class Researcher:
    def __init__(self, web: WebToolkit):
        self._web = web

    @policy(StagnationPolicy(max_repeats=3))
    @infer
    async def generate_report(self, topic: str) -> Report:
        """
        Generate a report on the given topic by searching the web and summarizing it.
        """
        ...


def _make_stores(serializer: Serializer):
    client = MemoryClient()
    return (
        GlyffMemoryStore(client=client, serializer=serializer),
        SefiaMemoryStore(client=client, serializer=serializer),
    )


async def test_stagnation_state_is_isolated_between_infer_calls():
    serializer = PydanticSerializer()
    hasher: ArgsHasher = PydanticArgsHasher()

    repeated_call_response = LLMResponse(
        content=json.dumps(
            {
                "tool_calls": [
                    {"name": "WebToolkit_search", "arguments": {"query": "sefia"}}
                ]
            }
        )
    )
    final_response = LLMResponse(
        content=json.dumps(
            {
                "final_answer": {
                    "topic": "sefia",
                    "summary": "Sefia is a framework for building LLM agents.",
                    "sources": [],
                }
            }
        )
    )

    mock_llm = MockLLMClient(
        responses=[
            repeated_call_response,
            LLMResponse(
                content=json.dumps(
                    {
                        "final_answer": {
                            "topic": "sefia",
                            "summary": "first call done",
                            "sources": [],
                        }
                    }
                )
            ),
            # The second @infer call repeats the same tool call twice before
            # finishing. If the StagnationDetector were shared across calls,
            # this would combine with the first call's history and raise.
            repeated_call_response,
            repeated_call_response,
            final_response,
        ]
    )

    glyff_store, sefia_store = _make_stores(serializer)

    async with glyff.Session(
        id="stagnation-isolation-test", store=glyff_store, hasher=hasher
    ) as gs:
        async with Session(
            llm_client=mock_llm, glyff_session=gs, session_store=sefia_store
        ):
            researcher = Researcher(WebToolkit())

            report1 = await researcher.generate_report(topic="first")
            assert report1.summary == "first call done"

            report2 = await researcher.generate_report(topic="sefia")
            assert report2.topic == "sefia"
