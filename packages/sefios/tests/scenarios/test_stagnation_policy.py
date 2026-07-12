import json
from dataclasses import dataclass

import glyff
from glyff import ArgsHasher, Serializer
from glyff.store import MemoryBackend
from sefia import Policy, Session, infer, policy
from sefia.llm import LLMResponse
from sefios.middleware import StagnationDetector


@dataclass
class SearchResult:
    title: str
    url: str


@dataclass
class WebToolkit:
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

    @policy(Policy(middleware=lambda: [StagnationDetector(max_repeats=3)]))
    @infer
    async def generate_report(self, topic: str) -> Report:
        """
        Generate a report on the given topic by searching the web and summarizing it.
        """
        ...


def _make_stores(serializer: Serializer):
    return MemoryBackend()


async def test_stagnation_state_is_isolated_between_infer_calls(
    serializer: Serializer, hasher: ArgsHasher, make_mock_llm
):
    repeated_call_response = LLMResponse(
        content=json.dumps(
            {
                "decision": "tool_calls",
                "tool_calls": [
                    {"name": "WebToolkit_search", "arguments": {"query": "sefia"}}
                ],
            }
        )
    )
    final_response = LLMResponse(
        content=json.dumps(
            {
                "decision": "result",
                "result": {
                    "topic": "sefia",
                    "summary": "Sefia is a framework for building LLM agents.",
                    "sources": [],
                },
            }
        )
    )

    mock_llm = make_mock_llm(
        [
            repeated_call_response,
            LLMResponse(
                content=json.dumps(
                    {
                        "decision": "result",
                        "result": {
                            "topic": "sefia",
                            "summary": "first call done",
                            "sources": [],
                        },
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

    glyff_store = _make_stores(serializer)

    async with glyff.Session(
        id="stagnation-isolation-test",
        backend=glyff_store,
        serializer=serializer,
        hasher=hasher,
    ) as gs:
        async with Session(llm_client=mock_llm, glyff_session=gs):
            researcher = Researcher(WebToolkit())

            report1 = await researcher.generate_report(topic="first")
            assert report1.summary == "first call done"

            report2 = await researcher.generate_report(topic="sefia")
            assert report2.topic == "sefia"
