from dataclasses import dataclass

from sefia import Policy, infer, policy
from sefia.testing import memory_session, result_response, tool_calls_response
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


async def test_stagnation_state_is_isolated_between_infer_calls(make_mock_llm):
    repeated_call_response = tool_calls_response(
        ("WebToolkit_search", {"query": "sefia"})
    )
    final_response = result_response(
        Report(
            topic="sefia",
            summary="Sefia is a framework for building LLM agents.",
            sources=[],
        )
    )

    mock_llm = make_mock_llm(
        [
            repeated_call_response,
            result_response(
                Report(topic="sefia", summary="first call done", sources=[])
            ),
            # The second @infer call repeats the same tool call twice before
            # finishing. If the StagnationDetector were shared across calls,
            # this would combine with the first call's history and raise.
            repeated_call_response,
            repeated_call_response,
            final_response,
        ]
    )

    async with memory_session(mock_llm, session_id="stagnation-isolation-test"):
        researcher = Researcher(WebToolkit())

        report1 = await researcher.generate_report(topic="first")
        assert report1.summary == "first call done"

        report2 = await researcher.generate_report(topic="sefia")
        assert report2.topic == "sefia"
