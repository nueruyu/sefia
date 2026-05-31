import asyncio
from typing import Any, Callable, Coroutine

import glyff
import pytest
from glyff.interfaces import ArgsHasher, Serializer
from pydantic import BaseModel

from sefia import LLMResponse, infer, tool
from sefia.llm.client import LLMClient
from sefia.llm.messages import Message
from sefia.serialization import SefiaArgsHasher, SefiaSerializer


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
                "messages": [m.model_dump(exclude_none=True) for m in messages],
                "tools": tools,
                "output_schema": output_schema,
                "stream_callback": stream_callback,
            }
        )
        if not self.responses:
            raise asyncio.InvalidStateError("MockLLMClient has no more responses.")
        return self.responses.pop(0)


class SearchResult(BaseModel):
    title: str
    url: str


@glyff.identify("WebToolkit")
class WebToolkit:
    """A simple toolkit for web operations."""

    @tool
    async def search(self, query: str) -> list[SearchResult]:
        """Search the web for a query."""
        if query == "sefia":
            return [
                SearchResult(title="sefia framework", url="https://example.com/sefia")
            ]
        return []

    @tool
    async def fetch_content(self, url: str) -> str:
        """Fetch content from a URL."""
        if url == "https://example.com/sefia":
            return "Sefia is a framework for building LLM agents."
        return "Not found."


class Report(BaseModel):
    topic: str
    summary: str
    sources: list[str]


@glyff.identify("Researcher")
class Researcher:
    """An agent that uses WebToolkit to research topics."""

    def __init__(self, web: WebToolkit):
        self._web = web

    @infer()
    async def generate_report(self, topic: str) -> Report:
        """
        Generate a report on the given topic by searching the web,
        fetching content, and summarizing it.
        """
        ...


@glyff.identify("BrokenToolkit")
class BrokenToolkit:
    """A toolkit where tools can fail."""

    @tool
    async def always_fail(self, reason: str) -> None:
        """This tool always raises an exception."""
        raise ValueError(f"Failed because: {reason}")


@glyff.identify("SimpleAgent")
class SimpleAgent:
    """An agent that has no tools."""

    def __init__(self):
        pass

    @infer()
    async def generate_report(self, topic: str) -> Report:
        """
        Generate a report on the given topic.
        This agent has no tools and must produce the report directly.
        """
        ...


@pytest.fixture
def web_toolkit() -> WebToolkit:
    return WebToolkit()


@pytest.fixture
def broken_toolkit() -> BrokenToolkit:
    return BrokenToolkit()


@pytest.fixture
def serializer() -> Serializer:
    return SefiaSerializer()


@pytest.fixture
def hasher() -> ArgsHasher:
    return SefiaArgsHasher()
