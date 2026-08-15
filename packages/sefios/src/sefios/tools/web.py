import asyncio
import importlib
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from typing import Any, Protocol, cast

from pydantic import BaseModel, Field


class WebSearchResult(BaseModel):
    """Represents a single web search result."""

    title: str = Field(description="The title of the search result page.")
    href: str = Field(description="The URL of the search result page.")
    body: str = Field(
        description="A snippet of the content from the search result page."
    )


class _DDGS(Protocol):
    def text(self, query: str, *, max_results: int) -> Iterator[dict[str, Any]]: ...


class WebSearch:
    """A toolkit for performing web searches using DuckDuckGo."""

    async def search(self, query: str, max_results: int = 5) -> list[WebSearchResult]:
        """
        Performs a web search for the given query using DuckDuckGo
        and returns a list of search results.
        """

        def _sync_search() -> list[dict[str, Any]]:
            try:
                module = importlib.import_module("ddgs")
            except ImportError as e:
                raise ImportError(
                    "The 'web' extra is required to use WebSearch. "
                    "Please install it with: pip install 'sefios[web]'"
                ) from e

            factory = cast(
                Callable[[], AbstractContextManager[_DDGS]],
                getattr(module, "DDGS"),
            )
            with factory() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        results = await asyncio.to_thread(_sync_search)
        return [WebSearchResult(**r) for r in results]
