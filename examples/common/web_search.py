import asyncio

from ddgs import DDGS
from pydantic import BaseModel, Field
from sefia import tool


class WebSearchResult(BaseModel):
    """Represents a single web search result."""

    title: str = Field(description="The title of the search result page.")
    href: str = Field(description="The URL of the search result page.")
    body: str = Field(
        description="A snippet of the content from the search result page."
    )


class WebSearchTool:
    """A toolkit for performing web searches using DuckDuckGo."""

    @tool
    async def search(self, query: str, max_results: int = 5) -> list[WebSearchResult]:
        """
        Performs a web search for the given query using DuckDuckGo
        and returns a list of search results.
        """

        def _sync_search():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        results = await asyncio.to_thread(_sync_search)
        return [WebSearchResult(**r) for r in results]
