from typing import Optional

import glyff
import glyff.exceptions
import sefia
from pydantic import BaseModel, Field


class NewsArticle(BaseModel):
    """Represents a finalized news article."""

    title: str
    summary: str
    sources: list[str]


class SessionState(BaseModel):
    """Represents the state of our long-running application."""

    topic: str
    answer: str | None = Field(default=None)


@glyff.identify("HumanInputTool")
class HumanInputTool:
    def __init__(self, answer: Optional[str] = None):
        self._answer = answer

    @sefia.tool
    async def get_human_input(self, question: str) -> str:
        """
        Asks the user a question and returns their answer.
        This tool interrupts the session to wait for user input.
        """
        if self._answer:
            return self._answer

        print(f"\n[USER_INPUT_REQUIRED] {question}\n")
        raise glyff.exceptions.YieldException()


@glyff.identify("Researcher")
class Researcher:
    def __init__(self, web_search: sefia.WebSearchTool):
        self._web = web_search

    @sefia.infer()
    async def research_topic(self, topic: str) -> list[str]:
        """
        Research the given topic to find relevant online sources.
        Your goal is to return a list of high-quality URLs related to the topic.

        **CRITICAL INSTRUCTIONS:**
        1. You MUST use the `WebSearchTool` tool to find the URLs.
        2. Do NOT answer from your own knowledge.
        3. The final answer MUST be a list of strings, where each string is a valid URL.
        """
        ...


@glyff.identify("NewsWriter")
class NewsWriter:
    def __init__(self, human_input: HumanInputTool):
        self._human_input = human_input

    @sefia.infer()
    async def write_article(self, topic: str, sources: list[str]) -> NewsArticle:
        """
        Write a news article on the given topic, using the provided sources.
        1. Briefly review the sources to understand the key points.
        2. Write a draft of the article.
        3. Ask the user for feedback on the draft's direction using the HumanInputTool.
        4. Finalize the article based on the user's feedback, incorporating their suggestions.
        5. Return the final article as a NewsArticle object.
        """
        ...


async def write_article(state: SessionState) -> tuple[NewsArticle, list[str]]:
    researcher = Researcher(sefia.WebSearchTool())
    writer = NewsWriter(HumanInputTool(answer=state.answer))

    print("> Stage 1: Researching topic...")
    sources = await researcher.research_topic(state.topic)
    print(f"\n   -> Found sources: {sources}")

    print("> Stage 2: Writing article...")
    article = await writer.write_article(topic=state.topic, sources=sources)
    return article, sources
