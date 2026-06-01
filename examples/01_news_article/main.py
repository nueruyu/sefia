from textwrap import dedent

import glyff
import sefia
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .cli import create_app
from .session import SessionState
from .tools import HumanInputTool

console = Console()


class NewsArticle(BaseModel):
    """Represents a finalized news article."""

    title: str
    summary: str
    sources: list[str]

    def to_markdown(self):
        return (
            dedent(
                """
                ## Title
                {title}

                ## Summary
                {summary}

                ## Sources
                {sources}
                """
            )
            .strip()
            .format(
                title=self.title,
                summary=self.summary,
                sources="\n".join(f"- {source}" for source in self.sources)
                or "- (none)",
            )
        )


@glyff.identify("Researcher")
class Researcher:
    def __init__(self, web_search: sefia.tools.web.WebSearchTool):
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


async def workflow(state: SessionState) -> None:
    researcher = Researcher(sefia.tools.web.WebSearchTool())
    writer = NewsWriter(HumanInputTool())

    @glyff.engrave
    async def research() -> list[str]:
        console.print("[bold]> Stage 1: Researching topic...[/bold]")
        sources = await researcher.research_topic(state.initial_topic)
        console.print(f"[dim]   -> Found sources: {sources}[/dim]")
        return sources

    sources = await research()

    @glyff.engrave
    async def write() -> NewsArticle:
        console.print("[bold]> Stage 2: Writing article...[/bold]")
        return await writer.write_article(topic=state.initial_topic, sources=sources)

    article = await write()

    console.print(
        Panel(
            Markdown(article.to_markdown()),
            title="FINAL ARTICLE",
            border_style="green",
            expand=False,
        )
    )


if __name__ == "__main__":
    create_app(workflow)()
