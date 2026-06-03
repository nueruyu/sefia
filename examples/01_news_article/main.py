from pathlib import Path

from glyff import engrave
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ..common.chat_cli import create_app
from ..common.chat_session import ChatSessionState
from ..common.human_input import HumanInputTool
from ..common.web_search import WebSearchTool
from .agents import NewsWriter, RequirementsClarifier, Researcher
from .models import ArticleRequest, NewsArticle
from .rendering import render_article_request, render_news_article

clarifier = RequirementsClarifier(HumanInputTool())
researcher = Researcher(WebSearchTool())
writer = NewsWriter(HumanInputTool(), researcher)

console = Console()
SESSION_DIR = Path(__file__).parent / ".local"


@engrave
async def _clarify(initial_topic: str) -> ArticleRequest:
    console.print("[bold]> Stage 1: Clarifying request...[/bold]")
    article_request = await clarifier.clarify_request(initial_topic)

    console.print("[dim]   -> Clarified request:[/dim]")
    console.print(Markdown(render_article_request(article_request)))
    return article_request


@engrave
async def _research(article_request: ArticleRequest) -> list[str]:
    console.print("[bold]> Stage 2: Researching topic...[/bold]")
    sources = await researcher.research_topic(article_request)
    console.print(f"[dim]   -> Found sources: {sources}[/dim]")
    return sources


@engrave
async def _write(article_request: ArticleRequest, sources: list[str]) -> NewsArticle:
    console.print("[bold]> Stage 3: Writing article...[/bold]")
    return await writer.write_article(article_request=article_request, sources=sources)


async def workflow(state: ChatSessionState) -> None:
    article_request = await _clarify(state.initial_topic)
    sources = await _research(article_request)
    article = await _write(article_request, sources)

    console.print(
        Panel(
            Markdown(render_news_article(article)),
            title="FINAL ARTICLE",
            border_style="green",
            expand=False,
        )
    )


if __name__ == "__main__":
    create_app(
        workflow,
        session_dir=SESSION_DIR,
        help_text="A multi-agent workflow for generating news articles with human-in-the-loop.",
    )()
