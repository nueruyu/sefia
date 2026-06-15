from pathlib import Path

from glyff import engrave
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from sefios.tools import HumanInputTool, WebSearchTool

from .._common.chat_cli import create_app
from .._common.session import SessionManager
from .agents import NewsWriter, RequirementsClarifier, Researcher
from .models import ArticleRequest, NewsArticle
from .rendering import render_article_request, render_news_article

clarifier = RequirementsClarifier(HumanInputTool())
researcher = Researcher(WebSearchTool())
writer = NewsWriter(HumanInputTool(), researcher)

console = Console()


@engrave
async def _clarify(initial_input: str) -> ArticleRequest:
    console.print("[bold]> Stage 1: Clarifying request...[/bold]")
    article_request = await clarifier.clarify_request(initial_input)

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


async def news_article_workflow(initial_input: str) -> None:
    """Orchestrates the news article generation workflow."""
    article_request = await _clarify(initial_input)
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


SESSION_DIR = Path(__file__).parent / ".local"
session_manager = SessionManager(SESSION_DIR)
app = create_app(
    workflow_coro=news_article_workflow,
    session_manager=session_manager,
    session_dir=SESSION_DIR,
    help_text="A multi-agent workflow for generating news articles with human-in-the-loop.",
)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    app()
