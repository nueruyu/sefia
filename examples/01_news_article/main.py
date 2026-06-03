from glyff import engrave
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .agents import NewsWriter, RequirementsClarifier, Researcher
from .cli import create_app
from .models import ArticleRequest, NewsArticle
from .session import SessionState
from .tools import HumanInputTool, WebSearchTool

clarifier = RequirementsClarifier(HumanInputTool())
researcher = Researcher(WebSearchTool())
writer = NewsWriter(HumanInputTool(), researcher)

console = Console()


@engrave
async def _clarify(initial_topic: str) -> ArticleRequest:
    console.print("[bold]> Stage 1: Clarifying request...[/bold]")
    article_request = await clarifier.clarify_request(initial_topic)

    console.print("[dim]   -> Clarified request:[/dim]")
    console.print(Markdown(article_request.to_markdown()))
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


async def workflow(state: SessionState) -> None:
    article_request = await _clarify(state.initial_topic)
    sources = await _research(article_request)
    article = await _write(article_request, sources)

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
