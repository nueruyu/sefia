from pathlib import Path

from glyff import engrave
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from sefia import get_context
from sefios.scaffolding.cli.runner import create_app
from sefios.tools import HumanInputTool, WebSearchTool

from .._common.workflow import WorkflowState
from .agents import NewsWriter, RequirementsClarifier, Researcher
from .models import ArticleRequest, NewsArticle
from .rendering import render_article_request, render_news_article

clarifier = RequirementsClarifier(HumanInputTool())
researcher = Researcher(WebSearchTool())
writer = NewsWriter(HumanInputTool(), researcher)

console = Console()
SESSION_DIR = Path(__file__).parent / ".local"


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


async def initialize_workflow(initial_input: str) -> None:
    """Initializes the workflow state for a new session."""
    session = get_context()
    state_store = session.get_state_store("workflow_state", WorkflowState)
    state = WorkflowState.from_initial_input(initial_input)
    await state_store.save(state)


async def workflow() -> None:
    """Runs the main application logic."""
    session = get_context()
    state_store = session.get_state_store("workflow_state", WorkflowState)
    state = await state_store.ensure()

    article_request = await _clarify(state.initial_input)
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
        init_callback=initialize_workflow,
        workflow_callback=workflow,
        session_dir=SESSION_DIR,
        help_text="A multi-agent workflow for generating news articles with human-in-the-loop.",
    )()
