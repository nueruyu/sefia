from pathlib import Path
from typing import Annotated

import typer
from glyff import engrave
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from sefios.cli import SefiaCLI, add_session_commands, async_command
from sefios.tools import WebSearchTool

from .._common.policies import VerbosePolicy
from .agents import NewsWriter, RequirementsClarifier, Researcher
from .models import ArticleRequest, NewsArticle
from .rendering import render_article_request, render_news_article

console = Console()
SESSION_DIR = Path(__file__).parent / ".local"
sefia_cli = SefiaCLI(session_dir=SESSION_DIR, stream=True)
human_input_tool = sefia_cli.human_input_tool

clarifier = RequirementsClarifier(human_input_tool)
researcher = Researcher(WebSearchTool())
writer = NewsWriter(human_input_tool, researcher)

app = typer.Typer(
    help="A multi-agent workflow for generating news articles with human-in-the-loop."
)


@engrave
async def _clarify() -> ArticleRequest:
    console.print("[bold]> Stage 1: Clarifying request...[/bold]")
    article_request = await clarifier.clarify_request()

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


@app.command()
@async_command
async def chat(
    message: Annotated[
        list[str],
        typer.Argument(
            help="The input for a new session, or an answer to resume an existing one."
        ),
    ],
    reply_to: Annotated[
        str | None,
        typer.Option(
            "--reply-to",
            help="The human input interaction ID to answer.",
        ),
    ] = None,
    session_id: Annotated[
        str | None,
        typer.Option(
            "--session-id",
            help="The session ID to use. If not provided, uses the active session.",
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="The LLM model to use. Can also be set via EXAMPLE_DEFAULT_MODEL env var.",
            envvar="EXAMPLE_DEFAULT_MODEL",
        ),
    ] = "gpt-4o",
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Enable verbose output for debugging, including LLM prompts.",
        ),
    ] = False,
) -> None:
    """Start a new workflow or provide an answer to continue the current session."""
    async with sefia_cli.session(
        session_id=session_id,
        model=model,
        policies=[VerbosePolicy()] if verbose else None,
    ) as session:
        await session.accept_input(message, reply_to=reply_to)

        article_request = await _clarify()
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


add_session_commands(app, sefia_cli)

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    app()
