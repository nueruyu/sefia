from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from sefios import SQLitePersistence
from sefios.cli import SefiaCLI
from sefios.sessions import FileActiveSessionStore
from sefios.tools import WebSearch
from typer.models import ArgumentInfo, OptionInfo

from .._common.policies import VerbosePolicy
from .._common.typer_utils import add_session_commands, async_command
from .agents import NewsWriter, RequirementsClarifier, Researcher
from .authoring import engrave
from .models import ArticleRequest, NewsArticle
from .rendering import render_article_request, render_news_article

console = Console()
SESSION_DIR = Path(__file__).parent / ".local"
sefia_cli = SefiaCLI(
    stream=True,
    persistence=SQLitePersistence(SESSION_DIR / "sessions.sqlite3"),
    active_session_store=FileActiveSessionStore(SESSION_DIR / "active_session.txt"),
)
input_tool = sefia_cli.input_tool

clarifier = RequirementsClarifier(input_tool)
researcher = Researcher(WebSearch())
writer = NewsWriter(input_tool, researcher)

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
        ArgumentInfo(
            default=...,
            help="The input for a new session, or an answer to resume an existing one.",
        ),
    ],
    reply_to: Annotated[
        str | None,
        OptionInfo(
            default=...,
            param_decls=("--reply-to",),
            help="The input interaction ID to answer.",
        ),
    ] = None,
    session_id: Annotated[
        str | None,
        OptionInfo(
            default=...,
            param_decls=("--session-id",),
            help="The session ID to use. If not provided, uses the active session.",
        ),
    ] = None,
    model: Annotated[
        str,
        OptionInfo(
            default=...,
            param_decls=("--model",),
            help="The LLM model to use. Can also be set via EXAMPLE_DEFAULT_MODEL env var.",
            envvar="EXAMPLE_DEFAULT_MODEL",
        ),
    ] = "gpt-4o",
    verbose: Annotated[
        bool,
        OptionInfo(
            default=...,
            param_decls=("--verbose",),
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
