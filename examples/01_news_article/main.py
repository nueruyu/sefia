from dataclasses import dataclass
from pathlib import Path

import typer
from glyff import engrave
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from sefia import get_context
from sefios.tools import WebSearchTool
from typing_extensions import Annotated

from .._common.sefia_cli import CLIParam, SefiaCLI
from .._common.session import UnknownSessionError
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
session_app = typer.Typer(help="Manage sessions.")
app.add_typer(session_app, name="session")


@dataclass
class ArticleState:
    initial_request: str | None = None

    def require_initial_request(self) -> str:
        if self.initial_request is None:
            raise RuntimeError("Article state has no initial request.")
        return self.initial_request


@session_app.command("new")
def new_session():
    """Create a new session and make it active."""
    session_id = sefia_cli.create_session()
    console.print(f"[bold]> Created and switched to new session: {session_id}[/bold]")


@session_app.command("switch")
def switch_session(
    session_id: Annotated[
        str,
        typer.Argument(help="The ID of the session to switch to."),
    ],
):
    """Switch the active session."""
    try:
        session_id = sefia_cli.switch_session(session_id)
    except UnknownSessionError as e:
        console.print(f"[bold red]> Unknown session:[/bold red] {e.session_id}")
        raise typer.Exit(code=1) from e

    console.print(f"[bold]> Switched active session to: {session_id}[/bold]")


async def _get_initial_request(message: list[str]) -> str:
    state_store = get_context().get_state_store("article_state", ArticleState)
    state = await state_store.ensure()

    if state.initial_request is None:
        state.initial_request = sefia_cli.to_input_text(message)
        await state_store.save(state)

    return state.require_initial_request()


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


@app.command()
@sefia_cli.scope
async def chat(
    message: Annotated[
        list[str],
        typer.Argument(
            help="The input for a new session, or an answer to resume an existing one."
        ),
        CLIParam.INPUT,
    ],
    reply_to: Annotated[
        str | None,
        typer.Option(
            "--reply-to",
            help="The human input interaction ID to answer.",
        ),
        CLIParam.REPLY_TO,
    ] = None,
    session_id: Annotated[
        str | None,
        typer.Option(
            "--session-id",
            help="The session ID to use. If not provided, uses the active session.",
        ),
        CLIParam.SESSION_ID,
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="The LLM model to use. Can also be set via EXAMPLE_DEFAULT_MODEL env var.",
            envvar="EXAMPLE_DEFAULT_MODEL",
        ),
        CLIParam.MODEL,
    ] = "gpt-4o",
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Enable verbose output for debugging, including LLM prompts.",
        ),
        CLIParam.VERBOSE,
    ] = False,
):
    """Start a new workflow or provide an answer to continue the current session."""
    initial_request = await _get_initial_request(message)

    article_request = await _clarify(initial_request)
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
    from dotenv import load_dotenv

    load_dotenv()
    app()
