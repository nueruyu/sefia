import asyncio
import sys
from pathlib import Path
from textwrap import dedent

import typer
from glyff import engrave
from glyff.exceptions import YieldException
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from sefia import get_context
from sefios import SefiaScope
from sefios.tools import HumanInputTool, WebSearchTool
from typing_extensions import Annotated

from .._common.session import SessionManager, create_session_cli
from .._common.workflow import WorkflowState
from .agents import NewsWriter, RequirementsClarifier, Researcher
from .models import ArticleRequest, NewsArticle
from .rendering import render_article_request, render_news_article

clarifier = RequirementsClarifier(HumanInputTool())
researcher = Researcher(WebSearchTool())
writer = NewsWriter(HumanInputTool(), researcher)

SESSION_DIR = Path(__file__).parent / ".local"
console = Console()

sefia_scope = SefiaScope(session_dir=SESSION_DIR)
app = typer.Typer(
    help="A multi-agent workflow for generating news articles with human-in-the-loop."
)
session_manager = SessionManager(SESSION_DIR)
create_session_cli(app, session_manager)


def _print_session_interrupted_hint() -> None:
    hint = dedent(
        """
        Session interrupted to wait for your input.
        To resume, run the script again with your answer.
        """
    ).strip()
    console.print()
    console.print(Panel(hint, title="WAITING FOR INPUT", border_style="yellow"))


async def _initialize_workflow(initial_input: str) -> None:
    """Initializes the workflow state for a new session."""
    session = get_context()
    state_store = session.get_state_store("workflow_state", WorkflowState)
    state = WorkflowState.from_initial_input(initial_input)
    await state_store.save(state)


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


@sefia_scope
async def _run_chat(
    *,
    session_id: str,
    input_text: str,
    is_new: bool,
    model: str,
    verbose: bool,
    stream: bool = True,
) -> None:
    try:
        session = get_context()
        pending = await session.session_store.get("pending_human_interaction", dict)
        if pending and not is_new:
            interaction_id = pending["id"]
            await session.session_store.set(
                f"human_input__{interaction_id}", input_text, str
            )

        if is_new:
            await _initialize_workflow(input_text)

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
    except YieldException:
        _print_session_interrupted_hint()
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")
        raise typer.Exit(code=1) from e


@app.command()
def chat(
    message: Annotated[
        list[str] | None,
        typer.Argument(
            help="The input for a new session, or an answer to resume an existing one."
        ),
    ] = None,
    session_id: Annotated[
        str | None,
        typer.Option(
            help="The session ID to use. If not provided, uses the active session."
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
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
):
    """
    Start a new workflow or provide an answer to continue the current session.
    """
    input_text = " ".join(message or []).strip()

    if not input_text and not sys.stdin.isatty():
        input_text = sys.stdin.read().strip()

    if not input_text:
        console.print("[bold red]Error:[/bold red] Message cannot be empty.")
        raise typer.Exit(code=1)

    is_new = session_id is None
    if is_new:
        session_id = session_manager.get_active_session_id()
        if session_id is None:
            session_id = session_manager.create_new_session_id()
            console.print(
                f"[bold]> No active session. Starting new session: {session_id}[/bold]"
            )
            session_manager.set_active_session_id(session_id)
        else:
            console.print(f"[bold]> Resuming session {session_id}[/bold]")
            is_new = False
    else:
        is_new = False

    asyncio.run(
        _run_chat(
            session_id=session_id,
            input_text=input_text,
            is_new=is_new,
            model=model,
            verbose=verbose,
            stream=True,
        )
    )


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    app()
