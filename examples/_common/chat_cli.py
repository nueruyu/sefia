import asyncio
import sys
from pathlib import Path
from typing import Any, Callable, Coroutine

import typer
from rich.console import Console
from sefios import SefiaScope
from typing_extensions import Annotated

from .runner import run_workflow
from .session import SessionManager


def create_app(
    workflow_coro: Callable[[str], Coroutine[Any, Any, None]],
    session_manager: SessionManager,
    session_dir: Path,
    help_text: str,
) -> typer.Typer:
    """
    Creates a Typer application with common chat and session management commands.
    """
    app = typer.Typer(help=help_text)
    console = Console()
    sefia_scope = SefiaScope(session_dir=session_dir)
    scoped_run_workflow = sefia_scope(run_workflow)

    session_app = typer.Typer(help="Manage user sessions.")
    app.add_typer(session_app, name="session")

    @session_app.command("switch")
    def switch_session(
        session_id: Annotated[
            str, typer.Argument(help="The ID of the session to switch to.")
        ],
    ):
        """
        Switch the active session.
        """
        session_manager.set_active_session_id(session_id)
        console.print(f"[bold]> Switched active session to: {session_id}[/bold]")

    @session_app.command("new")
    def new_session():
        """
        Create a new session and make it active.
        """
        session_id = session_manager.create_new_session_id()
        session_manager.set_active_session_id(session_id)
        console.print(
            f"[bold]> Created and switched to new session: {session_id}[/bold]"
        )

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

        is_new = False
        if session_id is None:
            session_id = session_manager.get_active_session_id()
            if session_id is None:
                session_id = session_manager.create_new_session_id()
                console.print(
                    f"[bold]> No active session. Starting new session: {session_id}[/bold]"
                )
                session_manager.set_active_session_id(session_id)
                is_new = True
            else:
                console.print(f"[bold]> Resuming session {session_id}[/bold]")

        asyncio.run(
            scoped_run_workflow(
                workflow_coro=workflow_coro,
                session_id=session_id,
                input_text=input_text,
                is_new=is_new,
                model=model,
                verbose=verbose,
                stream=True,
            )
        )

    return app
