import asyncio
from pathlib import Path
from typing import Any, Callable, Coroutine

import typer
from rich.console import Console
from sefios import SefiaScope
from typing_extensions import Annotated

from .human_input import ChatHumanInputAdapter
from .runner import run_workflow
from .session import ChatSession, SessionManager


def create_app(
    workflow_coro: Callable[[str], Coroutine[Any, Any, None]],
    session_manager: SessionManager,
    session_dir: Path,
    help_text: str,
    human_input: ChatHumanInputAdapter | None = None,
) -> typer.Typer:
    """
    Creates a Typer application with common chat and session management commands.
    """
    app = typer.Typer(help=help_text)
    console = Console()
    sefia_scope = SefiaScope(session_dir=session_dir, stream=True)
    scoped_run_workflow = sefia_scope(run_workflow)

    def print_chat_session_status(session: ChatSession) -> None:
        if session.source == "created":
            console.print(
                f"[bold]> No active session. Starting new session: {session.session_id}[/bold]"
            )
        elif session.source == "active":
            console.print(f"[bold]> Resuming session {session.session_id}[/bold]")

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
        session_id = session_manager.switch_active_session(session_id)
        console.print(f"[bold]> Switched active session to: {session_id}[/bold]")

    @session_app.command("new")
    def new_session():
        """
        Create a new session and make it active.
        """
        session_id = session_manager.create_new_active_session()
        console.print(
            f"[bold]> Created and switched to new session: {session_id}[/bold]"
        )

    @app.command()
    def chat(
        message: Annotated[
            list[str],
            typer.Argument(
                help="The input for a new session, or an answer to resume an existing one."
            ),
        ],
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
        input_text = " ".join(message).strip()
        chat_session = session_manager.prepare_chat_session(session_id)
        print_chat_session_status(chat_session)

        asyncio.run(
            scoped_run_workflow(
                session_id=chat_session.session_id,
                model=model,
                verbose=verbose,
                workflow_coro=workflow_coro,
                input_text=input_text,
                is_new=chat_session.is_new,
                human_input=human_input,
            )
        )

    return app
