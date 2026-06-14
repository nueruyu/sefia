import asyncio
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from textwrap import dedent
from typing import NoReturn

import glyff.exceptions
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from typing_extensions import Annotated

from ...presets.litellm import create_litellm_session
from .session_manager import SessionManager

WorkflowCallback = Callable[[], Awaitable[None]]
InitCallback = Callable[[str], Awaitable[None]]
console = Console()


def _exit_with_error(message: str) -> NoReturn:
    console.print(f"[bold red]Error:[/bold red] {message}")
    raise typer.Exit(code=1)


def _print_session_interrupted_hint() -> None:
    hint = dedent(
        """
        Session interrupted to wait for your input.
        To resume, run the script again with your answer.
        """
    ).strip()
    console.print()
    console.print(Panel(hint, title="WAITING FOR INPUT", border_style="yellow"))


async def _run_workflow(
    session_id: str,
    input_text: str,
    is_new: bool,
    model: str,
    verbose: bool,
    init_callback: InitCallback,
    workflow_callback: WorkflowCallback,
    session_dir: Path,
):
    """Encapsulates the main logic for running the sefia workflow."""
    try:
        async with create_litellm_session(
            model=model,
            session_id=session_id,
            stream=True,
            verbose=verbose,
            session_dir=session_dir,
        ) as session:
            pending = await session.session_store.get("pending_human_interaction", dict)
            if pending and not is_new:
                interaction_id = pending["id"]
                await session.session_store.set(
                    f"human_input__{interaction_id}", input_text, str
                )

            if is_new:
                await init_callback(input_text)

            await workflow_callback()

    except glyff.exceptions.YieldException:
        _print_session_interrupted_hint()


def create_app(
    init_callback: InitCallback,
    workflow_callback: WorkflowCallback,
    *,
    session_dir: Path,
    help_text: str = "A human-in-the-loop chat workflow.",
) -> typer.Typer:
    """Create a CLI app that routes execution to the injected workflow callback."""
    load_dotenv()
    app = typer.Typer(help=help_text)
    session_app = typer.Typer(help="Manage user sessions.")
    app.add_typer(session_app, name="session")
    session_manager = SessionManager(session_dir)

    @app.command("chat")
    def chat(
        message: Annotated[
            list[str] | None,
            typer.Argument(
                help="The input for a new session, or an answer to resume an existing one."
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
            _exit_with_error("Message cannot be empty.")

        session_id = session_manager.get_active_session_id()
        is_new = session_id is None

        if is_new:
            session_id = session_manager.create_new_session_id()
            console.print(
                f"[bold]> No active session. Starting new session: {session_id}[/bold]"
            )
            session_manager.set_active_session_id(session_id)
        else:
            console.print(f"[bold]> Resuming session {session_id}[/bold]")

        asyncio.run(
            _run_workflow(
                session_id=session_id,
                input_text=input_text,
                is_new=is_new,
                model=model,
                verbose=verbose,
                init_callback=init_callback,
                workflow_callback=workflow_callback,
                session_dir=session_dir,
            )
        )

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

    return app
