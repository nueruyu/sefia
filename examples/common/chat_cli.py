import asyncio
import sys
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from textwrap import dedent
from typing import Any

import glyff.exceptions
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from typing_extensions import Annotated

from .chat_session import ChatSessionState
from .session_setup import setup_session

WorkflowCallback = Callable[[ChatSessionState], Awaitable[None]]
console = Console()


def _get_active_session_file(session_dir: Path) -> Path:
    session_dir.mkdir(exist_ok=True)
    return session_dir / "active_session.txt"


def _get_active_session_id(session_dir: Path) -> str | None:
    session_file = _get_active_session_file(session_dir)
    if session_file.exists():
        return session_file.read_text(encoding="utf-8").strip()
    return None


def _set_active_session_id(session_dir: Path, session_id: str) -> None:
    _get_active_session_file(session_dir).write_text(session_id, encoding="utf-8")


def _print_session_interrupted_hint() -> None:
    hint = dedent(
        """
        Session interrupted to wait for your input.
        To resume, run the script again with your answer.
        """
    ).strip()
    console.print()
    console.print(Panel(hint, title="WAITING FOR INPUT", border_style="yellow"))


def _resolve_session_state(
    state: ChatSessionState | None, input_text: str, is_new: bool
) -> ChatSessionState:
    """Resolve the next session state from current state and latest user input."""
    if is_new or state is None:
        return ChatSessionState.from_initial_topic(input_text)

    if state.pending_interaction:
        # We are resuming a pending HumanInputTool interaction.
        state.update_pending_answer(input_text)
        return state

    # Previous run already completed, so replay the completed session result.
    console.print("[bold]> Previous session completed. Showing last result.[/bold]")
    return state


async def _run_workflow(
    session_id: str,
    input_text: str,
    is_new: bool,
    model: str,
    verbose: bool,
    workflow_callback: WorkflowCallback,
    session_dir: Path,
    text_block_selectors: dict[type, Callable[[Any], str]] | None,
):
    """Encapsulates the main logic for running the sefia workflow."""
    try:
        async with setup_session(
            model=model,
            session_id=session_id,
            stream=True,
            verbose=verbose,
            session_dir=session_dir,
            text_block_selectors=text_block_selectors,
        ) as session:
            state_store = session.get_state_store("session_state", ChatSessionState)
            state = await state_store.get()
            state = _resolve_session_state(state, input_text, is_new)

            await state_store.save(state)
            await workflow_callback(state)

    except glyff.exceptions.YieldException:
        _print_session_interrupted_hint()


def create_app(
    workflow_callback: WorkflowCallback,
    *,
    session_dir: Path,
    help_text: str = "A human-in-the-loop chat workflow.",
    text_block_selectors: dict[type, Callable[[Any], str]] | None = None,
) -> typer.Typer:
    """Create a CLI app that routes execution to the injected workflow callback."""
    load_dotenv()
    app = typer.Typer(help=help_text)
    session_app = typer.Typer(help="Manage user sessions.")
    app.add_typer(session_app, name="session")

    @app.command("chat")
    def chat(
        message: Annotated[
            list[str] | None,
            typer.Argument(
                help="The topic for a new session, or an answer to resume an existing one."
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
        Start a new topic or provide an answer to continue the current session.
        """
        input_text = " ".join(message or []).strip()

        if not input_text and not sys.stdin.isatty():
            input_text = sys.stdin.read().strip()

        if not input_text:
            raise typer.BadParameter("Message cannot be empty.")

        session_id = _get_active_session_id(session_dir)
        is_new = session_id is None

        if is_new:
            session_id = str(uuid.uuid4())
            console.print(
                f"[bold]> No active session. Starting new session: {session_id}[/bold]"
            )
            _set_active_session_id(session_dir, session_id)
        else:
            console.print(f"[bold]> Resuming session {session_id}[/bold]")

        asyncio.run(
            _run_workflow(
                session_id=session_id,
                input_text=input_text,
                is_new=is_new,
                model=model,
                verbose=verbose,
                workflow_callback=workflow_callback,
                session_dir=session_dir,
                text_block_selectors=text_block_selectors,
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
        # A more robust implementation would check if the session directory exists.
        _set_active_session_id(session_dir, session_id)
        console.print(f"[bold]> Switched active session to: {session_id}[/bold]")

    @session_app.command("new")
    def new_session():
        """
        Create a new session and make it active.
        """
        session_id = str(uuid.uuid4())
        _set_active_session_id(session_dir, session_id)
        console.print(
            f"[bold]> Created and switched to new session: {session_id}[/bold]"
        )

    return app
