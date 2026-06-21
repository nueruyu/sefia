import asyncio
import functools
from typing import Annotated, Any, Callable, Coroutine, TypeVar

import typer
from rich.console import Console
from typing_extensions import ParamSpec

from .sefia_cli import SefiaCLI
from .session import UnknownSessionError

_P = ParamSpec("_P")
_R = TypeVar("_R")

console = Console()


def async_command(
    f: Callable[_P, Coroutine[Any, Any, _R]],
) -> Callable[_P, _R]:
    """Wrap an async function so it can be used as a synchronous Typer command."""

    @functools.wraps(f)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        return asyncio.run(f(*args, **kwargs))

    return wrapper


def add_session_commands(app: typer.Typer, sefia_cli: SefiaCLI) -> None:
    """Register `session new` and `session switch` sub-commands on *app*."""
    session_app = typer.Typer(help="Manage sessions.")
    app.add_typer(session_app, name="session")

    @session_app.command("new")
    def new_session() -> None:
        """Create a new session and make it active."""
        session_id = sefia_cli.create_session()
        console.print(f"[bold]> Created and switched to new session: {session_id}[/bold]")

    @session_app.command("switch")
    def switch_session(
        session_id: Annotated[
            str,
            typer.Argument(help="The ID of the session to switch to."),
        ],
    ) -> None:
        """Switch the active session."""
        try:
            session_id = sefia_cli.switch_session(session_id)
        except UnknownSessionError as e:
            console.print(f"[bold red]> Unknown session:[/bold red] {e.session_id}")
            raise typer.Exit(code=1) from e
        console.print(f"[bold]> Switched active session to: {session_id}[/bold]")
