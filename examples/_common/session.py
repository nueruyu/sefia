import uuid
from pathlib import Path

import typer
from rich.console import Console
from typing_extensions import Annotated


class SessionManager:
    """Manages the lifecycle of user chat sessions, including the active session ID."""

    def __init__(self, session_dir: Path):
        self._session_dir = session_dir
        self._active_session_file = self._session_dir / "active_session.txt"
        self._session_dir.mkdir(exist_ok=True)

    def get_active_session_id(self) -> str | None:
        """Gets the ID of the currently active session, if one exists."""
        if self._active_session_file.exists():
            return self._active_session_file.read_text(encoding="utf-8").strip()
        return None

    def set_active_session_id(self, session_id: str) -> None:
        """Sets the active session ID."""
        self._active_session_file.write_text(session_id, encoding="utf-8")

    def create_new_session_id(self) -> str:
        """Generates a new unique session ID."""
        return str(uuid.uuid4())


def create_session_cli(app: typer.Typer, session_manager: SessionManager) -> None:
    """Adds session management commands (new, switch) to the Typer app."""
    session_app = typer.Typer(help="Manage user sessions.")
    app.add_typer(session_app, name="session")
    console = Console()

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
