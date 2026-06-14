import uuid
from pathlib import Path


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
