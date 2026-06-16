import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ChatSession:
    """A resolved chat session for a workflow run."""

    session_id: str
    is_new: bool
    source: Literal["explicit", "active", "created"]


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

    def switch_active_session(self, session_id: str) -> str:
        """Switches the active session and returns its ID."""
        self.set_active_session_id(session_id)
        return session_id

    def create_new_active_session(self) -> str:
        """Creates a new session, makes it active, and returns its ID."""
        session_id = self.create_new_session_id()
        self.set_active_session_id(session_id)
        return session_id

    def prepare_chat_session(self, session_id: str | None) -> ChatSession:
        """Resolves the session to use for a chat command."""
        if session_id is not None:
            return ChatSession(session_id=session_id, is_new=False, source="explicit")

        active_session_id = self.get_active_session_id()
        if active_session_id is not None:
            return ChatSession(
                session_id=active_session_id,
                is_new=False,
                source="active",
            )

        return ChatSession(
            session_id=self.create_new_active_session(),
            is_new=True,
            source="created",
        )
