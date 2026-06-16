import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SessionSource = Literal["explicit", "active", "created"]


class UnknownSessionError(Exception):
    """Raised when a requested CLI session is not known."""

    def __init__(self, session_id: str):
        super().__init__(f"Unknown session: {session_id}")
        self.session_id = session_id


@dataclass(frozen=True)
class ResolvedSession:
    """A resolved session for a CLI command invocation."""

    session_id: str
    is_new: bool
    source: SessionSource


class SessionManager:
    """Manages the lifecycle of CLI sessions, including the active session ID."""

    def __init__(self, session_dir: Path):
        self._active_session_file = session_dir / "active_session.txt"
        self._sessions_file = session_dir / "sessions.txt"
        self._glyff_sessions_dir = session_dir / "glyff_sessions"
        session_dir.mkdir(exist_ok=True)

    def get_active_session_id(self) -> str | None:
        """Gets the ID of the currently active session, if one exists."""
        if self._active_session_file.exists():
            session_id = self._active_session_file.read_text(encoding="utf-8").strip()
            return session_id or None
        return None

    def set_active_session_id(self, session_id: str) -> None:
        """Sets the active session ID."""
        self._active_session_file.write_text(session_id, encoding="utf-8")

    def create_new_session_id(self) -> str:
        """Generates a new unique session ID."""
        return str(uuid.uuid4())

    def session_exists(self, session_id: str) -> bool:
        """Returns whether the session is known to this CLI workspace."""
        if session_id in self._read_registered_session_ids():
            return True

        # Backward-compatibility for sessions created before the CLI registry existed.
        if (self._glyff_sessions_dir / session_id).exists():
            self._register_session(session_id)
            return True

        return False

    def switch_active_session(self, session_id: str) -> str:
        """Switches the active session and returns its ID."""
        if not self.session_exists(session_id):
            raise UnknownSessionError(session_id)

        self.set_active_session_id(session_id)
        return session_id

    def create_new_active_session(self) -> str:
        """Creates a new session, makes it active, and returns its ID."""
        while True:
            session_id = self.create_new_session_id()
            if not self.session_exists(session_id):
                break

        self._register_session(session_id)
        self.set_active_session_id(session_id)
        return session_id

    def resolve_session(self, session_id: str | None) -> ResolvedSession:
        """Resolves the session to use for a CLI command."""
        if session_id is not None:
            if not self.session_exists(session_id):
                raise UnknownSessionError(session_id)
            return ResolvedSession(
                session_id=session_id,
                is_new=False,
                source="explicit",
            )

        active_session_id = self.get_active_session_id()
        if active_session_id is not None:
            if not self.session_exists(active_session_id):
                raise UnknownSessionError(active_session_id)
            return ResolvedSession(
                session_id=active_session_id,
                is_new=False,
                source="active",
            )

        return ResolvedSession(
            session_id=self.create_new_active_session(),
            is_new=True,
            source="created",
        )

    def _read_registered_session_ids(self) -> set[str]:
        if not self._sessions_file.exists():
            return set()

        return {
            line.strip()
            for line in self._sessions_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    def _register_session(self, session_id: str) -> None:
        session_ids = self._read_registered_session_ids()
        if session_id in session_ids:
            return

        session_ids.add(session_id)
        self._sessions_file.write_text(
            "".join(f"{registered_id}\n" for registered_id in sorted(session_ids)),
            encoding="utf-8",
        )
