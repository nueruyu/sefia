from dataclasses import dataclass
from typing import Literal

from typing_extensions import final

from ._active import ActiveSessionStore
from ._registry import SessionRegistry

SessionSource = Literal["explicit", "active", "created"]


class UnknownSessionError(Exception):
    """Raised when a requested session is not registered."""

    def __init__(self, session_id: str):
        super().__init__(f"Unknown session: {session_id}")
        self.session_id = session_id


@dataclass(frozen=True)
class ResolvedSession:
    """A resolved session for one application invocation."""

    session_id: str
    is_new: bool
    source: SessionSource


@final
class SessionManager:
    """Manages the lifecycle of application sessions, including the active one.

    The registry is provided by the persistence layer. Active selection is
    workspace-local state supplied separately by the CLI integration.
    """

    def __init__(
        self, registry: SessionRegistry, active_session_store: ActiveSessionStore
    ) -> None:
        self._registry = registry
        self._active_session_store = active_session_store

    def get_active_session_id(self) -> str | None:
        """Gets the ID of the currently active session, if one exists."""
        return self._active_session_store.get_active_session_id()

    def set_active_session_id(self, session_id: str) -> None:
        """Sets the active session ID."""
        self._active_session_store.set_active_session_id(session_id)

    def session_exists(self, session_id: str) -> bool:
        """Returns whether the session is registered."""
        return self._registry.session_exists(session_id)

    def switch_active_session(self, session_id: str) -> str:
        """Switches the active session and returns its ID."""
        if not self.session_exists(session_id):
            raise UnknownSessionError(session_id)

        self.set_active_session_id(session_id)
        return session_id

    def create_new_active_session(self) -> str:
        """Creates a new session, makes it active, and returns its ID."""
        session_id = self._registry.create_session()
        self.set_active_session_id(session_id)
        return session_id

    def resolve_session(self, session_id: str | None) -> ResolvedSession:
        """Resolves the session to use for an application invocation."""
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
