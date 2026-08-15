from abc import ABC, abstractmethod
from pathlib import Path
from typing import final

from typing_extensions import override


class ActiveSessionStore(ABC):
    """Stores the session selected by one local CLI workspace."""

    @abstractmethod
    def get_active_session_id(self) -> str | None: ...

    @abstractmethod
    def set_active_session_id(self, session_id: str) -> None: ...


@final
class MemoryActiveSessionStore(ActiveSessionStore):
    def __init__(self) -> None:
        self._session_id: str | None = None

    @override
    def get_active_session_id(self) -> str | None:
        return self._session_id

    @override
    def set_active_session_id(self, session_id: str) -> None:
        self._session_id = session_id


@final
class FileActiveSessionStore(ActiveSessionStore):
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @override
    def get_active_session_id(self) -> str | None:
        try:
            session_id = self._path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        return session_id or None

    @override
    def set_active_session_id(self, session_id: str) -> None:
        self._path.write_text(session_id, encoding="utf-8")
