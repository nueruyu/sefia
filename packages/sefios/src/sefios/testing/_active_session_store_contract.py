"""Reusable pytest contract for ``ActiveSessionStore`` implementations."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeAlias

from ..sessions import ActiveSessionStore

ActiveSessionStoreFactory: TypeAlias = Callable[[], ActiveSessionStore]


class ActiveSessionStoreContract(ABC):
    """Shared selection behavior required by active-session stores."""

    @abstractmethod
    def make_active_session_store(self) -> ActiveSessionStore:
        """Reopen the same selection store, initially empty for each test."""
        ...

    def test_is_empty_initially(self) -> None:
        assert self.make_active_session_store().get_active_session_id() is None

    def test_stores_and_replaces_the_selection_across_reopened_handles(self) -> None:
        self.make_active_session_store().set_active_session_id("first")
        assert self.make_active_session_store().get_active_session_id() == "first"

        self.make_active_session_store().set_active_session_id("second")
        assert self.make_active_session_store().get_active_session_id() == "second"


__all__ = ["ActiveSessionStoreContract", "ActiveSessionStoreFactory"]
