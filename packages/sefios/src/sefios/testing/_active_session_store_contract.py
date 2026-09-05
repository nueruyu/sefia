"""Reusable pytest contract for ``ActiveSessionStore`` implementations."""

from collections.abc import Callable
from typing import TypeAlias

from ..sessions import ActiveSessionStore

ActiveSessionStoreFactory: TypeAlias = Callable[[], ActiveSessionStore]


class ActiveSessionStoreContract:
    """Shared selection behavior required by active-session stores."""

    def test_is_empty_initially(
        self, active_session_store_factory: ActiveSessionStoreFactory
    ) -> None:
        assert active_session_store_factory().get_active_session_id() is None

    def test_stores_and_replaces_the_selection_across_reopened_handles(
        self, active_session_store_factory: ActiveSessionStoreFactory
    ) -> None:
        active_session_store_factory().set_active_session_id("first")
        assert active_session_store_factory().get_active_session_id() == "first"

        active_session_store_factory().set_active_session_id("second")
        assert active_session_store_factory().get_active_session_id() == "second"


__all__ = ["ActiveSessionStoreContract", "ActiveSessionStoreFactory"]
