"""Reusable pytest contract for ``PersistenceProvider`` implementations."""

from glyff import Backend

from ..persistence import PersistenceProvider
from ..sessions import SessionRegistry
from ..storage import SessionStorage


class PersistenceProviderContract:
    """Shared resource-coherence behavior required by persistence providers."""

    def test_creates_each_persistence_component(
        self, persistence_provider: PersistenceProvider
    ) -> None:
        assert isinstance(persistence_provider.create_execution_backend(), Backend)
        assert isinstance(
            persistence_provider.create_session_storage("session"), SessionStorage
        )
        assert isinstance(
            persistence_provider.create_session_registry(), SessionRegistry
        )

    async def test_reuses_data_for_one_session_and_isolates_another(
        self, persistence_provider: PersistenceProvider
    ) -> None:
        writer = persistence_provider.create_session_storage("first")
        await writer.set("state", {"value": "kept"}, dict)

        assert await persistence_provider.create_session_storage("first").get(
            "state", dict
        ) == {"value": "kept"}
        assert (
            await persistence_provider.create_session_storage("second").get(
                "state", dict
            )
            is None
        )

    def test_reuses_the_session_registry(
        self, persistence_provider: PersistenceProvider
    ) -> None:
        persistence_provider.create_session_registry().register_session("session")

        assert persistence_provider.create_session_registry().session_exists("session")


__all__ = ["PersistenceProviderContract"]
