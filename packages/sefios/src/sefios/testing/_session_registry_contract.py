"""Reusable pytest contract for ``SessionRegistry`` implementations."""

from collections.abc import Callable
from typing import TypeAlias

from ..sessions import SessionRegistry

SessionRegistryFactory: TypeAlias = Callable[[], SessionRegistry]


class SessionRegistryContract:
    """Shared registration behavior required by session registries."""

    def test_registers_a_session_across_reopened_handles(
        self, session_registry_factory: SessionRegistryFactory
    ) -> None:
        registry = session_registry_factory()
        assert not registry.session_exists("session-1")

        registry.register_session("session-1")

        assert session_registry_factory().session_exists("session-1")

    def test_registering_twice_is_idempotent(
        self, session_registry_factory: SessionRegistryFactory
    ) -> None:
        registry = session_registry_factory()

        registry.register_session("session-1")
        registry.register_session("session-1")

        assert session_registry_factory().session_exists("session-1")

    def test_creates_unique_registered_sessions(
        self, session_registry_factory: SessionRegistryFactory
    ) -> None:
        registry = session_registry_factory()

        first = registry.create_session()
        second = registry.create_session()

        assert first != second
        assert registry.session_exists(first)
        assert registry.session_exists(second)


__all__ = ["SessionRegistryContract", "SessionRegistryFactory"]
