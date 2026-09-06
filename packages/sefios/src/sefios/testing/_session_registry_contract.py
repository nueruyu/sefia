"""Reusable pytest contract for ``SessionRegistry`` implementations."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeAlias

from ..sessions import SessionRegistry

SessionRegistryFactory: TypeAlias = Callable[[], SessionRegistry]


class SessionRegistryContract(ABC):
    """Shared registration behavior required by session registries."""

    @abstractmethod
    def make_session_registry(self) -> SessionRegistry:
        """Reopen the same registry, initially empty for each test."""
        ...

    def test_registers_a_session_across_reopened_handles(self) -> None:
        registry = self.make_session_registry()
        assert not registry.session_exists("session-1")

        registry.register_session("session-1")

        assert self.make_session_registry().session_exists("session-1")

    def test_registering_twice_is_idempotent(self) -> None:
        registry = self.make_session_registry()

        registry.register_session("session-1")
        registry.register_session("session-1")

        assert self.make_session_registry().session_exists("session-1")

    def test_creates_unique_registered_sessions(self) -> None:
        registry = self.make_session_registry()

        first = registry.create_session()
        second = registry.create_session()

        assert first != second
        assert registry.session_exists(first)
        assert registry.session_exists(second)


__all__ = ["SessionRegistryContract", "SessionRegistryFactory"]
