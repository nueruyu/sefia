import pytest
from sefios.sessions import (
    MemoryActiveSessionStore,
    MemorySessionRegistry,
    SessionManager,
    UnknownSessionError,
)


@pytest.fixture
def manager() -> SessionManager:
    return SessionManager(
        MemorySessionRegistry(),
        MemoryActiveSessionStore(),
    )


def test_no_active_session_initially(manager: SessionManager):
    assert manager.get_active_session_id() is None


def test_create_new_active_session_sets_active_and_registers(manager: SessionManager):
    session_id = manager.create_new_active_session()

    assert manager.get_active_session_id() == session_id
    assert manager.session_exists(session_id)


def test_create_new_session_ids_are_unique(manager: SessionManager):
    first = manager.create_new_active_session()
    second = manager.create_new_active_session()

    assert first != second
    assert manager.session_exists(first)
    assert manager.session_exists(second)


def test_switch_to_known_session(manager: SessionManager):
    target = manager.create_new_active_session()
    other = manager.create_new_active_session()
    assert manager.get_active_session_id() == other

    switched = manager.switch_active_session(target)

    assert switched == target
    assert manager.get_active_session_id() == target


def test_switch_to_unknown_session_raises(manager: SessionManager):
    with pytest.raises(UnknownSessionError) as exc_info:
        manager.switch_active_session("does-not-exist")

    assert exc_info.value.session_id == "does-not-exist"


def test_explicit_known_session(manager: SessionManager):
    session_id = manager.create_new_active_session()

    resolved = manager.resolve_session(session_id)

    assert resolved.session_id == session_id
    assert resolved.is_new is False
    assert resolved.source == "explicit"


def test_explicit_unknown_session_raises(manager: SessionManager):
    with pytest.raises(UnknownSessionError):
        manager.resolve_session("unknown")


def test_falls_back_to_active_session(manager: SessionManager):
    session_id = manager.create_new_active_session()

    resolved = manager.resolve_session(None)

    assert resolved.session_id == session_id
    assert resolved.is_new is False
    assert resolved.source == "active"


def test_creates_session_when_none_active(manager: SessionManager):
    resolved = manager.resolve_session(None)

    assert resolved.is_new is True
    assert resolved.source == "created"
    assert manager.get_active_session_id() == resolved.session_id
    assert manager.session_exists(resolved.session_id)


def test_dangling_active_session_raises(manager: SessionManager) -> None:
    manager.set_active_session_id("ghost")
    with pytest.raises(UnknownSessionError) as exc_info:
        manager.resolve_session(None)
    assert exc_info.value.session_id == "ghost"


def test_manager_reads_and_updates_the_supplied_active_store() -> None:
    registry = MemorySessionRegistry()
    first, second = registry.create_session(), registry.create_session()
    active = MemoryActiveSessionStore()
    active.set_active_session_id(first)
    manager = SessionManager(registry, active)

    assert manager.resolve_session(None).session_id == first
    manager.switch_active_session(second)
    assert active.get_active_session_id() == second
    active.set_active_session_id(first)
    assert manager.get_active_session_id() == first
