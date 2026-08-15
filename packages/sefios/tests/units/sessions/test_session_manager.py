from pathlib import Path

import pytest
from sefios.sessions import (
    FileActiveSessionStore,
    MemorySessionRegistry,
    SessionManager,
    UnknownSessionError,
)


@pytest.fixture
def manager(tmp_path: Path) -> SessionManager:
    return SessionManager(
        MemorySessionRegistry(),
        FileActiveSessionStore(tmp_path / "sessions" / "active_session.txt"),
    )


class TestSessionManager:
    def test_creates_session_directory(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sessions"
        assert not session_dir.exists()

        FileActiveSessionStore(session_dir / "active_session.txt")

        assert session_dir.is_dir()

    def test_creates_nested_session_directory(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "var" / "sefia" / "sessions"
        assert not session_dir.parent.exists()

        FileActiveSessionStore(session_dir / "active_session.txt")

        assert session_dir.is_dir()

    def test_no_active_session_initially(self, manager: SessionManager):
        assert manager.get_active_session_id() is None

    def test_create_new_active_session_sets_active_and_registers(
        self, manager: SessionManager
    ):
        session_id = manager.create_new_active_session()

        assert manager.get_active_session_id() == session_id
        assert manager.session_exists(session_id)

    def test_create_new_session_ids_are_unique(self, manager: SessionManager):
        first = manager.create_new_active_session()
        second = manager.create_new_active_session()

        assert first != second
        assert manager.session_exists(first)
        assert manager.session_exists(second)

    def test_active_session_persists_across_instances(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sessions"
        registry = MemorySessionRegistry()
        session_id = SessionManager(
            registry, FileActiveSessionStore(session_dir / "active_session.txt")
        ).create_new_active_session()

        reopened = SessionManager(
            registry, FileActiveSessionStore(session_dir / "active_session.txt")
        )

        assert reopened.get_active_session_id() == session_id
        assert reopened.session_exists(session_id)

    def test_blank_active_session_file_reads_as_none(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sessions"
        manager = SessionManager(
            MemorySessionRegistry(),
            FileActiveSessionStore(session_dir / "active_session.txt"),
        )
        (session_dir / "active_session.txt").write_text("   ", encoding="utf-8")

        assert manager.get_active_session_id() is None

    def test_switch_to_known_session(self, manager: SessionManager):
        target = manager.create_new_active_session()
        other = manager.create_new_active_session()
        assert manager.get_active_session_id() == other

        switched = manager.switch_active_session(target)

        assert switched == target
        assert manager.get_active_session_id() == target

    def test_switch_to_unknown_session_raises(self, manager: SessionManager):
        with pytest.raises(UnknownSessionError) as exc_info:
            manager.switch_active_session("does-not-exist")

        assert exc_info.value.session_id == "does-not-exist"


class TestResolveSession:
    def test_explicit_known_session(self, manager: SessionManager):
        session_id = manager.create_new_active_session()

        resolved = manager.resolve_session(session_id)

        assert resolved.session_id == session_id
        assert resolved.is_new is False
        assert resolved.source == "explicit"

    def test_explicit_unknown_session_raises(self, manager: SessionManager):
        with pytest.raises(UnknownSessionError):
            manager.resolve_session("unknown")

    def test_falls_back_to_active_session(self, manager: SessionManager):
        session_id = manager.create_new_active_session()

        resolved = manager.resolve_session(None)

        assert resolved.session_id == session_id
        assert resolved.is_new is False
        assert resolved.source == "active"

    def test_creates_session_when_none_active(self, manager: SessionManager):
        resolved = manager.resolve_session(None)

        assert resolved.is_new is True
        assert resolved.source == "created"
        assert manager.get_active_session_id() == resolved.session_id
        assert manager.session_exists(resolved.session_id)

    def test_dangling_active_session_raises(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sessions"
        manager = SessionManager(
            MemorySessionRegistry(),
            FileActiveSessionStore(session_dir / "active_session.txt"),
        )
        # Point the active session file at an unregistered id.
        manager.set_active_session_id("ghost")

        with pytest.raises(UnknownSessionError) as exc_info:
            manager.resolve_session(None)

        assert exc_info.value.session_id == "ghost"
