from pathlib import Path

import pytest
from sefios.sessions import (
    FileActiveSessionStore,
    MemorySessionRegistry,
    SessionManager,
)


@pytest.fixture
def manager(tmp_path: Path) -> SessionManager:
    return SessionManager(
        MemorySessionRegistry(),
        FileActiveSessionStore(tmp_path / "sessions" / "active_session.txt"),
    )


def test_creates_session_directory(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions"
    assert not session_dir.exists()

    FileActiveSessionStore(session_dir / "active_session.txt")

    assert session_dir.is_dir()


def test_creates_nested_session_directory(tmp_path: Path) -> None:
    session_dir = tmp_path / "var" / "sefia" / "sessions"
    assert not session_dir.parent.exists()

    FileActiveSessionStore(session_dir / "active_session.txt")

    assert session_dir.is_dir()


def test_active_session_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "sessions" / "active_session.txt"
    FileActiveSessionStore(path).set_active_session_id("session-1")
    assert FileActiveSessionStore(path).get_active_session_id() == "session-1"


def test_blank_active_session_file_reads_as_none(tmp_path: Path) -> None:
    path = tmp_path / "sessions" / "active_session.txt"
    store = FileActiveSessionStore(path)
    path.write_text("   ", encoding="utf-8")
    assert store.get_active_session_id() is None
