from pathlib import Path

from sefios import SQLitePersistence


def test_sqlite_persistence_has_a_project_local_default() -> None:
    assert SQLitePersistence().database == Path(".sefios/sessions.sqlite3")
