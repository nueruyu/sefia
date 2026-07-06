import pytest
from glyff_pydantic import PydanticSerializer
from sefia.stores import MemorySessionStore


@pytest.fixture
def session_store() -> MemorySessionStore:
    """An in-memory Sefia session store for exercising session-bound logic.

    Writes take effect immediately, so state set before a pause is visible when
    the session resumes.
    """
    return MemorySessionStore(serializer=PydanticSerializer())
