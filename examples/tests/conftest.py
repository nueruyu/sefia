import pytest
from glyff_pydantic import PydanticSerializer
from sefios.storage import MemorySessionStorage


@pytest.fixture
def session_storage() -> MemorySessionStorage:
    """An in-memory Sefia session storage for exercising session-bound logic.

    Writes take effect immediately, so state set before a pause is visible when
    the session resumes.
    """
    return MemorySessionStorage(serializer=PydanticSerializer())
