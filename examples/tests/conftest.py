import pytest
from glyff.stores import MemoryClient
from glyff_pydantic import PydanticSerializer
from sefia.stores import MemorySessionStore


@pytest.fixture
def memory_client() -> MemoryClient:
    """The transactional in-memory client backing the session store.

    Writes are staged until ``commit_staged`` is awaited, which mirrors how
    Sefia commits at transaction boundaries.
    """
    return MemoryClient()


@pytest.fixture
def session_store(memory_client: MemoryClient) -> MemorySessionStore:
    """An in-memory Sefia session store for exercising session-bound logic."""
    return MemorySessionStore(client=memory_client, serializer=PydanticSerializer())
