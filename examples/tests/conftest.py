import sys
from pathlib import Path

import pytest
from glyff.stores import MemoryClient
from glyff_pydantic import PydanticSerializer
from sefia.stores import MemorySessionStore

# Make the repository root importable so example entry points such as
# ``examples.01_news_article.main`` resolve. Their ``from .._common ...`` imports
# require the full ``examples`` package, which only exists when the repo root is
# on ``sys.path``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


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
